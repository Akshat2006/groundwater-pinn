"""Early zoning preview (PLAN.md §5 M4): does zoning help prediction in places the models have not seen?

    python scripts/zoning_preview.py      -> reports/zoning_preview.md, reports/tables/zoning_preview.csv

Validation blocks only, training years only (Task B, validation stage), pooled over the 3 block layouts.
The test blocks are never touched here.

Models:
- bucket model on level changes (primary; src/models/baselines/bucket.py): one fit for the whole region vs one fit
  per zone. It is the simplest model with the PINN's water balance, so it shows whether zone-specific parameters
  carry over to unseen places.
- random forest on water levels (secondary): without vs with the zone as an extra input.

Decision rule, fixed before running:
1. Candidates: every zoning method and k in configs/zoning.yaml (k = 3, 4, 5 and each layout's chosen k).
2. Score = pooled RMSE of the bucket model's level changes at validation wells.
3. Pick the candidate with the lowest score. If another candidate is within 1% of it, prefer fewer zones.
4. If no candidate beats the unzoned bucket by at least 2%, report "no gain from zoning" (risk register).

Added after the first run (reported as such): the primary score turned out uninformative, because no model
beats "no change" on single between-reading changes (reading noise dominates). A long-horizon check is therefore
also reported: each validation well's change since its first reading, where a wrong decline rate adds up over the
years. It is shown next to the rule's result and does not override it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import fraction_improved, paired_wilcoxon, per_well, rmse  # noqa: E402
from src.evaluation.tasks import shallow_levels, target_rows, training_rows, with_layout  # noqa: E402
from src.models.baselines import bucket, trees  # noqa: E402
from src.utils.config import data_dir, load_config  # noqa: E402

CANDIDATE_K = [3, 4, 5, "chosen"]


def md_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return (["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
            + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)])


def main() -> None:
    proc = data_dir() / "processed"
    zcfg = load_config(ROOT / "configs" / "zoning.yaml")
    scfg = load_config(ROOT / "configs" / "splits.yaml")
    master = pd.read_parquet(proc / "master.parquet")
    splits = pd.read_parquet(proc / "splits.parquet")
    assign = pd.read_parquet(proc / "zone_assign.parquet")
    summ = json.loads((proc / "zoning_summary.json").read_text(encoding="utf-8"))
    levels = shallow_levels(master)

    rain = xr.open_dataset(proc / "rain_daily.nc").rain_mm.load()
    rif = load_config(ROOT / "configs" / "study_area.yaml")["recharge"]["rif"]
    per_layout = {}
    for layout in scfg["layouts"]:
        df = with_layout(levels, splits, layout)
        tr, va = training_rows(df), target_rows(df, "B", "val")
        changes = bucket.level_changes(pd.concat([tr, va]))
        per_layout[layout] = {"tr": tr, "va": va, "ch_tr": changes[changes.split_space == "train"],
                              "ch_va": changes[changes.split_space == "val"],
                              "lh_tr": bucket.changes_from_first(tr, rain, rif),
                              "lh_va": bucket.changes_from_first(va, rain, rif)}

    def zones(method, layout, k):
        a = assign[(assign.method == method) & (assign.layout == layout) & (assign.k == k)]
        return a.set_index("well_id").zone_map

    cands = [("none", "unzoned", 1)]
    for method in zcfg["methods"]:
        for k in CANDIDATE_K:
            cands.append((method, f"k={k}", k))

    rows, preds_store = [], {}
    for method, label, k in cands:
        b_pred, rf_pred, lh_pred, n_zones, maps_seen = [], [], [], [], []
        for layout, d in per_layout.items():
            kk = summ["layouts"][str(layout)]["methods"][method]["chosen_k"] if k == "chosen" else k
            if method == "none":
                zc = None
                ch_tr, ch_va, tr, va = d["ch_tr"], d["ch_va"], d["tr"], d["va"]
                n_zones.append(1)
            else:
                z = zones(method, layout, kk)
                ch_tr = d["ch_tr"].assign(zone=d["ch_tr"].well_id.map(z))
                ch_va = d["ch_va"].assign(zone=d["ch_va"].well_id.map(z))
                tr = d["tr"].assign(**{f"zone_{i}": (d["tr"].well_id.map(z) == i).astype(float) for i in range(1, 9)})
                va = d["va"].assign(**{f"zone_{i}": (d["va"].well_id.map(z) == i).astype(float) for i in range(1, 9)})
                zc = "zone"
                n_zones.append(summ["layouts"][str(layout)]["methods"][method]["zones"][str(kk)]["k"])
                maps_seen.append(kk)
            p, _ = bucket.fit_predict_zoned(ch_tr, ch_va, zc)
            b_pred.append(ch_va.assign(pred=p.values, layout=layout))
            lh_tr, lh_va = d["lh_tr"], d["lh_va"]
            if zc:
                lh_tr, lh_va = lh_tr.assign(zone=lh_tr.well_id.map(z)), lh_va.assign(zone=lh_va.well_id.map(z))
            p, _ = bucket.fit_predict_zoned(lh_tr, lh_va, zc)
            lh_pred.append(lh_va.assign(pred=p.values, layout=layout))
            extra = None if zc is None else [f"zone_{i}" for i in range(1, 9)]
            rf_pred.append(va.assign(pred=trees.fit_predict(tr, va, "forest", seed=0, extra=extra).values,
                                     layout=layout))
        b = pd.concat(b_pred)
        r = pd.concat(rf_pred)
        lh = pd.concat(lh_pred)
        b["well_layout"] = b.well_id + "@" + b.layout.astype(str)
        r["well_layout"] = r.well_id + "@" + r.layout.astype(str)
        preds_store[(method, label)] = (b, r)
        rows.append({"method": method, "k": label, "zones (layouts 0/1/2)": "/".join(map(str, n_zones)),
                     "bucket_rmse": rmse(b.dh, b.pred), "rf_rmse": rmse(r.head_m, r.pred),
                     "longterm_rmse": rmse(lh.dh, lh.pred), "longterm_zero_rmse": rmse(lh.dh, 0 * lh.dh),
                     "zero_change_rmse": rmse(b.dh, 0 * b.dh),
                     "n_changes": len(b), "n_levels": len(r)})

    res = pd.DataFrame(rows)
    base = res.iloc[0]
    res["bucket_gain_pct"] = 100 * (1 - res.bucket_rmse / base.bucket_rmse)
    res["rf_gain_pct"] = 100 * (1 - res.rf_rmse / base.rf_rmse)
    res["longterm_gain_pct"] = 100 * (1 - res.longterm_rmse / base.longterm_rmse)
    b0, r0 = preds_store[("none", "unzoned")]
    pw0 = per_well(b0.rename(columns={"well_id": "w", "well_layout": "well_id"}), "pred", "dh")
    fi, pv = [], []
    for (m, lab), (b, _) in preds_store.items():
        pw = per_well(b.rename(columns={"well_id": "w", "well_layout": "well_id"}), "pred", "dh")
        fi.append(fraction_improved(pw.rmse, pw0.rmse) if m != "none" else np.nan)
        pv.append(paired_wilcoxon(pw.rmse, pw0.rmse)[1] if m != "none" else np.nan)
    res["wells_improved"] = fi
    res["wilcoxon_p"] = pv

    zoned = res[res.method != "none"].copy()
    # dedupe candidates that produced the same zones (e.g. contiguous k=4,5 often merge to the same map)
    best = zoned.bucket_rmse.min()
    near = zoned[zoned.bucket_rmse <= best * 1.01].copy()
    near["nz"] = near["zones (layouts 0/1/2)"].str.split("/").apply(lambda v: np.mean(list(map(int, v))))
    pick = near.sort_values(["nz", "bucket_rmse"]).iloc[0]
    gain = float(pick.bucket_gain_pct)
    verdict = ("zoning helps" if gain >= 5 else "small gain from zoning" if gain >= 2 else "no gain from zoning")
    k_pick = pick.k.replace("k=", "")
    out = {"method": pick.method, "k": k_pick if k_pick == "chosen" else int(k_pick), "bucket_gain_pct": gain,
           "verdict": verdict, "longterm_gain_pct": float(pick.longterm_gain_pct),
           "longterm_best": res.loc[res.longterm_rmse.idxmin(), ["method", "k"]].tolist()}
    (proc / "zoning_preview.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    tab = (ROOT / "reports" / "tables")
    tab.mkdir(parents=True, exist_ok=True)
    res.to_csv(tab / "zoning_preview.csv", index=False)

    show = res.assign(**{
        "Bucket: level-change error": res.bucket_rmse.map(lambda v: f"{v:.3f} m"),
        "Bucket: vs unzoned": res.bucket_gain_pct.map(lambda v: f"{v:+.1f}%"),
        "Bucket: wells improved": res.wells_improved.map(lambda v: "" if np.isnan(v) else f"{100 * v:.0f}%"),
        "Bucket: p-value": res.wilcoxon_p.map(lambda v: "" if np.isnan(v) else f"{v:.3f}"),
        "Forest: level error": res.rf_rmse.map(lambda v: f"{v:.2f} m"),
        "Forest: vs unzoned": res.rf_gain_pct.map(lambda v: f"{v:+.1f}%"),
        "Over the years: error": res.longterm_rmse.map(lambda v: f"{v:.2f} m"),
        "Over the years: vs unzoned": res.longterm_gain_pct.map(lambda v: f"{v:+.1f}%")})
    long_show = show[["method", "k", "zones (layouts 0/1/2)", "Over the years: error", "Over the years: vs unzoned"]]
    show = show[["method", "k", "zones (layouts 0/1/2)", "Bucket: level-change error", "Bucket: vs unzoned",
                 "Bucket: wells improved", "Bucket: p-value", "Forest: level error", "Forest: vs unzoned"]]
    md = [
        "# Early zoning preview",
        "",
        "*Generated by `scripts/zoning_preview.py`. Validation blocks only (the test blocks are untouched), training years, pooled over the 3 block layouts.*",
        "",
        f"**Result: {verdict}.** The pre-set rule picks **{pick.method}, k = {k_pick}**: the bucket model's error on level changes at unseen wells is {gain:+.1f}% vs no zoning.",
        "",
        "## What was compared",
        "",
        f"- **Bucket model** (primary): water balance without sideways flow, S·Δh = recharge − pumping·cropland + background. It predicts how much a well's level changes between readings ({len(b0)} changes at validation wells). It is fitted once for the region, or once per zone.",
        f"- **Random forest** (secondary): predicts the level itself from location, land cover, time and rainfall ({len(r0)} readings), without or with the zone as an extra input.",
        "",
        "Each validation well gets its zone from the zone map, as an unmonitored place would.",
        "",
        *md_table(show),
        "",
        f"**Caution:** predicting *no change at all* scores {base.zero_change_rmse:.3f} m on this measure, about the same as every model. Single between-reading changes are dominated by reading noise and local pumping, so this score can hardly show any effect of zoning.",
        "",
        "## Added after the first run: change over the years",
        "",
        "Because the first measure turned out uninformative, a long-horizon check was added *after seeing it*. It does not override the rule above.",
        "",
        f"For each validation well, the bucket model predicts the change in level since the well's first reading, up to 14 years later. A wrong decline rate adds up here. Predicting no change scores {base.longterm_zero_rmse:.2f} m.",
        "",
        *md_table(long_show),
        "",
        "## How to read it",
        "",
        "- **Error columns:** typical error (RMSE) in metres; lower is better.",
        "- **vs unzoned:** how much lower the error is than with no zones. Positive = zoning helps.",
        "- **Wells improved:** share of validation wells whose own error got smaller.",
        "- **p-value:** chance of improvements this consistent if zoning really made no difference (Wilcoxon test). Below 0.05 is usually taken as convincing.",
        "",
        "## Decision rule (fixed in the script before it was run)",
        "",
        "1. The score is the bucket model's level-change error at validation wells.",
        "2. Pick the lowest score. If another candidate is within 1% of it, prefer fewer zones.",
        "3. Gain under 2% → *no gain from zoning* (risk register); 2–5% → *small gain*; 5% or more → *zoning helps*.",
    ]
    (ROOT / "reports" / "zoning_preview.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(show.to_string(index=False))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
