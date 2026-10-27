"""M6/M7 report: real-data PINNs (regional and zoned) against the baselines on the same readings.

    python scripts/check_m6.py [--stage val|test] [--ref-tag tb_off1]   -> reports/m6_real_pinn_<stage>.md, tables/...csv

--ref-tag: the regional PINN setting the others are compared with ("wells improved", p-value); default: first found.
A shared zoned model (run_real.py --shared, E20) is one run covering every zone.

Groups run folders reports/runs/*_real_* by (layout, model, tag, seed). A zoned model is complete when every zone
of its zoning has a run with the same tag and seed; its zone predictions are stitched into one set.
Every model is scored only on readings that all compared models predicted (inner join), so differences come from
the models, not from different rows.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import fraction_improved, paired_wilcoxon, per_well, rmse, summary  # noqa: E402
from src.utils.config import data_dir, runs_dir  # noqa: E402

BASELINE_BEST = {"A": ["persistence", "seasonal trend"], "B": ["kriging", "depth kriging", "LightGBM"],
                 "C": ["trend + kriging", "depth trend + kriging", "LightGBM"]}


def md_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return (["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
            + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)])


def collect_runs() -> pd.DataFrame:
    rows = []
    for d in sorted(runs_dir().glob("*_real_*")):
        if not (d / "metrics.json").exists() or not (d / "predictions.parquet").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        name = d.name.split("_real_", 1)[1]
        tag = name.split(f"_s{cfg['seed']}", 1)[1].lstrip("_") if f"_s{cfg['seed']}" in name else ""
        model = ("regional" if cfg["zone"] is None and not cfg.get("shared")
                 else f"{cfg['method']}k{cfg['k']}" + (" shared" if cfg.get("shared") else ""))
        m = json.loads((d / "metrics.json").read_text())
        rows.append({"dir": d, "layout": cfg["layout"], "model": model, "zone": cfg["zone"], "seed": cfg["seed"],
                     "tag": tag, "S": m["S"], "T": m["T_fixed"], "pump": m["pumping_2005_2018_m_per_yr"],
                     "n_wells": m["n_wells"], "train_rmse": m["train_rmse_m"]})
    return pd.DataFrame(rows)


def stitched(group: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(d / "predictions.parquet") for d in group.dir], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="val", choices=["val", "test"])
    ap.add_argument("--ref-tag", default=None)
    args = ap.parse_args()
    proc = data_dir() / "processed"
    runs = collect_runs()
    if runs.empty:
        print("no real-data runs yet")
        return
    zsum = json.loads((proc / "zoning_summary.json").read_text(encoding="utf-8"))

    models = {}
    for (layout, model, tag, seed), g in runs.groupby(["layout", "model", "tag", "seed"]):
        if model != "regional" and not model.endswith("shared"):
            mth, k = re.match(r"(\w+?)k(\d+)$", model).groups()
            need = zsum["layouts"][str(layout)]["methods"][mth]["zones"][k]["k"]
            if g.zone.nunique() < need:
                print(f"skip {model} {tag} seed {seed}: {g.zone.nunique()} of {need} zones done")
                continue
            g = g.sort_values("dir").drop_duplicates("zone", keep="last")
        p = stitched(g)
        p = p[p.stage == args.stage]
        if p.empty:  # e.g. runs that were never predicted for the test stage
            continue
        models[(layout, model, tag, seed)] = (p, g)

    rows, pw_rows = [], []
    for layout in sorted({k[0] for k in models}):
        base = pd.read_parquet(proc / "predictions" / f"baselines_L{layout}_{args.stage}.parquet")
        for task in ("A", "B", "C"):
            keys = [k for k in models if k[0] == layout]
            frames = {f"PINN {k[1]}{' ' + k[2] if k[2] else ''} (seed {k[3]})": models[k][0][models[k][0].task == task]
                      for k in keys}
            # ensemble mean over seeds for each (model, tag)
            for (mdl, tag) in {(k[1], k[2]) for k in keys}:
                seeds = [models[k][0][models[k][0].task == task] for k in keys if k[1] == mdl and k[2] == tag]
                if len(seeds) > 1:
                    cat = pd.concat(seeds)
                    frames[f"PINN {mdl}{' ' + tag if tag else ''} (mean of {len(seeds)} seeds)"] = (
                        cat.groupby(["well_id", "date"], as_index=False).agg(head_m=("head_m", "first"),
                                                                            pred=("pred", "mean")))
            b = base[base.task == task]
            for col in BASELINE_BEST[task]:
                frames[f"baseline: {col}"] = b[["well_id", "date", "head_m", col]].rename(columns={col: "pred"})
            common = None
            for f in frames.values():
                k = f.dropna(subset=["pred"])[["well_id", "date"]].drop_duplicates()
                common = k if common is None else common.merge(k)
            if common is None or common.empty:
                continue
            ref_pre = "PINN regional" if args.ref_tag is None else f"PINN regional {args.ref_tag} ("
            ref_name = next((n for n in frames if n.startswith(ref_pre)), None)
            ref_pw = None
            for name, f in frames.items():
                f = f.merge(common).drop_duplicates(["well_id", "date"])
                s = summary(f, "pred")
                pw = per_well(f, "pred")
                if name == ref_name:
                    ref_pw = pw
                rows.append({"layout": layout, "task": task, "model": name, **s, "_pw": pw})
            for r in rows:
                if r["layout"] == layout and r["task"] == task and ref_pw is not None and r["model"] != ref_name:
                    r["vs_regional_wells_improved"] = fraction_improved(r["_pw"].rmse, ref_pw.rmse)
                    r["vs_regional_p"] = paired_wilcoxon(r["_pw"].rmse, ref_pw.rmse)[1]

    res = pd.DataFrame([{k: v for k, v in r.items() if k != "_pw"} for r in rows])
    ref_label = "regional PINN" + ("" if args.ref_tag is None else f" {args.ref_tag}")
    tab = ROOT / "reports" / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    res.to_csv(tab / f"m6_real_pinn_{args.stage}.csv", index=False)
    show = res.assign(**{"RMSE": res.rmse_m.map("{:.2f} m".format), "MAE": res.mae_m.map("{:.2f} m".format),
                         "Median well RMSE": res.median_well_rmse_m.map("{:.2f} m".format),
                         f"Wells improved vs {ref_label}": res.get("vs_regional_wells_improved", pd.Series(np.nan, index=res.index)).map(
                             lambda v: "" if pd.isna(v) else f"{100 * v:.0f}%"),
                         "p-value": res.get("vs_regional_p", pd.Series(np.nan, index=res.index)).map(
                             lambda v: "" if pd.isna(v) else f"{v:.3f}"),
                         "Readings / wells": res.n_readings.astype(str) + " / " + res.n_wells.astype(str)})
    show = show[["layout", "task", "model", "RMSE", "MAE", "Median well RMSE", f"Wells improved vs {ref_label}",
                 "p-value", "Readings / wells"]]
    par = runs[["layout", "model", "tag", "seed", "zone", "S", "T", "pump", "n_wells", "train_rmse"]].copy()
    par["S"] = par.S.map("{:.3f}".format)
    par["T"] = par["T"].map("{:.0f}".format)
    par["pump"] = par.pump.map("{:.2f}".format)
    par["train_rmse"] = par.train_rmse.map("{:.2f}".format)
    md = [
        f"# M6/M7 report: PINN on real data ({args.stage})",
        "",
        f"*Generated by `scripts/check_m6.py --stage {args.stage}`.* Every model is scored on the same readings (those all of them predicted).",
        "",
        "## Water-level errors",
        "",
        *md_table(show),
        "",
        "## Learned parameters per run",
        "",
        "S = storage (GEC norm for alluvium: 0.12; NAQUIM range 0.07–0.15). T is fixed from the CGWB reports.",
        "*pump* = average zone pumping in 2005–2018, in metres of water per year over the zone's area: an estimate, not a measurement.",
        "",
        *md_table(par),
        "",
        "## How to read it",
        "",
        "- Lower RMSE / MAE = better. *Wells improved* = share of wells whose own error is smaller than with the regional PINN.",
        "- *p-value*: chance of improvements this consistent if the two models were equally good (Wilcoxon). Below 0.05 = convincing.",
        "- Task A uses the well's learned offset (it is monitored); tasks B and C cannot (no well there).",
    ]
    out = ROOT / "reports" / f"m6_real_pinn_{args.stage}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(show.to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
