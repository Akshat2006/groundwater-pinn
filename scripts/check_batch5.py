"""Batch 5: apply the pre-set rules (reports/batch5_rules.md) and report the M8 experiments, validation only.

    python scripts/check_batch5.py    -> reports/batch5_results.md, reports/tables/batch5_*.csv

Reference: v1.0 zoned runs (tag tbc), every seed. Variants: seed 0 on layouts 0, 1, 2.
Score = A / 1.90 + B / 7.94 (validation RMSE over all of a run's validation readings, as in batches 2-3).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_m7 import md_table, pumping_per_cell  # noqa: E402
from src.training.real_data import load_real_run  # noqa: E402
from src.utils.config import data_dir, runs_dir  # noqa: E402

LAYOUTS = (0, 1, 2)
GATE_SCORE_TOL, GATE_RHO, GATE_PB_HR = 0.02, 0.5, 1.3
YEARS_CGWB = range(2017, 2021)
# label: (folder pattern with {L}, description)
RUNS = {
    "v1.0 (tbc)": ("*_real_L{L}_kmeansk5_shared_s*_tbc", "v1.0 zoned, every seed (reference)"),
    "pq": ("*_real_L{L}_kmeansk5_shared_s0_pq", "paddy share as pumping input"),
    "pqs": ("*_real_L{L}_kmeansk5_shared_s0_pqs", "paddy input + S fixed at 0.12"),
    "nophys": ("*_real_L{L}_kmeansk5_shared_s0_nophys", "no physics (data only)"),
    "noq": ("*_real_L{L}_kmeansk5_shared_s0_noq", "no pumping (Q = 0)"),
    "k3": ("*_real_L{L}_kmeansk3_shared_s0_ksweep", "3 zones"),
    "k8": ("*_real_L{L}_kmeansk8_shared_s0_ksweep", "8 zones (k-means k = 8)"),
}


def dirs(label: str, layout: int) -> list[Path]:
    pat = RUNS[label][0].format(L=layout)
    return sorted(d for d in runs_dir().glob(pat) if (d / "metrics.json").exists())


def val_rmse(d: Path) -> dict:
    p = pd.read_parquet(d / "predictions.parquet")
    p = p[p.stage == "val"]
    return {t: float(np.sqrt(((g.pred - g.head_m) ** 2).mean())) for t, g in p.groupby("task")}


def accuracy() -> pd.DataFrame:
    rows = []
    for label in RUNS:
        for L in LAYOUTS:
            for d in dirs(label, L):
                r = val_rmse(d)
                rows.append({"run": label, "layout": L, "dir": d.name, **r, "score": r["A"] / 1.90 + r["B"] / 7.94,
                             "S_mean": float(np.mean(json.loads((d / "metrics.json").read_text())["S_zones"]))})
    return pd.DataFrame(rows)


def district_pumping(label: str, static: xr.Dataset) -> tuple[pd.Series, int]:
    """Mean pumping (m/yr, 2017-2020) per district over the variant's seed-0 runs on the 3 layouts."""
    yy, xx = np.meshgrid(static.y.values, static.x.values, indexing="ij")
    area = static.district_id.values > 0
    dist = static.district_id.values[area]
    per_run = []
    for L in LAYOUTS:
        for d in dirs(label, L):
            if label.startswith("v1.0") and not d.name.split("_s")[-1].startswith("0_"):
                continue                                     # same footing as the variants: seed 0 only
            model, _ = load_real_run(d, static)
            q = pumping_per_cell(model, xx[area], yy[area], YEARS_CGWB)
            per_run.append(pd.Series(q).groupby(dist).mean())
    return (pd.DataFrame(per_run).mean() if per_run else pd.Series(dtype=float)), len(per_run)


def main() -> None:
    proc = data_dir() / "processed"
    with xr.open_dataset(proc / "static_1km.nc") as f:
        static = f.load()
    acc = accuracy()
    ref = acc[acc.run == "v1.0 (tbc)"].groupby("layout")[["A", "B", "C", "score"]].mean()
    ref_score = float(ref.score.mean())

    cg = pd.read_csv(ROOT / "data" / "reference" / "cgwb_district_2020.csv")
    cells = pd.Series(static.district_id.values[static.district_id.values > 0]).value_counts()
    cg["cgwb_m_per_yr"] = cg.extraction_ham / (cg.district_id.map(cells) * 100.0)
    cg_state = cg.groupby("state").cgwb_m_per_yr.mean()

    summary, pump_cols = [], {}
    for label, (_, desc) in RUNS.items():
        a = acc[acc.run == label]
        if a.empty:
            continue
        by_l = a.groupby("layout")[["A", "B", "C", "score"]].mean()
        row = {"run": label, "what": desc, "layouts": len(by_l), "runs": len(a),
               **{f"val {t}": round(float(by_l[t].mean()), 2) for t in ("A", "B", "C")},
               "score": round(float(by_l.score.mean()), 3),
               "vs v1.0": f"{100 * (by_l.score.mean() / ref.score.loc[by_l.index].mean() - 1):+.1f}%",
               "S (mean)": round(float(a.S_mean.mean()), 3)}
        if label in ("v1.0 (tbc)", "pq", "pqs"):
            q, n = district_pumping(label, static)
            if len(q):
                cg[label] = cg.district_id.map(q)
                pump_cols[label] = n
                st = cg.groupby("state")[label].mean()
                rho, p = spearmanr(cg.cgwb_m_per_yr, cg[label])
                row.update({"pumping m/yr": round(float(q.mean()), 3), "rank corr. CGWB": round(float(rho), 2),
                            "p": round(float(p), 3), "Punjab/Haryana": round(float(st["Punjab"] / st["Haryana"]), 2)})
        summary.append(row)
    summ = pd.DataFrame(summary)

    # ---- the rule
    lines, chosen = [], None
    for label in ("pqs", "pq"):
        r = summ[summ.run == label]
        if r.empty or int(r.layouts.iloc[0]) < 3:
            lines.append(f"- `{label}`: not all 3 layouts finished, so the rule cannot be applied yet.")
            continue
        r = r.iloc[0]
        ok_acc = r.score <= ref_score * (1 + GATE_SCORE_TOL)
        ok_q = r["rank corr. CGWB"] >= GATE_RHO and r["Punjab/Haryana"] >= GATE_PB_HR
        lines.append(f"- `{label}`: score {r.score:.3f} (gate ≤ {ref_score * (1 + GATE_SCORE_TOL):.3f}: "
                     f"{'pass' if ok_acc else 'FAIL'}); rank correlation {r['rank corr. CGWB']:.2f} and Punjab/Haryana "
                     f"{r['Punjab/Haryana']:.2f} (gates ≥ {GATE_RHO}, ≥ {GATE_PB_HR}: {'pass' if ok_q else 'FAIL'}).")
        if ok_acc and ok_q and chosen is None:
            chosen = label
    incomplete = any("not all 3 layouts" in line for line in lines)
    verdict = (f"**Rule result: `{chosen}` becomes v2.**" if chosen else
               "**Rule not applied yet: waiting for runs.**" if incomplete else
               "**Rule result: no variant passes both gates; v1.0 stays.**")

    out_t = ROOT / "reports" / "tables"
    acc.to_csv(out_t / "batch5_runs.csv", index=False)
    summ.to_csv(out_t / "batch5_summary.csv", index=False)
    cg.to_csv(out_t / "batch5_pumping_by_district.csv", index=False)

    dist_cols = ["district", "state", "cgwb_m_per_yr"] + list(pump_cols)
    md = ["# Batch 5 results: pumping v2 and M8 experiments (validation only)", "",
          "*Generated by `scripts/check_batch5.py`; rules in `reports/batch5_rules.md`, written before the runs.*", "",
          "Typical error (RMSE, metres) on the validation set, mean over the 3 block layouts. "
          "Score = A / 1.90 + B / 7.94 (lower is better). The v1.0 row averages its 5 seeds; every other row is "
          "one seed (seed 0), which varies by about 1.5% on its own.", "",
          *md_table(summ.fillna("")), "",
          f"Reference score (v1.0, mean single run): {ref_score:.3f}.", "",
          "## Rule (pumping v2)", "", *lines, "", verdict, "",
          "## Pumping by district, 2017-2020 mean (m/yr; seed-0 runs of each layout)", "",
          "CGWB figures are gross extraction (estimates themselves); model pumping is net. Compare the order of "
          "districts, not the size. CGWB's estimates are partly built from crop areas, so agreement after adding "
          "paddy is partly by construction.", "",
          *md_table(cg[dist_cols].round(3).sort_values("cgwb_m_per_yr", ascending=False)), ""]
    path = ROOT / "reports" / "batch5_results.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("wrote", path)


if __name__ == "__main__":
    main()
