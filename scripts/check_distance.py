"""M8: error against distance to the nearest training well (CLAUDE.md §6.4 item 3), places without a well.

    python scripts/check_distance.py    -> reports/m8_distance.md, reports/figures/m8_error_vs_distance.png,
                                           reports/tables/m8_error_vs_distance.csv

Tasks B and C (held-out blocks), validation and test, the 3 layouts pooled. Models: v1.0 zoned and regional PINN
(5-seed ensemble means, tag tbc), LightGBM and depth kriging. Each model is scored on the same readings.
Descriptive only: nothing is chosen from it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_m7 import md_table  # noqa: E402
from src.evaluation.tasks import shallow_levels, training_rows, with_layout  # noqa: E402
from src.utils.config import data_dir, runs_dir  # noqa: E402

BINS = [0, 5, 10, 15, 20, 100]
LABELS = ["< 5 km", "5-10 km", "10-15 km", "15-20 km", "> 20 km"]
PINNS = {"zoned PINN": "kmeansk5_shared", "regional PINN": "regional"}
BASELINES = ["LightGBM", "depth kriging"]


def ensemble(layout: int, name: str) -> pd.DataFrame:
    dirs = sorted(d for d in runs_dir().glob(f"*_real_L{layout}_{name}_s*_tbc") if (d / "model.pt").exists())
    p = pd.concat(pd.read_parquet(d / "predictions.parquet") for d in dirs)
    p = p[p.task.isin(["B", "C"])]
    return p.groupby(["stage", "task", "well_id", "date"]).agg(head_m=("head_m", "first"), x_m=("x_m", "first"),
                                                               y_m=("y_m", "first"), pred=("pred", "mean")).reset_index()


def main() -> None:
    proc = data_dir() / "processed"
    levels = shallow_levels(pd.read_parquet(proc / "master.parquet"))
    splits = pd.read_parquet(proc / "splits.parquet")
    frames = []
    for L in (0, 1, 2):
        tr = training_rows(with_layout(levels, splits, L))
        tw = tr.groupby("well_id")[["x_m", "y_m"]].first()
        tree = cKDTree(tw.values)
        base = ensemble(L, PINNS["zoned PINN"])
        base["dist_km"] = tree.query(base[["x_m", "y_m"]].values)[0] / 1000.0
        rows = base.rename(columns={"pred": "zoned PINN"})
        reg = ensemble(L, PINNS["regional PINN"]).rename(columns={"pred": "regional PINN"})
        rows = rows.merge(reg[["stage", "task", "well_id", "date", "regional PINN"]], on=["stage", "task", "well_id", "date"])
        for stage in ("val", "test"):
            b = pd.read_parquet(proc / "predictions" / f"baselines_L{L}_{stage}.parquet").assign(stage=stage)
            b["well_id"] = b.well_id.astype(str)
            rows = rows.merge(b[["stage", "task", "well_id", "date"] + BASELINES], on=["stage", "task", "well_id", "date"],
                              how="left", suffixes=("", f"_{stage}"))
            for c in BASELINES:
                if f"{c}_{stage}" in rows:
                    rows[c] = rows[c].fillna(rows.pop(f"{c}_{stage}"))
        frames.append(rows.assign(layout=L))
    df = pd.concat(frames, ignore_index=True)
    models = list(PINNS) + BASELINES
    df = df.dropna(subset=list(PINNS) + ["LightGBM"])          # depth kriging exists for Task B only
    df["band"] = pd.cut(df.dist_km, BINS, labels=LABELS, right=False)

    out = []
    for (task, band), g in df.groupby(["task", "band"], observed=True):
        r = {"task": task, "distance": band, "readings": len(g), "wells": g.well_id.nunique()}
        r.update({m: round(float(np.sqrt(((g[m] - g.head_m) ** 2).mean())), 2) if g[m].notna().all() else np.nan
                  for m in models})
        r["zoned vs regional"] = f"{100 * (r['zoned PINN'] / r['regional PINN'] - 1):+.0f}%"
        out.append(r)
    tab = pd.DataFrame(out)
    tab.to_csv(ROOT / "reports" / "tables" / "m8_error_vs_distance.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"zoned PINN": "#2a6fdb", "regional PINN": "#8a8f98", "LightGBM": "#d9822b", "depth kriging": "#3e9e62"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, task in zip(axs, ("B", "C")):
        t = tab[tab.task == task]
        for m in models:
            ax.plot(t.distance.astype(str), t[m], marker="o", ms=5, lw=2, color=colors[m], label=m)
        ax.set_title({"B": "Task B: no well, training years", "C": "Task C: no well, later years"}[task], fontsize=10, loc="left")
        ax.set_xlabel("distance to the nearest training well")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("typical error, RMSE (m)")
    axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "m8_error_vs_distance.png", dpi=130)
    plt.close(fig)

    corr = df.groupby("task").apply(lambda g: g.dist_km.corr((g["zoned PINN"] - g.head_m).abs(), method="spearman"),
                                     include_groups=False)
    md = ["# M8: error against distance to the nearest training well", "",
          "*Generated by `scripts/check_distance.py`.* Places without a well (held-out blocks): Task B (2005-2018) and "
          "Task C (2019-2024), validation and test readings, the 3 block layouts pooled. Every model is scored on the "
          "same readings. PINNs are the v1.0 5-seed ensembles. Descriptive only: nothing was chosen from this.", "",
          *md_table(tab.fillna("")), "", "Depth kriging was only run for Task B. Bands with fewer than 20 wells (15-20 km) are shaky.", "", "![error vs distance](figures/m8_error_vs_distance.png)", "",
          f"Rank correlation between distance and the zoned PINN's error per reading: "
          f"B {corr.get('B', np.nan):.2f}, C {corr.get('C', np.nan):.2f} (0 = unrelated, 1 = error always grows with distance).", ""]
    path = ROOT / "reports" / "m8_distance.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
