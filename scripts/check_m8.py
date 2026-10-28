"""M8: ablations (no physics, no pumping) and zone-count sweep, validation only (rules in reports/batch5_rules.md).

    python scripts/check_m8.py    -> reports/m8_experiments.md, reports/figures/m8_*.png, reports/tables/m8_*.csv

Reference: v1.0 zoned (tag tbc, 5 seeds). k = 1 is the regional PINN (tag tbc, 5 seeds). Variants: seed 0 on
layouts 0, 1, 2. Descriptive only: nothing is chosen from these runs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_m7 import md_table  # noqa: E402
from src.utils.config import runs_dir  # noqa: E402

LAYOUTS = (0, 1, 2)
TASKS = ("A", "B", "C")
# label: (folder pattern with {L}, description)
RUNS = {
    "v1.0 zoned": ("*_real_L{L}_kmeansk5_shared_s*_tbc", "v1.0: physics + pumping, k-means k = 5"),
    "no physics": ("*_real_L{L}_kmeansk5_shared_s0_nophys", "flow equation switched off (data only)"),
    "no pumping": ("*_real_L{L}_kmeansk5_shared_s0_noq", "pumping switched off (Q = 0)"),
    "k = 1 (regional)": ("*_real_L{L}_regional_s*_tbc", "one zone: the regional PINN"),
    "k = 3": ("*_real_L{L}_kmeansk3_shared_s0_ksweep", "k-means k = 3"),
    "k = 8": ("*_real_L{L}_kmeansk8_shared_s0_ksweep", "k-means k = 8"),
}
ABLATION = ["v1.0 zoned", "no physics", "no pumping"]
KSWEEP = ["k = 1 (regional)", "k = 3", "v1.0 zoned", "k = 8"]


def dirs(label: str, layout: int) -> list[Path]:
    pat = RUNS[label][0].format(L=layout)
    return sorted(d for d in runs_dir().glob(pat) if (d / "metrics.json").exists())


def val_preds(d: Path) -> pd.DataFrame:
    p = pd.read_parquet(d / "predictions.parquet")
    return p[p.stage == "val"]


def scores() -> pd.DataFrame:
    rows = []
    for label in RUNS:
        for L in LAYOUTS:
            for d in dirs(label, L):
                p = val_preds(d)
                r = {t: float(np.sqrt(((g.pred - g.head_m) ** 2).mean())) for t, g in p.groupby("task")}
                m = json.loads((d / "metrics.json").read_text())
                rows.append({"run": label, "layout": L, "dir": d.name, **r, "score": r["A"] / 1.90 + r["B"] / 7.94,
                             "zones": len(m.get("S_zones", [])) or 1})
    return pd.DataFrame(rows)


def per_well(label: str, layout: int, task: str, seed0: bool) -> pd.Series:
    """Validation RMSE per well; v1.0 restricted to seed 0 so every pair is seed 0 against seed 0."""
    ds = [d for d in dirs(label, layout) if not seed0 or "_s0_" in d.name]
    p = val_preds(ds[0])
    p = p[p.task == task]
    return p.groupby("well_id").apply(lambda g: np.sqrt(((g.pred - g.head_m) ** 2).mean()), include_groups=False)


def paired(label: str) -> dict:
    """Share of wells where the variant beats v1.0 (seed 0 vs seed 0), wells pooled over the 3 layouts."""
    out = {}
    for t in TASKS:
        a, b = [], []
        for L in LAYOUTS:
            v, r = per_well(label, L, t, True), per_well("v1.0 zoned", L, t, True)
            j = v.index.intersection(r.index)
            a.append(v[j]), b.append(r[j])
        a, b = pd.concat(a), pd.concat(b)
        out[t] = (float((a < b).mean()), float(wilcoxon(a, b).pvalue), len(a))
    return out


def summarise(acc: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    ref = acc[acc.run == "v1.0 zoned"].groupby("layout")[list(TASKS) + ["score"]].mean()
    rows = []
    for label in labels:
        a = acc[acc.run == label]
        by_l = a.groupby("layout")[list(TASKS) + ["score"]].mean()
        row = {"model": label, "what": RUNS[label][1], "runs": len(a),
               "zones (after merging)": "/".join(str(int(z)) for z in a.groupby("layout").zones.first()),
               **{f"val {t} (m)": round(float(by_l[t].mean()), 2) for t in TASKS},
               "score": round(float(by_l.score.mean()), 3),
               "vs v1.0": f"{100 * (by_l.score.mean() / ref.score.mean() - 1):+.1f}%"}
        if label != "v1.0 zoned":
            worse = (by_l.score > ref.score).sum()
            row["layouts worse than v1.0"] = f"{worse} of 3"
        rows.append(row)
    return pd.DataFrame(rows)


def figures(acc: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"A": "#2a6fdb", "B": "#d9822b", "C": "#3e9e62"}
    names = {"A": "A: wells, later year", "B": "B: no well, training years", "C": "C: no well, later year"}

    # zone-count sweep: one line per task, layout means as small dots
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ks = [1, 3, 5, 8]
    for t in TASKS:
        lay = [acc[acc.run == lab].groupby("layout")[t].mean() for lab in KSWEEP]
        ax.plot(ks, [v.mean() for v in lay], marker="o", ms=6, lw=2, color=colors[t], label=names[t])
        for k, v in zip(ks, lay):
            ax.scatter([k] * len(v), v, s=12, color=colors[t], alpha=0.35, lw=0)
    ax.set_xticks(ks, ["1\n(regional)", "3", "5\n(v1.0)", "8"])
    ax.set_xlabel("k-means zone count")
    ax.set_ylabel("typical error, validation RMSE (m)")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / "m8_zone_count.png", dpi=130)
    plt.close(fig)

    # ablations: grouped bars per task
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bar = {"v1.0 zoned": "#2a6fdb", "no physics": "#8a8f98", "no pumping": "#c4453c"}
    w = 0.26
    for i, lab in enumerate(ABLATION):
        v = [acc[acc.run == lab].groupby("layout")[t].mean().mean() for t in TASKS]
        xs = np.arange(len(TASKS)) + (i - 1) * w
        ax.bar(xs, v, width=w - 0.03, color=bar[lab], label=lab)
        for x, y in zip(xs, v):
            ax.text(x, y + 0.2, f"{y:.1f}", ha="center", fontsize=7, color="#333")
    ax.set_xticks(range(len(TASKS)), [names[t] for t in TASKS], fontsize=8)
    ax.set_ylabel("typical error, validation RMSE (m)")
    ax.grid(alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("What happens when a part of the model is removed (mean of 3 layouts)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "m8_ablations.png", dpi=130)
    plt.close(fig)


def main() -> None:
    acc = scores()
    abl, ksw = summarise(acc, ABLATION), summarise(acc, KSWEEP)
    pair_rows = []
    for label in ["no physics", "no pumping", "k = 3", "k = 8"]:
        pr = paired(label)
        pair_rows.append({"model": label, **{f"{t}: wells better than v1.0": f"{100 * pr[t][0]:.0f}% of {pr[t][2]} (p {pr[t][1]:.3f})"
                                             for t in TASKS}})
    pairs = pd.DataFrame(pair_rows)

    out_t, fig_dir = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
    acc.to_csv(out_t / "m8_runs.csv", index=False)
    pd.concat([abl, ksw]).drop_duplicates("model").to_csv(out_t / "m8_summary.csv", index=False)
    pairs.to_csv(out_t / "m8_paired_wells.csv", index=False)
    figures(acc, fig_dir)

    ref = abl.iloc[0]
    md = ["# M8: ablations and zone count (validation only)", "",
          "*Generated by `scripts/check_m8.py`. Rules and run list in `reports/batch5_rules.md`, written before the "
          "runs. Nothing was chosen from these results.*", "",
          "Validation = 2019. Typical error = RMSE in metres. Three tasks:",
          "- **A**: wells the model trained on, forecast one year ahead.",
          "- **B**: places with no well (held-out 20 km blocks), training years 2005-2018.",
          "- **C**: places with no well, the later year.", "",
          "Score = A / 1.90 + B / 7.94 (lower is better), the same score used for every earlier decision. v1.0 and "
          "the regional PINN average 5 seeds; each other row is one seed (seed 0) per layout. One seed alone moves the "
          "score by about 1.5%, so differences under about 2% are noise.", "",
          "## 1. Removing parts of the model", "",
          *md_table(abl.fillna("")), "", "![ablations](figures/m8_ablations.png)", "",
          "## 2. Number of zones", "",
          "k is the k-means zone count before small zones are merged into neighbours (each zone needs at least 10 "
          "wells), so the zones actually used differ per layout (column 'zones').", "",
          *md_table(ksw.fillna("")), "", "![zone count](figures/m8_zone_count.png)", "",
          "Line = mean of the 3 layouts; faint dots = each layout. The spread between layouts (dots) is much larger "
          "than the differences between zone counts.", "",
          "## 3. Well by well (seed 0 against v1.0 seed 0, 3 layouts pooled)", "",
          "Share of wells where the variant's error is lower than v1.0's. 50% = no difference. p = Wilcoxon "
          "signed-rank test on the paired per-well errors (below 0.05 = unlikely to be chance).", "",
          *md_table(pairs), "",
          f"Reference: v1.0 val A {ref['val A (m)']}, B {ref['val B (m)']}, C {ref['val C (m)']}, score {ref.score}.", "",
          "## What it means (written after reading the numbers above)", "",
          "1. **Physics helps forecasting through time.** Without the flow equation, the one-year-ahead forecast at "
          "monitored wells is much worse (A 3.6 m vs 2.2 m; worse at two thirds of wells). At places without a "
          "well there is no clear difference either way (B and C: about half the wells better, not significant). "
          "So the physics earns its place in the forecast, not in filling the map between wells.",
          "2. **Inferred pumping is needed.** Without pumping, the error at places without a well almost doubles "
          "(B 14.0 m vs 7.9 m, worse at 62% of wells, p = 0.001), and forecasts get worse too. The water levels "
          "fall, and the model needs a water sink to explain that. This supports treating pumping as a hidden "
          "variable, but it does not show that the pumping *map* is right: M7 found its spatial pattern is not "
          "recovered (`reports/m7_report.md`, `reports/s_identifiability.md`). The no-pumping runs were not "
          "retuned.",
          "3. **Zones help at places without a well, up to a point.** k = 5 (v1.0) beats one zone (B 7.9 vs 8.4 m; "
          "the test set showed the same, −8.1%). k = 8 is within noise of k = 5 (score +0.8%; B slightly lower, "
          "not significant). k = 3 is no better than one zone. The gain is not a smooth function of k, and the "
          "layout-to-layout spread is larger than the differences between k. k = 5 was chosen earlier by the "
          "pre-set zoning rule, not from this sweep, and stays.", "",
          "Limits: one seed per variant on 3 layouts; validation is a single year (2019).", ""]
    path = ROOT / "reports" / "m8_experiments.md"
    path.write_text("\n".join(md), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
