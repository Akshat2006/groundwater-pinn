"""M2 report: every synthetic PINN run so far in one table, plus figures for the best gate attempt.

    python scripts/check_m2.py          -> reports/m2_synthetic.md, reports/figures/m2_*.png

The gate (PLAN.md §5 M2-A) is judged on run1 attempts only: case C, 60 wells, monthly, clean data,
no prior on S or T. It passes if S, T and the seasonal pumping totals are all within 10%.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import data_dir, runs_dir  # noqa: E402
from synthetic.generate import pumping_net  # noqa: E402
from synthetic.pinn_data import load_run, model_pumping_grid, season_keys  # noqa: E402

FIG = ROOT / "reports" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TOL = 0.10


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def s_drift(history: Path, last_steps: int) -> float:
    """Relative change of S over the last `last_steps` Adam steps: is S still moving when training stops?"""
    h = pd.read_csv(history)
    adam = h[h.stage == "joint"]
    if adam.empty:
        return float("nan")
    end = adam.iloc[-1]
    before = adam[adam.step <= end.step - last_steps]
    return float(end.S / before.iloc[-1].S - 1) if len(before) else float("nan")


def collect() -> pd.DataFrame:
    rows = []
    for d in sorted(runs_dir().glob("*_m2_*")):
        m = d / "metrics.json"
        if not m.exists():
            continue
        r = json.loads(m.read_text())
        r["dir"] = d.name
        import yaml
        stages = yaml.safe_load((d / "config.yaml").read_text())["training"]["stages"]
        r["joint_steps"] = stages.get("joint", 0)
        r["lbfgs_steps"] = stages.get("lbfgs", 0)
        r["S_drift"] = s_drift(d / "history.csv", 4000)
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["tag"] = df.get("tag", pd.Series([""] * len(df))).fillna("")
    df["gate_pass"] = ((df.S_rel_err.abs() <= TOL) & (df.T_rel_err.abs() <= TOL)
                       & (df.Q_net_season_mean_abs_err <= TOL))
    return df


def plot_trajectory(run_dir: Path, label: str) -> None:
    h = pd.read_csv(run_dir / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), dpi=150)
    for ax, col, true, name in ((axes[0], "S", 0.12, "storage S"), (axes[1], "T", 1500.0, "flow ease T (m²/day)")):
        ax.plot(h.step, h[col], color=BLUE, lw=1.8, label="PINN estimate")
        ax.axhline(true, color=INK, lw=1, ls="--", label="true value")
        ax.axhspan(true * (1 - TOL), true * (1 + TOL), color=AQUA, alpha=0.15, lw=0, label="±10% target")
        joint = h.loc[h.stage == "joint", "step"]
        if len(joint):
            ax.axvline(joint.iloc[0], color=MUTED, lw=0.8, ls=":")
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("training step", fontsize=8, color=MUTED)
        style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Gate run {label}: parameters during training (dotted line: S and T start learning)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "m2_gate_trajectory.png")
    plt.close(fig)


def plot_pumping(run_dir: Path, label: str) -> dict:
    ds = xr.open_dataset(data_dir() / "synthetic" / "case_C" / "truth.nc").load()
    model = load_run(run_dir, ds)
    qm = model_pumping_grid(model, ds, every_days=5)
    qt = pumping_net(ds).sel(time=qm.time)
    keys = season_keys(qm.time.values)
    frame = pd.DataFrame({"key": keys, "model": qm.mean(("x", "y")).values * 365.25,
                          "true": qt.mean(("x", "y")).values * 365.25})
    order = list(dict.fromkeys(keys))
    tot = frame.groupby("key")[["model", "true"]].mean().loc[order]

    fig = plt.figure(figsize=(10, 3.2), dpi=150)
    ax = fig.add_axes([0.06, 0.2, 0.52, 0.68])
    x = np.arange(len(tot))
    ax.plot(x, tot.true, color=INK, lw=1.2, ls="--", label="true")
    ax.plot(x, tot.model, color=ORANGE, lw=1.8, marker="o", ms=3, label="PINN")
    paddy = [i for i, k in enumerate(tot.index) if k.startswith("paddy")]
    ax.set_xticks(x[paddy], [tot.index[i].split("-")[1] for i in paddy], fontsize=7)
    ax.set_xlabel("year (tick = paddy season, Jun–Sep)", fontsize=7, color=MUTED)
    ax.set_ylabel("domain-mean pumping (m/yr)", fontsize=8, color=INK)
    ax.set_title("Pumping per crop season (paddy, wheat, other)", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    vmax = float(max(qt.mean("time").max(), qm.mean("time").max())) * 365.25
    for k, (field, name) in enumerate(((qt.mean("time"), "true pumping"), (qm.mean("time"), "PINN pumping"))):
        a = fig.add_axes([0.63 + k * 0.165, 0.2, 0.15, 0.68])
        im = a.imshow(field * 365.25, origin="lower", cmap="Blues", vmin=0, vmax=vmax,
                      extent=[0, 60, 0, 60])
        a.set_title(name, fontsize=8, color=INK, loc="left")
        a.tick_params(labelsize=6, colors=MUTED, labelleft=(k == 0))
    cb = fig.colorbar(im, ax=fig.axes[-2:], shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("m/yr (10-yr mean)", fontsize=7)
    fig.savefig(FIG / "m2_gate_pumping.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    tot["err"] = tot.model / tot.true - 1
    tot["season"] = [k.split("-")[0] for k in tot.index]
    g = tot.groupby("season").err
    by_type = pd.DataFrame({"mean": g.mean(), "mean_abs": g.apply(lambda v: v.abs().mean()),
                            "worst": g.apply(lambda v: v.abs().max())})
    return {"by_type": by_type, "n_within": int((tot.err.abs() <= TOL).sum()), "n": len(tot)}


def pct(v):
    return f"{100 * v:+.0f}%"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = collect()
    if df.empty:
        print("no M2 runs with metrics yet")
        return
    gate = df[df.run == "run1"].copy()
    gate["score"] = gate[["S_rel_err", "T_rel_err"]].abs().max(axis=1) + gate.Q_net_season_mean_abs_err
    best = gate.sort_values("score").iloc[0]
    best_dir = runs_dir() / best["dir"]
    label = best["tag"] or "attempt 1"
    plot_trajectory(best_dir, label)
    seasons = plot_pumping(best_dir, label)

    lines = ["| Run | Settings | S | T | Pumping, seasons (mean abs) | Pumping, total | Hidden-well level error | Gate |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        gate_txt = ("✅" if r.gate_pass else "❌") if r.run == "run1" else "n/a"
        lines.append(f"| {r.run} | {r['tag'] or 'default'}, {r.joint_steps // 1000}k joint steps | {pct(r.S_rel_err)} | {pct(r.T_rel_err)} | "
                     f"{100 * r.Q_net_season_mean_abs_err:.0f}% | {pct(r.Q_net_total_rel_err)} | "
                     f"{r.head_rmse_test_m:.2f} m | {gate_txt} |")
    passed = bool(gate.gate_pass.any())
    final = gate[gate.lbfgs_steps > 0]
    status = (f"passed on {int(final.gate_pass.sum())} of {len(final)} runs with the final settings "
              f"(same settings, different random seed and well layout)" if len(final)
              else ("PASSED" if passed else "NOT PASSED YET"))

    matrix = df[df.run != "run1"].copy()
    matrix["order"] = matrix.run.str[3:].astype(int)
    mlines = ["| Run | What it tests | S | T | Pumping, seasons | Pumping, total | Hidden-well level error | S change, last 4k Adam steps |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in matrix.sort_values("order").iterrows():
        mlines.append(f"| {r.run} | {r.note} | {pct(r.S_rel_err)} | {pct(r.T_rel_err)} | "
                      f"{100 * r.Q_net_season_mean_abs_err:.0f}% | {pct(r.Q_net_total_rel_err)} | "
                      f"{r.head_rmse_test_m:.2f} m | {pct(r.S_drift)} |")
    by_run = matrix.drop_duplicates("run", keep="last").set_index("run")
    r_sens = []
    if {"run3", "run6", "run7"} <= set(by_run.index):
        base = by_run.loc["run3"]
        for run, factor in (("run6", 0.8), ("run7", 1.2)):
            r = by_run.loc[run]
            r_sens.append(f"| R × {factor} ({run}) | {r.S / base.S:.2f} | "
                          f"{r.Q_model_mean_m_per_year / base.Q_model_mean_m_per_year:.2f} | {factor:.2f} |")
    seed_lines = [f"| {r['tag'] or 'attempt 1'} | {pct(r.S_rel_err)} | {pct(r.T_rel_err)} | "
                  f"{100 * r.Q_net_season_mean_abs_err:.0f}% (worst {100 * r.Q_net_season_max_abs_err:.0f}%) | "
                  f"{pct(r.Q_net_total_rel_err)} | {pct(r.S_drift)} | {'✅' if r.gate_pass else '❌'} |"
                  for _, r in final.iterrows()]
    md = [
        "# M2 report: single-zone PINN on synthetic aquifers",
        "",
        "*Generated by `scripts/check_m2.py` from every run in `reports/runs/*_m2_*`.*",
        "",
        "## The gate (PLAN.md §5, M2-A)",
        "",
        "**Setup:** case C, 60 wells read monthly, clean data.",
        "",
        "**Starting values:** the PINN starts from S = 0.20 and T = 3,000 m²/day, while the truth is 0.12 and 1,500. No prior pulls it toward the truth.",
        "",
        "**To pass,** all three must be within 10%:",
        "- S",
        "- T",
        "- pumping per season",
        "",
        f"**Gate status: {status}.**",
        "",
        "Errors are the PINN estimate relative to the truth (−34% = 34% too low). Full runs have 20k or 40k joint steps;",
        "8k runs were quick comparisons.",
        "",
        *lines,
        "",
        f"![trajectory](figures/m2_gate_trajectory.png)",
        "",
        f"![pumping](figures/m2_gate_pumping.png)",
        "",
        f"*Figures show the best gate attempt so far ({label}).*",
        "",
        f"### Pumping by season type ({label})",
        "",
        "| Season | Mean error | Mean abs error | Worst season |",
        "|---|---|---|---|",
        *[f"| {k} | {pct(r['mean'])} | {100 * r['mean_abs']:.0f}% | {100 * r['worst']:.0f}% |"
          for k, r in seasons["by_type"].iterrows()],
        "",
        f"{seasons['n_within']} of {seasons['n']} individual seasons are within 10%.",
        "",
        "**How the gate rule is applied.** The rule (written in this script before the passing run) uses the *average* seasonal error across all seasons.",
        "",
        "Judged season by season instead, the low-pumping shoulder months ('other': Apr, May, Oct; about a tenth of yearly pumping) are overestimated. This is a **known limitation** to state in the paper.",
        "",
        "Likely cause: the 31-day time window still mixes May with the start of paddy pumping in June.",
        "",
        "### Does the pass repeat? (final settings, different seeds)",
        "",
        "| Run | S | T | Pumping, seasons | Pumping, total | S change, last 4k Adam steps | Gate |",
        "|---|---|---|---|---|---|---|",
        *seed_lines,
        "",
        "The *S change* column shows how much S changed over the last 4,000 Adam steps before L-BFGS. A large value means S was still",
        "climbing when training stopped, so the final value depends on how long training ran rather than on a settled answer.",
        "",
        "## Run matrix (M2-B realistic cases and M2-D mismatched case: reported, not gated)",
        "",
        "Same settings as the gate run. Cases A and D have 30 wells (run8: 15) read four times a year, like CGWB.",
        "",
        *mlines,
        "",
        *(["### Recharge sensitivity (runs 6 and 7 compared with run 3)",
           "",
           "If recharge R is wrong by a factor, theory (PLAN.md §2.1) says S and pumping should come out wrong by the same factor.",
           "",
           "| Recharge given to the PINN | S ratio | Pumping ratio | Theory |",
           "|---|---|---|---|",
           *r_sens,
           ""] if r_sens else []),
        "## What the metrics mean",
        "",
        "- **S, T:** relative error of the recovered storage and ease of flow.",
        "- **Pumping, seasons:** the PINN's total pumping in each crop season (paddy Jun–Sep, wheat Nov–Mar, other) is compared with the truth. The column gives the average size of the error across the 30 seasons.",
        "- **Pumping, total:** error in the 10-year total pumping.",
        "- **Hidden-well level error:** typical water-level error (RMSE) at 30 wells the PINN never saw.",
    ]
    (ROOT / "reports" / "m2_synthetic.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"gate {status}; best attempt {label}: "
          f"S {pct(best.S_rel_err)}, T {pct(best.T_rel_err)}, Q seasons {100 * best.Q_net_season_mean_abs_err:.0f}%")
    print("wrote reports/m2_synthetic.md")


if __name__ == "__main__":
    main()
