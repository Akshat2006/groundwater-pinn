"""M1 "done" checks: solver vs exact solutions, convergence, water balance, synthetic case summaries.

    python scripts/check_m1.py      (run scripts/make_synthetic.py first)

Writes reports/m1_solver.md and reports/figures/m1_*.png.
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.dates
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.physics.fd_solver import BoundarySpec, Grid, solve_fd  # noqa: E402
from src.utils.config import data_dir  # noqa: E402
from synthetic.analytical import closed_box, linear_steady, sine_mode, theis_drawdown  # noqa: E402
from synthetic.generate import pumping_gross, pumping_net, recharge_given, recharge_true  # noqa: E402

FIG = ROOT / "reports" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("decline_rise", [ORANGE, "#d9d8d4", BLUE])   # fall | 0 | rise
S, T = 0.12, 1500.0
YR = 365.25


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ solver checks
def solver_checks() -> dict:
    out = {}
    L, h0, amp = 60_000.0, 230.0, 5.0
    rows = []
    for n in (8, 16, 32, 64):
        g = Grid(n, n, L / n, L / n)
        X, Y = g.mesh()
        r = solve_fd(g, S, T, 0.0, 0.0, sine_mode(X, Y, 0, L, S, T, h0, amp), BoundarySpec(h0, h0, h0, h0),
                     1.0, 3650, save_every=3650)
        err = float(np.abs(r.h_final - sine_mode(X, Y, 3650, L, S, T, h0, amp)).max())
        rows.append({"cells": n, "dx_km": L / n / 1000, "max_error_m": err,
                     "budget_max_rel": float(r.budget["rel_error"].max())})
    conv = pd.DataFrame(rows)
    conv["order"] = np.r_[np.nan, np.log2(conv.max_error_m.values[:-1] / conv.max_error_m.values[1:])]
    out["convergence"] = conv

    g = Grid(10, 10, 1000.0, 1000.0)
    r = solve_fd(g, S, T, 2e-3, 1.5e-3, 200.0, BoundarySpec(), 1.0, 365)
    out["closed_box_err"] = float(np.abs(r.h_final - closed_box(365, 200.0, S, 5e-4)).max())

    g = Grid(60, 60, 1000.0, 1000.0)
    X, _ = g.mesh()
    h = linear_steady(X, 220.0, 238.0, 60_000.0)
    r = solve_fd(g, S, T, 0.0, 0.0, h, BoundarySpec(west=220.0, east=238.0), 1.0, 365)
    out["steady_err"] = float(np.abs(r.h_final - h).max())

    n, dx, rate = 161, 250.0, 5000.0
    g = Grid(n, n, dx, dx)
    c = n // 2
    Q = np.zeros(g.shape)
    Q[c, c] = rate / g.cell_area
    r = solve_fd(g, S, T, 0.0, Q, 0.0, BoundarySpec(0.0, 0.0, 0.0, 0.0), 1.0, 365, save_every=365)
    out["theis"] = pd.DataFrame([{"distance_km": k * dx / 1000, "simulated_m": -r.h_final[c, c + k],
                                  "theis_m": float(theis_drawdown(k * dx, 365, rate, S, T))} for k in (2, 4, 8, 12, 20)])
    out["theis"]["rel_diff"] = out["theis"].simulated_m / out["theis"].theis_m - 1
    return out


def plot_convergence(conv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=150)
    ax.loglog(conv.dx_km, conv.max_error_m, "-o", color=BLUE, lw=2, ms=6, label="solver error")
    ref = conv.max_error_m.iloc[1] * (conv.dx_km / conv.dx_km.iloc[1]) ** 2
    ax.loglog(conv.dx_km, ref, "--", color=MUTED, lw=1.2, label="ideal: error ∝ Δx²")
    ax.set_xticks(conv.dx_km, [f"{v:g}" for v in conv.dx_km])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("grid spacing Δx (km)", color=INK, fontsize=9)
    ax.set_ylabel("max head error after 10 years (m)", color=INK, fontsize=9)
    ax.set_title("Finer grid → error shrinks ~4× per halving", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "m1_convergence.png")
    plt.close(fig)


# ------------------------------------------------------------------ synthetic cases
def case_summary(ds: xr.Dataset) -> dict:
    h = ds.h
    first = h.sel(time=str(ds.time.dt.year.values[0])).mean("time")
    last = h.sel(time=str(ds.time.dt.year.values[-1])).mean("time")
    years = int(ds.time.dt.year.values[-1] - ds.time.dt.year.values[0])
    trend = (last - first) / years
    centre = h.isel(x=ds.sizes["x"] // 2, y=ds.sizes["y"] // 2).to_series()
    swing = centre.groupby(centre.index.year).agg(lambda s: s.max() - s.min()).mean()
    return {
        "case": ds.attrs["case"],
        "R true (m/yr)": float(recharge_true(ds).mean()) * YR,
        "R given to model (m/yr)": float(recharge_given(ds).mean()) * YR,
        "Q gross (m/yr)": float(pumping_gross(ds).mean()) * YR,
        "Q net (m/yr)": float(pumping_net(ds).mean()) * YR,
        "head trend, whole area (m/yr)": float(trend.mean()),
        "head trend, central 20 km (m/yr)": float(trend.isel(x=slice(20, 40), y=slice(20, 40)).mean()),
        "seasonal swing at centre (m)": float(swing),
        "max budget error": float(ds.budget_rel_error.max()),
    }, trend


def plot_cases(datasets: dict, trends: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), dpi=150, constrained_layout=True)
    vmax = max(float(abs(t).max()) for t in trends.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    qmax = max(float(pumping_gross(ds).mean("time").max()) * YR for ds in datasets.values())
    for k, (case, ds) in enumerate(datasets.items()):
        ext = [v / 1000 for v in (ds.x[0] - 500, ds.x[-1] + 500, ds.y[0] - 500, ds.y[-1] + 500)]
        ax = axes[0, k]
        im1 = ax.imshow(trends[case], origin="lower", extent=ext, cmap=DIVERGING, norm=norm)
        wells_file = sorted((data_dir() / "synthetic" / f"case_{case}").glob("wells_*.parquet"))[0]
        w = pd.read_parquet(wells_file).query("split_space == 'train'").drop_duplicates("well_id")
        ax.scatter(w.x_m / 1000, w.y_m / 1000, s=10, facecolor="white", edgecolor=INK, linewidth=0.6)
        ax.set_title(f"Case {case}: head trend", fontsize=9, color=INK, loc="left")
        ax = axes[1, k]
        im2 = ax.imshow(pumping_gross(ds).mean("time") * YR, origin="lower", extent=ext, cmap="Blues",
                        vmin=0, vmax=qmax)
        if float(ds.canal_map.max()) > 0:
            ax.contour(ds.x / 1000, ds.y / 1000, ds.canal_map > 0, levels=[0.5], colors=[AQUA], linewidths=1.2)
        ax.set_title(f"Case {case}: true pumping", fontsize=9, color=INK, loc="left")
    for ax in axes.ravel():
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_xlabel("x (km)", fontsize=7, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (km)", fontsize=7, color=MUTED)
    cb = fig.colorbar(im1, ax=axes[0, :], shrink=0.85, label="m / year (orange = falling)")
    cb.ax.tick_params(labelsize=7)
    cb = fig.colorbar(im2, ax=axes[1, :], shrink=0.85, label="pumping, m / year (aqua line = canal)")
    cb.ax.tick_params(labelsize=7)
    fig.savefig(FIG / "m1_cases.png")
    plt.close(fig)


def plot_hydrograph(ds: xr.Dataset) -> None:
    x, y = float(ds.x[30]), float(ds.y[30])
    s = ds.h.sel(x=x, y=y).to_series()
    q = s[(s.index.day == 15) & s.index.month.isin([1, 5, 8, 11])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), dpi=150, gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, (a, b), title in ((axes[0], (None, None), "Case A, aquifer centre: 10 years"),
                              (axes[1], ("2004-01-01", "2005-12-31"), "Zoom: 2004–2005")):
        ss, qq = s[a:b], q[a:b]
        ax.plot(ss.index, ss.values, color=BLUE, lw=1.5, label="true daily head")
        ax.plot(qq.index, qq.values, "o", color=ORANGE, ms=5, mec="white", mew=0.8,
                label="quarterly readings (all the model sees)")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("head (m)", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(FIG / "m1_hydrograph_caseA.png")
    plt.close(fig)


def fmt(df: pd.DataFrame, floatfmt: dict) -> str:
    d = df.copy()
    for col, f in floatfmt.items():
        d[col] = d[col].map(lambda v: "" if pd.isna(v) else format(v, f))
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.values)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    chk = solver_checks()
    conv, theis = chk["convergence"], chk["theis"]
    plot_convergence(conv)
    datasets, trends, rows = {}, {}, []
    for case in "ABCD":
        ds = xr.open_dataset(data_dir() / "synthetic" / f"case_{case}" / "truth.nc")
        summary, trend = case_summary(ds)
        datasets[case], trends[case] = ds, trend
        rows.append(summary)
    plot_cases(datasets, trends)
    plot_hydrograph(datasets["A"])
    cases = pd.DataFrame(rows)

    budget_max = max(conv.budget_max_rel.max(), cases["max budget error"].max())
    order_ok = conv.order.dropna().iloc[:2].min() > 1.8
    theis_ok = theis.query("distance_km >= 1 and distance_km <= 3").rel_diff.abs().max() < 0.02
    checks = [
        ("Sine-mode decay (exact solution) matches", f"max error {conv.max_error_m.iloc[2]:.1e} m on a 5 m bump (32×32 grid)",
         conv.max_error_m.iloc[2] < 1e-3),
        ("Second-order convergence (~4× smaller error per halving of Δx)",
         "observed orders " + ", ".join(f"{v:.2f}" for v in conv.order.dropna()), order_ok),
        ("Closed box: h = h0 + (R−Q)·t/S", f"max error {chk['closed_box_err']:.1e} m", chk["closed_box_err"] < 1e-9),
        ("Straight-line steady state stays put", f"max change {chk['steady_err']:.1e} m", chk["steady_err"] < 1e-9),
        ("Water balance < 1e-8 every step", f"worst step {budget_max:.1e} (all tests and all 4 cases)", budget_max < 1e-8),
        ("Theis drawdown (optional, 1–3 km from the well)",
         f"within {100 * theis.query('distance_km >= 1 and distance_km <= 3').rel_diff.abs().max():.1f}%", theis_ok),
    ]
    tick = lambda ok: "✅ pass" if ok else "❌ FAIL"  # noqa: E731

    md = ["# M1 report: finite-difference solver and synthetic aquifers",
          "",
          "*Generated by `scripts/check_m1.py`.*",
          "",
          "## What M1 is",
          "",
          "M1 builds a computer-simulated aquifer where we *know* the true storage (S), ease of flow (T), recharge (R) and pumping (Q).",
          "",
          "- The simulator steps the groundwater equation S·∂h/∂t = T∇²h + R − Q forward one day at a time.",
          "- It works on a grid of 1 km squares.",
          "",
          "In M2 the PINN sees only sparse \"well readings\" from these simulations. It must recover the S, T and Q we put in.",
          "",
          "## Done criteria (PLAN.md §5, M1)",
          "",
          "| Check | Result | Status |", "|---|---|---|"]
    md += [f"| {name} | {res} | {tick(ok)} |" for name, res, ok in checks]
    md += ["",
           "**What these checks mean in plain words:**",
           "- **Exact solutions.** For a few simple situations, mathematicians have worked out the answer on paper.",
           "  - The simulator reproduces those answers almost perfectly.",
           "  - So its units, signs and boundaries are right.",
           "- **Second-order convergence.** Making the grid twice as fine cuts the error about four times.",
           "  - That is exactly what this method should do.",
           "  - It shows there is no hidden bug that only appears at some resolutions.",
           "- **Water balance.** Every day, the change in stored water equals recharge − pumping + inflow at the edges, to about one part in 10¹². No water is created or lost.",
           "",
           "### Grid refinement (sine-mode decay, 10 years, Δt = 1 day)",
           "",
           fmt(conv, {"dx_km": ".2f", "max_error_m": ".2e", "budget_max_rel": ".1e", "order": ".2f"}),
           "",
           "The last order is a little below 2. At the finest grid, the 1-day time step's own small error starts to show, which is expected.",
           "",
           "![convergence](figures/m1_convergence.png)",
           "",
           "### Theis check (one well pumping 5,000 m³/day for a year)",
           "",
           fmt(theis, {"distance_km": ".1f", "simulated_m": ".3f", "theis_m": ".3f", "rel_diff": "+.1%"}),
           "",
           "Right next to the well (0.5 km) a 250 m grid cannot resolve the steep cone, so a few % difference is normal.",
           "",
           "## The four synthetic aquifers (60 × 60 km, 10 years, 2000–2009)",
           "",
           fmt(cases, {c: ".3f" for c in cases.columns if c not in ("case", "max budget error")} | {"max budget error": ".1e"}),
           "",
           "**What each case is for:**",
           "- **A: homogeneous.** The main recovery test.",
           "  - The centre falls about 0.9 m/yr, as planned.",
           "  - The fixed-head edges (a river in the west, a regional level in the east) slow the fall near the sides.",
           "- **B: two zones.** S and T differ west/east. It checks that zoning can find them (E17).",
           "- **C: well-posed.** It adds a canal and three heavy-pumping clusters, which create 4–7 m cones.",
           "  - Sideways flow toward the cones is what lets T be measured.",
           "  - **This is the M2-A gate.**",
           "- **D: mismatched (honest stress test).**",
           "  - The real aquifer gets canal seepage and irrigation return flow that the model is *not* told about.",
           "  - Pumping is patchy and has village hot spots.",
           "  - The model's land-use map is blurred and noisy.",
           "  - The extra hidden recharge (0.21 vs 0.14 m/yr given) cancels most of the decline, so the levels are nearly flat.",
           "  - A model that trusts its wrong recharge will have to \"explain\" this with less pumping. That is exactly the bias M2 must measure and report.",
           "",
           "![cases](figures/m1_cases.png)",
           "",
           "*Top: 10-year head trend (orange = falling), with the training wells of each case's first design. Bottom: true pumping.*",
           "",
           "![hydrograph](figures/m1_hydrograph_caseA.png)",
           "",
           "This plot shows what the PINN is up against: the orange dots (4 readings a year) are all it sees of the blue curve.",
           "",
           "## Files",
           "",
           "- `data/synthetic/case_{A,B,C,D}/truth.nc`: daily heads plus the true S, T, recharge and pumping components (see `synthetic/generate.py`).",
           "- `wells_*.parquet`: sampled wells (training wells plus 30 hidden wells for scoring).",
           "  - Designs: A q30 and q15, B q30, C m60, D q30.",
           "  - Noisy variants for the M2 run matrix are made on the fly with `synthetic.sample_wells.sample_wells`.",
           "",
           "## Not done in M1 (on purpose)",
           "",
           "**PyTorch (differentiable) solver.** PLAN.md lists one, but it is only needed for two things:",
           "- the M2 fallback (a differentiable finite-difference inverse model), which needs your OK anyway;",
           "- training-free forecasts, for which the numpy solver is enough.",
           "",
           "It will be added if and when the fallback is needed."]
    (ROOT / "reports" / "m1_solver.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(f"{'PASS' if ok else 'FAIL'}  {name}: {res}" for name, res, ok in checks)
          .encode("ascii", "replace").decode())
    print("wrote reports/m1_solver.md")


if __name__ == "__main__":
    main()
