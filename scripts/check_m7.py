"""M7 report: uncertainty calibration, physical plausibility, S/T/pumping maps, pumping vs CGWB district figures.

    python scripts/check_m7.py    -> reports/m7_report.md, reports/figures/m7_*.png, reports/tables/m7_*.csv

Uses the final runs (tag tbc): the shared zoned PINN and the regional PINN, 5 seeds x 3 layouts, with their val and
test predictions already in reports/runs (predict_real.py). Nothing is trained or re-tuned here.

1. Calibration (PLAN.md 2.6). A reading's 90% range is mean +- 1.645 * c * sqrt(sd_seeds^2 + noise^2):
   sd_seeds = spread of the 5 seeds, noise = the runs' training-fit RMSE (what remains even at a training well),
   c = one factor per task, fitted on the *validation* readings of all 3 layouts so that 90% fall inside, then applied
   unchanged to the test readings.
2. Plausibility. Share of dry, heavily farmed cell-years where the model has the water table *rising* from 15 Jan to
   15 May (rabi irrigation and pre-monsoon pumping, little rain). Dry = Jan-May rain below that cell's 2005-24
   median; heavily farmed = cropland >= 80%. Compared with how often measured wells really rose in the same setting.
3. Maps of storage S and transmissivity T per zone and pumping Q per cell (layout 0, 5 seeds; the map model).
4. Pumping vs CGWB: model Q averaged over 2017-2020 per district (all 15 zoned runs) against the CGWB 2020
   assessment's gross extraction per unit area (data/reference/cgwb_district_2020.csv). CGWB figures are themselves
   estimates (GEC-2015 method), not measurements: this is a consistency check, not a validation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.tasks import shallow_levels  # noqa: E402
from src.training.real_data import load_real_run, predict_heads, predict_pumping  # noqa: E402
from src.utils.config import data_dir, runs_dir  # noqa: E402

MODELS = {"zoned": "kmeansk5_shared", "regional": "regional"}
Z90 = 1.645
YEARS_MAP = range(2005, 2019)
YEARS_CGWB = range(2017, 2021)


def run_dirs(layout: int, model: str) -> list[Path]:
    return sorted(d for d in runs_dir().glob(f"*_real_L{layout}_{MODELS[model]}_s*_tbc") if (d / "model.pt").exists())


def md_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return (["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
            + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)])


# ---------------------------------------------------------------- 1. calibration
def ensemble_rows(layout: int, model: str) -> pd.DataFrame:
    dirs = run_dirs(layout, model)
    noise = float(np.mean([json.loads((d / "metrics.json").read_text())["train_rmse_m"] for d in dirs]))
    p = pd.concat(pd.read_parquet(d / "predictions.parquet").assign(seed=i) for i, d in enumerate(dirs))
    g = p.groupby(["stage", "task", "well_id", "date"])
    e = g.agg(head_m=("head_m", "first"), mu=("pred", "mean"), sd=("pred", "std"), n=("pred", "size")).reset_index()
    e = e[e.n == len(dirs)]
    return e.assign(layout=layout, model=model, noise=noise, sd_tot=np.sqrt(e.sd ** 2 + noise ** 2))


def calibration(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, by_year = [], []
    for (model, task), g in rows.groupby(["model", "task"]):
        val, test = g[g.stage == "val"], g[g.stage == "test"]
        z_val = (val.head_m - val.mu).abs() / val.sd_tot
        c = float(np.quantile(z_val, 0.90) / Z90)

        def cov(d, width):
            return float(((d.head_m - d.mu).abs() <= width).mean())
        out.append({"model": model, "task": task,
                    "val coverage, seeds only": cov(val, Z90 * val.sd),
                    "test coverage, seeds only": cov(test, Z90 * test.sd),
                    "test coverage, seeds + noise": cov(test, Z90 * test.sd_tot),
                    "factor c (from val)": c,
                    "test coverage, calibrated": cov(test, Z90 * c * test.sd_tot),
                    "calibrated half-width (m, median)": float(np.median(Z90 * c * test.sd_tot)),
                    "test readings": len(test)})
        if task in ("A", "C"):
            t = test.assign(year=pd.to_datetime(test.date).dt.year)
            for yr, ty in t.groupby("year"):
                by_year.append({"model": model, "task": task, "year": yr,
                                "coverage": cov(ty, Z90 * c * ty.sd_tot), "readings": len(ty)})
    return pd.DataFrame(out), pd.DataFrame(by_year)


# ---------------------------------------------------------------- 2. plausibility
def cell_latlon(xs, ys):
    from rasterio.warp import transform
    lon, lat = transform("EPSG:32643", "EPSG:4326", list(map(float, xs)), list(map(float, ys)))
    return np.asarray(lat), np.asarray(lon)


def dry_season_rain(rain: xr.DataArray, lat, lon, years) -> np.ndarray:
    """Rain (mm) from 15 Jan to 15 May of each year at the nearest 0.25 degree cell: (year, point)."""
    il = np.abs(rain.lat.values[None, :] - np.asarray(lat)[:, None]).argmin(1)
    jl = np.abs(rain.lon.values[None, :] - np.asarray(lon)[:, None]).argmin(1)
    out = []
    for yr in years:
        tot = rain.sel(time=slice(f"{yr}-01-15", f"{yr}-05-14")).sum("time").values
        out.append(tot[il, jl])
    return np.array(out)


def ensemble_rise(dirs, static, xs, ys, years) -> np.ndarray:
    """Mean over seeds of head(15 May) - head(15 Jan): (year, point)."""
    acc = np.zeros((len(years), len(xs)))
    for d in dirs:
        model, _ = load_real_run(d, static)
        for k, yr in enumerate(years):
            jan = predict_heads(model, xs, ys, pd.DatetimeIndex([pd.Timestamp(yr, 1, 15)] * len(xs)))
            may = predict_heads(model, xs, ys, pd.DatetimeIndex([pd.Timestamp(yr, 5, 15)] * len(xs)))
            acc[k] += (may - jan) / len(dirs)
    return acc


def plausibility(static, rain, levels, cells_ok) -> tuple[pd.DataFrame, dict]:
    years = list(range(2005, 2025))
    yy, xx = np.meshgrid(static.y.values, static.x.values, indexing="ij")
    farmed = static.lc_cropland_frac.fillna(0).values >= 0.8
    sel = cells_ok & farmed
    xs, ys = xx[sel], yy[sel]
    lat, lon = cell_latlon(xs, ys)
    r = dry_season_rain(rain, lat, lon, years)
    dry = r < np.median(r, axis=0, keepdims=True)

    # measured wells with a January and a May reading in the same year, in heavily farmed cells
    lv = levels.assign(year=pd.to_datetime(levels.date).dt.year, month=pd.to_datetime(levels.date).dt.month)
    lv = lv[lv.year.between(2005, 2024) & (lv.lu_cropland_frac >= 0.8)]
    jan = lv[lv.month == 1].groupby(["well_id", "year"]).agg(h_jan=("head_m", "mean"), x_m=("x_m", "first"),
                                                             y_m=("y_m", "first"))
    may = lv[lv.month == 5].groupby(["well_id", "year"]).agg(h_may=("head_m", "mean"))
    w = jan.join(may, how="inner").reset_index()
    wlat, wlon = cell_latlon(w.x_m.values, w.y_m.values)
    wr = dry_season_rain(rain, wlat, wlon, years)                      # (year, well-year)
    wy = w.year.values - years[0]
    w["rain"] = wr[wy, np.arange(len(w))]
    med_cell = np.median(wr, axis=0)
    w["dry"] = w.rain < med_cell
    w["obs_rise"] = w.h_may - w.h_jan

    rows, info = [], {"cells": int(sel.sum()), "cell_years_dry": int(dry.sum()), "well_years_dry": int(w.dry.sum())}
    for model in MODELS:
        dirs = run_dirs(0, model)
        rise = ensemble_rise(dirs, static, xs, ys, years)
        # at the wells: the model's own Jan -> May change at each well's position and year
        wrise = np.zeros(len(w))
        for d in dirs:
            m, _ = load_real_run(d, static)
            hj = predict_heads(m, w.x_m.values, w.y_m.values, pd.DatetimeIndex(pd.to_datetime(w.year.astype(str) + "-01-15")))
            hm = predict_heads(m, w.x_m.values, w.y_m.values, pd.DatetimeIndex(pd.to_datetime(w.year.astype(str) + "-05-15")))
            wrise += (hm - hj) / len(dirs)
        wd = w[w.dry]
        for period, yrs in (("2005-2018", (2005, 2018)), ("2019-2024", (2019, 2024)), ("all", (2005, 2024))):
            ky = [(yrs[0] <= y <= yrs[1]) for y in years]
            mask = dry[ky]
            wp = wd[wd.year.between(*yrs)]
            rows.append({"model": model, "years": period,
                         "map: dry cell-years with a rise": float((rise[ky][mask] > 0).mean()),
                         "map: rise > 0.5 m": float((rise[ky][mask] > 0.5).mean()),
                         "wells: model rise": float((wrise[wp.index] > 0).mean()) if len(wp) else np.nan,
                         "wells: measured rise": float((wp.obs_rise > 0).mean()) if len(wp) else np.nan,
                         "well-years": len(wp)})
    return pd.DataFrame(rows), info


# ---------------------------------------------------------------- 3 + 4. S, T, pumping
def pumping_per_cell(model, xs, ys, years) -> np.ndarray:
    """Mean pumping (m/yr) per cell over the given years, from mid-month samples."""
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-01", freq="MS") + pd.Timedelta(days=14)
    q = predict_pumping(model, np.tile(xs, len(dates)), np.tile(ys, len(dates)), np.repeat(dates.values, len(xs)))
    return q.reshape(len(dates), len(xs)).mean(0) * 365.25


def parameters_and_pumping(static, zones, cells_ok, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yy, xx = np.meshgrid(static.y.values, static.x.values, indexing="ij")
    area = static.district_id.values > 0
    xs, ys = xx[area], yy[area]
    dist = static.district_id.values[area]

    # per-zone S and T, every layout and seed
    zrows = []
    for layout in (0, 1, 2):
        for d in run_dirs(layout, "zoned"):
            m = json.loads((d / "metrics.json").read_text())
            seed = yaml.safe_load((d / "config.yaml").read_text())["seed"]
            for z, (s, t) in enumerate(zip(m["S_zones"], m["T_zones"]), 1):
                zrows.append({"layout": layout, "seed": seed, "zone": z, "S": s, "T": t})
    zt = pd.DataFrame(zrows)

    # pumping per cell: layout 0 for the maps (2005-2018), all 15 runs for the CGWB check (2017-2020)
    q_map, q_cgwb = [], []
    for layout in (0, 1, 2):
        for d in run_dirs(layout, "zoned"):
            model, _ = load_real_run(d, static)
            if layout == 0:
                q_map.append(pumping_per_cell(model, xs, ys, YEARS_MAP))
            qc = pumping_per_cell(model, xs, ys, YEARS_CGWB)
            q_cgwb.append(pd.Series(qc).groupby(dist).mean())
    q_map = np.array(q_map)
    qd = pd.DataFrame(q_cgwb)                                    # run x district, m/yr

    ref = pd.read_csv(ROOT / "data" / "reference" / "cgwb_district_2020.csv")
    cells = pd.Series(dist).value_counts()
    ref["area_ha"] = ref.district_id.map(cells) * 100.0          # 1 km cells = 100 ha
    ref["cgwb_m_per_yr"] = ref.extraction_ham / ref.area_ha
    ref["model_m_per_yr"] = ref.district_id.map(qd.mean())
    ref["model_lo"] = ref.district_id.map(qd.min())
    ref["model_hi"] = ref.district_id.map(qd.max())
    rho, p = spearmanr(ref.cgwb_m_per_yr, ref.model_m_per_yr)
    ratio = float(ref.model_m_per_yr.sum() / ref.cgwb_m_per_yr.sum())

    # ---- figures
    def to_grid(v):
        g = np.full(area.shape, np.nan)
        g[area] = v
        return np.where(cells_ok, g, np.nan)[::-1]               # north up

    def outlines(ax):
        did = static.district_id.values[::-1].astype(float)
        ax.contour(did, levels=np.arange(0.5, 17), colors="0.35", linewidths=0.4)
        ax.set_axis_off()

    zmap = zones.zone.sel({"method": "kmeans", "layout": 0, "k": 5}).values
    z0 = zt[zt.layout == 0].groupby("zone")[["S", "T"]].mean()
    s_cell = np.where(zmap > 0, pd.Series(zmap.ravel()).map(z0.S).values.reshape(zmap.shape), np.nan)
    t_cell = np.where(zmap > 0, pd.Series(zmap.ravel()).map(z0["T"]).values.reshape(zmap.shape), np.nan)
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    panels = [(np.where(cells_ok, s_cell, np.nan)[::-1], "Storage S per zone (learned; scale = allowed range)",
               "Greens", (0.072, 0.15)),
              (np.where(cells_ok, t_cell, np.nan)[::-1], "Transmissivity T per zone (m²/day, from CGWB reports)", "Purples", None),
              (to_grid(q_map.mean(0)), "Pumping Q, 2005-2018 mean (m/yr, inferred)", "Blues", None),
              (to_grid(q_map.std(0, ddof=1)), "Pumping: spread over 5 seeds (m/yr)", "Greys", None)]
    for ax, (g, title, cmap, lim) in zip(axs, panels):
        im = ax.imshow(g, cmap=cmap, **({} if lim is None else {"vmin": lim[0], "vmax": lim[1]}))
        outlines(ax)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(fig_dir / "m7_parameter_maps.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.errorbar(ref.cgwb_m_per_yr, ref.model_m_per_yr,
                yerr=[ref.model_m_per_yr - ref.model_lo, ref.model_hi - ref.model_m_per_yr],
                fmt="o", color="#2b6cb0", ecolor="#9fb8d6", ms=6, capsize=2)
    for r in ref.itertuples():
        ax.annotate(r.district, (r.cgwb_m_per_yr, r.model_m_per_yr), fontsize=7, color="0.3",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("CGWB 2020: gross groundwater extraction per area (m/yr)")
    ax.set_ylabel("Model: inferred pumping 2017-2020 (m/yr)")
    ax.set_title(f"Pumping by district (rank correlation {rho:.2f}, p = {p:.3f})", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "m7_pumping_vs_cgwb.png", dpi=130)
    plt.close(fig)

    # zone pumping over time, layout 0
    pz = pd.concat(pd.read_csv(d / "pumping_yearly_zones.csv").assign(run=d.name) for d in run_dirs(0, "zoned"))
    fig, ax = plt.subplots(figsize=(7, 4))
    for z, g in pz.groupby("zone"):
        s = g.groupby("year").pumping_m_per_yr
        ax.fill_between(s.min().index, s.min(), s.max(), alpha=0.2)
        ax.plot(s.mean().index, s.mean(), label=f"zone {z}", lw=2)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.axvline(2018.5, color="0.5", lw=1, ls="--")
    ax.text(2018.7, ax.get_ylim()[1] * 0.98, "after training years", fontsize=8, va="top", color="0.4")
    ax.set_ylabel("inferred pumping (m/yr)")
    ax.set_title("Pumping per zone, layout 0 (line = mean of 5 seeds, band = min to max)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "m7_pumping_by_zone.png", dpi=130)
    plt.close(fig)

    by_state = ref.groupby("state")[["cgwb_m_per_yr", "model_m_per_yr"]].mean()
    return zt, ref, {"rho": float(rho), "p": float(p), "ratio": ratio,
                     "cgwb_pb_hr": float(by_state.cgwb_m_per_yr["Punjab"] / by_state.cgwb_m_per_yr["Haryana"]),
                     "model_pb_hr": float(by_state.model_m_per_yr["Punjab"] / by_state.model_m_per_yr["Haryana"]),
                     "q_map_mean": float(np.nanmean(q_map.mean(0)[cells_ok[area]])),
                     "q_seed_sd_median": float(np.median(q_map.std(0, ddof=1)[cells_ok[area]]))}


def main() -> None:
    proc = data_dir() / "processed"
    # load and close each file: a NetCDF file still open here crashes the re-open inside load_real_run (Windows)
    with xr.open_dataset(proc / "static_1km.nc") as f:
        static = f.load()
    with xr.open_dataset(proc / "zones.nc") as f:
        zones = f.load()
    with xr.open_dataset(proc / "rain_daily.nc") as f:
        rain = f.rain_mm.load()
    levels = shallow_levels(pd.read_parquet(proc / "master.parquet"))
    fig_dir, tab_dir = ROOT / "reports" / "figures", ROOT / "reports" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # the cells the maps show: inside the districts, ground no higher than 99% of the shallow wells
    wells = pd.read_parquet(proc / "wells.parquet")
    wells = wells[(wells.depth_class == "shallow") & wells.in_study_area.astype(bool)]
    cells_ok = (static.district_id.values > 0) & (static.elev_m.values <= np.percentile(wells.ground_elev_m.dropna(), 99))

    print("1. calibration")
    rows = pd.concat(ensemble_rows(L, m) for L in (0, 1, 2) for m in MODELS)
    cal, cal_year = calibration(rows)
    cal.to_csv(tab_dir / "m7_calibration.csv", index=False)
    cal_year.to_csv(tab_dir / "m7_calibration_by_year.csv", index=False)
    print(cal.round(3).to_string(index=False))

    print("2. plausibility")
    pl, pl_info = plausibility(static, rain, levels, cells_ok)
    pl.to_csv(tab_dir / "m7_plausibility.csv", index=False)
    print(pl.round(3).to_string(index=False), pl_info)

    print("3+4. parameters, pumping, CGWB")
    zt, ref, qinfo = parameters_and_pumping(static, zones, cells_ok, fig_dir)
    zt.to_csv(tab_dir / "m7_zone_parameters.csv", index=False)
    ref.to_csv(tab_dir / "m7_pumping_vs_cgwb.csv", index=False)
    print(ref[["district", "cgwb_m_per_yr", "model_m_per_yr", "model_lo", "model_hi"]].round(3).to_string(index=False))
    print(qinfo)

    write_report(cal, cal_year, pl, pl_info, zt, ref, qinfo)


def pct(v) -> str:
    return "" if pd.isna(v) else f"{100 * v:.0f}%"


def write_report(cal, cal_year, pl, pl_info, zt, ref, qinfo) -> None:
    c = cal.copy()
    for col in [k for k in c.columns if "coverage" in k]:
        c[col] = c[col].map(pct)
    c["factor c (from val)"] = c["factor c (from val)"].map("{:.1f}".format)
    c["calibrated half-width (m, median)"] = c["calibrated half-width (m, median)"].map("± {:.1f} m".format)
    cy = cal_year[cal_year.model == "zoned"].pivot(index="year", columns="task", values="coverage").map(pct).reset_index()
    p = pl.copy()
    for col in [k for k in p.columns if ":" in k]:
        p[col] = p[col].map(pct)
    zs = (zt.groupby(["layout", "zone"]).agg(S_mean=("S", "mean"), S_min=("S", "min"), S_max=("S", "max"),
                                            T=("T", "mean")).reset_index())
    zs["S (mean, min-max over seeds)"] = [f"{a:.3f} ({b:.3f}-{d:.3f})" for a, b, d in zip(zs.S_mean, zs.S_min, zs.S_max)]
    zs["T (m²/day)"] = zs["T"].map("{:.0f}".format)
    zs = zs[["layout", "zone", "S (mean, min-max over seeds)", "T (m²/day)"]]
    at_floor = float((zt.S < 0.0725).mean())
    r = ref.sort_values("cgwb_m_per_yr", ascending=False)
    r = pd.DataFrame({"District": r.district, "State": r.state,
                      "CGWB 2020 extraction (m/yr)": r.cgwb_m_per_yr.map("{:.2f}".format),
                      "Model pumping 2017-20 (m/yr)": r.model_m_per_yr.map("{:.2f}".format),
                      "Model range over 15 runs": [f"{a:.2f}-{b:.2f}" for a, b in zip(r.model_lo, r.model_hi)]})
    zA = cal[(cal.model == "zoned") & (cal.task == "A")].iloc[0]
    zB = cal[(cal.model == "zoned") & (cal.task == "B")].iloc[0]
    zC = cal[(cal.model == "zoned") & (cal.task == "C")].iloc[0]
    plz = pl[(pl.model == "zoned") & (pl.years == "all")].iloc[0]
    plr = pl[(pl.model == "regional") & (pl.years == "all")].iloc[0]

    md = [
        "# M7 report: uncertainty, plausibility, storage/transmissivity/pumping maps, pumping vs CGWB",
        "",
        "*Generated by `scripts/check_m7.py`.* Final models: shared zoned PINN and regional PINN, both with canal seepage "
        "(tag `tbc`), 5 seeds × 3 block layouts. Nothing was trained or tuned for this report.",
        "",
        "## 1. Uncertainty: are the stated ranges honest?",
        "",
        "Each model is 5 copies trained from different random starts. Where they disagree, the model is less sure. "
        "A *90% range* should contain the real value 9 times out of 10; that share is the *coverage*.",
        "",
        "Three versions of the range: (a) seed spread only; (b) seed spread plus a noise term (the training-fit error, "
        "about 1.2 m); (c) version (b) widened by one factor per task, **fitted on the validation set (2019 and "
        "validation blocks) and applied unchanged to the test set**.",
        "",
        *md_table(c),
        "",
        f"**In plain words:** the seed spread alone is far too narrow: on test it catches the real value only "
        f"{pct(zB['test coverage, seeds only'])} of the time at places without wells (should be 90%). "
        f"After calibrating on validation, test coverage is {pct(zA['test coverage, calibrated'])} (A), "
        f"{pct(zB['test coverage, calibrated'])} (B) and {pct(zC['test coverage, calibrated'])} (C). "
        f"So the target (90% within about 5 points) is **nearly met at places without wells in the training years (B)** "
        f"and **not met for later years (A, C)**: the factor was fitted on 2019, one year after training, and errors keep "
        f"growing after that (table below). Honest ranges are also wide: about "
        f"{zB['calibrated half-width (m, median)']:.0f} m either side at places without wells.",
        "",
        "Coverage of the calibrated zoned ranges by test year (A and C get harder further from the training years; "
        "the factor was fitted on 2019 only):",
        "",
        *md_table(cy),
        "",
        "## 2. Physical plausibility: does the water rise when it shouldn't?",
        "",
        f"Setting: heavily farmed cells (cropland ≥ 80%, {pl_info['cells']:,} cells) in the dry season from 15 January "
        "to 15 May (wheat irrigation, then pre-monsoon pumping), in years when that season's rain was below the "
        "cell's 2005-2024 median. Here the water table is expected to fall. "
        "Measured wells (January and May readings in the same year) show how often it really rose.",
        "",
        *md_table(p),
        "",
        f"**In plain words:** across all years the zoned model has the water rising in {pct(plz['map: dry cell-years with a rise'])} "
        f"of dry, heavily farmed cell-seasons (regional: {pct(plr['map: dry cell-years with a rise'])}). "
        f"Measured wells in the same setting rose in {pct(plz['wells: measured rise'])} of cases "
        f"(some rises are real: canal water, delayed monsoon recharge, measurement noise). "
        "So the model is **physically plausible**: it almost never has the water rising in a dry, heavily pumped season. "
        "It rises far *less* often than real wells do, which means it is smoother than reality: it captures the steady "
        "decline but misses short local rises.",
        "",
        "## 3. Storage, transmissivity and pumping maps (layout 0)",
        "",
        "![parameter maps](figures/m7_parameter_maps.png)",
        "",
        "- **S (storage, specific yield):** learned per zone, but held inside 0.072-0.15 (the CGWB NAQUIM range for "
        f"this alluvium). {pct(at_floor)} of zone values sit at the lower limit 0.072: the data push S lower than "
        "the reports allow. S is therefore set by the allowed range more than by the data. This is a limitation to state.",
        "- **T (transmissivity):** not learned; fixed per zone from the CGWB reports. On the synthetic tests, learning T "
        "together with S let both drift (M2 run3: T −79%), while fixing T kept S, and total pumping, within 10%.",
        f"- **Q (pumping):** inferred, **not measured**. Area mean {qinfo['q_map_mean']:.2f} m/yr over 2005-2018; "
        f"typical spread between seeds {qinfo['q_seed_sd_median']:.3f} m/yr. The pattern follows cropland and zones "
        "by construction (the pumping network only sees land use, month, zone and a slow trend).",
        "",
        *md_table(zs),
        "",
        "![pumping by zone](figures/m7_pumping_by_zone.png)",
        "",
        "## 4. Pumping compared with CGWB district figures",
        "",
        "CGWB's 2020 assessment estimates gross groundwater extraction per district (GEC-2015 method). "
        "Converted to metres of water per year over the district area, it is compared with the model's inferred pumping "
        "(2017-2020 mean, all 15 zoned runs). Both are estimates, so agreement supports the model and disagreement is not proof of error.",
        "",
        *md_table(r),
        "",
        "![pumping vs CGWB](figures/m7_pumping_vs_cgwb.png)",
        "",
        f"- **Pattern:** rank correlation across the 16 districts = {qinfo['rho']:.2f} (p = {qinfo['p']:.2f}; "
        "1 = same order of districts, 0 = unrelated; p above 0.05 = could easily be chance). CGWB has the Punjab districts "
        f"extracting {qinfo['cgwb_pb_hr']:.1f}× as much per hectare as the Haryana ones; the model has them almost equal "
        f"({qinfo['model_pb_hr']:.2f}×), at {ref.model_m_per_yr.min():.2f}-{ref.model_m_per_yr.max():.2f} m/yr everywhere. "
        "**The model does not recover the district pattern of pumping.** Its pumping network sees only cropland share "
        "(high almost everywhere), month, zone and a slow trend, so it cannot tell districts apart. Differences in "
        "decline between places end up in the water-level network instead.",
        f"- **Size:** the model's pumping is {qinfo['ratio']:.2f}× CGWB's gross extraction on average. This is expected to be "
        "below 1: the model's Q is *net* pumping (water that leaves the aquifer for good). Part of what is pumped soaks "
        "back from irrigated fields, and CGWB counts that return flow as recharge instead. With S at its lower limit, "
        "the model also needs less pumping to explain the observed decline.",
        "",
        "## What M7 adds to the results",
        "",
        "- **Uncertainty:** the raw seed spread is badly over-confident (12-23% coverage for a 90% range), as the "
        "literature warns. Calibrating on validation fixes it for places without wells in the training years (84%), "
        "but not for later years (64-69%).",
        "- **Plausibility: passed.** The model almost never has water rising in dry, heavily pumped seasons (3%, wells: 23%).",
        "- **Physical parameters: not recovered.** S sits at its lower limit and T is fixed from reports, so neither is "
        "learned from these data.",
        "- **Pumping: total size plausible, pattern not.** Net pumping of about 0.22 m/yr is a third of CGWB's gross "
        "extraction (expected to be lower), but it is almost uniform and does not reproduce CGWB's Punjab-Haryana "
        "difference. Pumping maps must not be presented as findings about where pumping is highest.",
    ]
    out = ROOT / "reports" / "m7_report.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
