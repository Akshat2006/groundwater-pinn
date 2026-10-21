"""M3 report: what the master table covers, what QC removed, and the known gaps.

    python scripts/check_m3.py      -> reports/m3_data_gaps.md, reports/figures/m3_*.png, reports/tables/m3_*.csv
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

from src.data.boundaries import load_districts  # noqa: E402
from src.data.canals import load_canals  # noqa: E402
from src.utils.config import data_dir, load_config  # noqa: E402

FIG, TAB = ROOT / "reports" / "figures", ROOT / "reports" / "tables"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REQUIRED = {"well_id": "object", "district": "object", "lat": "float64", "lon": "float64", "x_m": "float64",
            "y_m": "float64", "date": "datetime64", "season": "category", "dtw_m": "float64", "is_dry": "bool",
            "well_depth_m": "float64", "well_type": "object", "aquifer_type": "object", "depth_class": "object",
            "ground_elev_m": "float64", "head_m": "float64", "rain_prev_season_mm": "float64",
            "rain_monsoon_mm": "float64", "recharge_prev_m": "float64", "lu_cropland_frac": "float64",
            "canal_dist_km": "float64", "canal_density": "float64", "qc_flag": "object", "usable": "bool",
            "split_time": "category"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join([str(i)] + [f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in r]) + " |"
              for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join(lines)


def schema_problems(m: pd.DataFrame, bbox: dict) -> list[str]:
    bad = [f"missing column {c}" for c in REQUIRED if c not in m]
    ok = m[m.usable]
    if not ok.dtw_m.between(0, 150).all():
        bad.append("usable depth to water outside 0-150 m")
    if not (ok.lat.between(bbox["south"], bbox["north"]).all() and ok.lon.between(bbox["west"], bbox["east"]).all()):
        bad.append("well outside the download bbox")
    if m.duplicated(["well_id", "date"]).any() and not (m[m.duplicated(["well_id", "date"])].qc_flag == "duplicate").all():
        bad.append("unflagged duplicate readings")
    if m.loc[m.is_dry, "head_m"].notna().any():
        bad.append("dry readings with a head value")
    return bad


def map_figure(static, districts, canals, wells):
    fig, ax = plt.subplots(figsize=(7.2, 7.6), dpi=150)
    ext = [static.x[0] - 500, static.x[-1] + 500, static.y[0] - 500, static.y[-1] + 500]
    elev = static.elev_m.where(static.district_id > 0)
    im = ax.imshow(elev, origin="lower", extent=[e / 1000 for e in ext], cmap="Greys", alpha=0.55,
                   vmin=float(elev.quantile(0.02)), vmax=float(elev.quantile(0.98)))
    for g in districts.geometry.boundary:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.xy
            ax.plot(np.array(xs) / 1000, np.array(ys) / 1000, color=MUTED, lw=0.6)
    for g in canals.geometry:
        xs, ys = g.xy
        ax.plot(np.array(xs) / 1000, np.array(ys) / 1000, color=BLUE, lw=0.3, alpha=0.5)
    sets = [(wells[wells.depth_class == "deep"], GREY, "deep / confined (not in main analysis)", 6),
            (wells[(wells.depth_class == "shallow") & ~wells.tier1], ORANGE, "shallow, other", 8),
            (wells[wells.tier1], AQUA, "shallow, Tier 1 (zoning wells)", 12)]
    for d, c, lab, s in sets:
        ax.scatter(d.x_m / 1000, d.y_m / 1000, s=s, color=c, edgecolor="white", lw=0.3, label=f"{lab}: {len(d)}")
    for _, r in districts.iterrows():
        p = r.geometry.representative_point()
        ax.text(p.x / 1000, p.y / 1000, r.district, fontsize=6, color=INK, ha="center", alpha=0.8)
    ax.set_xlim(ext[0] / 1000, ext[1] / 1000)
    ax.set_ylim(ext[2] / 1000, ext[3] / 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("UTM 43N easting (km)", fontsize=8, color=MUTED)
    ax.set_ylabel("northing (km)", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cb.set_label("ground elevation (m)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Study area: 16 districts, wells, OSM canals (blue)", fontsize=9, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "m3_map.png")
    plt.close(fig)


def coverage_figure(m):
    s = m[(m.depth_class == "shallow") & m.head_m.notna()]
    counts = s.groupby([s.date.dt.year, "season"], observed=False).well_id.nunique().unstack(fill_value=0)
    counts = counts[["jan", "pre", "aug", "post"]]
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=150)
    x = counts.index.values
    w = 0.2
    for k, (col, c) in enumerate(zip(counts.columns, (GREY, ORANGE, BLUE, AQUA))):
        ax.bar(x + (k - 1.5) * w, counts[col], width=w, color=c, label=col)
    top = counts.values.max() * 1.12
    for edge in (2004.5, 2018.5, 2019.5, 2024.5):
        ax.axvline(edge, color=MUTED, lw=0.8, ls="--")
    for xm, lab in ((1997, "history (trends only)"), (2011.5, "train"), (2019, "val"), (2022, "test")):
        ax.text(xm, top, lab, ha="center", fontsize=7, color=MUTED)
    ax.set_ylim(0, top * 1.08)
    ax.set_ylabel("shallow wells with a usable reading", fontsize=8)
    ax.set_title("Wells read per year and season (Jan, pre-monsoon Mar-Jun, Aug, post-monsoon Oct-Dec)",
                 fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="center left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "m3_coverage.png")
    plt.close(fig)
    return counts


def wells_figure(m, wells, seed=0):
    pick = wells[wells.tier1].sample(10, random_state=seed).well_id
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.2), dpi=150, sharex=True)
    for ax, wid in zip(axes.ravel(), pick):
        d = m[m.well_id == wid].sort_values("date")
        ok = d[d.head_m.notna()]
        ax.plot(ok.date, ok.head_m, color=BLUE, lw=0.9, marker="o", ms=1.8)
        bad = d[~d.usable]
        if len(bad):
            ax.scatter(bad.date, d.loc[bad.index, "ground_elev_m"] - bad.dtw_m, color=ORANGE, s=10, zorder=3,
                       label="removed by QC")
        dry = d[d.is_dry]
        if len(dry):
            ax.scatter(dry.date, dry.bottom_head_m, color=GREY, marker="v", s=10, zorder=3, label="dry (bound)")
        w = wells.set_index("well_id").loc[wid]
        ax.set_title(f"{w.station_name[:18]} ({w.district[:9]})", fontsize=7, color=INK, loc="left")
        style(ax)
        ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("head (m above sea level)", fontsize=7)
    axes[1, 0].set_ylabel("head (m above sea level)", fontsize=7)
    handles = {h.get_label(): h for ax in axes.ravel() for h in ax.collections}
    if handles:
        fig.legend(handles.values(), handles.keys(), frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("10 random Tier-1 wells: hydraulic head over time", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "m3_wells.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "study_area.yaml")
    proc, raw = data_dir() / "processed", data_dir() / "raw"
    m = pd.read_parquet(proc / "master.parquet")
    wells = pd.read_parquet(proc / "wells.parquet")
    tel = pd.read_parquet(proc / "telemetry_daily.parquet")
    static = xr.open_dataset(proc / "static_1km.nc").load()
    summary = json.loads((proc / "build_summary.json").read_text())
    districts = load_districts(raw / "boundaries" / "gadm41_IND_2.json.zip", cfg["districts"])
    canals = load_canals(raw / "canals" / "osm" / "canals_overpass.json")

    problems = schema_problems(m, cfg["bbox_deg"])
    map_figure(static, districts, canals, wells)
    counts = coverage_figure(m)
    wells_figure(m, wells)

    sh = m[m.depth_class == "shallow"]
    per_d = (wells.groupby("district")
             .agg(wells=("well_id", "size"), shallow=("depth_class", lambda s: (s == "shallow").sum()),
                  tier1=("tier1", "sum"), outside_gadm=("in_study_area", lambda s: (~s).sum())))
    ok = sh[sh.head_m.notna()]
    for sp in ("history", "train", "val", "test"):
        per_d[f"{sp}_readings"] = ok[ok.split_time == sp].groupby("district").size()
    per_d = per_d.fillna(0).astype(int)
    per_d.loc["Total"] = per_d.sum()
    per_d.to_csv(TAB / "m3_by_district.csv")
    qc = m.qc_flag.value_counts()
    dry_by_type = m[m.usable].groupby(m.well_type.fillna("unknown")).is_dry.agg(["sum", "mean"])
    s_ok = sh[sh.head_m.notna()].sort_values(["well_id", "date"])
    jumps = s_ok.groupby("well_id").head_m.diff().abs()
    long_gap = s_ok.groupby("well_id").date.diff().dt.days > 3 * 365
    meta_missing = {c: float(wells[c].isna().mean()) for c in ("well_type", "well_depth_m", "aquifer_type")}
    tel_wells = tel.groupby("well_id").date.agg(["min", "max", "size"])
    elev_diff = (wells.dem30_elev_m - wells.ground_elev_m)
    split_wells = {sp: ok[ok.split_time == sp].well_id.nunique() for sp in ("train", "val", "test")}

    import yaml
    priors = yaml.safe_load((ROOT / "configs" / "priors.yaml").read_text(encoding="utf-8"))

    def fmt_T(t):
        if t is None:
            return "not available"
        return f"{t[0]:,}-{t[1]:,}" if isinstance(t, list) else f"{t:,}"

    def pct(v):
        return f"{100 * v:.1f}%"

    md = [
        "# M3 report: real-data master table and gaps",
        "",
        "*Generated by `scripts/check_m3.py` from `data/processed/` (built by `scripts/build_master.py`).*",
        "",
        "## What was built",
        "",
        "| File | What | Size |",
        "|---|---|---|",
        f"| `master.parquet` | one row per well × reading, QC flags, head, rainfall windows | {len(m):,} readings, {m.well_id.nunique():,} wells |",
        f"| `wells.parquet` | one row per well: location, type, depth, depth class, land use, canals, Tier 1 | {len(wells):,} wells |",
        f"| `static_1km.nc` | 1 km UTM grid ({summary['grid']['nx']} × {summary['grid']['ny']} cells): districts, elevation, land cover, canals | {(static.district_id > 0).sum().item():,} cells in the study area |",
        "| `rain_daily.nc` | IMD daily rain, native 0.25° grid, 1990-2024 | |",
        f"| `telemetry_daily.parquet` | cleaned daily telemetry (DWLR) depth to water | {tel.well_id.nunique()} wells, {len(tel):,} well-days |",
        "",
        f"**Schema check:** {'all checks pass' if not problems else '; '.join(problems)}.",
        "",
        "## Wells in the analysis",
        "",
        f"- **Shallow wells** (unconfined aquifer): {(wells.depth_class == 'shallow').sum()}. These are the main analysis.",
        f"- **Deep wells** (confined / semi-confined, nested medium or deep piezometers, or ≥ {cfg['qc']['deep_depth_m']:.0f} m deep): {(wells.depth_class == 'deep').sum()}. These are excluded from the main analysis because they tap a different aquifer layer.",
        f"- **Tier 1** (shallow, ≥ {cfg['wells']['tier1_min_years']} training years with both a pre- and post-monsoon head): **{int(wells.tier1.sum())} wells**. These are the zoning wells (M4).",
        f"- **Shallow wells with ≥1 usable head:** train {split_wells['train']}, validation {split_wells['val']}, test {split_wells['test']}.",
        "",
        "**Correction to G0.** G0 counted 193 Tier-1 wells as \"unconfined or unknown aquifer\". But many nested",
        "piezometers named \"... M\" or \"... D\" (medium/deep, median depth about 200-300 m) have a *blank* aquifer field, so G0 wrongly",
        f"included some of them. With the depth class above, Tier 1 is {int(wells.tier1.sum())}. This is still enough for k = 1, 3 and 5 zones.",
        "",
        "![map](figures/m3_map.png)",
        "",
        "### By district (shallow wells; readings = usable heads)",
        "",
        md_table(per_d),
        "",
        "![coverage](figures/m3_coverage.png)",
        "",
        "## Quality control",
        "",
        "| Flag | Readings | Meaning | Used? |",
        "|---|---|---|---|",
        *[f"| {k} | {qc.get(k, 0):,} | {v} | {u} |" for k, v, u in (
            ("ok", "passes all checks", "yes"),
            ("suspect_jump", f"a lasting step of > {cfg['qc']['jump_m']:.0f} m from the previous reading", "yes, flagged"),
            ("spike", f"one or two readings > {cfg['qc']['spike_m']:.0f} m away from both neighbours, which agree", "no"),
            ("outlier", f"> {cfg['qc']['outlier_m']:.0f} m (+1 m per year of gap) from the well's nearest-in-time readings", "no"),
            ("below_bottom", "deeper than the well itself", "no"),
            ("out_of_range", f"depth to water > {cfg['qc']['max_dtw_m']:.0f} m", "no"),
            ("nonpositive", "water at or above ground", "no"),
            ("duplicate", f"same well and day, or the same value repeated within {cfg['qc']['dup_days']} days (a copied entry)", "no"))],
        "",
        f"**Dry readings** (within {cfg['qc']['dry_margin_m']:.0f} m of the well bottom): {int(m.is_dry.sum())}. These are treated as *censored*: the true level is at",
        "or below the bottom, so only that bound is kept (no head value). By well type:",
        "",
        "| Well type | Dry readings | Share |",
        "|---|---|---|",
        *[f"| {k} | {int(r['sum'])} | {pct(r['mean'])} |" for k, r in dry_by_type.iterrows()],
        "",
        f"**Sanity check (PLAN.md M3: no 20 m jumps):** among shallow wells, {int((jumps > 20).sum())} of {int(jumps.notna().sum()):,} consecutive",
        f"head changes exceed 20 m ({int((jumps > 10).sum())} exceed 10 m). {int(((jumps > 20) & long_gap).sum())} of the 20 m changes span a gap of",
        "more than 3 years between readings, which fits real long-term decline (about 1 m a year here). The rest are a handful",
        "of messy wells, kept but flagged where the step is within a year. Spot-check plot of 10 random Tier-1 wells:",
        "",
        "![wells](figures/m3_wells.png)",
        "",
        "## Inputs at each well and on the grid",
        "",
        "| Input | Source | How |",
        "|---|---|---|",
        "| Ground elevation | Copernicus GLO-30 DEM | averaged to ~1 km (smooths village buildings and DEM noise), bilinear at wells |",
        f"| Head | elevation − depth to water | 1 km elevation vs the raw 30 m value at wells: median difference {elev_diff.median():+.1f} m, 90% within ±{elev_diff.abs().quantile(0.9):.1f} m |",
        "| Rainfall | IMD 0.25° daily | bilinear at wells; rain since the previous reading; last monsoon (Jun-Sep) total |",
        f"| Recharge (first estimate) | rain × {cfg['recharge']['rif']} | GEC-2015 norm for Indo-Gangetic alluvium (report Table 4: 22%, range 20-25%) |",
        f"| Land use | ESA WorldCover 2021 | fraction of cropland / built-up / trees per 1 km cell; cropland covers {pct(float(static.lc_cropland_frac.where(static.district_id > 0).mean()))} of the area |",
        f"| Canals | OpenStreetMap | distance to nearest canal (median {wells.canal_dist_km.median():.1f} km at wells) and canal length per km² |",
        "",
        "## Starting values for S and T (from CGWB district reports)",
        "",
        "Full table with page numbers: `configs/priors.yaml`. Belt-wide starting value: "
        f"S = {priors['belt']['S']}, T = {priors['belt']['T_m2_per_day']:,} m²/day.",
        "",
        "| District | T, shallow aquifer (m²/day) | Note |",
        "|---|---|---|",
        *[f"| {d} | {fmt_T(v['T'])} | {v.get('note', '')} |" for d, v in priors["districts"].items()],
        "",
        "**Caveat:** specific yield is almost never measured. The reports use the GEC norm of 12% (or 60% of it, 0.072).",
        "So the S starting value is a convention, and the PINN's S estimate on real data can't be checked against an",
        "independent measurement. T comes from a few pumping tests per district and varies by more than 10× within a district.",
        "",
        "## Known gaps and limitations",
        "",
        f"1. **Karnal before 2019.** WRIS has Karnal only for 2019-2022. The India Data Portal file fills 2013-2021 ({int((m.source == 'idp').sum())} readings). {summary.get('idp_matched_wells', 0)} of its wells were matched to WRIS wells by location and depth suffix, and {summary.get('idp_dropped_duplicates', 0)} duplicate readings were dropped. Unmatched IDP wells have no type, depth or aquifer metadata.",
        f"2. **Missing metadata.** Well type is missing for {pct(meta_missing['well_type'])} of wells, depth for {pct(meta_missing['well_depth_m'])}, aquifer type for {pct(meta_missing['aquifer_type'])}. Wells without depth can't be checked for dry readings.",
        "3. **Network change in 2012** (from G0). Dug wells dominate 2005-2011, tube wells 2012 onwards. Zones (M4) must be checked against well type.",
        "4. **Land use is static.** WorldCover is a single year (2021), so changes in cropping over 2005-2024 aren't captured. Bhuvan LULC 50K (2005-06, 2011-12, 2015-16) is on disk only as rendered map tiles; decoding it into crop-season fractions is a possible later improvement.",
        "5. **Canals are incomplete.** OSM maps main canals and branches well but misses many small distributaries, so canal density is a lower bound. Canal seepage is not in the recharge estimate yet.",
        "6. **Elevation is a surface model.** Copernicus is a DSM (includes buildings and trees). The 1 km averaging reduces this, but heads can carry a few metres of constant error per well. This affects head gradients (and so T), not water-level changes over time.",
        f"7. **Wells outside the GADM district outlines:** {int((~wells.in_study_area).sum())}. These are near borders (WRIS and GADM boundaries differ); they are kept and flagged.",
        f"8. **Telemetry** is noisy: {summary.get('telemetry_raw_rows', 0):,} raw 6-hourly values were cleaned to {len(tel):,} well-days ({tel.well_id.nunique()} wells, median {int(tel_wells['size'].median())} days each). Used only as a bonus within-season test.",
        "9. **Few readings per year.** Most wells are read 4 times a year (Jan, pre-monsoon, Aug, post-monsoon). M2 showed this sampling is where the PINN currently struggles.",
        "",
        "## What the terms mean",
        "",
        "- **Depth to water:** what CGWB measures, metres from the ground down to the water.",
        "- **Head:** the height of the water table above sea level (elevation − depth to water). The flow equation works with head.",
        "- **Censored reading:** we only know the water is *at least* this deep (the well was dry), not the exact level.",
        "- **Tier 1:** wells with long enough records to describe their seasonal and long-term behaviour, used to form zones.",
    ]
    (ROOT / "reports" / "m3_data_gaps.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("schema:", "ok" if not problems else problems)
    print("wrote reports/m3_data_gaps.md")


if __name__ == "__main__":
    main()
