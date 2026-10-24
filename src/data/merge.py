"""Build the M3 outputs (PLAN.md §4.1):

    data/processed/master.parquet     one row per well × reading (manual CGWB readings, QC'd)
    data/processed/wells.parquet      one row per well (static attributes, tiers)
    data/processed/static_1km.nc      1 km UTM grid: district id, elevation, land cover, canals
    data/processed/rain_daily.nc      IMD daily rain on its native 0.25° grid, study bbox
    data/processed/telemetry_daily.parquet   cleaned daily DWLR depth to water (bonus test data)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data.boundaries import km_outside_district, load_districts, rasterize_districts
from src.data.canals import canal_layers, distance_to_canal, load_canals
from src.data.cgwb import assemble_manual, depth_class, nest_level, telemetry_daily
from src.data.imd import load_rain, rain_at_points
from src.data.rasters import dem_at_points, dem_on_grid, landcover_on_grid, lonlat_bounds
from src.features.head import add_head, qc_levels, season_of
from src.utils.geo import Grid, bilinear, to_utm

SPLITS = ("history", "train", "val", "test", "after")


def build_static(cfg: dict, raw: Path) -> tuple[Grid, xr.Dataset, "gpd.GeoDataFrame", "gpd.GeoDataFrame"]:
    districts = load_districts(raw / "boundaries" / "gadm41_IND_2.json.zip", cfg["districts"])
    grid = Grid.around(tuple(districts.total_bounds), cfg["grid"]["res_m"], cfg["grid"]["buffer_m"])
    did = rasterize_districts(districts, grid)
    elev = dem_on_grid(raw / "dem", grid)
    lc = landcover_on_grid(raw / "worldcover", grid)
    canals = load_canals(raw / "canals" / "osm" / "canals_overpass.json")
    cdist, cdens = canal_layers(canals, grid)
    dims = ("y", "x")
    ds = xr.Dataset(
        {"district_id": (dims, did), "elev_m": (dims, elev),
         **{f"lc_{k}_frac": (dims, v) for k, v in lc.items()},
         "canal_dist_km": (dims, cdist), "canal_density_km_per_km2": (dims, cdens)},
        coords={"x": grid.x, "y": grid.y})
    ds.attrs.update(crs="EPSG:32643", res_m=grid.res,
                    district_names=";".join(f"{i}={d}" for i, d in zip(districts.district_id, districts.district)),
                    sources="Copernicus GLO-30 (elev, 30 arc-sec block mean then 1 km average); "
                            "ESA WorldCover 2021 (land-cover fractions); OSM waterway=canal; GADM 4.1")
    return grid, ds, districts, canals


def split_time(dates: pd.Series, period: dict) -> pd.Series:
    t0, t1 = pd.Timestamp(period["train"][0]), pd.Timestamp(period["train"][1])
    v1, s1 = pd.Timestamp(period["val"][1]), pd.Timestamp(period["test"][1])
    lab = np.select([dates < t0, dates <= t1, dates <= v1, dates <= s1], ["history", "train", "val", "test"], "after")
    return pd.Categorical(lab, categories=SPLITS)


def rain_windows(df: pd.DataFrame, wells: pd.DataFrame, rain: xr.DataArray, max_gap_days: int) -> pd.DataFrame:
    """rain_prev_season_mm: rain at the well since its previous usable reading (NaN if none within max_gap_days).
    rain_monsoon_mm: Jun-Sep rain of the latest monsoon finished by the reading date."""
    series = rain_at_points(rain, wells.lon.values, wells.lat.values)          # (time, point)
    cum = np.vstack([np.zeros((1, len(wells)), dtype=np.float64), np.nancumsum(series.values, axis=0)])
    t0 = pd.Timestamp(rain.time.values[0])
    col = pd.Series(np.arange(len(wells)), index=wells.well_id)
    month = series.time.dt.month
    monsoon = series.where((month >= 6) & (month <= 9)).groupby("time.year").sum("time")   # (year, point)

    df = df.copy()
    df["rain_prev_season_mm"] = np.nan
    df["rain_monsoon_mm"] = np.nan
    p = col.reindex(df.well_id).to_numpy()
    day = (df["date"] - t0).dt.days.to_numpy()
    inside = (day >= 0) & (day < series.sizes["time"])
    mon_year = np.where(df["date"].dt.month >= 10, df["date"].dt.year, df["date"].dt.year - 1)
    yrs = monsoon.year.values
    ok = inside & np.isin(mon_year, yrs)
    yi = np.searchsorted(yrs, mon_year[ok])
    df.loc[ok, "rain_monsoon_mm"] = monsoon.values[yi, p[ok]]
    use = df["usable"].to_numpy()
    for _, idx in df[use].groupby("well_id").indices.items():
        rows = np.flatnonzero(use)[idx]
        d, pp = day[rows], p[rows]
        prev = np.r_[np.nan, d[:-1]]
        good = np.isfinite(prev) & (d - prev <= max_gap_days) & (prev >= 0) & (d < series.sizes["time"])
        r = rows[good]
        df.loc[df.index[r], "rain_prev_season_mm"] = (cum[d[good] + 1, pp[good]] - cum[prev[good].astype(int) + 1, pp[good]])
    return df


def tier1(df: pd.DataFrame, period: dict, min_years: int) -> pd.Series:
    """Wells with >= min_years training years that have both a usable pre- and post-monsoon head."""
    t = df[(df.split_time == "train") & df.head_m.notna()]
    yrs = t.assign(year=t.date.dt.year).groupby(["well_id", "year"]).season.agg(lambda s: {"pre", "post"} <= set(s))
    return yrs.groupby("well_id").sum() >= min_years


def build_all(cfg: dict, raw: Path, out: Path, verbose: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    q = cfg["qc"]

    log("static layers ...")
    grid, static, districts, canals = build_static(cfg, raw)
    static.to_netcdf(out / "static_1km.nc")

    log("rainfall ...")
    p = cfg["period"]
    y0, y1 = int(p["history_start"][:4]), int(p["test"][1][:4])
    rain = load_rain(raw / "imd_rain", y0, y1, lonlat_bounds(grid))
    rain.to_netcdf(out / "rain_daily.nc")

    log("well readings + QC ...")
    idp = raw / "cgwb_levels" / "idp" / "cgwb-changes-in-depth-to-water-level.csv"
    fill = {s: [d for d in ds if d in cfg.get("idp_fill_districts", [])] for s, ds in cfg["districts"].items()}
    m = assemble_manual(raw / "cgwb_levels" / "wris", idp, {s: v for s, v in fill.items() if v})
    attrs = dict(m.attrs)
    m = qc_levels(m, spike_m=q["spike_m"], jump_m=q["jump_m"], max_dtw_m=q["max_dtw_m"],
                  dry_margin_m=q["dry_margin_m"], max_gap_days=q["max_gap_days"],
                  spike_gap_days=q["spike_gap_days"], dup_days=q["dup_days"], outlier_m=q["outlier_m"])

    wells = (m.groupby("well_id")
             .agg(station_name=("station_name", "first"), state=("state", "first"), district=("district", "first"),
                  lat=("lat", "first"), lon=("lon", "first"), well_type=("well_type", "first"),
                  well_depth_m=("well_depth_m", "first"), aquifer_type=("aquifer_type", "first"),
                  source=("source", "first"), n_readings=("dtw_m", "size"))
             .reset_index())
    wells["x_m"], wells["y_m"] = to_utm(wells.lon.values, wells.lat.values)
    wells["nest"] = nest_level(wells.station_name)
    wells["depth_class"] = depth_class(wells.aquifer_type, wells.nest, wells.well_depth_m, q["deep_depth_m"])
    wells["ground_elev_m"] = bilinear(static.elev_m.values, grid, wells.x_m, wells.y_m)
    wells["dem30_elev_m"] = dem_at_points(raw / "dem", wells.lon, wells.lat)
    wells["in_study_area"] = grid.sample(static.district_id.values, wells.x_m, wells.y_m) > 0
    # coordinates far outside the well's own district are typing errors in the source: the well's elevation and
    # neighbours would be wrong, so its readings are not used (district outlines differ by a few km between sources)
    wells["km_outside_district"] = km_outside_district(districts, wells.district, wells.x_m, wells.y_m)
    bad_loc = wells.well_id[wells.km_outside_district > q["max_km_outside_district"]]
    m.loc[m.well_id.isin(bad_loc), ["qc_flag", "usable"]] = ["bad_location", False]
    for k in ("cropland", "builtup", "tree"):
        wells[f"lu_{k}_frac"] = grid.sample(static[f"lc_{k}_frac"].values, wells.x_m, wells.y_m)
    wells["canal_dist_km"] = distance_to_canal(canals, wells.x_m, wells.y_m)
    wells["canal_density"] = grid.sample(static.canal_density_km_per_km2.values, wells.x_m, wells.y_m)

    m = m.merge(wells[["well_id", "x_m", "y_m", "ground_elev_m", "depth_class", "nest", "lu_cropland_frac",
                       "lu_builtup_frac", "canal_dist_km", "canal_density"]], on="well_id", how="left")
    m = add_head(m, m.pop("ground_elev_m").to_numpy())
    m["season"] = season_of(m["date"])
    m["split_time"] = split_time(m["date"], p)

    log("rain windows ...")
    m = rain_windows(m, wells, rain, q["max_gap_days"])
    m["recharge_prev_m"] = cfg["recharge"]["rif"] * m["rain_prev_season_mm"] / 1000.0
    for c in ("split_space", "zone_behaviour", "zone_map"):        # filled in M4/M5
        m[c] = pd.NA

    t1 = tier1(m[m.depth_class == "shallow"], p, cfg["wells"]["tier1_min_years"])
    wells["tier1"] = wells.well_id.map(t1).fillna(False).astype(bool)
    wells.to_parquet(out / "wells.parquet", index=False)
    m.to_parquet(out / "master.parquet", index=False)

    log("telemetry ...")
    tel = telemetry_daily(raw / "cgwb_levels" / "wris", q["telemetry_max_dtw_m"], q["spike_m"])
    tel_attrs = dict(tel.attrs)
    tel.to_parquet(out / "telemetry_daily.parquet", index=False)
    summary = {"grid": grid.to_dict(), "n_readings": len(m), "n_wells": len(wells), **attrs,
               "telemetry_raw_rows": tel_attrs.get("n_raw"), "telemetry_days": len(tel)}
    log(summary)
    return summary
