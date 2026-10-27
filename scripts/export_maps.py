"""Export ready-made water-level maps from trained PINN runs, for the website (no PyTorch needed to show them).

    python scripts/export_maps.py --version draft-batch3 --stage val     # draft from the batch 3 model
    python scripts/export_maps.py --version model-v1.0 --stage test      # final, after the test set is scored

Default model: the shared zoned PINN with canals (tag tbc) on layout 0, every seed found (an ensemble).
Each month 2005-01 .. 2024-12 is predicted on the 1 km study-area grid (data/processed/static_1km.nc) on the 15th.
Map = mean over seeds; uncertainty = spread (standard deviation) over seeds (0 with a single seed).
Depth to water = ground elevation (same 1 km grid) - head. Per-well offsets are not used: every cell is treated
as a place with no well, so the map's accuracy is the "no well" accuracy away from wells.

Output (exports/<version>/, gitignored; ship it as a zip):
- water_level_monthly.nc        head_m, depth_m, spread_m on (time, y, x), UTM 43N (EPSG:32643)
- geotiff/depth_YYYY-MM.tif     depth to water [m], north-up, EPSG:32643, NaN outside the study area
- geotiff/spread_YYYY-MM.tif    uncertainty [m]
- study_area.geojson, districts.geojson, zones.geojson, wells.geojson   (WGS84, for web maps)
- not_shown.geojson             cells left blank (foothills above the wells' range, or water above ground)
- metadata.json                 grid, dates, units, model, accuracy, disclaimers
- preview.png                   two example months, to check the export by eye
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.real_data import load_real_run, predict_heads  # noqa: E402
from src.utils.config import data_dir, runs_dir  # noqa: E402

CRS = "EPSG:32643"
DISCLAIMERS = [
    "Model estimate (physics-informed neural network), not measured data and not a live feed.",
    "Detail is about 1 km: inputs (land use, canals) are on a 1 km grid; the map is smooth below that.",
    "Years after 2018 are a hindcast: the model was trained on 2005-2018 readings and driven by observed rainfall.",
    "Shallow water table only, inside the 16-district Punjab-Haryana study area.",
    "Pumping and aquifer values behind the map are inferred by the model, not measured.",
    "The uncertainty layer shows how much 5 versions of the model disagree (about 1 m). The real error is larger "
    "(see the accuracy line), so treat it as a relative guide to where the model is less sure, not as an error bar.",
]


def shapes_to_gdf(values: np.ndarray, transform, name: str, labels: dict | None = None):
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape
    from shapely.ops import unary_union

    geoms = {}
    for geom, v in features.shapes(values.astype(np.int16), mask=values > 0, transform=transform):
        geoms.setdefault(int(v), []).append(shape(geom))
    rows = [{name: k, **({"name": labels[k]} if labels else {}), "geometry": unary_union(g)} for k, g in sorted(geoms.items())]
    return gpd.GeoDataFrame(rows, crs=CRS).to_crs("EPSG:4326")


def accuracy_from_table(stage: str, model_prefix: str) -> dict:
    """Typical errors of the exported model from reports/tables/m6_real_pinn_<stage>.csv (ensemble row if present)."""
    path = ROOT / "reports" / "tables" / f"m6_real_pinn_{stage}.csv"
    if not path.exists():
        return {}
    t = pd.read_csv(path)
    out = {}
    for task, label in [("A", "near_monitored_wells_rmse_m"), ("B", "no_well_rmse_m"), ("C", "no_well_later_years_rmse_m")]:
        rows = t[(t.task == task) & t.model.str.startswith(model_prefix)]
        if rows.empty:
            continue
        ens = rows[rows.model.str.contains("mean of")]
        r = (ens if len(ens) else rows).groupby("model").rmse_m.mean()
        out[label] = round(float(r.iloc[0] if len(ens) else r.mean()), 2)
    if out:
        out["stage"] = stage
        out["meaning"] = "RMSE = typical error in metres on wells the model never saw (" + (
            "final test set" if stage == "test" else "validation set; final numbers follow") + ")."
        if "no_well_rmse_m" in out and "no_well_later_years_rmse_m" in out:
            # the map has no per-well corrections, so the no-well errors are the ones that apply to it
            out["text"] = (f"Typical error: about {out['no_well_rmse_m']:.0f} m at places without a monitoring well "
                           f"(2005-2018) and about {out['no_well_later_years_rmse_m']:.0f} m in 2019-2024.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="*_real_L0_kmeansk5_shared_s*_tbc", help="run folders to average (glob)")
    ap.add_argument("--version", required=True)
    ap.add_argument("--stage", default="val", choices=["val", "test"], help="which accuracy table to quote")
    ap.add_argument("--start", default="2005-01")
    ap.add_argument("--end", default="2024-12")
    args = ap.parse_args()

    import rasterio
    from rasterio.transform import from_origin

    proc = data_dir() / "processed"
    static = xr.open_dataset(proc / "static_1km.nc").load()
    runs = sorted(d for d in runs_dir().glob(args.runs) if (d / "model.pt").exists())
    if not runs:
        sys.exit(f"no trained runs match {args.runs} in {runs_dir()}")
    out = ROOT / "exports" / args.version
    (out / "geotiff").mkdir(parents=True, exist_ok=True)

    mask = static.district_id.values > 0
    yy, xx = np.meshgrid(static.y.values, static.x.values, indexing="ij")
    xs, ys = xx[mask], yy[mask]
    dates = pd.date_range(args.start, args.end, freq="MS") + pd.Timedelta(days=14)
    n = len(xs)

    # running mean and variance over seeds (Welford), one (time, cell) array each
    t0 = time.time()
    mean = np.zeros((len(dates), n))
    m2 = np.zeros_like(mean)
    cfgs = []
    for i, d in enumerate(runs, 1):
        model, cfg = load_real_run(d, static)
        cfgs.append(cfg)
        h = predict_heads(model, np.tile(xs, len(dates)), np.tile(ys, len(dates)),
                          np.repeat(dates.values, n)).reshape(len(dates), n)
        delta = h - mean
        mean += delta / i
        m2 += delta * (h - mean)
        print(f"  {d.name}: done ({time.time() - t0:.0f} s)")
    spread = np.sqrt(m2 / (len(runs) - 1)) if len(runs) > 1 else np.zeros_like(mean)
    elev = static.elev_m.values[mask]
    depth = elev[None, :] - mean

    # Cells the model cannot speak for are left blank (and listed in not_shown.geojson), not clipped:
    # ground higher than 99% of the monitoring wells (Shivalik foothills: hilly, few wells, outside the alluvium the
    # model is built for) or water above ground in any month (physically impossible for this aquifer).
    w = pd.read_parquet(proc / "wells.parquet")
    w = w[(w.depth_class == "shallow") & w.in_study_area.astype(bool)]
    elev_lim = float(np.percentile(w.ground_elev_m.dropna(), 99))
    high, above = elev > elev_lim, (depth < 0).any(axis=0)
    blank = high | above
    for a in (mean, depth, spread):
        a[:, blank] = np.nan
    shown = mask.copy()
    shown[mask] = ~blank

    def grid(v):   # (time, cell) -> (time, y, x) with NaN outside the study area
        g = np.full((len(dates), *mask.shape), np.nan, dtype=np.float32)
        g[:, mask] = v
        return g

    enc = {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -32768, "zlib": True, "complevel": 5}
    ds = xr.Dataset(
        {"head_m": (("time", "y", "x"), grid(mean), {"long_name": "water-table elevation (hydraulic head)", "units": "m above sea level"}),
         "depth_m": (("time", "y", "x"), grid(depth), {"long_name": "depth to water below ground", "units": "m"}),
         "spread_m": (("time", "y", "x"), grid(spread), {"long_name": "uncertainty: spread over the ensemble seeds", "units": "m"})},
        coords={"time": dates, "y": static.y.values, "x": static.x.values},
        attrs={"crs": CRS, "res_m": 1000.0, "model_version": args.version, "n_seeds": len(runs)})
    ds.to_netcdf(out / "water_level_monthly.nc", encoding={v: enc for v in ds.data_vars})

    # GeoTIFFs: north-up (row 0 = northernmost), cell-centre coordinates -> corner origin
    res = 1000.0
    transform = from_origin(static.x.values.min() - res / 2, static.y.values.max() + res / 2, res, res)
    y_ascending = static.y.values[0] < static.y.values[-1]   # grid rows run south -> north
    flip = (lambda a: a[::-1]) if y_ascending else (lambda a: a)
    prof = {"driver": "GTiff", "height": mask.shape[0], "width": mask.shape[1], "count": 1, "dtype": "float32",
            "crs": CRS, "transform": transform, "nodata": np.nan, "compress": "deflate", "predictor": 3}
    for var, data in (("depth", ds.depth_m.values), ("spread", ds.spread_m.values)):
        for k, dt in enumerate(dates):
            with rasterio.open(out / "geotiff" / f"{var}_{dt:%Y-%m}.tif", "w", **prof) as f:
                f.write(flip(data[k]).astype(np.float32), 1)

    # map layers in WGS84
    names = dict(kv.split("=") for kv in static.attrs["district_names"].split(";"))
    shapes_to_gdf(flip(mask.astype(np.int16)), transform, "study_area").to_file(out / "study_area.geojson", driver="GeoJSON")
    shapes_to_gdf(flip(static.district_id.values), transform, "district_id",
                  {int(k): v for k, v in names.items()}).to_file(out / "districts.geojson", driver="GeoJSON")
    c = cfgs[0]
    if c.get("shared"):
        zones = xr.open_dataset(proc / "zones.nc").load()
        zmap = zones.zone.sel({"method": c["method"], "layout": c["layout"], "k": c["k"]}).values * mask
        shapes_to_gdf(flip(zmap), transform, "zone").to_file(out / "zones.geojson", driver="GeoJSON")
    if blank.any():
        shapes_to_gdf(flip((mask & ~shown).astype(np.int16)), transform, "not_shown").to_file(
            out / "not_shown.geojson", driver="GeoJSON")
    import geopandas as gpd
    gpd.GeoDataFrame(w[["well_id", "station_name", "district", "well_type", "n_readings"]].astype({"well_id": str}),
                     geometry=gpd.points_from_xy(w.lon, w.lat), crs="EPSG:4326").to_file(out / "wells.geojson", driver="GeoJSON")

    meta = {
        "model_version": args.version,
        "model": "shared zoned physics-informed neural network (PINN) with canal seepage",
        "runs": [d.name for d in runs], "n_seeds": len(runs),
        "grid": {"crs": CRS, "cell_m": 1000, "width": int(mask.shape[1]), "height": int(mask.shape[0]),
                 "bounds_m": [float(transform.c), float(transform.f - res * mask.shape[0]),
                              float(transform.c + res * mask.shape[1]), float(transform.f)],
                 "cells_in_study_area": int(n), "cells_shown": int(shown.sum())},
        "not_shown": {"rule": f"blank where the ground is higher than 99% of the monitoring wells ({elev_lim:.0f} m: "
                              "Shivalik foothills, outside the alluvium the model is built for) or where the model "
                              "puts water above ground in any month",
                      "cells_too_high": int(high.sum()), "cells_water_above_ground": int((above & ~high).sum()),
                      "layer": "not_shown.geojson"},
        "dates": [f"{d:%Y-%m}" for d in dates], "date_day": 15,
        "variables": {"depth": "depth to water below ground [m] (bigger = deeper water)",
                      "spread": "uncertainty [m]: spread over the seeds (0 with one seed); smaller than the real error, a relative guide only",
                      "head_m (NetCDF only)": "water-table elevation [m above sea level]"},
        "accuracy": accuracy_from_table(args.stage, "PINN kmeansk5 shared tbc"),
        "disclaimers": DISCLAIMERS,
        "summary": {"depth_m_min": round(float(np.nanmin(depth)), 1), "depth_m_median": round(float(np.nanmedian(depth)), 1),
                    "depth_m_max": round(float(np.nanmax(depth)), 1)},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = [k for k, d in enumerate(dates) if f"{d:%Y-%m}" in ("2005-11", "2018-11")] or [0, len(dates) - 1]
    vmin, vmax = np.nanpercentile(depth, [2, 98])
    fig, axs = plt.subplots(1, len(pick), figsize=(6 * len(pick), 6))
    for ax, k in zip(np.atleast_1d(axs), pick):
        im = ax.imshow(flip(ds.depth_m.values[k]), cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"Depth to water, {dates[k]:%b %Y} (model estimate)")
        ax.axis("off")
    fig.colorbar(im, ax=axs, shrink=0.7, label="m below ground")
    fig.savefig(out / "preview.png", dpi=110, bbox_inches="tight")
    print(f"wrote {out} ({len(runs)} seeds, {len(dates)} months, {n} cells, {time.time() - t0:.0f} s); "
          f"depth median {meta['summary']['depth_m_median']} m")


if __name__ == "__main__":
    main()
