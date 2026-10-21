"""OpenStreetMap canals (waterway=canal) -> distance and density on the model grid and at wells.

OSM is the fallback chosen in PLAN.md §2.7 (India-WRIS canal layers are not downloadable). Its coverage
of small distributaries is incomplete, so canal density is a lower bound; this is noted in the M3 report.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString

from src.utils.geo import CRS_GEO, CRS_MODEL, Grid


def load_canals(path: str | Path) -> gpd.GeoDataFrame:
    """Overpass JSON (ways with geometry) -> lines in UTM 43N with name, usage, width."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for e in data["elements"]:
        pts = e.get("geometry") or []
        if e.get("type") != "way" or len(pts) < 2:
            continue
        tags = e.get("tags", {})
        rows.append({"osm_id": e["id"], "name": tags.get("name"), "usage": tags.get("usage"),
                     "geometry": LineString([(p["lon"], p["lat"]) for p in pts])})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_GEO).to_crs(CRS_MODEL)


def canal_layers(canals: gpd.GeoDataFrame, grid: Grid, fine: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """(distance from cell centre to nearest canal [km], canal length per cell [km/km²]), both (ny, nx).

    Lines are burned onto a grid `fine` times finer; density = burned fine cells × fine-cell size.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    fres = grid.res / fine
    shape = (grid.ny * fine, grid.nx * fine)
    tr = from_origin(grid.x0, grid.y0 + grid.ny * grid.res, fres, fres)
    burned = rasterize(((g, 1) for g in canals.geometry), out_shape=shape, transform=tr, fill=0, dtype="uint8")
    burned = burned[::-1]
    length_km = burned.reshape(grid.ny, fine, grid.nx, fine).sum(axis=(1, 3)) * fres / 1000.0
    density = length_km / (grid.res / 1000.0) ** 2
    dist_fine = distance_transform_edt(burned == 0) * fres / 1000.0
    centre = fine // 2
    return dist_fine[centre::fine, centre::fine].astype(np.float32), density.astype(np.float32)


def distance_to_canal(canals: gpd.GeoDataFrame, x, y) -> np.ndarray:
    """Exact distance (km) from each point to the nearest canal line."""
    pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=CRS_MODEL)
    idx = canals.sindex.nearest(pts, return_all=False)
    near = canals.geometry.values[idx[1]]
    return np.asarray(pts.values[idx[0]].distance(near)) / 1000.0
