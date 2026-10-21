"""District boundaries (GADM 4.1 level 2) and the study-area mask on the model grid."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np

from src.utils.geo import CRS_MODEL, Grid


def _key(name: str) -> str:
    return "".join(str(name).lower().split())          # GADM writes "FatehgarhSahib"


def load_districts(path: str | Path, districts: dict[str, list[str]]) -> gpd.GeoDataFrame:
    """The configured districts as polygons in UTM 43N, columns: state, district, district_id (1..n)."""
    path = Path(path)
    g = gpd.read_file(f"zip://{path}" if path.suffix == ".zip" else path)
    g["k1"], g["k2"] = g.NAME_1.map(_key), g.NAME_2.map(_key)
    rows = []
    for state, names in districts.items():
        for name in names:
            hit = g[(g.k1 == _key(state)) & (g.k2 == _key(name))]
            if len(hit) != 1:
                raise ValueError(f"district {state}/{name}: {len(hit)} matches in {path.name}")
            rows.append(hit.assign(state=state, district=name))
    out = gpd.GeoDataFrame(np.concatenate([r[["state", "district", "geometry"]].values for r in rows]),
                           columns=["state", "district", "geometry"], geometry="geometry", crs=g.crs)
    out["district_id"] = np.arange(1, len(out) + 1)
    return out.to_crs(CRS_MODEL)


def rasterize_districts(districts: gpd.GeoDataFrame, grid: Grid) -> np.ndarray:
    """(ny, nx) int array: district_id of the district containing each cell centre, 0 outside."""
    from rasterio.features import rasterize
    shapes = zip(districts.geometry, districts.district_id)
    north_up = rasterize(shapes, out_shape=(grid.ny, grid.nx), transform=grid.transform_north_up(),
                         fill=0, dtype="int16")
    return north_up[::-1].copy()
