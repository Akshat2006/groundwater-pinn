"""Large lon/lat rasters (Copernicus DEM, ESA WorldCover) -> 1 km model grid.

Tiles are read window by window, reduced to a coarse lon/lat grid by block averaging, then
reprojected to the UTM grid with area-weighted averaging. Nothing larger than one window is in memory.

- elevation: 30 m DEM -> mean over 30 arc-second blocks (~0.8-0.9 km) -> 1 km cells. This is the
  ~1 km smoothing PLAN.md §2.7 asks for; the 30 m value at each well is kept separately for QC.
- land cover: 10 m classes -> fraction of each class over 0.01° blocks (~1 km) -> 1 km cells.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from src.utils.geo import CRS_GEO, CRS_MODEL, Grid, to_lonlat

WORLDCOVER = {"tree": 10, "cropland": 40, "builtup": 50, "water": 80}


def lonlat_bounds(grid: Grid, pad_deg: float = 0.05) -> tuple[float, float, float, float]:
    xs = np.linspace(grid.bounds[0], grid.bounds[2], 50)
    ys = np.linspace(grid.bounds[1], grid.bounds[3], 50)
    X, Y = np.meshgrid(xs, ys)
    lon, lat = to_lonlat(X.ravel(), Y.ravel())
    return lon.min() - pad_deg, lat.min() - pad_deg, lon.max() + pad_deg, lat.max() + pad_deg


def coarsen_tiles(files: list[Path], bbox: tuple[float, float, float, float], block: int,
                  reduce: Callable[[np.ndarray], dict[str, np.ndarray]], window: int = 3600) -> tuple[dict, object]:
    """Block-reduce lon/lat tiles onto one coarse lon/lat mosaic covering `bbox`.

    reduce(arr (h, w)) -> {name: array (h // block, w // block)}. All tiles must share one pixel size.
    Returns ({name: mosaic (rows north->south)}, affine transform of the mosaic).
    """
    with rasterio.open(files[0]) as r:
        px = r.res[0]
    cdeg = px * block
    west, south, east, north = bbox
    west, north = np.floor(west / cdeg) * cdeg, np.ceil(north / cdeg) * cdeg
    ncol, nrow = int(np.ceil((east - west) / cdeg)), int(np.ceil((north - south) / cdeg))
    out: dict[str, np.ndarray] = {}
    for f in files:
        with rasterio.open(f) as r:
            b = r.bounds
            if b.right <= west or b.left >= east or b.top <= south or b.bottom >= north:
                continue
            for row0 in range(0, r.height, window):
                for col0 in range(0, r.width, window):
                    h, w = min(window, r.height - row0), min(window, r.width - col0)
                    h, w = h - h % block, w - w % block
                    if h == 0 or w == 0:
                        continue
                    lon0 = b.left + col0 * px
                    lat0 = b.top - row0 * px
                    if lon0 >= east or lon0 + w * px <= west or lat0 <= south or lat0 - h * px >= north:
                        continue
                    arr = r.read(1, window=rasterio.windows.Window(col0, row0, w, h))
                    for name, red in reduce(arr).items():
                        m = out.setdefault(name, np.full((nrow, ncol), np.nan, dtype=np.float32))
                        c0 = int(round((lon0 - west) / cdeg))
                        r0 = int(round((north - lat0) / cdeg))
                        rr0, cc0 = max(r0, 0), max(c0, 0)
                        rr1, cc1 = min(r0 + red.shape[0], nrow), min(c0 + red.shape[1], ncol)
                        if rr1 > rr0 and cc1 > cc0:
                            m[rr0:rr1, cc0:cc1] = red[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
    return out, from_origin(west, north, cdeg, cdeg)


def block_mean(arr: np.ndarray, block: int) -> np.ndarray:
    h, w = arr.shape
    return arr.reshape(h // block, block, w // block, block).mean(axis=(1, 3))


def to_grid(mosaic: np.ndarray, transform, grid: Grid, resampling=Resampling.average) -> np.ndarray:
    """Reproject a lon/lat mosaic to the model grid; returns (ny, nx) with y increasing."""
    dst = np.full((grid.ny, grid.nx), np.nan, dtype=np.float32)
    reproject(mosaic, dst, src_transform=transform, src_crs=CRS_GEO, src_nodata=np.nan,
              dst_transform=grid.transform_north_up(), dst_crs=CRS_MODEL, dst_nodata=np.nan,
              resampling=resampling)
    return dst[::-1].copy()


def dem_on_grid(dem_dir: str | Path, grid: Grid) -> np.ndarray:
    """Mean ground elevation (m, EGM2008) per 1 km cell from Copernicus GLO-30 tiles."""
    files = sorted(Path(dem_dir).glob("Copernicus_DSM_*_DEM.tif"))
    mos, tr = coarsen_tiles(files, lonlat_bounds(grid), 30, lambda a: {"elev": block_mean(a.astype(np.float32), 30)})
    return to_grid(mos["elev"], tr, grid)


def dem_at_points(dem_dir: str | Path, lon, lat) -> np.ndarray:
    """Unsmoothed 30 m DEM value at each point (QC only)."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    out = np.full(lon.shape, np.nan)
    for f in sorted(Path(dem_dir).glob("Copernicus_DSM_*_DEM.tif")):
        with rasterio.open(f) as r:
            b = r.bounds
            inside = (lon >= b.left) & (lon < b.right) & (lat > b.bottom) & (lat <= b.top)
            if inside.any():
                out[inside] = [v[0] for v in r.sample(zip(lon[inside], lat[inside]))]
    return out


def landcover_on_grid(wc_dir: str | Path, grid: Grid, classes: dict[str, int] = WORLDCOVER) -> dict[str, np.ndarray]:
    """Fraction (0-1) of each WorldCover class per 1 km cell."""
    files = sorted(Path(wc_dir).glob("ESA_WorldCover_10m_*_Map.tif"))
    block = 120                                           # 120 × 10 m pixels (1/12000°) = 0.01°

    def reduce(a):
        valid = a > 0
        n = block_mean(valid.astype(np.float32), block)
        return {k: np.where(n > 0, block_mean((a == v).astype(np.float32), block) / np.maximum(n, 1e-9), np.nan)
                for k, v in classes.items()}
    mos, tr = coarsen_tiles(files, lonlat_bounds(grid), block, reduce, window=3600)
    return {k: to_grid(v, tr, grid) for k, v in mos.items()}
