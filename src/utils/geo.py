"""Coordinates and the 1 km model grid.

All modelling uses UTM zone 43N (EPSG:32643, metres). The model grid is a north-up raster whose cell
edges sit on whole multiples of the resolution, so grids built at different times line up exactly.
Arrays on the grid are stored (y, x) with y increasing (row 0 = southernmost), like the synthetic cases.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

CRS_MODEL = "EPSG:32643"
CRS_GEO = "EPSG:4326"
_to_utm = Transformer.from_crs(CRS_GEO, CRS_MODEL, always_xy=True)
_to_geo = Transformer.from_crs(CRS_MODEL, CRS_GEO, always_xy=True)


def to_utm(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    return _to_utm.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


def to_lonlat(x, y) -> tuple[np.ndarray, np.ndarray]:
    return _to_geo.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


@dataclass(frozen=True)
class Grid:
    x0: float          # west edge (m)
    y0: float          # south edge (m)
    res: float         # cell size (m)
    nx: int
    ny: int

    @classmethod
    def around(cls, bounds: tuple[float, float, float, float], res: float = 1000.0, buffer: float = 0.0) -> "Grid":
        """Smallest grid with edges on multiples of `res` covering (xmin, ymin, xmax, ymax) + buffer."""
        xmin, ymin, xmax, ymax = bounds
        x0 = np.floor((xmin - buffer) / res) * res
        y0 = np.floor((ymin - buffer) / res) * res
        nx = int(np.ceil((xmax + buffer - x0) / res))
        ny = int(np.ceil((ymax + buffer - y0) / res))
        return cls(float(x0), float(y0), float(res), nx, ny)

    @property
    def x(self) -> np.ndarray:
        return self.x0 + self.res * (np.arange(self.nx) + 0.5)

    @property
    def y(self) -> np.ndarray:
        return self.y0 + self.res * (np.arange(self.ny) + 0.5)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x0 + self.nx * self.res, self.y0 + self.ny * self.res

    def transform_north_up(self):
        """rasterio Affine for the same cells stored north-up (row 0 = northernmost); flip with [::-1]."""
        from rasterio.transform import from_origin
        return from_origin(self.x0, self.y0 + self.ny * self.res, self.res, self.res)

    def index(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """(row, col) of the cell containing each point, in the y-increasing layout; -1 outside."""
        col = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        row = np.floor((np.asarray(y) - self.y0) / self.res).astype(int)
        bad = (col < 0) | (col >= self.nx) | (row < 0) | (row >= self.ny)
        return np.where(bad, -1, row), np.where(bad, -1, col)

    def sample(self, field: np.ndarray, x, y) -> np.ndarray:
        """Value of a (ny, nx) field at the cell containing each point (NaN outside the grid)."""
        r, c = self.index(x, y)
        out = np.full(np.shape(r), np.nan)
        ok = r >= 0
        out[ok] = field[r[ok], c[ok]]
        return out

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "res": self.res, "nx": self.nx, "ny": self.ny, "crs": CRS_MODEL}


def bilinear(field: np.ndarray, grid: Grid, x, y) -> np.ndarray:
    """Bilinear interpolation of a (ny, nx) field between cell centres; NaN cells propagate."""
    fx = (np.asarray(x, dtype=float) - grid.x0) / grid.res - 0.5
    fy = (np.asarray(y, dtype=float) - grid.y0) / grid.res - 0.5
    i0 = np.clip(np.floor(fx).astype(int), 0, grid.nx - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, grid.ny - 2)
    wx = np.clip(fx - i0, 0.0, 1.0)
    wy = np.clip(fy - j0, 0.0, 1.0)
    return ((1 - wx) * (1 - wy) * field[j0, i0] + wx * (1 - wy) * field[j0, i0 + 1]
            + (1 - wx) * wy * field[j0 + 1, i0] + wx * wy * field[j0 + 1, i0 + 1])
