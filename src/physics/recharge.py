"""Recharge and other gridded inputs, looked up at arbitrary (x, y, t) points on the GPU.

GridField holds a (time, y, x) or (y, x) grid and returns values at collocation points:
- bilinear interpolation between cell centres in space (clamped at the edges)
- nearest grid day in time.

Daily recharge from rain is very spiky, but the network's head surface is smooth. So the recharge used in
the PDE residual is first smoothed with a running mean (`smooth_days`). That keeps the total water
the same and only blurs the timing within a few weeks.
"""
from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from scipy.ndimage import uniform_filter1d
from torch import Tensor


def compute_recharge(rain_mm_day: xr.DataArray, rif: float,
                     canal_seepage_m_day: xr.DataArray | None = None) -> xr.DataArray:
    """R = rif · rain (+ canal seepage), converted to m/day."""
    r = rif * rain_mm_day / 1000.0
    return r + canal_seepage_m_day if canal_seepage_m_day is not None else r


def running_mean(values: np.ndarray, window_days: int) -> np.ndarray:
    """Centred running mean along axis 0 (time). Edges use the nearest values."""
    if window_days <= 1:
        return values
    return uniform_filter1d(values, size=int(window_days), axis=0, mode="nearest")


class GridField:
    def __init__(self, values: np.ndarray, x_centres: np.ndarray, y_centres: np.ndarray,
                 t_days: np.ndarray | None = None, device: str | torch.device = "cpu"):
        v = np.asarray(values, dtype=np.float32)
        if v.ndim == 2:
            v = v[None]
        self.values = torch.as_tensor(v, device=device)
        self.nt, self.ny, self.nx = self.values.shape
        self.x0, self.dx = float(x_centres[0]), float(x_centres[1] - x_centres[0])
        self.y0, self.dy = float(y_centres[0]), float(y_centres[1] - y_centres[0])
        if t_days is None:
            self.t0, self.dt = 0.0, 1.0
        else:
            self.t0 = float(t_days[0])
            self.dt = float(t_days[1] - t_days[0]) if len(t_days) > 1 else 1.0
        self._flat = self.values.reshape(-1)

    def __call__(self, x_m: Tensor, y_m: Tensor, t_days: Tensor | None = None) -> Tensor:
        """Values at points; inputs shape (N,) or (N, 1); output (N, 1)."""
        x, y = x_m.reshape(-1), y_m.reshape(-1)
        fx = ((x - self.x0) / self.dx).clamp(0, self.nx - 1)
        fy = ((y - self.y0) / self.dy).clamp(0, self.ny - 1)
        i0 = fx.floor().long().clamp(max=max(self.nx - 2, 0))
        j0 = fy.floor().long().clamp(max=max(self.ny - 2, 0))
        wx, wy = (fx - i0).unsqueeze(-1), (fy - j0).unsqueeze(-1)
        if self.nt == 1 or t_days is None:
            k = torch.zeros_like(i0)
        else:
            k = torch.round((t_days.reshape(-1) - self.t0) / self.dt).long().clamp(0, self.nt - 1)
        i1 = (i0 + 1).clamp(max=self.nx - 1)
        j1 = (j0 + 1).clamp(max=self.ny - 1)
        base = k * (self.ny * self.nx)

        def at(j, i):
            return self._flat[base + j * self.nx + i].unsqueeze(-1)

        return ((1 - wy) * ((1 - wx) * at(j0, i0) + wx * at(j0, i1))
                + wy * ((1 - wx) * at(j1, i0) + wx * at(j1, i1)))
