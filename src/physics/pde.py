"""PDE residual of S·∂h/∂t = T∇²h + R − Q, evaluated through the network's input/output scalers.

The network works in normalised units: x_n, y_n, t_n ∈ [−1, 1] and h_n = (h − h_mean)/h_std.
Derivatives are taken with autograd in those units and converted to physical ones by the chain rule:
    ∂h/∂t   = (h_std / t_half) · ∂h_n/∂t_n            [m/day]
    ∂²h/∂x² = (h_std / x_half²) · ∂²h_n/∂x_n²          [1/m]
The residual is computed in m/day, then divided by a typical flux, so a good fit has |residual| ≪ 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PhysScales:
    x_half_m: float
    y_half_m: float
    t_half_days: float
    h_mean_m: float
    h_std_m: float
    flux_scale_m_per_day: float


def derivatives(h_norm: Tensor, xyt_norm: Tensor, sc: PhysScales) -> dict[str, Tensor]:
    """Physical h_t [m/day] and h_xx, h_yy [1/m]. `xyt_norm` must have requires_grad=True and be the
    tensor `h_norm` was computed from (columns x_n, y_n, t_n)."""
    (g,) = torch.autograd.grad(h_norm.sum(), xyt_norm, create_graph=True)
    if g.requires_grad:
        (gxx,) = torch.autograd.grad(g[:, 0].sum(), xyt_norm, create_graph=True, allow_unused=True,
                                     materialize_grads=True)
        (gyy,) = torch.autograd.grad(g[:, 1].sum(), xyt_norm, create_graph=True, allow_unused=True,
                                     materialize_grads=True)
    else:                      # h is linear in all inputs: every second derivative is zero
        gxx = gyy = torch.zeros_like(g)
    return {
        "h_t": g[:, 2:3] * (sc.h_std_m / sc.t_half_days),
        "h_xx": gxx[:, 0:1] * (sc.h_std_m / sc.x_half_m**2),
        "h_yy": gyy[:, 1:2] * (sc.h_std_m / sc.y_half_m**2),
    }


def pde_residual(h_norm: Tensor, xyt_norm: Tensor, S: Tensor, T: Tensor, R: Tensor, Q: Tensor,
                 sc: PhysScales, return_terms: bool = False):
    """(S·h_t − T·(h_xx + h_yy) − R + Q) / flux_scale.  R, Q in m/day, shape (N, 1)."""
    d = derivatives(h_norm, xyt_norm, sc)
    storage = S * d["h_t"]
    lateral = T * (d["h_xx"] + d["h_yy"])
    res = (storage - lateral - R + Q) / sc.flux_scale_m_per_day
    if return_terms:
        return res, {"storage": storage, "lateral": lateral, **d}
    return res
