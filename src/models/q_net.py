"""q_net: pumping Q ≥ 0 [m/day].

Two modes (PLAN.md §2.1). M2 decides between them on synthetic data:
- "restricted" (default): Q = q_scale · softplus(MLP(land use, season harmonics)) · exp(β · t_n).
  Q can only vary in space through the land-use map and in time through the crop season plus one slow
  trend β. Without raw x, y it cannot simply soak up whatever the head data leave unexplained.
- "full" (the brief's original form): Q = q_scale · softplus(MLP(x_n, y_n, t_n, land use, season harmonics)).
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from .h_net import mlp


class QNet(nn.Module):
    def __init__(self, n_feat: int, width: int = 16, depth: int = 3, q_scale_m_day: float = 1e-3,
                 mode: Literal["restricted", "full"] = "restricted", act: str = "tanh"):
        super().__init__()
        if mode not in ("restricted", "full"):
            raise ValueError(f"mode must be restricted or full, got {mode!r}")
        self.mode, self.q_scale = mode, q_scale_m_day
        self.net = mlp(n_feat, width, depth, 1, act)
        self.trend = nn.Parameter(torch.zeros(())) if mode == "restricted" else None

    def forward(self, feats: Tensor, t_norm: Tensor | None = None) -> Tensor:
        q = self.q_scale * softplus(self.net(feats))
        if self.mode == "restricted" and t_norm is not None:
            q = q * torch.exp(self.trend * t_norm)
        return q
