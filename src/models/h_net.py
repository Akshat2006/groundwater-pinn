"""h_net: MLP (x_n, y_n, t_n) -> normalised head h_n.

Inside the network, t_n is expanded with annual harmonics sin/cos(2πk·t/365.25), k = 1..n_harmonics,
computed from physical days. These are fixed functions of t, so h_net still depends only on
(x, y, t) and autograd's ∂h/∂t includes them automatically. They let a small network
represent ten seasonal cycles, which a plain tanh MLP finds hard.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


def mlp(in_dim: int, width: int, depth: int, out_dim: int, act: str = "tanh") -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(depth):
        lin = nn.Linear(d, width)
        nn.init.xavier_normal_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, ACTIVATIONS[act]()]
        d = width
    out = nn.Linear(d, out_dim)
    nn.init.xavier_normal_(out.weight)
    nn.init.zeros_(out.bias)
    return nn.Sequential(*layers, out)


def annual_harmonics(t_norm: Tensor, t_centre_days: float, t_half_days: float, n: int) -> Tensor:
    t_days = t_centre_days + t_half_days * t_norm
    w = 2 * math.pi / 365.25
    return torch.cat([f(k * w * t_days) for k in range(1, n + 1) for f in (torch.sin, torch.cos)], dim=-1)


class HNet(nn.Module):
    def __init__(self, in_dim: int = 3, width: int = 48, depth: int = 7, act: str = "tanh",
                 t_centre_days: float = 0.0, t_half_days: float = 1.0, n_harmonics: int = 2):
        super().__init__()
        self.t_centre, self.t_half, self.n_harm = t_centre_days, t_half_days, n_harmonics
        self.net = mlp(in_dim + 2 * n_harmonics, width, depth, 1, act)

    def forward(self, xyt_norm: Tensor) -> Tensor:
        feats = [xyt_norm]
        if self.n_harm:
            feats.append(annual_harmonics(xyt_norm[:, 2:3], self.t_centre, self.t_half, self.n_harm))
        return self.net(torch.cat(feats, dim=-1))
