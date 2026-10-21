"""Adaptive loss weighting by gradient size ("learning-rate annealing", Wang, Teng & Perdikaris 2021).

The PDE loss keeps weight 1. Every `every` steps the data weight is set so that the data loss's
gradients are about as large as the PDE loss's largest gradients. This uses the network weights only:
    λ_new = max|∇θ L_pde| / mean|∇θ L_data|,   λ ← (1 − α)·λ + α·λ_new
This stops one loss from drowning out the other while their sizes change during training.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class GradNormWeighting:
    def __init__(self, params: list[nn.Parameter], init: float = 1.0, alpha: float = 0.1,
                 every: int = 100, lo: float = 1e-2, hi: float = 1e4):
        self.params = [p for p in params if p.requires_grad]
        self.weight, self.alpha, self.every, self.lo, self.hi = init, alpha, every, lo, hi

    def _grads(self, loss: Tensor) -> Tensor:
        gs = torch.autograd.grad(loss, self.params, retain_graph=True, allow_unused=True)
        return torch.cat([g.reshape(-1).abs() for g in gs if g is not None])

    def update(self, step: int, l_pde: Tensor, l_data: Tensor) -> float:
        if step % self.every == 0:
            g_pde, g_data = self._grads(l_pde), self._grads(l_data)
            new = float(g_pde.max() / (g_data.mean() + 1e-12))
            self.weight = min(max((1 - self.alpha) * self.weight + self.alpha * new, self.lo), self.hi)
        return self.weight
