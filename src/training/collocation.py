"""Collocation points: random (x, y, t) where the PDE is enforced. No measurement is needed there.

Points are drawn uniformly in normalised space ([−1, 1]³), and fresh points are drawn every step.
An optional mask (a boolean grid over the domain, e.g. a zone polygon rasterised in M4) rejects points
outside the zone.
"""
from __future__ import annotations

import torch
from torch import Tensor


class CollocationSampler:
    def __init__(self, n_points: int, device, mask: Tensor | None = None, generator: torch.Generator | None = None):
        self.n, self.device, self.mask, self.gen = n_points, device, mask, generator

    def sample(self) -> Tensor:
        if self.mask is None:
            return torch.rand(self.n, 3, device=self.device, generator=self.gen) * 2 - 1
        out = []
        need = self.n
        ny, nx = self.mask.shape
        while need > 0:
            pts = torch.rand(2 * need, 3, device=self.device, generator=self.gen) * 2 - 1
            i = ((pts[:, 0] + 1) / 2 * nx).long().clamp(0, nx - 1)
            j = ((pts[:, 1] + 1) / 2 * ny).long().clamp(0, ny - 1)
            keep = pts[self.mask[j, i]]
            out.append(keep[:need])
            need -= len(out[-1])
        return torch.cat(out)
