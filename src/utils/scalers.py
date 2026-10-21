"""Scalers that map physical values to the ranges networks like, with save/load (JSON).

MinMaxScaler: [lo, hi] -> [-1, 1]   (used for x, y, t)
StandardScaler: (v - mean) / std     (used for heads)
Both work on numpy arrays and torch tensors.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MinMaxScaler:
    lo: float
    hi: float

    @property
    def centre(self) -> float:
        return 0.5 * (self.lo + self.hi)

    @property
    def half(self) -> float:
        return 0.5 * (self.hi - self.lo)

    def transform(self, v):
        return (v - self.centre) / self.half

    def inverse(self, v):
        return v * self.half + self.centre


@dataclass
class StandardScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, values) -> "StandardScaler":
        import numpy as np
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        return cls(float(v.mean()), float(v.std()) or 1.0)

    def transform(self, v):
        return (v - self.mean) / self.std

    def inverse(self, v):
        return v * self.std + self.mean


def save_scalers(scalers: dict, path: str | Path) -> None:
    out = {k: {"type": type(s).__name__, **asdict(s)} for k, s in scalers.items()}
    Path(path).write_text(json.dumps(out, indent=2))


def load_scalers(path: str | Path) -> dict:
    kinds = {"MinMaxScaler": MinMaxScaler, "StandardScaler": StandardScaler}
    raw = json.loads(Path(path).read_text())
    return {k: kinds[v.pop("type")](**v) for k, v in raw.items()}
