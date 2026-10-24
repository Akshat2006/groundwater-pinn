"""Checks used by the M4 report: is well behaviour a spatial pattern that a map can capture?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


def neighbour_correlation(values: np.ndarray, xy_km: np.ndarray) -> float:
    """Correlation between each well's value and its nearest other well's value (0 = unrelated, 1 = identical)."""
    _, idx = cKDTree(xy_km).query(xy_km, k=2)
    return float(np.corrcoef(values, values[idx[:, 1]])[0, 1])


def block_folds(blocks: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ub = rng.permutation(np.unique(blocks))
    fold_of = dict(zip(ub, np.arange(len(ub)) % n_folds))
    return np.array([fold_of[b] for b in blocks])


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def block_cv_r2(y: np.ndarray, static: np.ndarray, xy_km: np.ndarray, blocks: np.ndarray, n_folds: int = 5,
                seed: int = 0) -> dict:
    """How well a well's value is predicted in held-out blocks: from static features (random forest) and from the
    5 nearest wells. R² = 1 perfect, 0 = no better than the overall average."""
    fold = block_folds(blocks, n_folds, seed)
    out = {}
    for name, make, X in (("static features", lambda: RandomForestRegressor(500, min_samples_leaf=3, random_state=seed,
                                                                           n_jobs=-1), static),
                          ("5 nearest wells", lambda: KNeighborsRegressor(5, weights="distance"), xy_km)):
        p = np.zeros_like(y, dtype=float)
        for f in range(n_folds):
            p[fold == f] = make().fit(X[fold != f], y[fold != f]).predict(X[fold == f])
        out[name] = r2(y, p)
    return out


def correlation_by_distance(values: np.ndarray, xy_km: np.ndarray, bands=((0, 10), (10, 20), (20, 40), (40, 80))) -> pd.Series:
    """Average product of standardised values for well pairs in each distance band (a correlation by distance)."""
    z = (values - values.mean()) / values.std()
    d = np.sqrt(((xy_km[:, None] - xy_km[None]) ** 2).sum(-1))
    prod = np.outer(z, z)
    off = ~np.eye(len(z), dtype=bool)
    return pd.Series({f"{lo}-{hi} km": float(prod[(d > lo) & (d <= hi) & off].mean()) for lo, hi in bands})
