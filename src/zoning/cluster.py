"""Group wells into zones that behave alike (PLAN.md §5 M4).

k-means on standardised behaviour features. The number of zones k is chosen with two standard scores:
- elbow: the k after which adding zones stops reducing the within-zone spread much (the "knee" of the curve);
- silhouette: −1 to 1, how much closer each well is to its own zone than to the next one (higher = cleaner groups).
Rule (fixed before looking at the zones): take the elbow k, then the best silhouette among elbow−1, elbow, elbow+1.
Zones with fewer than `min_size` wells are merged into the zone with the nearest centre, one at a time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class FeatureScaler:
    """Clip each feature to its [p, 100−p] percentiles, then standardise (mean 0, sd 1). Fitted on training wells."""
    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, pct: float = 2.0) -> "FeatureScaler":
        lo, hi = np.percentile(X, pct, axis=0), np.percentile(X, 100 - pct, axis=0)
        Xc = np.clip(X, lo, hi)
        return cls(lo, hi, Xc.mean(0), Xc.std(0) + 1e-12)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.clip(X, self.lo, self.hi) - self.mean) / self.sd


def cluster_wells(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    if k == 1:
        return np.zeros(len(X), int)
    return KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(X)


def elbow(ks: np.ndarray, inertia: np.ndarray) -> int:
    """Knee of the inertia curve: the point farthest from the straight line joining its two ends."""
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    return int(ks[np.argmax(np.abs(1 - x - y))])


@dataclass
class KSelection:
    table: pd.DataFrame          # k, inertia, silhouette
    elbow: int
    chosen: int


def choose_k(X: np.ndarray, k_range=range(2, 13), seed: int = 0) -> KSelection:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    t = pd.DataFrame(rows)
    e = elbow(t.k.to_numpy(float), t.inertia.to_numpy(float))
    near = t[t.k.between(e - 1, e + 1)]
    return KSelection(t, e, int(near.loc[near.silhouette.idxmax(), "k"]))


def merge_small_zones(labels: np.ndarray, X: np.ndarray, min_size: int) -> np.ndarray:
    """Merge the smallest zone below `min_size` into the zone whose centre is nearest, until none is left.
    Returns labels 0..k'−1 (numbered by first appearance order of the remaining zones)."""
    lab = np.asarray(labels).copy()
    while True:
        ids, n = np.unique(lab, return_counts=True)
        if len(ids) <= 1 or n.min() >= min_size:
            break
        small = ids[np.argmin(n)]
        centres = {i: X[lab == i].mean(0) for i in ids}
        others = [i for i in ids if i != small]
        target = min(others, key=lambda i: np.sum((centres[i] - centres[small]) ** 2))
        lab[lab == small] = target
    return pd.factorize(lab)[0]


def order_by(labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Renumber zones 1..k so that zone 1 has the lowest mean of `values` (e.g. the steepest decline)."""
    ids = np.unique(labels)
    means = [values[labels == i].mean() for i in ids]
    rank = {ids[j]: r + 1 for r, j in enumerate(np.argsort(means))}
    return np.array([rank[v] for v in labels])
