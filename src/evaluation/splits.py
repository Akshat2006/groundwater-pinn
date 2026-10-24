"""Spatial block and temporal splits (PLAN.md §2.6, §4.2).

Spatial: the area is cut into square blocks (default 20 km) on a grid whose offset depends on the seed. Blocks are
taken in random order into the test set until it holds about `test_frac` of the eligible wells, then into the
validation set; all other blocks are training. Splitting whole blocks (not single wells) means a test well never
has a training well from its own block, so the score measures prediction in places the model has not seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def block_ids(x, y, block_m: float, offset: tuple[float, float]) -> np.ndarray:
    """Integer id of the block containing each point (x, y in metres)."""
    bx = np.floor((np.asarray(x) - offset[0]) / block_m).astype(np.int64)
    by = np.floor((np.asarray(y) - offset[1]) / block_m).astype(np.int64)
    return by * 100_000 + bx


def spatial_block_split(x, y, eligible=None, block_km: float = 20.0, test_frac: float = 0.175,
                        val_frac: float = 0.10, tol: float = 0.025, seed: int = 0) -> pd.DataFrame:
    """Label each well train / val / test by block. Shares are counted over `eligible` wells only.

    Returns a frame with columns block (id) and split_space, in the input order.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    eligible = np.ones(len(x), bool) if eligible is None else np.asarray(eligible, bool)
    rng = np.random.default_rng(seed)
    block_m = block_km * 1000.0
    blk = block_ids(x, y, block_m, tuple(rng.uniform(0, block_m, 2)))
    counts = pd.Series(blk[eligible]).value_counts()
    order = rng.permutation(np.sort(counts.index.to_numpy()))
    n = int(eligible.sum())
    split = {}
    for name, frac in (("test", test_frac), ("val", val_frac)):
        got = 0
        for b in order:
            if b in split or got >= (frac - tol) * n:
                continue
            if got + counts[b] <= (frac + tol) * n:
                split[b] = name
                got += counts[b]
    labels = np.array([split.get(b, "train") for b in blk], dtype=object)
    return pd.DataFrame({"block": blk, "split_space": labels})
