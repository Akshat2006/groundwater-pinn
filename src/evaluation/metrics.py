"""Error measures (PLAN.md §7). Plain-language meaning:

- RMSE: typical error in metres; big misses count extra (errors are squared before averaging).
- MAE: plain average size of the error in metres.
- NSE (Nash–Sutcliffe efficiency), per well: 1 = perfect; 0 = no better than always predicting that well's
  average level; below 0 = worse than that. Only computed for wells with at least `min_n` readings.
- Fraction improved: share of wells where model A's error is smaller than model B's.
- Wilcoxon signed-rank test: are the per-well improvements consistent, or could they be luck? The p-value is the
  chance of seeing improvements this consistent if the two models were really equally good (small = convincing).
- Coverage: how often the true value falls inside the stated range (a 90% range should hold ~90% of values).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def rmse(y, yhat) -> float:
    e = np.asarray(yhat, float) - np.asarray(y, float)
    return float(np.sqrt(np.nanmean(e ** 2)))


def mae(y, yhat) -> float:
    return float(np.nanmean(np.abs(np.asarray(yhat, float) - np.asarray(y, float))))


def nse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    den = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((yhat - y) ** 2) / den) if den > 0 else np.nan


def per_well(df: pd.DataFrame, pred_col: str, obs_col: str = "head_m", min_n: int = 4) -> pd.DataFrame:
    """One row per well: number of readings, RMSE, MAE and NSE (NSE only with >= min_n readings)."""
    d = df[df[pred_col].notna() & df[obs_col].notna()]
    rows = []
    for w, g in d.groupby("well_id"):
        y, p = g[obs_col].to_numpy(), g[pred_col].to_numpy()
        rows.append({"well_id": w, "n": len(g), "rmse": rmse(y, p), "mae": mae(y, p),
                     "nse": nse(y, p) if len(g) >= min_n else np.nan})
    return pd.DataFrame(rows).set_index("well_id")


def summary(df: pd.DataFrame, pred_col: str, obs_col: str = "head_m", min_n: int = 4) -> dict:
    d = df[df[pred_col].notna() & df[obs_col].notna()]
    pw = per_well(d, pred_col, obs_col, min_n)
    return {"n_readings": int(len(d)), "n_wells": int(len(pw)),
            "coverage_of_targets": float(len(d) / max(1, df[obs_col].notna().sum())),
            "rmse_m": rmse(d[obs_col], d[pred_col]), "mae_m": mae(d[obs_col], d[pred_col]),
            "median_well_rmse_m": float(pw.rmse.median()) if len(pw) else np.nan,
            "median_well_nse": float(pw.nse.median()) if pw.nse.notna().any() else np.nan}


def fraction_improved(err_a: pd.Series, err_b: pd.Series) -> float:
    a, b = err_a.align(err_b, join="inner")
    return float((a < b).mean())


def paired_wilcoxon(err_a: pd.Series, err_b: pd.Series) -> tuple[float, float]:
    a, b = err_a.align(err_b, join="inner")
    if len(a) < 5 or np.allclose(a, b):
        return np.nan, np.nan
    res = wilcoxon(a, b)
    return float(res.statistic), float(res.pvalue)


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))
