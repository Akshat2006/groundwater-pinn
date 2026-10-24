"""Simple per-well forecasts for Task A (monitored wells, later years). They need the well's own history.

- persistence: the well's last training-period level.
- seasonal_mean: the well's average level in the same season (jan / pre / aug / post) over the training years.
- seasonal_trend: the same-season levels continued along their Theil–Sen trend line (falls back to the seasonal
  mean with fewer than `min_points` readings). In a steadily declining aquifer this is a hard baseline to beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import theilslopes


def decimal_year(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return (d.dt.year + (d.dt.dayofyear - 1) / 365.25).to_numpy(float)


def persistence(train: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    last = train.sort_values("date").groupby("well_id").head_m.last()
    return target.well_id.map(last)


def seasonal_mean(train: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    m = train.groupby(["well_id", "season"], observed=True).head_m.mean()
    key = pd.MultiIndex.from_arrays([target.well_id, target.season.astype(str)])
    m.index = pd.MultiIndex.from_arrays([m.index.get_level_values(0), m.index.get_level_values(1).astype(str)])
    return pd.Series(m.reindex(key).to_numpy(), index=target.index)


def seasonal_trend(train: pd.DataFrame, target: pd.DataFrame, min_points: int = 4) -> pd.Series:
    fits = {}
    for (w, s), g in train.groupby(["well_id", "season"], observed=True):
        t, h = decimal_year(g.date), g.head_m.to_numpy(float)
        if len(g) >= min_points and np.ptp(t) >= 3:
            slope, icept = theilslopes(h, t)[:2]
            fits[(w, str(s))] = (slope, icept)
        else:
            fits[(w, str(s))] = (0.0, float(np.mean(h)))
    t_target = decimal_year(target.date)
    out = np.full(len(target), np.nan)
    for i, (w, s) in enumerate(zip(target.well_id, target.season.astype(str))):
        if (w, s) in fits:
            slope, icept = fits[(w, s)]
            out[i] = icept + slope * t_target[i]
    return pd.Series(out, index=target.index)
