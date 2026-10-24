"""Bucket model (experiment E18 and the early zoning preview): water balance with no sideways flow.

Between two readings of a well, the level change follows
    S · Δh = R − Q · cropland + c · days
with R = recharge since the previous reading (0.22 × rain), Q split into paddy (Jun–Sep), wheat (Nov–Mar) and
other days, and c a constant background in- or outflow. Dividing by S gives a linear model
    Δh = a·R − cropland·(b_paddy·days_paddy + b_wheat·days_wheat + b_other·days_other) + c'·days
with a, b ≥ 0 (non-negative least squares; c' may take either sign), fitted per zone or for the whole region.
S = 1 / a. It predicts level *changes*, which is what zoning should improve: the regional decline and the size of
the seasonal swing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

COEFS = ["a_recharge", "b_paddy", "b_wheat", "b_other", "c_up", "c_down"]


def season_days(start: pd.Series, end: pd.Series) -> np.ndarray:
    """(n, 3) days in paddy (Jun–Sep), wheat (Nov–Mar) and other months between each start and end date."""
    out = np.zeros((len(start), 3))
    for i, (a, b) in enumerate(zip(pd.to_datetime(start), pd.to_datetime(end))):
        m = pd.date_range(a, b - pd.Timedelta(days=1), freq="D").month
        out[i] = [np.isin(m, [6, 7, 8, 9]).sum(), np.isin(m, [11, 12, 1, 2, 3]).sum(), np.isin(m, [4, 5, 10]).sum()]
    return out


def level_changes(df: pd.DataFrame, max_gap_days: int = 400) -> pd.DataFrame:
    """Consecutive usable readings of each well: Δh, recharge since the previous reading, season days, cropland."""
    d = df.sort_values(["well_id", "date"]).copy()
    prev = d.groupby("well_id")[["date", "head_m"]].shift()
    d["dh"] = d.head_m - prev.head_m
    d["gap"] = (d.date - prev.date).dt.days
    d = d[d.dh.notna() & d.gap.between(1, max_gap_days) & d.recharge_prev_m.notna()].copy()
    days = season_days(prev.loc[d.index, "date"], d.date)
    d[["d_paddy", "d_wheat", "d_other"]] = days
    return d


def design(d: pd.DataFrame) -> np.ndarray:
    crop = d.lu_cropland_frac.fillna(d.lu_cropland_frac.median()).to_numpy()
    n = d[["d_paddy", "d_wheat", "d_other"]].to_numpy() / 365.25
    days = d.gap.to_numpy() / 365.25
    return np.column_stack([d.recharge_prev_m.to_numpy(), -crop * n[:, 0], -crop * n[:, 1], -crop * n[:, 2],
                            days, -days])


def fit(d: pd.DataFrame) -> np.ndarray:
    beta, _ = nnls(design(d), d.dh.to_numpy())
    return beta


def predict(d: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    return design(d) @ beta


def fit_predict_zoned(train: pd.DataFrame, target: pd.DataFrame, zone_col: str | None,
                      min_changes: int = 30) -> tuple[pd.Series, dict]:
    """Fit per zone (zone_col) or once (None); zones with too few changes use the regional fit."""
    beta_all = fit(train)
    out = pd.Series(predict(target, beta_all), index=target.index)
    betas = {0: beta_all}
    if zone_col:
        for z, g in train.groupby(zone_col):
            if len(g) < min_changes:
                continue
            betas[int(z)] = fit(g)
            sel = target[zone_col] == z
            if sel.any():
                out[sel] = predict(target[sel], betas[int(z)])
    return out, betas


def changes_from_first(df: pd.DataFrame, rain, rif: float) -> pd.DataFrame:
    """Long-horizon version: each reading's level change since the well's *first* reading in `df`, with recharge,
    season days and cropland accumulated over that whole span. Errors in a zone's decline rate add up over the
    years here, while single-interval changes are dominated by reading noise."""
    from src.data.imd import rain_at_points

    d = df.sort_values(["well_id", "date"]).copy()
    first = d.groupby("well_id")[["date", "head_m"]].transform("first")
    d["dh"] = d.head_m - first.head_m
    d["gap"] = (d.date - first.date).dt.days
    d = d[d.gap > 0].copy()
    wells = d.groupby("well_id")[["lon", "lat"]].first()
    series = rain_at_points(rain, wells.lon.values, wells.lat.values)
    cum = np.vstack([np.zeros((1, len(wells))), np.nancumsum(series.values, axis=0)])
    t0 = pd.Timestamp(series.time.values[0])
    col = pd.Series(np.arange(len(wells)), index=wells.index).reindex(d.well_id).to_numpy()
    i1 = (d.date - t0).dt.days.to_numpy()
    i0 = (first.loc[d.index, "date"] - t0).dt.days.to_numpy()
    d["recharge_prev_m"] = rif * (cum[i1 + 1, col] - cum[i0 + 1, col]) / 1000.0
    d[["d_paddy", "d_wheat", "d_other"]] = season_days(first.loc[d.index, "date"], d.date)
    return d
