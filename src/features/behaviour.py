"""Zoning features: how each well's water level behaves (PLAN.md §2.5, §5 M4).

Computed from readings inside the training period only, so nothing from validation or test years can leak into
the zones. One value per well:
- amplitude_m: mean yearly rise from the pre-monsoon to the post-monsoon reading (negative = the level falls
  through the monsoon, e.g. because paddy pumping outweighs recharge).
- trend_m_per_yr: Theil–Sen slope of the pre-monsoon water level over the years (negative = long-term decline).
  Theil–Sen is the median of all pairwise slopes, so a few bad readings cannot swing it.
- monsoon_resp_m_per_100mm: how much more the level rises in a wetter monsoon, per 100 mm of extra monsoon rain
  (Theil–Sen slope of the yearly rise against that year's June–September rain).
The quarterly CGWB data cannot resolve a rainfall lag in months (it can only be 0, 3, 6 or 9), which is why the
monsoon response replaces the lag feature of the brief.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

FEATURES = ["amplitude_m", "trend_m_per_yr", "monsoon_resp_m_per_100mm"]


def yearly_rise(df: pd.DataFrame) -> pd.DataFrame:
    """Per well and year: pre- and post-monsoon level (median of that season's readings), rise and monsoon rain."""
    d = df[df.head_m.notna() & df.season.isin(["pre", "post"])].copy()
    d["year"] = d.date.dt.year
    lv = d.pivot_table(index=["well_id", "year"], columns="season", values="head_m", aggfunc="median",
                       observed=True)
    rain = (d[d.season == "post"].groupby(["well_id", "year"]).rain_monsoon_mm.median())
    out = lv.reindex(columns=["pre", "post"]).join(rain)
    out["rise_m"] = out["post"] - out["pre"]
    return out.reset_index()


def _theil(y, x) -> float:
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return np.nan
    return float(theilslopes(y[ok], x[ok])[0])


def behaviour_features(df: pd.DataFrame, train_period: tuple[str, str], min_years: int = 5) -> pd.DataFrame:
    """Features for every well with at least `min_years` training years that have both a pre and a post reading.

    `df` is the master table (any rows); only rows dated inside `train_period` are used.
    """
    start, end = pd.Timestamp(train_period[0]), pd.Timestamp(train_period[1])
    yr = yearly_rise(df[(df.date >= start) & (df.date <= end)])
    rows = []
    for well, g in yr.groupby("well_id"):
        both = g.dropna(subset=["rise_m"])
        if len(both) < min_years:
            continue
        pre = g.dropna(subset=["pre"])
        rows.append({
            "well_id": well,
            "n_years": len(both),
            "amplitude_m": float(both.rise_m.mean()),
            "trend_m_per_yr": _theil(pre["pre"].to_numpy(float), pre.year.to_numpy(float)),
            "monsoon_resp_m_per_100mm": 100 * _theil(both.rise_m.to_numpy(float), both.rain_monsoon_mm.to_numpy(float)),
        })
    return pd.DataFrame(rows, columns=["well_id", "n_years", *FEATURES]).set_index("well_id")
