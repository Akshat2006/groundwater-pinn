"""Quality control of depth-to-water readings, dry-well flags, and conversion to hydraulic head.

QC flags (one per reading, first match wins):
    duplicate     same well, same day as an earlier reading, or the identical value repeated within
                  dup_days of the previous reading (a copied entry, not a new measurement)
    nonpositive   depth to water <= 0 (water at or above ground: not expected in this belt)
    out_of_range  depth to water > max_dtw_m
    below_bottom  depth to water more than dry_margin_m below the recorded well bottom
    spike         one or two consecutive readings that jump > spike_m away from both neighbours in the
                  same direction while the neighbours agree with each other
    outlier       more than outlier_m (+ 1 m per year of time gap) away from the median of the well's
                  4 readings nearest in time: catches bad end-of-record values such as a round 85.00 m
                  after years of ~25 m, while real long-term decline (~1 m/yr) passes
    suspect_jump  change of > jump_m from the previous reading within max_gap_days that persists (kept, but
                  flagged); a large change across a long gap can be real decline
    ok
Readings flagged duplicate/nonpositive/out_of_range/below_bottom/spike are not used (`usable` = False).

Dry (censored) readings: within dry_margin_m of the well bottom. The true level is at or below the
bottom, so the head is unknown (NaN) and only the bound `bottom_head_m` is kept (PLAN.md §2.7).

Head: head_m = ground_elev_m - dtw_m.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNUSABLE = ("duplicate", "nonpositive", "out_of_range", "below_bottom", "spike", "outlier")
SEASONS = {1: "jan", 2: "jan", 3: "pre", 4: "pre", 5: "pre", 6: "pre",
           7: "aug", 8: "aug", 9: "aug", 10: "post", 11: "post", 12: "post"}


def spike_mask(dtw: np.ndarray, days: np.ndarray, spike_m: float, max_gap_days: float,
               max_run: int = 2) -> np.ndarray:
    """Runs of 1..max_run readings that stand out from both neighbours in one well's time-sorted series."""
    n = len(dtw)
    out = np.zeros(n, dtype=bool)
    for run in range(1, max_run + 1):
        for i in range(1, n - run):
            if out[i - 1] or out[i:i + run].any():
                continue
            a, b = i - 1, i + run                              # neighbours either side of the run
            if days[i] - days[a] > max_gap_days or days[b] - days[b - 1] > max_gap_days:
                continue
            seg = dtw[i:i + run]
            d1, d2 = seg - dtw[a], seg - dtw[b]
            if not (np.all(np.sign(d1) == np.sign(d1[0])) and np.all(np.sign(d2) == np.sign(d1[0]))):
                continue
            smallest = min(np.abs(d1).min(), np.abs(d2).min())
            if smallest > spike_m and abs(dtw[a] - dtw[b]) < 0.5 * smallest:
                out[i:i + run] = True
    return out


def outlier_mask(dtw: np.ndarray, days: np.ndarray, outlier_m: float, n_near: int = 4) -> np.ndarray:
    """Readings far from the median of their n_near nearest-in-time readings (threshold grows 1 m/yr of gap)."""
    n = len(dtw)
    out = np.zeros(n, dtype=bool)
    if n < n_near:
        return out
    for i in range(n):
        dist = np.abs(days - days[i])
        dist[i] = np.inf
        near = np.argsort(dist)[:min(n_near, n - 1)]
        gap_years = dist[near].min() / 365.25
        out[i] = abs(dtw[i] - np.median(dtw[near])) > outlier_m + gap_years
    return out


def qc_levels(df: pd.DataFrame, spike_m: float = 5.0, jump_m: float = 15.0, max_dtw_m: float = 100.0,
              dry_margin_m: float = 1.0, max_gap_days: float = 400.0, spike_gap_days: float = 800.0,
              dup_days: float = 31.0, outlier_m: float = 20.0) -> pd.DataFrame:
    """Add qc_flag, usable and is_dry columns. Needs well_id, date, dtw_m, well_depth_m."""
    df = df.sort_values(["well_id", "date"]).reset_index(drop=True).copy()
    flag = np.full(len(df), "ok", dtype=object)
    dtw = df["dtw_m"].to_numpy(float)
    days = (df["date"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy(float)
    dup = df.duplicated(["well_id", "date"], keep="first").to_numpy()
    same_well = np.r_[False, df["well_id"].to_numpy()[1:] == df["well_id"].to_numpy()[:-1]]
    copied = same_well & np.r_[False, (np.abs(np.diff(dtw)) < 0.005) & (np.diff(days) <= dup_days)]
    flag[dup | copied] = "duplicate"
    depth = df["well_depth_m"].to_numpy(float)
    free = flag == "ok"
    for name, cond in (("nonpositive", dtw <= 0), ("out_of_range", dtw > max_dtw_m),
                       ("below_bottom", np.isfinite(depth) & (dtw > depth + dry_margin_m))):
        hit = free & cond
        flag[hit] = name
        free &= ~hit
    for _, idx in df.groupby("well_id").indices.items():
        idx = idx[free[idx]]                              # spikes judged among plausible readings only
        if len(idx) < 2:
            continue
        sp = spike_mask(dtw[idx], days[idx], spike_m, spike_gap_days)
        flag[idx[sp]] = "spike"
        keep = idx[~sp]
        out = outlier_mask(dtw[keep], days[keep], outlier_m)
        flag[keep[out]] = "outlier"
        keep = keep[~out]
        jumps = (np.abs(np.diff(dtw[keep])) > jump_m) & (np.diff(days[keep]) <= max_gap_days)
        flag[keep[1:][jumps]] = "suspect_jump"
    df["qc_flag"] = flag
    df["usable"] = ~df["qc_flag"].isin(UNUSABLE)
    df["is_dry"] = df["usable"] & np.isfinite(depth) & (dtw >= depth - dry_margin_m)
    return df


def add_head(df: pd.DataFrame, ground_elev_m: np.ndarray) -> pd.DataFrame:
    """head_m = ground elevation - depth to water (NaN if unusable or dry); bottom_head_m for dry readings."""
    df = df.copy()
    df["ground_elev_m"] = ground_elev_m
    head = df["ground_elev_m"] - df["dtw_m"]
    df["head_m"] = head.where(df["usable"] & ~df["is_dry"])
    df["bottom_head_m"] = (df["ground_elev_m"] - df["well_depth_m"]).where(df["is_dry"])
    return df


def season_of(dates: pd.Series) -> pd.Series:
    return dates.dt.month.map(SEASONS).astype("category")
