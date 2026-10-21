"""Sample sparse 'monitoring wells' from a synthetic truth Dataset (PLAN.md §6.4).

The output follows the master-table columns where they apply (PLAN.md §4.1), plus truth columns.
- Training wells (`split_space == "train"`) are spread evenly by best-candidate sampling.
- Hidden wells (`"test"`) are placed at random. They are used only to score heads where the model saw no data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

QUARTERLY_MONTHS = (1, 5, 8, 11)          # CGWB-style rounds: Jan, May, Aug, Nov (15th)


def place_wells(n: int, extent, rng: np.random.Generator, margin_m: float = 2000.0,
                candidates: int = 30) -> np.ndarray:
    """Evenly spread points (Mitchell's best-candidate sampling): each new well is the candidate farthest
    from the wells already placed. Returns (n, 2) array of x, y [m]."""
    x0, x1, y0, y1 = extent
    lo = np.array([x0 + margin_m, y0 + margin_m])
    hi = np.array([x1 - margin_m, y1 - margin_m])
    pts = [rng.uniform(lo, hi)]
    for _ in range(n - 1):
        cand = rng.uniform(lo, hi, size=(candidates, 2))
        d = np.min(np.linalg.norm(cand[:, None, :] - np.array(pts)[None, :, :], axis=2), axis=1)
        pts.append(cand[np.argmax(d)])
    return np.array(pts)


def schedule(times: pd.DatetimeIndex, kind: str) -> pd.DatetimeIndex:
    """Reading dates within the record: the 15th of every month, or of the CGWB quarterly months."""
    days = times[times.day == 15]
    if kind == "monthly":
        return days
    if kind == "quarterly":
        return days[days.month.isin(QUARTERLY_MONTHS)]
    raise ValueError(f"schedule must be monthly or quarterly, got {kind!r}")


def season_label(months) -> np.ndarray:
    m = np.asarray(months)
    return np.select([m == 1, (m >= 3) & (m <= 6), m == 8, (m >= 10) & (m <= 12)],
                     ["jan", "pre", "aug", "post"], "other")


def heads_at(ds: xr.Dataset, xy: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Bilinear interpolation of the true heads at points xy on the given dates -> (n_dates, n_points)."""
    h = ds["h"].sel(time=dates).values
    y, x = ds.y.values, ds.x.values
    out = np.empty((len(dates), len(xy)))
    pts = xy[:, ::-1]                              # interpolator wants (y, x)
    for k in range(len(dates)):
        out[k] = RegularGridInterpolator((y, x), h[k])(pts)
    return out


def field_at(da: xr.DataArray, xy: np.ndarray) -> np.ndarray:
    return RegularGridInterpolator((da.y.values, da.x.values), da.values)(xy[:, ::-1])


def sample_wells(ds: xr.Dataset, n_wells: int, schedule_kind: str = "quarterly", seed: int = 0,
                 noise_sd: float = 0.0, offset_sd: float = 0.0, missing_frac: float = 0.0,
                 dry_frac: float = 0.0, n_hidden: int = 30, hidden_schedule: str = "monthly",
                 margin_m: float = 2000.0) -> pd.DataFrame:
    """Observations at `n_wells` training wells, plus noise-free readings at `n_hidden` hidden wells.

    noise_sd:     random measurement error per reading [m]
    offset_sd:    constant error per well [m] (like a wrong ground elevation)
    missing_frac: share of training readings dropped at random
    dry_frac:     roughly this share of training readings are made 'dry'. A well bottom is set inside the
                  range of some wells; readings below it get head_m = NaN and is_dry = True (censored).
    """
    rng = np.random.default_rng(seed)
    dx = float(ds.x[1] - ds.x[0])
    extent = (float(ds.x[0]) - dx / 2, float(ds.x[-1]) + dx / 2, float(ds.y[0]) - dx / 2, float(ds.y[-1]) + dx / 2)
    times = pd.DatetimeIndex(ds.time.values)

    frames = []
    for split, n, kind, xy in (
        ("train", n_wells, schedule_kind, place_wells(n_wells, extent, rng, margin_m)),
        ("test", n_hidden, hidden_schedule,
         rng.uniform([extent[0] + margin_m, extent[2] + margin_m], [extent[1] - margin_m, extent[3] - margin_m],
                     size=(n_hidden, 2))),
    ):
        if n == 0:
            continue
        dates = schedule(times, kind)
        h_true = heads_at(ds, xy, dates)                                  # (n_dates, n)
        prefix = "W" if split == "train" else "H"
        df = pd.DataFrame({
            "well_id": np.tile([f"{prefix}{i:03d}" for i in range(n)], len(dates)),
            "x_m": np.tile(xy[:, 0], len(dates)),
            "y_m": np.tile(xy[:, 1], len(dates)),
            "date": np.repeat(dates.values, n),
            "head_true_m": h_true.ravel(),
        })
        df["lu_ag_frac"] = np.tile(field_at(ds["landuse_input"], xy), len(dates))
        df["split_space"] = split
        df["head_m"] = df["head_true_m"]
        df["is_dry"] = False
        df["well_bottom_m"] = np.nan
        if split == "train":
            offsets = rng.normal(0, offset_sd, n) if offset_sd > 0 else np.zeros(n)
            df["head_m"] = (df["head_true_m"] + np.tile(offsets, len(dates))
                            + (rng.normal(0, noise_sd, len(df)) if noise_sd > 0 else 0.0))
            if dry_frac > 0:
                _make_dry(df, h_true, dry_frac, rng)
            if missing_frac > 0:
                df = df[rng.random(len(df)) >= missing_frac]
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["season"] = season_label(out["date"].dt.month)
    out["source"] = f"synthetic_{ds.attrs.get('case', '?')}"
    cols = ["well_id", "x_m", "y_m", "date", "season", "head_m", "head_true_m", "is_dry", "well_bottom_m",
            "lu_ag_frac", "split_space", "source"]
    return out[cols].sort_values(["split_space", "well_id", "date"]).reset_index(drop=True)


def _make_dry(df: pd.DataFrame, h_true: np.ndarray, dry_frac: float, rng) -> None:
    """Give a quarter of the wells a bottom inside their range so that ~dry_frac of all readings are dry."""
    n = h_true.shape[1]
    chosen = rng.choice(n, size=max(1, n // 4), replace=False)
    share = min(dry_frac * n / len(chosen), 0.9)            # dry share needed within the chosen wells
    for i in chosen:
        bottom = np.quantile(h_true[:, i], share)
        rows = df.index[df["well_id"] == f"W{i:03d}"]
        df.loc[rows, "well_bottom_m"] = bottom
        dry = rows[df.loc[rows, "head_true_m"].values < bottom]
        df.loc[dry, "is_dry"] = True
        df.loc[dry, "head_m"] = np.nan
