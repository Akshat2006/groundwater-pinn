"""Ordinary kriging per survey (Task B): the standard way to estimate a value between measured points.

Readings are grouped into surveys (year + season). For each survey the training wells' water levels are kriged
(spherical variogram fitted automatically) to the target wells of the same survey. Surveys with fewer than
`min_wells` training wells are skipped, so their target readings get no prediction (reported as coverage < 100%).
Wells at the same spot (nested piezometers) are averaged first, since kriging cannot use two values at one point.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def survey_key(df: pd.DataFrame) -> pd.Series:
    return df.date.dt.year.astype(str) + "-" + df.season.astype(str)


def ordinary_kriging(train: pd.DataFrame, target: pd.DataFrame, min_wells: int = 15,
                     variogram: str = "spherical") -> pd.Series:
    from pykrige.ok import OrdinaryKriging

    out = pd.Series(np.nan, index=target.index)
    tk, gk = survey_key(train), survey_key(target)
    for key, tgt in target.groupby(gk):
        tr = train[tk == key]
        tr = tr.assign(xr=(tr.x_m / 50).round(), yr=(tr.y_m / 50).round()).groupby(["xr", "yr"]).agg(
            x=("x_m", "mean"), y=("y_m", "mean"), h=("head_m", "mean"))
        if len(tr) < min_wells:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ok = OrdinaryKriging(tr.x.to_numpy() / 1000, tr.y.to_numpy() / 1000, tr.h.to_numpy(),
                                     variogram_model=variogram, nlags=12)
                z, _ = ok.execute("points", tgt.x_m.to_numpy() / 1000, tgt.y_m.to_numpy() / 1000)
        except np.linalg.LinAlgError:
            continue
        out.loc[tgt.index] = np.asarray(z)
    return out
