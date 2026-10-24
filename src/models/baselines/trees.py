"""Random forest and LightGBM baselines (Tasks A, B, C).

One model for the whole region predicts the water level from location, elevation, land cover, canal proximity,
time and rainfall. It knows nothing about a well's own history, so it can be applied anywhere. Tree models cannot
extend a trend beyond the training years: in later years they repeat the latest pattern they learned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baselines.naive import decimal_year

FEATURES = ["x_km", "y_km", "ground_elev_m", "lu_cropland_frac", "lu_builtup_frac", "canal_dist_km",
            "canal_density", "t_year", "month_sin", "month_cos", "rain_prev_season_mm", "rain_monsoon_mm"]


def tree_features(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    month = df.date.dt.month.to_numpy()
    X = pd.DataFrame({"x_km": df.x_m / 1000, "y_km": df.y_m / 1000, "ground_elev_m": df.ground_elev_m,
                      "lu_cropland_frac": df.lu_cropland_frac, "lu_builtup_frac": df.lu_builtup_frac,
                      "canal_dist_km": df.canal_dist_km, "canal_density": df.canal_density,
                      "t_year": decimal_year(df.date), "month_sin": np.sin(2 * np.pi * month / 12),
                      "month_cos": np.cos(2 * np.pi * month / 12),
                      "rain_prev_season_mm": df.rain_prev_season_mm, "rain_monsoon_mm": df.rain_monsoon_mm},
                     index=df.index)
    for col in extra or []:
        X[col] = df[col].to_numpy()
    return X


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, kind: str = "forest", seed: int = 0,
                target_col: str = "head_m", extra: list[str] | None = None) -> pd.Series:
    Xtr, Xte = tree_features(train, extra), tree_features(target, extra)
    y = train[target_col].to_numpy(float)
    if kind == "forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=seed,
                                      n_jobs=-1)
        model.fit(Xtr.fillna(-1), y)
        return pd.Series(model.predict(Xte.fillna(-1)), index=target.index)
    import lightgbm as lgb
    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    model.fit(Xtr, y)
    return pd.Series(model.predict(Xte), index=target.index)
