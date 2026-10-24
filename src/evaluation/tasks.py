"""The three prediction tasks (PLAN.md §2.6), for one spatial layout.

Every model learns from the same data: shallow wells in *training blocks*, readings in the *training years*
(2005–2018), usable water levels only. They are then asked to predict:

- Task A, forecast: the same training-block wells in later years (test 2020–2024; validation 2019).
- Task B, unmonitored places: wells in held-out blocks during the training years.
- Task C, both: wells in held-out blocks in later years. The hardest and most realistic task.

`stage="val"` gives the tuning version of each task (validation blocks / 2019); `stage="test"` the final one.
"""
from __future__ import annotations

import pandas as pd


def shallow_levels(master: pd.DataFrame) -> pd.DataFrame:
    return master[(master.depth_class == "shallow") & master.head_m.notna()].copy()


def with_layout(df: pd.DataFrame, splits: pd.DataFrame, layout: int) -> pd.DataFrame:
    sp = splits[splits.layout == layout].set_index("well_id")
    out = df.copy()
    out["split_space"] = out.well_id.map(sp.split_space)
    out["block"] = out.well_id.map(sp.block)
    return out


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.split_space == "train") & (df.split_time == "train")]


def target_rows(df: pd.DataFrame, task: str, stage: str = "test") -> pd.DataFrame:
    held = "test" if stage == "test" else "val"
    later = "test" if stage == "test" else "val"
    if task == "A":
        return df[(df.split_space == "train") & (df.split_time == later)]
    if task == "B":
        return df[(df.split_space == held) & (df.split_time == "train")]
    if task == "C":
        return df[(df.split_space == held) & (df.split_time == later)]
    raise ValueError(task)
