import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import coverage, fraction_improved, mae, nse, paired_wilcoxon, rmse
from src.evaluation.tasks import target_rows
from src.models.baselines import bucket, naive


def test_nse_rmse_mae_reference_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nse(y, y) == 1.0
    assert nse(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert rmse(y, y + 2) == pytest.approx(2.0) and mae(y, y - 1) == pytest.approx(1.0)


def test_coverage_and_fraction_improved():
    assert coverage([1, 2, 3, 4], [0, 0, 0, 5], [1.5, 2.5, 2.5, 6]) == pytest.approx(0.5)
    a = pd.Series([1.0, 2.0, 3.0, 1.0], index=list("abcd"))
    b = pd.Series([2.0, 1.0, 4.0, 3.0], index=list("abcd"))
    assert fraction_improved(a, b) == pytest.approx(0.75)
    _, p = paired_wilcoxon(pd.Series(np.arange(20.0)), pd.Series(np.arange(20.0) + 1))
    assert p < 0.01


def test_tasks_select_the_right_rows():
    df = pd.DataFrame({"split_space": ["train", "train", "val", "test", "test"],
                       "split_time": ["train", "test", "train", "train", "test"]})
    assert target_rows(df, "A").index.tolist() == [1]
    assert target_rows(df, "B").index.tolist() == [3]
    assert target_rows(df, "C").index.tolist() == [4]
    assert target_rows(df, "B", "val").index.tolist() == [2]


def test_seasonal_trend_continues_a_straight_decline():
    dates = pd.to_datetime([f"{y}-05-15" for y in range(2005, 2019)])
    train = pd.DataFrame({"well_id": "W", "season": "pre", "date": dates,
                          "head_m": 200 - 0.5 * (np.arange(14))})
    target = pd.DataFrame({"well_id": ["W"], "season": ["pre"], "date": pd.to_datetime(["2022-05-15"])})
    assert naive.seasonal_trend(train, target).iloc[0] == pytest.approx(200 - 0.5 * 17, abs=0.01)
    assert naive.persistence(train, target).iloc[0] == pytest.approx(200 - 0.5 * 13)


def test_bucket_recovers_storage_and_pumping():
    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame({"recharge_prev_m": rng.uniform(0, 0.15, n), "lu_cropland_frac": rng.uniform(0.5, 1, n),
                      "d_paddy": rng.integers(0, 120, n), "d_wheat": rng.integers(0, 150, n),
                      "d_other": rng.integers(0, 90, n)})
    d["gap"] = d.d_paddy + d.d_wheat + d.d_other
    S, q_paddy = 0.12, 0.6                                      # q in m/yr of pumping per unit cropland
    d["dh"] = (d.recharge_prev_m - d.lu_cropland_frac * q_paddy * d.d_paddy / 365.25) / S
    beta = bucket.fit(d)
    assert 1 / beta[0] == pytest.approx(S, rel=1e-3)
    assert beta[1] / beta[0] == pytest.approx(q_paddy, rel=1e-3)
