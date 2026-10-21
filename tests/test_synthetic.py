"""Synthetic aquifer generator and well sampler (M1)."""
import numpy as np
import pandas as pd
import pytest

from src.utils.config import REPO_ROOT, load_config
from synthetic.generate import (build_case, month_factor, pumping_gross, pumping_net, recharge_given,
                                recharge_true)
from synthetic.sample_wells import heads_at, sample_wells

SMALL = {"grid": {"nx": 20, "ny": 20, "dx_m": 1000.0, "dy_m": 1000.0},
         "time": {"spinup_days": 60, "record_start": "2000-01-01", "record_end": "2001-12-31"}}


def small_case(case: str, **extra):
    cfg = load_config(REPO_ROOT / "configs" / "synthetic" / f"case_{case}.yaml", overrides={**SMALL, **extra})
    if case in ("C", "D"):        # keep features inside the 20 km test domain
        cfg["extras"]["canal"]["line_km"] = [[2.0, 4.0], [18.0, 16.0]]
    if case == "C":
        cfg["extras"]["clusters"]["centres_km"] = [[4.5, 15.5], [15.5, 4.5]]   # on cell centres
    if case == "B":
        cfg["zones"]["split_x_m"] = 10_000.0
    return build_case(cfg)


@pytest.fixture(scope="module")
def case_a():
    return small_case("A")


def test_month_factor_has_mean_one_and_crop_calendar():
    pcfg = load_config(REPO_ROOT / "configs" / "synthetic" / "base.yaml")["pumping"]
    f = month_factor(np.arange(1, 13), pcfg)
    assert f.mean() == pytest.approx(1.0)
    assert f[6] > f[0] > f[4]                         # Jul (paddy) > Jan (wheat) > May (other)


def test_case_a_targets(case_a):
    yr = 365.25
    assert float(pumping_gross(case_a).mean()) * yr == pytest.approx(0.25, rel=1e-9)
    assert case_a.landuse_true.min() == pytest.approx(0.3) and case_a.landuse_true.max() == pytest.approx(1.0)
    assert float(case_a.budget_rel_error.max()) < 1e-8
    np.testing.assert_allclose(recharge_true(case_a), recharge_given(case_a))   # A: model knows R exactly
    assert (case_a.rain_map.isel(y=-1) > case_a.rain_map.isel(y=0)).all()      # wetter to the north


def test_case_b_two_zones():
    ds = small_case("B")
    west, east = ds.isel(x=0), ds.isel(x=-1)
    assert float(west.S[0]) == 0.08 and float(east.S[0]) == 0.16
    assert float(west.T[0]) == 800.0 and float(east.T[0]) == 2500.0


def test_case_c_canal_and_clusters():
    ds = small_case("C")
    assert float(ds.canal_map.max()) == pytest.approx(0.3 / 365.25)
    assert 0 < float((ds.canal_map > 0).mean()) < 0.5
    far = dict(x=19500, y=19500)                                               # > 15 km from both clusters
    ratio = (ds.q_map / ds.landuse_true) / float(ds.q_map.sel(**far) / ds.landuse_true.sel(**far))
    assert float(ratio.max()) == pytest.approx(3.0, rel=0.01)                   # 3x at cluster centres
    q_per_input = ds.q_map / ds.landuse_input                                   # well-posed: the model's map
    assert float(q_per_input.std() / q_per_input.mean()) < 1e-12                # matches true pumping pattern


def test_case_d_hides_things_from_the_model():
    ds = small_case("D")
    assert float((recharge_true(ds) - recharge_given(ds)).min()) > 0           # unknown canal + return flow
    np.testing.assert_allclose(pumping_net(ds), 0.8 * pumping_gross(ds))
    a, li = ds.landuse_true.values.ravel(), ds.landuse_input.values.ravel()
    assert not np.allclose(a, li) and np.corrcoef(a, li)[0, 1] > 0.7


def test_sampling_quarterly_clean(case_a):
    df = sample_wells(case_a, n_wells=12, schedule_kind="quarterly", seed=3, n_hidden=5)
    train, test = df[df.split_space == "train"], df[df.split_space == "test"]
    assert train.well_id.nunique() == 12 and test.well_id.nunique() == 5
    assert set(train.date.dt.month) == {1, 5, 8, 11} and set(train.date.dt.day) == {15}
    assert len(train) == 12 * 8 and len(test) == 5 * 24                        # 2 years
    np.testing.assert_array_equal(train.head_m, train.head_true_m)
    assert train.x_m.between(2000, 18000).all() and train.y_m.between(2000, 18000).all()
    assert df.equals(sample_wells(case_a, n_wells=12, schedule_kind="quarterly", seed=3, n_hidden=5))


def test_sampling_noise_offset_missing_dry(case_a):
    df = sample_wells(case_a, 20, "monthly", seed=1, noise_sd=0.0, offset_sd=2.0, missing_frac=0.1,
                      dry_frac=0.05, n_hidden=0)
    ok = df[~df.is_dry]
    per_well = (ok.head_m - ok.head_true_m).groupby(ok.well_id).agg(["min", "max"])
    np.testing.assert_allclose(per_well["min"], per_well["max"], atol=1e-9)    # offset is constant per well
    assert 0.8 < len(df) / (20 * 24) < 0.97
    dry = df[df.is_dry]
    assert len(dry) > 0 and dry.head_m.isna().all() and (dry.head_true_m < dry.well_bottom_m).all()


def test_heads_at_cell_centre_matches_grid(case_a):
    xy = np.array([[case_a.x.values[3], case_a.y.values[7]]])
    date = pd.DatetimeIndex([case_a.time.values[100]])
    assert heads_at(case_a, xy, date)[0, 0] == pytest.approx(float(case_a.h.isel(time=100, y=7, x=3)))
