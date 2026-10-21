import numpy as np
import pandas as pd
import pytest

from src.data.cgwb import depth_class, nest_level
from src.features.head import add_head, outlier_mask, qc_levels, season_of, spike_mask
from src.utils.config import data_dir
from src.utils.geo import Grid, bilinear, to_lonlat, to_utm


def readings(dtw, dates=None, depth=60.0, well="W1"):
    dates = dates or pd.date_range("2010-01-15", periods=len(dtw), freq="91D")
    return pd.DataFrame({"well_id": well, "date": pd.to_datetime(dates), "dtw_m": np.asarray(dtw, float),
                         "well_depth_m": depth})


def test_head_is_elevation_minus_depth_and_dry_has_no_head():
    df = qc_levels(readings([10.0, 11.0, 14.6], depth=15.0))
    out = add_head(df, np.full(len(df), 230.0))
    assert out.head_m.iloc[0] == pytest.approx(220.0)
    assert out.is_dry.tolist() == [False, False, True]          # within 1 m of the 15 m well bottom
    assert np.isnan(out.head_m.iloc[2]) and out.bottom_head_m.iloc[2] == pytest.approx(215.0)


def test_qc_flags_impossible_values():
    df = qc_levels(readings([-0.5, 12.0, 120.0, 70.0, 12.5], depth=60.0))
    assert df.qc_flag.tolist() == ["nonpositive", "ok", "out_of_range", "below_bottom", "ok"]
    assert df.usable.tolist() == [False, True, False, False, True]


def test_single_and_double_spikes_flagged_but_seasonal_swing_kept():
    swing = [10, 13, 10, 13, 10, 13, 10]
    assert not spike_mask(np.array(swing, float), np.arange(7) * 91.0, 5.0, 800).any()
    one = np.array([10, 10.5, 25, 10.8, 11], float)
    assert spike_mask(one, np.arange(5) * 91.0, 5.0, 800).tolist() == [False, False, True, False, False]
    two = np.array([30, 31, 3.2, 3.4, 32, 33], float)
    assert spike_mask(two, np.arange(6) * 91.0, 5.0, 800).tolist() == [False, False, True, True, False, False]


def test_outlier_catches_bad_end_value_but_not_slow_decline():
    days = np.r_[np.arange(8) * 91.0, 8 * 91.0 + 5 * 365]          # last reading 5 years later
    bad = np.r_[np.full(8, 26.0), 94.0]
    assert outlier_mask(bad, days, 20.0)[-1]
    decline = np.r_[np.full(8, 11.0), 31.0]                          # 20 m over ~7 years: real decline
    days_long = np.r_[np.arange(8) * 91.0, 8 * 91.0 + 19 * 365]
    assert not outlier_mask(decline, days_long, 20.0).any()


def test_copied_reading_marked_duplicate():
    df = qc_levels(readings([12.0, 12.0, 13.0], dates=["2018-10-20", "2018-11-01", "2019-01-05"]))
    assert df.qc_flag.tolist() == ["ok", "duplicate", "ok"]


def test_seasons_and_depth_class():
    s = season_of(pd.Series(pd.to_datetime(["2010-01-05", "2010-05-15", "2010-08-25", "2010-11-05"])))
    assert s.tolist() == ["jan", "pre", "aug", "post"]
    names = pd.Series(["Phurlak M", "Karsola-D_1", "Garhikhajur(medium)", "Doraha Pz", "Ram S", "Kalan"])
    nest = nest_level(names)
    assert nest.tolist()[:3] == ["M", "D", "M"] and nest.iloc[4] == "S" and pd.isna(nest.iloc[3])
    cls = depth_class(pd.Series(["Unconfined", pd.NA, pd.NA, pd.NA, "Confined", pd.NA]), nest,
                      pd.Series([60.0, 300.0, 180.0, 70.0, 80.0, 200.0]), 150.0)
    assert cls.tolist() == ["deep", "deep", "deep", "shallow", "deep", "deep"]


def test_utm_round_trip_and_grid_lookup():
    x, y = to_utm([75.85, 76.9], [30.9, 29.7])
    lon, lat = to_lonlat(x, y)
    assert np.allclose(lon, [75.85, 76.9], atol=1e-7) and np.allclose(lat, [30.9, 29.7], atol=1e-7)
    g = Grid.around((1000.0, 2000.0, 4500.0, 5200.0), res=1000.0)
    assert (g.x0, g.y0, g.nx, g.ny) == (1000.0, 2000.0, 4, 4)
    field = np.add.outer(np.arange(g.ny) * 10.0, np.arange(g.nx))   # value = 10·row + col
    assert g.sample(field, [1500.0, 3500.0], [2500.0, 4500.0]).tolist() == [0.0, 22.0]
    assert bilinear(field, g, [2000.0], [3000.0])[0] == pytest.approx(5.5)   # between 4 cell centres


@pytest.mark.data
def test_master_table_schema():
    proc = data_dir() / "processed"
    if not (proc / "master.parquet").exists():
        pytest.skip("run scripts/build_master.py first")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_m3 import schema_problems
    from src.utils.config import REPO_ROOT, load_config
    m = pd.read_parquet(proc / "master.parquet")
    cfg = load_config(REPO_ROOT / "configs" / "study_area.yaml")
    assert schema_problems(m, cfg["bbox_deg"]) == []
    wells = pd.read_parquet(proc / "wells.parquet")
    assert wells.tier1.sum() >= 150 and (wells[wells.tier1].depth_class == "shallow").all()
    assert m.split_time.isin(["history", "train", "val", "test", "after"]).all()
