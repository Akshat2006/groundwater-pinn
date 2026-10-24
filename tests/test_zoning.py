import numpy as np
import pandas as pd
import pytest

from src.evaluation.splits import spatial_block_split
from src.features.behaviour import behaviour_features
from src.utils.config import data_dir
from src.zoning.assign import absorb_small_patches, majority_filter
from src.zoning.cluster import choose_k, cluster_wells, merge_small_zones, order_by


def fake_master(n_wells=3, years=range(2005, 2023), seed=0):
    """Pre (May) and post (Nov) readings per well and year with a known trend and seasonal rise."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        trend, rise = -0.5 * w, 1.0 - w
        for y in years:
            rain = 400 + 50 * (y % 5)
            pre = 200 + trend * (y - 2005)
            rows += [{"well_id": f"W{w}", "date": pd.Timestamp(f"{y}-05-15"), "season": "pre", "head_m": pre,
                      "rain_monsoon_mm": np.nan},
                     {"well_id": f"W{w}", "date": pd.Timestamp(f"{y}-11-10"), "season": "post",
                      "head_m": pre + rise + 0.002 * (rain - 500) + rng.normal(0, 1e-3), "rain_monsoon_mm": rain}]
    return pd.DataFrame(rows)


def test_behaviour_features_recover_known_values():
    f = behaviour_features(fake_master(), ("2005-01-01", "2018-12-31"), min_years=5)
    assert f.loc["W1", "trend_m_per_yr"] == pytest.approx(-0.5, abs=1e-6)
    assert f.loc["W2", "amplitude_m"] == pytest.approx(-1.0, abs=0.05)
    assert f.loc["W0", "monsoon_resp_m_per_100mm"] == pytest.approx(0.2, abs=0.01)
    assert (f.n_years == 14).all()


def test_behaviour_features_ignore_validation_and_test_years():
    """Leakage guard: changing readings after the training period must not change any feature."""
    df = fake_master()
    base = behaviour_features(df, ("2005-01-01", "2018-12-31"), 5)
    late = df.date > "2018-12-31"
    df.loc[late, "head_m"] += 50.0
    df.loc[late, "rain_monsoon_mm"] *= 3
    pd.testing.assert_frame_equal(base, behaviour_features(df, ("2005-01-01", "2018-12-31"), 5))


def test_merge_guarantees_minimum_zone_size_and_is_deterministic():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(c, 0.3, (n, 2)) for c, n in ((0, 40), (5, 30), (10, 4), (-5, 6))])
    lab = cluster_wells(X, 4, seed=0)
    merged = merge_small_zones(lab, X, min_size=12)
    assert np.bincount(merged).min() >= 12 and merged.max() + 1 == 2
    assert np.array_equal(merged, merge_small_zones(cluster_wells(X, 4, seed=0), X, 12))


def test_choose_k_finds_three_clear_groups_and_order_by_trend():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(c, 0.2, (30, 2)) for c in ((0, 0), (4, 0), (0, 4))])
    assert choose_k(X, range(2, 9), seed=0).chosen == 3
    lab = np.array([0, 0, 1, 1, 2, 2])
    trend = np.array([-1.0, -1.2, 0.1, 0.0, -0.5, -0.4])
    assert order_by(lab, trend).tolist() == [1, 1, 3, 3, 2, 2]     # zone 1 = steepest decline


def test_majority_filter_and_small_patches():
    zone = np.ones((30, 30), int)
    zone[:, 15:] = 2
    zone[5, 5] = 2                               # one-cell speckle
    zone[20:24, 3:7] = 2                         # 16-cell patch
    mask = np.ones_like(zone, bool)
    smooth = majority_filter(zone, mask, 3)
    assert smooth[5, 5] == 1
    clean = absorb_small_patches(smooth, mask, min_cells=20)
    assert (clean[:, :12] == 1).all() and (clean[:, 18:] == 2).all()


def test_block_split_shares_disjoint_blocks_and_deterministic():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 200_000, 600), rng.uniform(0, 200_000, 600)
    a = spatial_block_split(x, y, block_km=20, test_frac=0.175, val_frac=0.10, tol=0.025, seed=0)
    b = spatial_block_split(x, y, block_km=20, test_frac=0.175, val_frac=0.10, tol=0.025, seed=0)
    pd.testing.assert_frame_equal(a, b)
    share = a.split_space.value_counts(normalize=True)
    assert 0.15 <= share["test"] <= 0.20 and 0.075 <= share["val"] <= 0.125
    assert a.groupby("block").split_space.nunique().max() == 1    # no block is split between sets
    c = spatial_block_split(x, y, block_km=20, seed=1)
    assert not a.split_space.equals(c.split_space)                # another layout differs


@pytest.mark.data
def test_zoning_outputs():
    proc = data_dir() / "processed"
    if not (proc / "zoning_summary.json").exists():
        pytest.skip("run scripts/build_zones.py first")
    import json
    summ = json.loads((proc / "zoning_summary.json").read_text(encoding="utf-8"))
    m = pd.read_parquet(proc / "master.parquet")
    used = m[(m.depth_class == "shallow") & m.head_m.notna()]
    assert used.split_space.notna().all() and used.zone_map.notna().all()
    for lay in summ["layouts"].values():
        for z in lay["zones"].values():
            assert min(z["wells_per_zone"].values()) >= 12
    # held-out wells never define a zone
    assert m.loc[m.split_space.isin(["val", "test"]), "zone_behaviour"].isna().all()
