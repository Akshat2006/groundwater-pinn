"""Scalers, grid lookups, q_net, and a short training smoke test (PLAN.md §8)."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.pinn import Normaliser, ZonePINN
from src.models.q_net import QNet
from src.physics.recharge import GridField, running_mean
from src.utils.config import REPO_ROOT, load_config
from src.utils.scalers import MinMaxScaler, StandardScaler, load_scalers, save_scalers


def test_scalers_round_trip_and_save(tmp_path):
    mm, st = MinMaxScaler(10.0, 50.0), StandardScaler.fit([1.0, 2.0, 3.0, np.nan])
    v = np.array([10.0, 30.0, 50.0])
    np.testing.assert_allclose(mm.transform(v), [-1, 0, 1])
    np.testing.assert_allclose(mm.inverse(mm.transform(v)), v)
    np.testing.assert_allclose(st.inverse(st.transform(v)), v)
    save_scalers({"x": mm, "h": st}, tmp_path / "s.json")
    assert load_scalers(tmp_path / "s.json") == {"x": mm, "h": st}


def test_gridfield_bilinear_and_nearest_day():
    x, y = np.array([0.0, 1000.0, 2000.0]), np.array([0.0, 1000.0])
    X, Y = np.meshgrid(x, y)
    vals = np.stack([X + 2 * Y, 10 + X + 2 * Y])                # linear in space, two days
    f = GridField(vals, x, y, t_days=np.array([0.5, 1.5]))
    pts_x, pts_y = torch.tensor([250.0, 1999.0, -50.0]), torch.tensor([500.0, 0.0, 2000.0])
    out = f(pts_x, pts_y, torch.tensor([0.2, 1.4, 1.9])).squeeze(-1)
    torch.testing.assert_close(out, torch.tensor([1250.0, 1999.0 + 10, 0 + 2000 + 10]))   # last point clamped


def test_running_mean_keeps_totals_in_the_interior():
    v = np.zeros(100)
    v[50] = 31.0                                                 # one rainy day
    sm = running_mean(v[:, None, None], 31)[:, 0, 0]
    assert sm.sum() == pytest.approx(31.0) and sm.max() == pytest.approx(1.0)


def test_qnet_nonnegative_and_restricted_ignores_location():
    q = QNet(5, q_scale_m_day=1e-3)
    out = q(torch.randn(1000, 5) * 10, t_norm=torch.rand(1000, 1) * 2 - 1)
    assert (out >= 0).all()
    with pytest.raises(ValueError):
        QNet(3, mode="other")

    cfg = load_config(REPO_ROOT / "configs" / "pinn.yaml")
    norm = Normaliser(MinMaxScaler(0, 20_000), MinMaxScaler(0, 20_000), MinMaxScaler(0, 730),
                      StandardScaler(220.0, 2.0), 1e-3)
    x = np.arange(500.0, 20_000, 1000)
    flat = GridField(np.full((20, 20), 0.5), x, x)
    model = ZonePINN(cfg, norm, flat, 0.12, 1500.0)
    t = torch.full((50, 1), 0.3)
    pts = torch.cat([torch.rand(50, 2) * 2 - 1, t], dim=1)
    Q = model.pumping(pts)
    torch.testing.assert_close(Q, Q[:1].expand_as(Q))          # same land use + time -> same Q anywhere


def test_calendar_month():
    from src.models.pinn import calendar_month
    t = torch.tensor([0.5, 31.5, 190.0, 364.0, 365.25 + 200.0, 3650.0 + 15])
    assert calendar_month(t).tolist() == [0, 1, 6, 11, 6, 0]
    assert calendar_month(torch.tensor([0.5]), doy_at_t0=181.0).tolist() == [5]   # record starting 1 July


def test_month_season_features_change_pumping_by_month():
    cfg = load_config(REPO_ROOT / "configs" / "pinn.yaml", overrides={"q_net": {"season_features": "month"}})
    norm = Normaliser(MinMaxScaler(0, 20_000), MinMaxScaler(0, 20_000), MinMaxScaler(0, 730),
                      StandardScaler(220.0, 2.0), 1e-3)
    x = np.arange(500.0, 20_000, 1000)
    model = ZonePINN(cfg, norm, GridField(np.full((20, 20), 0.5), x, x), 0.12, 1500.0)
    same_month = norm.to_norm(torch.tensor([5e3, 5e3]), torch.tensor([5e3, 5e3]), torch.tensor([190.0, 200.0]))
    q = model.pumping(same_month)
    assert model.q_net.net[0].in_features == 13
    torch.testing.assert_close(q[0], q[1] * torch.exp(model.q_net.trend * (same_month[0, 2] - same_month[1, 2])))


def test_pumping_window_mean_matches_manual_average():
    cfg = load_config(REPO_ROOT / "configs" / "pinn.yaml",
                      overrides={"q_net": {"season_features": "month", "smooth_days": 31, "smooth_points": 5}})
    norm = Normaliser(MinMaxScaler(0, 20_000), MinMaxScaler(0, 20_000), MinMaxScaler(0, 730),
                      StandardScaler(220.0, 2.0), 1e-3)
    x = np.arange(500.0, 20_000, 1000)
    model = ZonePINN(cfg, norm, GridField(np.linspace(0, 1, 400).reshape(20, 20), x, x), 0.12, 1500.0)
    pts = norm.to_norm(torch.tensor([3e3, 9e3]), torch.tensor([4e3, 15e3]), torch.tensor([170.0, 300.0]))
    manual = torch.stack([model.pumping(pts + torch.tensor([0.0, 0.0, d / norm.t.half]))
                          for d in (-15.0, -7.5, 0.0, 7.5, 15.0)]).mean(0)
    torch.testing.assert_close(model.pumping_window_mean(pts), manual)
    model.q_smooth_days = 1
    torch.testing.assert_close(model.pumping_window_mean(pts), model.pumping(pts))


@pytest.mark.slow
def test_smoke_train_small_synthetic_case():
    """300 steps on a tiny case on the CPU: loss falls, nothing becomes NaN, S and T stay finite."""
    from synthetic.generate import build_case
    from synthetic.pinn_data import zone_from_synthetic
    from synthetic.sample_wells import sample_wells
    from src.training.trainer import train_zone

    small = {"grid": {"nx": 16, "ny": 16, "dx_m": 1000.0, "dy_m": 1000.0},
             "time": {"spinup_days": 30, "record_start": "2000-01-01", "record_end": "2001-12-31"}}
    ds = build_case(load_config(REPO_ROOT / "configs" / "synthetic" / "case_A.yaml", overrides=small))
    wells = sample_wells(ds, 10, "monthly", seed=0, n_hidden=0, margin_m=1500)
    zone = zone_from_synthetic(ds, wells, 0.2, 3000.0)
    cfg = load_config(REPO_ROOT / "configs" / "pinn.yaml", overrides={
        "h_net": {"width": 24, "depth": 3},
        "training": {"n_colloc": 256, "stages": {"data_only": 100, "physics_frozen": 50, "joint": 150, "lbfgs": 0},
                     "log_every": 50}})
    res = train_zone(cfg, zone, seed=0, device="cpu", verbose=False)
    h = res.history
    assert h.data_rmse_m.iloc[-1] < 0.5 * h.data_rmse_m.iloc[0]
    for col in ("data", "data_rmse_m", "S", "T"):
        assert np.isfinite(h[col]).all(), col
    assert np.isfinite(h["pde"].dropna()).all() and h["pde"].notna().any()     # NaN only in data-only rows
    assert np.isfinite([res.info["S"], res.info["T"]]).all() and res.info["S"] > 0


def test_fix_T_holds_T_and_prior_runs_through_lbfgs():
    """training.fix_T keeps T at its starting value through Adam and L-BFGS; the prior term runs in both."""
    from synthetic.generate import build_case
    from synthetic.pinn_data import zone_from_synthetic
    from synthetic.sample_wells import sample_wells
    from src.training.trainer import train_zone

    small = {"grid": {"nx": 12, "ny": 12, "dx_m": 1000.0, "dy_m": 1000.0},
             "time": {"spinup_days": 30, "record_start": "2000-01-01", "record_end": "2000-12-31"}}
    ds = build_case(load_config(REPO_ROOT / "configs" / "synthetic" / "case_A.yaml", overrides=small))
    wells = sample_wells(ds, 8, "monthly", seed=0, n_hidden=0, margin_m=1500)
    zone = zone_from_synthetic(ds, wells, 0.2, 2000.0, S_prior=0.15, T_prior=2000.0)
    cfg = load_config(REPO_ROOT / "configs" / "pinn.yaml", overrides={
        "h_net": {"width": 16, "depth": 2},
        "training": {"n_colloc": 128, "lbfgs_colloc": 128, "fix_T": True, "prior": {"weight": 0.1, "sigma_log": 0.5},
                     "stages": {"data_only": 20, "physics_frozen": 10, "joint": 60, "lbfgs": 5}, "log_every": 20}})
    res = train_zone(cfg, zone, seed=0, device="cpu", verbose=False)
    assert res.info["T"] == pytest.approx(2000.0, rel=1e-5)
    assert res.info["S"] != pytest.approx(0.2, rel=1e-6)          # S still learns
    assert "prior" in res.history.columns and np.isfinite(res.history["prior"].dropna()).all()


def test_well_offsets_absorb_a_constant_shift_per_well():
    """Two wells read the same surface, one shifted by +3 m (elevation error): the offset takes the shift."""
    from synthetic.generate import build_case
    from synthetic.pinn_data import zone_from_synthetic
    from synthetic.sample_wells import sample_wells
    from src.training.trainer import train_zone

    small = {"grid": {"nx": 12, "ny": 12, "dx_m": 1000.0, "dy_m": 1000.0},
             "time": {"spinup_days": 30, "record_start": "2000-01-01", "record_end": "2000-12-31"}}
    ds = build_case(load_config(REPO_ROOT / "configs" / "synthetic" / "case_A.yaml", overrides=small))
    wells = sample_wells(ds, 8, "monthly", seed=0, n_hidden=0, margin_m=1500)
    ids = pd.Index(wells.well_id.unique())
    shift = np.where(wells.well_id == ids[0], 3.0, 0.0)
    wells = wells.assign(head_m=wells.head_m + shift)
    zone = zone_from_synthetic(ds, wells, 0.12, 1500.0)
    ok = wells[wells.split_space.eq("train") & ~wells.is_dry & wells.head_m.notna()]
    zone.well_index, zone.n_wells = ids.get_indexer(ok.well_id), len(ids)
    cfg = load_config(REPO_ROOT / "configs" / "pinn.yaml", overrides={
        "h_net": {"width": 16, "depth": 2},
        "training": {"n_colloc": 128, "fix_T": True, "well_offsets": {"weight": 0.1, "sigma_m": 5.0},
                     "stages": {"data_only": 400, "physics_frozen": 0, "joint": 0, "lbfgs": 0}, "log_every": 100}})
    res = train_zone(cfg, zone, seed=0, device="cpu", verbose=False)
    off = res.model.well_offset.detach().numpy() * res.model.norm.h.std
    assert off[0] - np.median(off[1:]) > 1.5                     # the shifted well carries most of the 3 m


def test_S_range_squash_starts_at_S_init_and_stays_inside():
    cfg = load_config(REPO_ROOT / "configs" / "pinn.yaml", overrides={"training": {"S_range": [0.05, 0.2]}})
    norm = Normaliser(MinMaxScaler(0, 20_000), MinMaxScaler(0, 20_000), MinMaxScaler(0, 730),
                      StandardScaler(220.0, 2.0), 1e-3)
    x = np.arange(500.0, 20_000, 1000)
    model = ZonePINN(cfg, norm, GridField(np.full((20, 20), 0.5), x, x), 0.12, 1500.0)
    assert float(model.S) == pytest.approx(0.12, rel=1e-5)
    for v in (-50.0, 50.0):
        model.log_S.data.fill_(v)
        assert 0.05 - 1e-6 <= float(model.S) <= 0.2 + 1e-6


def test_terrain_base_adds_to_head_and_enters_the_pde_as_a_source():
    """head = base + network; the residual equals the base-free residual with R + T·∇²base."""
    cfg = load_config(REPO_ROOT / "configs" / "pinn.yaml")
    norm = Normaliser(MinMaxScaler(0, 20_000), MinMaxScaler(0, 20_000), MinMaxScaler(0, 730),
                      StandardScaler(0.0, 2.0), 1e-3)
    x = np.arange(500.0, 20_000, 1000)
    X, Y = np.meshgrid(x, x)
    base = GridField(250.0 + 1e-3 * X, x, x)
    lap = GridField(np.full((20, 20), 2e-8), x, x)
    land = GridField(np.full((20, 20), 0.5), x, x)
    torch.manual_seed(0)
    plain = ZonePINN(cfg, norm, land, 0.12, 1500.0)
    with_base = ZonePINN(cfg, norm, land, 0.12, 1500.0, base, lap)
    with_base.load_state_dict(plain.state_dict())
    pts = norm.to_norm(torch.tensor([3e3, 9e3]), torch.tensor([4e3, 15e3]), torch.tensor([170.0, 300.0]))
    torch.testing.assert_close(with_base.head(pts) - plain.head(pts), torch.tensor([[253.0], [259.0]]))
    R = torch.full((2, 1), 1e-4)
    torch.testing.assert_close(with_base.residual(pts.clone(), R), plain.residual(pts.clone(), R + 1500.0 * 2e-8))


def test_shared_zoned_model_uses_each_zones_S_T_and_pumping():
    """E20: one network, S and T per zone; the residual inside a zone equals a one-zone model with that zone's S, T."""
    from src.training.real_data import zone_weight_fields
    import xarray as xr

    cfg = {**load_config(REPO_ROOT / "configs" / "pinn.yaml")}
    cfg["q_net"] = {**cfg["q_net"], "season_features": "month"}
    norm = Normaliser(MinMaxScaler(0, 20_000), MinMaxScaler(0, 20_000), MinMaxScaler(0, 730),
                      StandardScaler(0.0, 2.0), 1e-3)
    x = np.arange(500.0, 20_000, 1000)
    land = GridField(np.full((20, 20), 0.5), x, x)
    zmap = np.where(x[None, :] < 10_000, 1, 2) * np.ones((20, 1), int)     # west half zone 1, east half zone 2
    mask = np.ones((20, 20), bool)
    mask[:, :2] = False                                                     # outside the area: nearest zone
    ids, fields = zone_weight_fields(zmap, mask, xr.Dataset(coords={"x": x, "y": x}))
    assert ids == [1, 2] and float(fields[0](torch.tensor([600.0]), torch.tensor([600.0]))) == 1.0
    torch.manual_seed(0)
    shared = ZonePINN(cfg, norm, land, [0.10, 0.14], [800.0, 2500.0], zone_w=fields)
    pts = norm.to_norm(torch.tensor([3e3, 16e3]), torch.tensor([4e3, 15e3]), torch.tensor([170.0, 300.0]))
    S, T = shared.S_T_at(pts)
    torch.testing.assert_close(S, torch.tensor([[0.10], [0.14]]))
    torch.testing.assert_close(T, torch.tensor([[800.0], [2500.0]]))
    R = torch.full((2, 1), 1e-4)
    terms = shared.residual(pts.clone(), R, return_terms=True)[1]
    torch.testing.assert_close(terms["storage"], S * terms["h_t"])
    torch.testing.assert_close(terms["lateral"], T * (terms["h_xx"] + terms["h_yy"]))
    with torch.no_grad():                                 # pumping trend differs per zone
        shared.q_trend_z.copy_(torch.tensor([0.0, 1.0]))
    p_late = norm.to_norm(torch.tensor([3e3, 16e3]), torch.tensor([4e3, 4e3]), torch.tensor([700.25, 700.25]))   # same month, a year later
    p_early = norm.to_norm(torch.tensor([3e3, 16e3]), torch.tensor([4e3, 4e3]), torch.tensor([335.0, 335.0]))
    ratio = shared.pumping(p_late) / shared.pumping(p_early)
    torch.testing.assert_close(ratio[0], torch.tensor([1.0]))
    assert float(ratio[1]) > 1.5


def test_report_T_and_time_axis():
    from src.training.real_data import report_T, t_days
    assert report_T({"T": [400, 1600]}, 1500) == pytest.approx(800.0)
    assert report_T({"T": 2200}, 1500) == 2200.0 and report_T({"T": None}, 1500) == 1500.0
    assert t_days(pd.to_datetime(["2005-01-01", "2005-01-11"])).tolist() == [0.5, 10.5]


def test_canal_recharge_follows_the_gec_formula():
    """Recharge = density × wetted area per km × seepage factor × running days / year, spread without losing water."""
    import xarray as xr
    from src.training.real_data import canal_recharge

    x = np.arange(500.0, 30_000, 1000)
    dens = np.zeros((30, 30))
    dens[15, :] = 1.0                                     # one canal: 1 km per km² along a row
    st = xr.Dataset({"canal_density_km_per_km2": (("y", "x"), dens)}, coords={"x": x, "y": x})
    ccfg = {"strength": 1.0, "seepage_m_per_day": 0.175, "wetted_m": 10, "days_per_yr": 180, "sigma_km": 0}
    R = canal_recharge(st, ccfg).values[0].numpy()
    per_km2 = 10 * 1000 * 0.175 * 180 / 365.25 / 1e6
    assert R[15, 3] == pytest.approx(per_km2, rel=1e-5) and R[3, 3] == 0
    spread = canal_recharge(st, {**ccfg, "sigma_km": 2, "strength": 2.0}).values[0].numpy()
    assert spread.sum() == pytest.approx(2 * R.sum(), rel=1e-3) and spread[13, 3] > 0
