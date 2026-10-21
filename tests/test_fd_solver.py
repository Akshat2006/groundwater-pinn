"""Finite-difference solver against exact solutions (PLAN.md §6.3, §8)."""
import numpy as np
import pytest

from src.physics.fd_solver import BoundarySpec, FDModel, Grid, solve_fd
from synthetic.analytical import closed_box, linear_steady, sine_mode, theis_drawdown

S, T = 0.12, 1500.0


def sine_error(n: int, L=60_000.0, h0=230.0, amp=5.0, days=3650) -> float:
    g = Grid(n, n, L / n, L / n)
    X, Y = g.mesh()
    res = solve_fd(g, S, T, 0.0, 0.0, sine_mode(X, Y, 0, L, S, T, h0, amp), BoundarySpec(h0, h0, h0, h0),
                   dt_days=1.0, n_steps=days, save_every=days)
    return float(np.abs(res.h_final - sine_mode(X, Y, days, L, S, T, h0, amp)).max())


def test_sine_mode_accuracy_and_second_order_convergence():
    e8, e16, e32 = (sine_error(n) for n in (8, 16, 32))
    assert e32 < 1e-3                                  # metres, for a 5 m bump
    for coarse, fine in ((e8, e16), (e16, e32)):       # halving dx should cut the error ~4x
        assert np.log2(coarse / fine) > 1.8


@pytest.mark.parametrize("net", [5e-4, -7e-4])
def test_closed_box_uniform_source_units_and_sign(net):
    g = Grid(10, 8, 1000.0, 500.0)
    res = solve_fd(g, S, T, R=max(net, 0) + 1e-3, Q=max(-net, 0) + 1e-3, h0=200.0, bc=BoundarySpec(),
                   dt_days=1.0, n_steps=365)
    np.testing.assert_allclose(res.h_final, closed_box(365, 200.0, S, net), atol=1e-9)
    assert (res.h_final > 200).all() == (net > 0)      # more recharge than pumping -> rise


def test_linear_profile_is_steady():
    g = Grid(60, 60, 1000.0, 1000.0)
    X, _ = g.mesh()
    h = linear_steady(X, 220.0, 238.0, 60_000.0)
    res = solve_fd(g, S, T, 0.0, 0.0, h, BoundarySpec(west=220.0, east=238.0), dt_days=1.0, n_steps=200)
    np.testing.assert_allclose(res.h_final, h, atol=1e-9)


def test_two_zone_series_flow_conserves_flux():
    """Steady 1D flow through two zones in series. Heads are piecewise linear and the flux through both
    zones is the same: (h_w - h_e) / (L1/T1 + L2/T2). This checks the harmonic-mean face transmissivity."""
    n, dx, T1, T2, hw, he = 40, 500.0, 800.0, 2500.0, 230.0, 220.0
    g = Grid(n, 3, dx, dx)
    Tf = np.where(g.mesh()[0] < n * dx / 2, T1, T2)
    res = solve_fd(g, S, Tf, 0.0, 0.0, 225.0, BoundarySpec(west=hw, east=he), dt_days=1e7, n_steps=5)
    L1 = L2 = n * dx / 2
    q = (hw - he) / (L1 / T1 + L2 / T2)                # m²/day per metre width
    x = g.xc
    exact = np.where(x < L1, hw - q * x / T1, hw - q * L1 / T1 - q * (x - L1) / T2)
    np.testing.assert_allclose(res.h_final[1], exact, atol=1e-8)


def test_water_balance_closes_every_step_heterogeneous():
    rng = np.random.default_rng(1)
    g = Grid(25, 20, 800.0, 1200.0)
    Sf = rng.uniform(0.05, 0.2, g.shape)
    Tf = rng.uniform(300.0, 3000.0, g.shape)
    bc = BoundarySpec(west=220.0, north=np.linspace(225, 235, g.nx))
    R = lambda t: 1e-3 * (1 + np.sin(2 * np.pi * t / 365)) * rng.uniform(0, 1, g.shape)  # noqa: E731
    Q = lambda t: 8e-4 * rng.uniform(0, 2, g.shape)                                       # noqa: E731
    res = solve_fd(g, Sf, Tf, R, Q, rng.uniform(215, 240, g.shape), bc, dt_days=3.0, n_steps=200)
    assert res.budget["rel_error"].max() < 1e-8


def test_theis_drawdown_single_well():
    """Point well in a large grid vs the Theis solution after one year (1 km to 3 km from the well)."""
    n, dx, rate = 161, 250.0, 5000.0
    g = Grid(n, n, dx, dx)
    c = n // 2
    Q = np.zeros(g.shape)
    Q[c, c] = rate / g.cell_area
    res = solve_fd(g, S, T, 0.0, Q, 0.0, BoundarySpec(0.0, 0.0, 0.0, 0.0), dt_days=1.0, n_steps=365,
                   save_every=365)
    for cells in (4, 8, 12):                            # 1, 2, 3 km east of the well
        sim = -res.h_final[c, c + cells]
        assert sim == pytest.approx(theis_drawdown(cells * dx, 365, rate, S, T), rel=0.02)


def test_saved_states_and_reuse_of_factorisation():
    g = Grid(5, 5, 1000.0, 1000.0)
    model = FDModel(g, S, T, BoundarySpec(), dt_days=1.0)
    r1 = model.run(100.0, 1e-3, 0.0, n_steps=10, save_every=5)
    assert r1.heads.shape == (3, 5, 5) and list(r1.times) == [0, 5, 10]
    r2 = model.run(r1.h_final, 1e-3, 0.0, n_steps=10, t0_days=10)
    np.testing.assert_allclose(r2.h_final, closed_box(20, 100.0, S, 1e-3), atol=1e-10)


def test_rejects_nonpositive_parameters():
    with pytest.raises(ValueError):
        FDModel(Grid(3, 3, 1.0, 1.0), 0.0, T, BoundarySpec(), 1.0)
