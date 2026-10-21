"""PDE residual through the scalers (PLAN.md §8, 'critical'): exact functions in, known residual out."""
import math

import pytest
import torch

from src.physics.pde import PhysScales, derivatives, pde_residual
from src.utils.scalers import MinMaxScaler, StandardScaler

XS, YS, TS = MinMaxScaler(0.0, 60_000.0), MinMaxScaler(-5_000.0, 45_000.0), MinMaxScaler(0.0, 3650.0)
HS = StandardScaler(225.0, 4.0)
SC = PhysScales(XS.half, YS.half, TS.half, HS.mean, HS.std, flux_scale_m_per_day=0.25 / 365.25)


def points(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, generator=g) * 2 - 1).requires_grad_(True)


def through_scalers(h_phys_fn, xyt_n):
    """Wrap a physical head function h(x, y, t) as a 'network' in normalised units."""
    x, y, t = XS.inverse(xyt_n[:, 0]), YS.inverse(xyt_n[:, 1]), TS.inverse(xyt_n[:, 2])
    return HS.transform(h_phys_fn(x, y, t)).unsqueeze(-1), (x, y, t)


def test_polynomial_derivatives_in_physical_units():
    a, b, c = 3e-8, -5e-8, 2.5e-3                       # 1/m, 1/m, m/day
    xyt = points()
    h_n, _ = through_scalers(lambda x, y, t: 220 + a * x**2 + b * y**2 + c * t + 1e-9 * x * y, xyt)
    d = derivatives(h_n, xyt, SC)
    torch.testing.assert_close(d["h_t"], torch.full_like(d["h_t"], c), rtol=1e-9, atol=1e-15)
    torch.testing.assert_close(d["h_xx"], torch.full_like(d["h_xx"], 2 * a), rtol=1e-9, atol=1e-18)
    torch.testing.assert_close(d["h_yy"], torch.full_like(d["h_yy"], 2 * b), rtol=1e-9, atol=1e-18)


def test_sine_mode_has_zero_residual():
    S, T, L, A = 0.12, 1500.0, 60_000.0, 5.0
    lam = 2 * math.pi**2 * (T / S) / L**2
    xyt = points()
    h_n, _ = through_scalers(lambda x, y, t: 230 + A * torch.sin(math.pi * x / L) * torch.sin(math.pi * y / L)
                             * torch.exp(-lam * t), xyt)
    zeros = torch.zeros(len(xyt), 1)
    res = pde_residual(h_n, xyt, torch.tensor(S), torch.tensor(T), zeros, zeros, SC)
    assert res.abs().max() < 1e-8


def test_closed_box_units_and_sign():
    """h = h0 + (R − Q)·t/S solves the PDE with no sideways flow: residual 0. Extra pumping δ raises the
    residual by exactly δ/flux_scale, i.e. the model would need h to fall faster."""
    S, R, Q = 0.12, 4e-4, 7e-4                          # m/day
    xyt = points()
    h_n, _ = through_scalers(lambda x, y, t: 200 + (R - Q) * t / S + 0 * x, xyt)
    ones = torch.ones(len(xyt), 1)
    res = pde_residual(h_n, xyt, torch.tensor(S), torch.tensor(1500.0), R * ones, Q * ones, SC)
    assert res.abs().max() < 1e-9
    delta = 1e-4
    res2 = pde_residual(h_n, xyt, torch.tensor(S), torch.tensor(1500.0), R * ones, (Q + delta) * ones, SC)
    torch.testing.assert_close(res2 - res, torch.full_like(res, delta / SC.flux_scale_m_per_day))


def test_residual_terms_returned():
    xyt = points(10)
    h_n, _ = through_scalers(lambda x, y, t: 220 + 1e-3 * t + 0 * x, xyt)
    z = torch.zeros(10, 1)
    _, terms = pde_residual(h_n, xyt, torch.tensor(0.1), torch.tensor(1000.0), z, z, SC, return_terms=True)
    torch.testing.assert_close(terms["storage"], torch.full_like(z, 1e-4))
    assert set(terms) >= {"storage", "lateral", "h_t", "h_xx", "h_yy"}


@pytest.fixture(autouse=True)
def _float64():
    """Exact-solution checks need double precision; restore the default afterwards."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)
