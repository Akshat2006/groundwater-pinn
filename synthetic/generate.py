"""Build a synthetic aquifer with known S, T, R and Q, and simulate heads (PLAN.md §6).

    ds = build_case(load_config("configs/synthetic/case_A.yaml"))

The output Dataset (saved as data/synthetic/case_X/truth.nc) holds:
- `h` (time, y, x): daily heads [m]; the record starts after the spin-up year.
- `h_init` (y, x): the state at the start of the record.
- `S`, `T` (y, x): the true aquifer parameters.
- Forcing, stored as maps × time series (every field here is separable, so this is exact and compact):
  - `rain_mm` (time), `rain_map` (y, x): rain = rain_mm · rain_map [mm/day]
  - `canal_map` (y, x): canal seepage [m/day], constant in time
  - `q_map` (y, x) [m/day] × `q_time` (time) [–] = gross pumping
  - `landuse_true` a(x, y); `landuse_input`, the land-use field the model is given (degraded in case D)
- `budget_*` (time): the solver's daily water budget.

Use `recharge_true`, `recharge_given`, `pumping_gross` and `pumping_net` to rebuild full fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from src.physics.fd_solver import BoundarySpec, FDModel, Grid

DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------- building blocks
def make_grid(cfg: dict) -> Grid:
    g = cfg["grid"]
    return Grid(nx=g["nx"], ny=g["ny"], dx=g["dx_m"], dy=g["dy_m"])


def synthetic_rain(dates: pd.DatetimeIndex, rcfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Daily basin rain [mm/day]. Expected monthly totals put `monsoon_fraction` of the annual total in the
    monsoon months; wet days are random, with exponentially distributed amounts."""
    monsoon = set(rcfg["monsoon_months"])
    n_mon = len(monsoon)
    month_total = {m: rcfg["annual_mm"] * (rcfg["monsoon_fraction"] / n_mon if m in monsoon
                                            else (1 - rcfg["monsoon_fraction"]) / (12 - n_mon))
                   for m in range(1, 13)}
    months = dates.month.values
    p = np.where(np.isin(months, list(monsoon)), rcfg["wet_day_prob"]["monsoon"], rcfg["wet_day_prob"]["other"])
    mean_amount = np.array([month_total[m] for m in months]) / (dates.days_in_month.values * p)
    wet = rng.random(len(dates)) < p
    return np.where(wet, rng.exponential(mean_amount), 0.0)


def smooth_field(grid: Grid, scale_km: float, rng: np.random.Generator, lo=0.0, hi=1.0) -> np.ndarray:
    """Smooth random field rescaled to [lo, hi] (Gaussian-filtered white noise)."""
    z = gaussian_filter(rng.standard_normal(grid.shape), sigma=scale_km * 1000 / grid.dx, mode="reflect")
    z = (z - z.min()) / (z.max() - z.min())
    return lo + (hi - lo) * z


def month_factor(months: np.ndarray, pcfg: dict) -> np.ndarray:
    """Crop-calendar pumping factor, normalised so the 12-month mean is 1."""
    mf = pcfg["month_factor"]
    table = np.full(13, mf["other_value"])
    table[mf["paddy"]] = mf["paddy_value"]
    table[mf["wheat"]] = mf["wheat_value"]
    table[1:] /= table[1:].mean()
    return table[months]


def distance_to_segment(X, Y, p0, p1) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    s = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / (d @ d), 0, 1)
    return np.hypot(X - (p0[0] + s * d[0]), Y - (p0[1] + s * d[1]))


def gaussian_bumps(X, Y, centres_m, sigma_m) -> np.ndarray:
    """Sum of unit-height Gaussian bumps."""
    out = np.zeros_like(X)
    for cx, cy in centres_m:
        out += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma_m**2))
    return out


def coarsen_noisy(field: np.ndarray, block: int, noise_sd: float, rng, lo, hi) -> np.ndarray:
    """Block-average to `block` cells, expand back, add noise: the 'imperfect land-use map' of case D."""
    ny, nx = field.shape
    by, bx = -(-ny // block), -(-nx // block)
    padded = np.pad(field, ((0, by * block - ny), (0, bx * block - nx)), mode="edge")
    coarse = padded.reshape(by, block, bx, block).mean(axis=(1, 3))
    fine = np.kron(coarse, np.ones((block, block)))[:ny, :nx]
    return np.clip(fine + rng.normal(0, noise_sd, field.shape), lo, hi)


# ---------------------------------------------------------------- the case
def build_case(cfg: dict) -> xr.Dataset:
    rng = np.random.default_rng(cfg["seed"])
    grid = make_grid(cfg)
    X, Y = grid.mesh()
    tcfg, pcfg, ex = cfg["time"], cfg["pumping"], cfg.get("extras") or {}
    dt = tcfg["dt_days"]
    if dt != 1.0:
        raise ValueError("forcing is defined per day; use dt_days = 1")

    record = pd.date_range(tcfg["record_start"], tcfg["record_end"], freq="D")
    spin = pd.date_range(end=record[0] - pd.Timedelta(days=1), periods=tcfg["spinup_days"], freq="D")
    days = spin.append(record)                       # forcing day k covers step k (spin-up first)

    # ---- rain and recharge
    rain_mm = synthetic_rain(days, cfg["rain"], rng)
    g = cfg["rain"]["ns_gradient"]
    y_mid, y_half = grid.y0 + grid.ny * grid.dy / 2, grid.ny * grid.dy / 2
    rain_map = 1.0 + g * (Y - y_mid) / y_half
    rif = cfg["recharge"]["rif"]
    canal_map = np.zeros(grid.shape)
    canal_known = True
    if "canal" in ex:
        c = ex["canal"]
        dist = distance_to_segment(X, Y, np.array(c["line_km"][0]) * 1000, np.array(c["line_km"][1]) * 1000)
        canal_map = np.where(dist <= c["half_width_km"] * 1000, c["seepage_m_per_year"] / DAYS_PER_YEAR, 0.0)
        canal_known = bool(c["model_knows"])

    # ---- land use and pumping
    lo, hi = pcfg["landuse_range"]
    a = smooth_field(grid, pcfg["landuse_smooth_km"], rng, lo, hi)
    year0 = record[0].year
    q_time = month_factor(days.month.values, pcfg) * (1 + pcfg["growth_per_year"] * (days.year.values - year0))
    rec = slice(len(spin), None)
    q0 = pcfg["mean_m_per_year"] / DAYS_PER_YEAR / (a.mean() * q_time[rec].mean())
    multiplier = np.ones(grid.shape)
    if "clusters" in ex:
        c = ex["clusters"]
        multiplier += (c["peak_multiplier"] - 1) * gaussian_bumps(X, Y, np.array(c["centres_km"]) * 1000,
                                                                  c["sigma_km"] * 1000)
    if "patches" in ex:
        p = ex["patches"]
        multiplier *= 1 + p["amplitude"] * smooth_field(grid, p["scale_km"] / 2, rng, -1.0, 1.0)
    if "hotspots" in ex:
        h = ex["hotspots"]
        centres = rng.uniform([X.min(), Y.min()], [X.max(), Y.max()], size=(h["n"], 2))
        multiplier += (h["peak_multiplier"] - 1) * gaussian_bumps(X, Y, centres, h["sigma_km"] * 1000)
    q_map = q0 * a * multiplier
    return_frac = float(ex.get("return_flow_fraction", 0.0))
    landuse_input = a
    if "landuse_input" in ex:
        li = ex["landuse_input"]
        block = int(round(li["coarsen_km"] * 1000 / grid.dx))
        landuse_input = coarsen_noisy(a, block, li["noise_sd"], rng, lo, hi)

    # ---- aquifer
    S = np.full(grid.shape, cfg["aquifer"]["S"])
    T = np.full(grid.shape, cfg["aquifer"]["T_m2_day"])
    if "zones" in cfg:
        z = cfg["zones"]
        west = X < z["split_x_m"]
        S = np.where(west, z["west"]["S"], z["east"]["S"])
        T = np.where(west, z["west"]["T_m2_day"], z["east"]["T_m2_day"])

    # ---- simulate: spin-up (discarded) then the record
    bcfg = cfg["boundary"]
    bc = BoundarySpec(west=bcfg["west_head_m"], east=bcfg["east_head_m"])
    L = grid.nx * grid.dx
    h_plane = bcfg["west_head_m"] + (bcfg["east_head_m"] - bcfg["west_head_m"]) * (X - grid.x0) / L
    model = FDModel(grid, S, T, bc, dt)

    def R_of(t_mid):
        k = int(t_mid // dt)
        return rif * rain_mm[k] / 1000.0 * rain_map + canal_map + return_frac * q_map * q_time[k]

    def Q_of(t_mid):
        return q_map * q_time[int(t_mid // dt)]

    n_spin = len(spin)
    spun = model.run(h_plane, R_of, Q_of, n_spin, t0_days=0.0, save_every=n_spin, return_budget=False)
    out = model.run(spun.h_final, R_of, Q_of, len(record), t0_days=n_spin * dt, save_every=1,
                    dtype=np.float32)

    # ---- assemble
    coords = {"x": grid.xc, "y": grid.yc, "time": record}
    yx = ("y", "x")
    ds = xr.Dataset(
        {
            "h": (("time",) + yx, out.heads[1:], {"units": "m", "long_name": "hydraulic head (end of day)"}),
            "h_init": (yx, out.heads[0].astype(np.float64), {"units": "m", "long_name": "head at record start"}),
            "S": (yx, S, {"units": "1"}),
            "T": (yx, T, {"units": "m2/day"}),
            "rain_mm": (("time",), rain_mm[rec], {"units": "mm/day", "long_name": "basin rain series"}),
            "rain_map": (yx, rain_map, {"units": "1", "long_name": "spatial rain factor"}),
            "canal_map": (yx, canal_map, {"units": "m/day"}),
            "q_map": (yx, q_map, {"units": "m/day", "long_name": "gross pumping at q_time = 1"}),
            "q_time": (("time",), q_time[rec], {"units": "1"}),
            "landuse_true": (yx, a, {"units": "1"}),
            "landuse_input": (yx, landuse_input, {"units": "1", "long_name": "land use given to the model"}),
        },
        coords=coords,
    )
    for k, v in out.budget.items():
        ds[f"budget_{k}"] = (("time",), v)
    ds.attrs.update(
        case=cfg["case"], rif=rif, return_flow_fraction=return_frac, canal_known_to_model=int(canal_known),
        dx_m=grid.dx, dy_m=grid.dy, west_head_m=bcfg["west_head_m"], east_head_m=bcfg["east_head_m"],
        config_json=json.dumps(cfg),
    )
    return ds


# ---------------------------------------------------------------- full fields from a truth Dataset
def recharge_true(ds: xr.Dataset) -> xr.DataArray:
    """All recharge that really reached the water table [m/day]: rain + canal + irrigation return flow."""
    return (ds.attrs["rif"] * ds.rain_mm / 1000 * ds.rain_map + ds.canal_map
            + ds.attrs["return_flow_fraction"] * pumping_gross(ds))


def recharge_given(ds: xr.Dataset, factor: float = 1.0) -> xr.DataArray:
    """The recharge the model is told about [m/day]: rain recharge × `factor` (to test a wrong R),
    plus the canal only if the model knows about it."""
    r = factor * ds.attrs["rif"] * ds.rain_mm / 1000 * ds.rain_map
    return r + ds.canal_map if ds.attrs["canal_known_to_model"] else r + 0 * ds.canal_map


def pumping_gross(ds: xr.Dataset) -> xr.DataArray:
    return ds.q_map * ds.q_time


def pumping_net(ds: xr.Dataset) -> xr.DataArray:
    """Gross pumping minus irrigation return flow: what the model's Q can at best represent (PLAN.md §2.1)."""
    return (1 - ds.attrs["return_flow_fraction"]) * pumping_gross(ds)


def save_truth(ds: xr.Dataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
