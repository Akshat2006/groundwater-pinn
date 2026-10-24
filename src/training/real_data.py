"""Turn the real master table into PINN inputs for one zone (or the whole region), and predict at wells.

Time: t_days = days since `t_start` (2005-01-01, the start of the training years), measured to midday.
The physics domain runs to the end of the test years (2024), but well readings come only from training
blocks and training years. Beyond 2018 the model's levels are therefore driven by the flow equation with the
observed rainfall and the learned pumping (a hindcast), not by data.

Inputs per zone:
- wells: training-block wells whose *map* zone is the zone (the same rule held-out places use);
- domain: the zone's cells on the zone map (collocation points are drawn inside it);
- recharge: 0.22 × IMD daily rain (0.25°, ~25 km), regridded to `rain_res_m` (10 km: no detail lost, little memory)
  and smoothed over `smooth_days`;
- land use: cropland fraction on the 1 km grid (input to the pumping network);
- T: fixed at the CGWB report value for the zone (configs/priors.yaml); S starts at the GEC norm 0.12.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.data.imd import rain_at_points
from src.physics.recharge import GridField, running_mean
from src.training.trainer import ZoneData
from src.utils.geo import to_lonlat

T_START = pd.Timestamp("2005-01-01")


def t_days(dates) -> np.ndarray:
    return (pd.DatetimeIndex(dates) - T_START).days.values.astype(float) + 0.5


def report_T(entry, belt_T: float) -> float:
    """One T value (m²/day) from a priors.yaml district entry: a number, the geometric mean of a range, or the
    belt value when the report gives none."""
    T = entry.get("T") if isinstance(entry, dict) else None
    if T is None:
        return belt_T
    if isinstance(T, (list, tuple)):
        return float(np.sqrt(T[0] * T[1]))
    return float(T)


def zone_T(static: xr.Dataset, cells: np.ndarray, priors: dict) -> float:
    """Area-weighted geometric mean of the district report T values over the zone's cells."""
    names = dict(item.split("=") for item in static.attrs["district_names"].split(";"))
    belt = priors["belt"]["T_m2_per_day"]
    ids = static.district_id.values[cells]
    ids = ids[ids > 0]
    if ids.size == 0:
        return belt
    vals = np.array([report_T(priors["districts"].get(names[str(i)], {}), belt) for i in ids])
    return float(np.exp(np.log(vals).mean()))


def recharge_grid(rain: xr.DataArray, x_edges: tuple[float, float], y_edges: tuple[float, float], res_m: float,
                  t_end: pd.Timestamp, rif: float, smooth_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(time, y, x) recharge in m/day on a regular UTM grid covering the box, days from T_START to t_end."""
    xs = np.arange(x_edges[0] + res_m / 2, x_edges[1], res_m)
    ys = np.arange(y_edges[0] + res_m / 2, y_edges[1], res_m)
    X, Y = np.meshgrid(xs, ys)
    lon, lat = to_lonlat(X.ravel(), Y.ravel())
    r = rain.sel(time=slice(T_START, t_end))
    series = rain_at_points(r, lon, lat).values                         # (time, point) mm/day
    R = rif * np.nan_to_num(series, nan=0.0) / 1000.0
    R = running_mean(R.reshape(len(r.time), len(ys), len(xs)), smooth_days)
    return R.astype(np.float32), xs, ys


def zone_inputs(train: pd.DataFrame, zone_cells: np.ndarray, static: xr.Dataset, rain: xr.DataArray, priors: dict,
                cfg: dict, t_end: str = "2024-12-31", device="cpu", rain_res_m: float = 10000.0) -> ZoneData:
    """ZoneData for the wells in `train` (already restricted to the zone) and the zone's map cells."""
    x, y = static.x.values, static.y.values
    res = float(static.attrs.get("res_m", x[1] - x[0]))
    rows, cols = np.nonzero(zone_cells)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    bx = (float(x[c0] - res / 2), float(x[c1 - 1] + res / 2))
    by = (float(y[r0] - res / 2), float(y[r1 - 1] + res / 2))
    t_hi = float(t_days([pd.Timestamp(t_end)])[0] + 0.5)

    ok = train[train.head_m.notna()]
    dry = train[train.is_dry & train.bottom_head_m.notna()]
    wells = pd.Index(ok.well_id.unique())
    R, rx, ry = recharge_grid(rain, bx, by, rain_res_m, pd.Timestamp(t_end), priors["rif"],
                              cfg["training"]["recharge_smooth_days"])
    T_fixed = zone_T(static, zone_cells, priors)
    return ZoneData(
        x_m=ok.x_m.values, y_m=ok.y_m.values, t_days=t_days(ok.date), head_m=ok.head_m.values,
        bounds={"x": bx, "y": by, "t": (0.0, t_hi)},
        recharge=GridField(R, rx, ry, np.arange(R.shape[0]) + 0.5, device=device),
        landuse=GridField(static.lc_cropland_frac.fillna(0).values, x, y, device=device),
        S_init=priors["belt"]["S"], T_init=T_fixed,
        dry=None if dry.empty else {"x_m": dry.x_m.values, "y_m": dry.y_m.values, "t_days": t_days(dry.date),
                                    "bottom_m": dry.bottom_head_m.values},
        mask=torch.as_tensor(zone_cells[r0:r1, c0:c1], device=device),
        well_index=wells.get_indexer(ok.well_id), n_wells=len(wells),
    )


@torch.no_grad()
def predict_heads(model, x_m, y_m, dates, chunk: int = 100_000) -> np.ndarray:
    """Model head (m) at points and dates, without per-well offsets (as for a place with no well)."""
    dev = next(model.parameters()).device
    t = t_days(dates)
    out = np.empty(len(t), dtype=np.float64)
    for a in range(0, len(t), chunk):
        b = slice(a, a + chunk)
        xyt = model.norm.to_norm(*(torch.as_tensor(np.asarray(v[b], dtype=np.float32), device=dev)
                                   for v in (np.asarray(x_m), np.asarray(y_m), t)))
        out[b] = model.head(xyt).squeeze(-1).cpu().numpy()
    return out


def well_offsets_m(model, wells: pd.Index) -> pd.Series:
    """Learned per-well offsets in metres (empty if offsets were off)."""
    off = getattr(model, "well_offset", None)
    if off is None:
        return pd.Series(dtype=float)
    return pd.Series(off.detach().cpu().numpy() * model.norm.h.std, index=wells)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def load_real_run(run_dir, static: xr.Dataset, device="cpu"):
    """Rebuild a trained real-data ZonePINN from its run folder (config.yaml, normaliser.json, model.pt)."""
    import json
    from pathlib import Path

    import yaml

    from src.models.pinn import Normaliser, ZonePINN

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    norm = Normaliser.from_dict(json.loads((run_dir / "normaliser.json").read_text()))
    landuse = GridField(static.lc_cropland_frac.fillna(0).values, static.x.values, static.y.values, device=device)
    state = torch.load(run_dir / "model.pt", map_location=device)
    qcfg = {**cfg["q_net"], "lu_mean": float(landuse.values.mean()), "lu_std": float(landuse.values.std()) or 1.0}
    model = ZonePINN({**cfg, "q_net": qcfg}, norm, landuse, 0.1, 1000.0).to(device)
    if "well_offset" in state:
        model.well_offset = torch.nn.Parameter(torch.zeros_like(state["well_offset"]))
    model.load_state_dict(state)
    return model.eval(), cfg
