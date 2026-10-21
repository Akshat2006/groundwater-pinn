"""Turn a synthetic truth Dataset + sampled wells into PINN training inputs, and score a trained model.

Time convention: t_days = days since the start of the record.
- t = 0 is the start of the first day, so a reading taken at the end of record day k has t = k + 1.
- Recharge for day k applies during [k, k + 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.models.pinn import ZonePINN
from src.physics.recharge import GridField, running_mean
from src.training.trainer import ZoneData

from .generate import pumping_net, recharge_given, recharge_true, pumping_gross

SEASONS = {6: "paddy", 7: "paddy", 8: "paddy", 9: "paddy", 11: "wheat", 12: "wheat", 1: "wheat", 2: "wheat",
           3: "wheat", 4: "other", 5: "other", 10: "other"}


def t_days_of(ds: xr.Dataset, dates) -> np.ndarray:
    start = pd.Timestamp(ds.time.values[0])
    return (pd.DatetimeIndex(dates) - start).days.values.astype(float) + 1.0


def zone_from_synthetic(ds: xr.Dataset, wells: pd.DataFrame, S_init: float, T_init: float,
                        r_factor: float = 1.0, smooth_days: int = 31, device="cpu",
                        S_prior: float | None = None, T_prior: float | None = None) -> ZoneData:
    train = wells[wells.split_space == "train"]
    ok = train[~train.is_dry & train.head_m.notna()]
    dry = train[train.is_dry]
    dx = float(ds.x[1] - ds.x[0])
    dy = float(ds.y[1] - ds.y[0])
    n_days = ds.sizes["time"]
    R = running_mean(recharge_given(ds, r_factor).transpose("time", "y", "x").values, smooth_days)
    t_centres = np.arange(n_days) + 0.5
    return ZoneData(
        x_m=ok.x_m.values, y_m=ok.y_m.values, t_days=t_days_of(ds, ok.date), head_m=ok.head_m.values,
        bounds={"x": (float(ds.x[0]) - dx / 2, float(ds.x[-1]) + dx / 2),
                "y": (float(ds.y[0]) - dy / 2, float(ds.y[-1]) + dy / 2), "t": (0.0, float(n_days))},
        recharge=GridField(R, ds.x.values, ds.y.values, t_centres, device=device),
        landuse=GridField(ds.landuse_input.values, ds.x.values, ds.y.values, device=device),
        S_init=S_init, T_init=T_init, S_prior=S_prior, T_prior=T_prior,
        dry=None if dry.empty else {"x_m": dry.x_m.values, "y_m": dry.y_m.values,
                                    "t_days": t_days_of(ds, dry.date), "bottom_m": dry.well_bottom_m.values},
    )


@torch.no_grad()
def model_pumping_grid(model: ZonePINN, ds: xr.Dataset, every_days: int = 5, chunk: int = 200_000) -> xr.DataArray:
    """Model Q [m/day] at every cell centre on every `every_days`-th day."""
    dev = next(model.parameters()).device
    X, Y = np.meshgrid(ds.x.values, ds.y.values)
    idx = np.arange(0, ds.sizes["time"], every_days)
    t = idx + 0.5
    xs = np.tile(X.ravel(), len(t))
    ys = np.tile(Y.ravel(), len(t))
    ts = np.repeat(t, X.size)
    out = np.empty(len(xs), dtype=np.float32)
    for a in range(0, len(xs), chunk):
        b = slice(a, a + chunk)
        xyt = model.norm.to_norm(*(torch.as_tensor(v[b], dtype=torch.float32, device=dev) for v in (xs, ys, ts)))
        out[b] = model.pumping(xyt).squeeze(-1).cpu().numpy()
    return xr.DataArray(out.reshape(len(t), *X.shape), dims=("time", "y", "x"),
                        coords={"time": ds.time.values[idx], "y": ds.y, "x": ds.x})


def season_keys(times) -> list[str]:
    ti = pd.DatetimeIndex(times)
    out = []
    for d in ti:
        s = SEASONS[d.month]
        year = d.year - 1 if (s == "wheat" and d.month <= 3) else d.year
        out.append(f"{s}-{year}")
    return out


@torch.no_grad()
def score(model: ZonePINN, ds: xr.Dataset, wells: pd.DataFrame, r_factor: float = 1.0, every_days: int = 5) -> dict:
    """Recovery metrics. Targets:
    - Q_net   = gross pumping − irrigation return flow. This is what we would claim; it equals gross pumping
                in cases A–C.
    - Q_equiv = the pumping the model's equation *should* find given the recharge it was told:
                Q_gross − (R_true − R_given). It differs from Q_net only when R is wrong or incomplete.
    """
    out = {"S_true": float(ds.S.mean()), "T_true": float(ds["T"].mean()), "S": float(model.S), "T": float(model.T)}
    out["S_rel_err"] = out["S"] / out["S_true"] - 1
    out["T_rel_err"] = out["T"] / out["T_true"] - 1

    q_model = model_pumping_grid(model, ds, every_days)
    sel = dict(time=q_model.time)
    targets = {"net": pumping_net(ds).sel(**sel),
               "equiv": (pumping_gross(ds) - (recharge_true(ds) - recharge_given(ds, r_factor))).sel(**sel)}
    keys = season_keys(q_model.time.values)
    for name, q_true in targets.items():
        q_true = q_true.transpose("time", "y", "x")
        frame = pd.DataFrame({"key": keys, "model": q_model.mean(("x", "y")).values,
                              "true": q_true.mean(("x", "y")).values})
        tot = frame.groupby("key")[["model", "true"]].mean()
        rel = (tot.model / tot.true - 1).abs()
        out[f"Q_{name}_season_mean_abs_err"] = float(rel.mean())
        out[f"Q_{name}_season_max_abs_err"] = float(rel.max())
        out[f"Q_{name}_total_rel_err"] = float(q_model.mean() / q_true.mean() - 1)
        out[f"Q_{name}_field_rel_l2"] = float(np.sqrt(((q_model.values - q_true.values) ** 2).sum()
                                                      / (q_true.values ** 2).sum()))
    out["Q_true_mean_m_per_year"] = float(targets["net"].mean()) * 365.25
    out["Q_model_mean_m_per_year"] = float(q_model.mean()) * 365.25

    dev = next(model.parameters()).device
    for split in ("train", "test"):
        w = wells[(wells.split_space == split) & wells.head_true_m.notna()]
        xyt = model.norm.to_norm(*(torch.tensor(v, dtype=torch.float32, device=dev)
                                   for v in (w.x_m.values, w.y_m.values, t_days_of(ds, w.date))))
        pred = model.head(xyt).squeeze(-1).cpu().numpy()
        out[f"head_rmse_{split}_m"] = float(np.sqrt(np.mean((pred - w.head_true_m.values) ** 2)))
    return out


def load_run(run_dir, ds: xr.Dataset, device="cpu") -> ZonePINN:
    """Rebuild a trained ZonePINN from reports/runs/<run>/ (config.yaml, normaliser.json, model.pt)."""
    import json
    from pathlib import Path

    import yaml

    from src.models.pinn import Normaliser

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    norm = Normaliser.from_dict(json.loads((run_dir / "normaliser.json").read_text()))
    landuse = GridField(ds.landuse_input.values, ds.x.values, ds.y.values, device=device)
    state = torch.load(run_dir / "model.pt", map_location=device)
    qcfg = {**cfg["q_net"], "lu_mean": float(landuse.values.mean()), "lu_std": float(landuse.values.std()) or 1.0}
    model = ZonePINN({**cfg, "q_net": qcfg}, norm, landuse, 0.1, 1000.0).to(device)
    model.load_state_dict(state)
    return model.eval()
