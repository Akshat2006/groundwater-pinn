"""Staged PINN training for one zone (PLAN.md §2.4, §5 M2).

Stages:
1. data_only:      fit h_net to the well readings
2. physics_frozen: add the PDE loss; S, T held at their starting values; q_net and h_net learn
3. joint:          everything learns, including S and T, with a decaying learning rate (S and T can
                   ramp up over `param_warmup` steps so they don't jump while the networks catch up)
4. lbfgs:          optional L-BFGS polish (fixed collocation set), 0 steps = off

Loss in stages 2–4: L_pde + λ·L_data (+ censored + prior). λ comes from the gradient-size rule in
weighting.py (or is fixed if training.weighting.type = "fixed").

Optional per-well offsets (training.well_offsets.weight > 0, real data): each well gets a constant correction to
its measured level, with a Gaussian prior of spread `sigma_m` against reading noise `sigma_noise_m`. It absorbs the error of the elevation model at the well
(PLAN.md §2.8 item 10), so the network's surface need not bend to fit each well's elevation error. A place
without a well gets offset 0.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.pinn import Normaliser, ZonePINN
from src.physics.recharge import GridField
from src.utils.config import set_seed
from src.utils.scalers import MinMaxScaler, StandardScaler

from .collocation import CollocationSampler
from .losses import censored_loss, data_loss, pde_loss, prior_loss
from .weighting import GradNormWeighting


@dataclass
class ZoneData:
    """Everything one zone's PINN needs. Coordinates in metres, time in days, heads in metres."""
    x_m: np.ndarray
    y_m: np.ndarray
    t_days: np.ndarray
    head_m: np.ndarray
    bounds: dict                      # {"x": (lo, hi), "y": (lo, hi), "t": (lo, hi)}
    recharge: GridField               # m/day, already on the training device
    landuse: GridField
    S_init: float
    T_init: float
    S_prior: float | None = None
    T_prior: float | None = None
    dry: dict | None = None           # {"x_m", "y_m", "t_days", "bottom_m"} for censored readings
    mask: torch.Tensor | None = None  # optional zone mask over the bounding box
    well_index: np.ndarray | None = None   # per reading: index 0..n_wells-1 of its well (for per-well offsets)
    n_wells: int = 0


@dataclass
class TrainResult:
    model: ZonePINN
    history: pd.DataFrame
    seconds: float
    info: dict = field(default_factory=dict)


def make_normaliser(zone: ZoneData, flux_scale_m_per_day: float) -> Normaliser:
    b = zone.bounds
    return Normaliser(MinMaxScaler(*b["x"]), MinMaxScaler(*b["y"]), MinMaxScaler(*b["t"]),
                      StandardScaler.fit(zone.head_m), flux_scale_m_per_day)


def train_zone(cfg: dict, zone: ZoneData, seed: int = 0, run_dir: Path | None = None,
               device: str | torch.device | None = None, verbose: bool = True) -> TrainResult:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)
    tcfg = cfg["training"]
    flux = tcfg["flux_scale_m_per_year"] / 365.25
    norm = make_normaliser(zone, flux)
    qcfg = dict(cfg["q_net"])
    lu_vals = zone.landuse.values
    qcfg.setdefault("lu_mean", float(lu_vals.mean()))
    qcfg.setdefault("lu_std", float(lu_vals.std()) or 1.0)
    model = ZonePINN({**cfg, "q_net": qcfg}, norm, zone.landuse, zone.S_init, zone.T_init).to(device)

    def tens(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)

    xyt_obs = norm.to_norm(tens(zone.x_m), tens(zone.y_m), tens(zone.t_days))
    h_obs_n = norm.h.transform(tens(zone.head_m)).unsqueeze(-1)
    ocfg = tcfg.get("well_offsets", {"weight": 0.0})
    use_offsets = ocfg.get("weight", 0.0) > 0 and zone.well_index is not None
    if use_offsets:
        model.well_offset = torch.nn.Parameter(torch.zeros(zone.n_wells, device=device))
        w_idx = torch.as_tensor(zone.well_index, dtype=torch.long, device=device)
        # Gaussian prior on offsets, on the same footing as the data term: with reading noise sigma_noise and
        # offset spread sigma_m, the most likely fit minimises  Σ err² / sigma_noise² + Σ offset² / sigma_m².
        # Divided by N / sigma_noise² (data_loss is a mean over N readings, in normalised units) that is
        # data_loss + (sigma_noise / sigma_m)² · (n_wells / N) · mean(offset²), offsets in normalised units.
        off_coef = (ocfg.get("weight", 1.0) * (ocfg.get("sigma_noise_m", 1.0) / ocfg.get("sigma_m", 3.0)) ** 2
                    * zone.n_wells / len(zone.head_m))

    def h_at_obs(idx=None):
        h = model.head_norm(xyt_obs if idx is None else xyt_obs[idx])
        if use_offsets:
            h = h + model.well_offset[w_idx if idx is None else w_idx[idx]].unsqueeze(-1)
        return h

    def offset_penalty():
        return off_coef * torch.mean(model.well_offset ** 2) if use_offsets else 0.0
    if zone.dry:
        xyt_dry = norm.to_norm(tens(zone.dry["x_m"]), tens(zone.dry["y_m"]), tens(zone.dry["t_days"]))
        bottom_n = norm.h.transform(tens(zone.dry["bottom_m"])).unsqueeze(-1)
    gen = torch.Generator(device=device).manual_seed(seed)
    sampler = CollocationSampler(tcfg["n_colloc"], device, zone.mask, gen)

    net_params = list(model.h_net.parameters()) + list(model.q_net.parameters())
    if use_offsets:
        net_params.append(model.well_offset)
    opt = torch.optim.Adam([{"params": net_params, "lr": tcfg["lr"]},
                            {"params": [model.log_S, model.log_T], "lr": tcfg["lr_params"]}])
    wcfg = tcfg.get("weighting", {"type": "gradnorm"})
    weighting = GradNormWeighting(list(model.h_net.parameters()), init=wcfg.get("init", 1.0),
                                  alpha=wcfg.get("alpha", 0.1), every=wcfg.get("every", 100))
    pcfg = tcfg.get("prior", {"weight": 0.0})
    use_prior = pcfg.get("weight", 0.0) > 0 and zone.S_prior is not None
    fix_T = bool(tcfg.get("fix_T", False))     # PLAN.md §10 fallback: T held at its starting (report) value
    stages = tcfg["stages"]
    batch = tcfg.get("batch_data", 0)

    def R_at(xyt_n):
        x_m, y_m, t_d = norm.to_phys(xyt_n)
        return zone.recharge(x_m, y_m, t_d)

    def losses(physics: bool):
        if batch and batch < len(xyt_obs):
            idx = torch.randint(len(xyt_obs), (batch,), device=device, generator=gen)
            l_data = data_loss(h_at_obs(idx), h_obs_n[idx])
        else:
            l_data = data_loss(h_at_obs(), h_obs_n)
        out = {"data": l_data}
        if zone.dry:
            out["censored"] = censored_loss(model.head_norm(xyt_dry), bottom_n)
        if physics:
            pts = sampler.sample()
            out["pde"] = pde_loss(model.residual(pts, R_at(pts)))
            if use_prior:
                out["prior"] = prior_loss(model.log_S, model.log_T, zone.S_prior, zone.T_prior, pcfg["sigma_log"])
        return out

    rows, step, t0 = [], 0, time.time()
    log_every = tcfg.get("log_every", 500)
    plan = [("data_only", stages.get("data_only", 0), False, False),
            ("physics_frozen", stages.get("physics_frozen", 0), True, False),
            ("joint", stages.get("joint", 0), True, True)]
    for name, n_steps, physics, learn_params in plan:
        model.log_S.requires_grad_(learn_params)
        model.log_T.requires_grad_(learn_params and not fix_T)
        decay = (tcfg.get("lr_final_factor", 1.0) ** (1 / max(n_steps, 1))) if name == "joint" else 1.0
        base_lrs = [g["lr"] for g in opt.param_groups]
        for i in range(n_steps):
            if name == "joint":
                warm = min(1.0, (i + 1) / max(tcfg.get("param_warmup", 0), 1))   # S, T ramp up gently
                for k, (g, lr0) in enumerate(zip(opt.param_groups, base_lrs)):
                    g["lr"] = lr0 * decay**i * (warm if k == 1 else 1.0)
            opt.zero_grad(set_to_none=True)
            L = losses(physics)
            if physics:
                lam = (weighting.update(step, L["pde"], L["data"]) if wcfg.get("type", "gradnorm") == "gradnorm"
                       else wcfg.get("init", 1.0))
                total = L["pde"] + lam * L["data"]
            else:
                lam, total = float("nan"), L["data"]
            total = (total + L.get("censored", 0.0) * (lam if physics else 1.0) + pcfg.get("weight", 0.0) * L.get("prior", 0.0)
                     + (lam if physics else 1.0) * offset_penalty())
            total.backward()
            opt.step()
            if step % log_every == 0 or i == n_steps - 1:
                rows.append(_log_row(step, name, L, lam, model, norm, time.time() - t0))
                if verbose:
                    r = rows[-1]
                    print(f"{step:6d} {name:15s} data_rmse={r['data_rmse_m']:.3f}m pde={r.get('pde', float('nan')):.2e} "
                          f"lam={lam:.3g} S={r['S']:.4f} T={r['T']:.0f} ({r['seconds']:.0f}s)", flush=True)
            step += 1
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0

    n_lbfgs = stages.get("lbfgs", 0)
    if n_lbfgs:
        model.log_S.requires_grad_(True)
        model.log_T.requires_grad_(not fix_T)
        fixed_pts = CollocationSampler(tcfg.get("lbfgs_colloc", tcfg["n_colloc"]), device, zone.mask, gen).sample()
        R_fixed = R_at(fixed_pts)
        lam = weighting.weight
        lb = torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad], lr=1.0, max_iter=n_lbfgs,
                               history_size=50, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad(set_to_none=True)
            t = pde_loss(model.residual(fixed_pts.clone(), R_fixed)) + lam * (data_loss(h_at_obs(), h_obs_n) + offset_penalty())
            if use_prior:
                t = t + pcfg["weight"] * prior_loss(model.log_S, model.log_T, zone.S_prior, zone.T_prior, pcfg["sigma_log"])
            t.backward()
            return t
        lb.step(closure)
        L = losses(True)
        rows.append(_log_row(step, "lbfgs", L, lam, model, norm, time.time() - t0))

    history = pd.DataFrame(rows)
    result = TrainResult(model, history, time.time() - t0,
                         {"S": float(model.S.detach()), "T": float(model.T.detach()), "device": str(device), "seed": seed})
    if run_dir is not None:
        save_model(result, run_dir)
    return result


def _log_row(step, stage, L, lam, model, norm, seconds) -> dict:
    row = {"step": step, "stage": stage, "lambda": lam, "S": float(model.S.detach()), "T": float(model.T.detach()),
           "seconds": seconds, "data_rmse_m": math.sqrt(float(L["data"].detach())) * norm.h.std}
    for k, v in L.items():
        row[k] = float(v.detach()) if torch.is_tensor(v) else float(v)
    if model.q_net.trend is not None:
        row["q_trend"] = float(model.q_net.trend.detach())
    return row


def save_model(result: TrainResult, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), run_dir / "model.pt")
    (run_dir / "normaliser.json").write_text(json.dumps(result.model.norm.to_dict(), indent=2))
    result.history.to_csv(run_dir / "history.csv", index=False)
