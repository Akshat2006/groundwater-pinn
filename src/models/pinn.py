"""ZonePINN: h_net + q_net + learnable log S, log T for one zone (or the whole region when k = 1).

All public methods take normalised inputs xyt_n (columns x_n, y_n, t_n in [−1, 1]); `Normaliser`
converts between those and physical metres/days/heads.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from src.physics.pde import PhysScales, pde_residual
from src.physics.recharge import GridField
from src.utils.scalers import MinMaxScaler, StandardScaler

from .h_net import HNet, annual_harmonics
from .q_net import QNet


def calendar_month(t_days: Tensor, doy_at_t0: float = 0.0) -> Tensor:
    """Month index 0–11 from days since t = 0. Uses equal 365.25/12-day months, so a month boundary can be
    off by about a day from the real calendar, which is harmless for crop seasons."""
    doy = torch.remainder(t_days + doy_at_t0, 365.25)
    return torch.clamp((doy / (365.25 / 12)).long(), 0, 11)


@dataclass
class Normaliser:
    x: MinMaxScaler
    y: MinMaxScaler
    t: MinMaxScaler
    h: StandardScaler
    flux_scale_m_per_day: float

    def phys_scales(self) -> PhysScales:
        return PhysScales(self.x.half, self.y.half, self.t.half, self.h.mean, self.h.std, self.flux_scale_m_per_day)

    def to_norm(self, x_m, y_m, t_days) -> Tensor:
        return torch.stack([self.x.transform(x_m), self.y.transform(y_m), self.t.transform(t_days)], dim=-1)

    def to_phys(self, xyt_n: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.x.inverse(xyt_n[:, 0]), self.y.inverse(xyt_n[:, 1]), self.t.inverse(xyt_n[:, 2])

    def to_dict(self) -> dict:
        return {"x": asdict(self.x), "y": asdict(self.y), "t": asdict(self.t), "h": asdict(self.h),
                "flux_scale_m_per_day": self.flux_scale_m_per_day}

    @classmethod
    def from_dict(cls, d: dict) -> "Normaliser":
        return cls(MinMaxScaler(**d["x"]), MinMaxScaler(**d["y"]), MinMaxScaler(**d["t"]),
                   StandardScaler(**d["h"]), d["flux_scale_m_per_day"])


class ZonePINN(nn.Module):
    """Optional real-data extras:
    - `base` (+ its Laplacian `lap_base`): a fixed smooth terrain surface [m]. The network then models head − base,
      so in the gaps between wells the water table follows the land (depth to water is far smoother in space
      than head). The PDE still acts on the full head: T∇²base is passed to the residual as a known source.
    - training.S_range [lo, hi]: S is kept inside this range (log-uniform squash). Real data push S towards 0
      when the network fits noise (M2 noise runs; real batch 1), which switches the time physics off.
    - `zone_w` (PLAN.md E20, "shared" zoned model): one indicator field per zone (bilinear, so weights sum to 1
      and blend over one grid cell at borders). One h_net covers the whole region, so every well informs the
      water-level surface, while S, T and pumping are set per zone: log_S and log_T become one value per zone,
      the zone weights join the pumping inputs, and each zone has its own pumping trend. With T changing
      across a border the ∇T·∇h term is left out (the constant-T-per-zone form of the brief).
    """

    def __init__(self, cfg: dict, norm: Normaliser, landuse: GridField, S_init, T_init,
                 base: GridField | None = None, lap_base: GridField | None = None,
                 zone_w: list[GridField] | None = None):
        super().__init__()
        self.norm, self.sc, self.landuse = norm, norm.phys_scales(), landuse
        self.base, self.lap_base = base, lap_base
        self.zone_w = zone_w
        n_z = len(zone_w) if zone_w else 0
        h_cfg, q_cfg = cfg["h_net"], cfg["q_net"]
        self.n_harm_q = q_cfg.get("n_harmonics", 2)
        # season input for pumping: "harmonics" (smooth annual waves) or "month" (one-hot calendar month,
        # which can represent sharp crop-calendar switches such as paddy irrigation starting in June)
        self.season_features = q_cfg.get("season_features", "harmonics")
        self.doy_at_t0 = float(q_cfg.get("doy_at_t0", 0.0))      # day of year (0 = 1 Jan) at t = 0
        self.lu_mean, self.lu_std = q_cfg.get("lu_mean", 0.0), q_cfg.get("lu_std", 1.0)
        self.h_net = HNet(3, h_cfg["width"], h_cfg["depth"], h_cfg.get("act", "tanh"),
                          norm.t.centre, norm.t.half, h_cfg.get("n_harmonics", 2))
        self.q_mode = q_cfg.get("mode", "restricted")
        n_season = 12 if self.season_features == "month" else 2 * self.n_harm_q
        n_feat = 1 + n_season + (3 if self.q_mode == "full" else 0) + n_z
        self.q_net = QNet(n_feat, q_cfg["width"], q_cfg["depth"], q_cfg.get("q_scale_m_day", norm.flux_scale_m_per_day),
                          self.q_mode, q_cfg.get("act", "tanh"))
        self.q_trend_z = nn.Parameter(torch.zeros(n_z)) if n_z and self.q_mode == "restricted" else None

        def per_zone(v):                        # scalar, or one value per zone for a shared zoned model
            v = torch.as_tensor(v, dtype=torch.float32)
            return v.expand(n_z).clone() if n_z else v.reshape(())
        S_range = cfg.get("training", {}).get("S_range")
        self.S_log_range = None if S_range is None else (math.log(S_range[0]), math.log(S_range[1]))
        if self.S_log_range is None:
            self.log_S = nn.Parameter(torch.log(per_zone(S_init)))
        else:                                   # log_S holds the unsquashed value: log S = lo + (hi − lo)·sigmoid(log_S)
            lo, hi = self.S_log_range
            u = ((torch.log(per_zone(S_init)) - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
            self.log_S = nn.Parameter(torch.log(u / (1 - u)))
        self.log_T = nn.Parameter(torch.log(per_zone(T_init)))
        # In the PDE residual, Q is averaged over the same running-mean window as the recharge, so both
        # sources are compared at the same time resolution (see recharge.py). 1 = off.
        self.q_smooth_days = int(q_cfg.get("smooth_days", 1))
        self.q_smooth_points = int(q_cfg.get("smooth_points", 11))

    @property
    def S(self) -> Tensor:
        if self.S_log_range is None:
            return torch.exp(self.log_S)
        lo, hi = self.S_log_range
        return torch.exp(lo + (hi - lo) * torch.sigmoid(self.log_S))

    @property
    def T(self) -> Tensor:
        return torch.exp(self.log_T)

    def zone_weights(self, xyt_n: Tensor) -> Tensor:
        """(N, n_zones) zone weights at the points (shared zoned model only)."""
        x_m, y_m, _ = self.norm.to_phys(xyt_n.detach())
        return torch.cat([f(x_m, y_m) for f in self.zone_w], dim=-1)

    def S_T_at(self, xyt_n: Tensor) -> tuple[Tensor, Tensor]:
        """S and T at the points: the scalars, or (N, 1) zone blends for a shared zoned model."""
        if self.zone_w is None:
            return self.S, self.T
        w = self.zone_weights(xyt_n)
        return w @ self.S.unsqueeze(-1), w @ self.T.unsqueeze(-1)

    def head_norm(self, xyt_n: Tensor) -> Tensor:
        return self.h_net(xyt_n)

    def head(self, xyt_n: Tensor) -> Tensor:
        """Head in metres."""
        return self.norm.h.inverse(self.h_net(xyt_n)) + self.base_at(xyt_n)

    def base_at(self, xyt_n: Tensor) -> Tensor | float:
        """Terrain base [m] at the points (0 without one)."""
        if self.base is None:
            return 0.0
        x_m, y_m, _ = self.norm.to_phys(xyt_n)
        return self.base(x_m, y_m)

    def pumping(self, xyt_n: Tensor) -> Tensor:
        """Q [m/day] at the points."""
        x_m, y_m, _ = self.norm.to_phys(xyt_n)
        lu = (self.landuse(x_m, y_m) - self.lu_mean) / self.lu_std
        if self.season_features == "month":
            month = calendar_month(self.norm.t.inverse(xyt_n[:, 2]), self.doy_at_t0)
            season = torch.nn.functional.one_hot(month, 12).to(xyt_n.dtype)
        else:
            season = annual_harmonics(xyt_n[:, 2:3], self.norm.t.centre, self.norm.t.half, self.n_harm_q)
        feats = [lu, season]
        if self.q_mode == "full":
            feats.insert(0, xyt_n)
        if self.zone_w is None:
            return self.q_net(torch.cat(feats, dim=-1), t_norm=xyt_n[:, 2:3])
        w = self.zone_weights(xyt_n)
        q = self.q_net(torch.cat(feats + [w], dim=-1))
        if self.q_trend_z is not None:          # each zone's own slow trend
            q = q * torch.exp((w @ self.q_trend_z.unsqueeze(-1)) * xyt_n[:, 2:3])
        return q

    def pumping_window_mean(self, xyt_n: Tensor) -> Tensor:
        """Q averaged over a centred window of `q_smooth_days` (evenly spaced samples), matching the
        running mean applied to recharge. Equals pumping() when smoothing is off."""
        if self.q_smooth_days <= 1:
            return self.pumping(xyt_n)
        half = (self.q_smooth_days - 1) / 2
        offsets = torch.linspace(-half, half, self.q_smooth_points, device=xyt_n.device) / self.norm.t.half
        n = len(xyt_n)
        shifted = xyt_n.detach().repeat(len(offsets), 1)
        shifted[:, 2] += offsets.repeat_interleave(n)
        return self.pumping(shifted).reshape(len(offsets), n, 1).mean(0)

    def residual(self, xyt_n: Tensor, R: Tensor, return_terms: bool = False):
        """Scaled PDE residual at collocation points; R [m/day] shape (N, 1). xyt_n gets requires_grad."""
        xyt_n = xyt_n.requires_grad_(True)
        h_n = self.h_net(xyt_n)
        Q = self.pumping_window_mean(xyt_n)
        S, T = self.S_T_at(xyt_n)
        if self.lap_base is not None:            # S h_t − T(∇²net + ∇²base) − R + Q: the base term joins R
            x_m, y_m, _ = self.norm.to_phys(xyt_n.detach())
            R = R + T * self.lap_base(x_m, y_m)
        out = pde_residual(h_n, xyt_n, S, T, R, Q, self.sc, return_terms=return_terms)
        if return_terms:
            res, terms = out
            return res, {**terms, "Q": Q}
        return out
