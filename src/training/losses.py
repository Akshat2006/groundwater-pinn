"""Loss terms. Each returns a scalar tensor.

data:     mean squared error of normalised heads at observations
censored: one-sided penalty for dry readings. The predicted head must not be above the well bottom.
pde:      mean squared scaled PDE residual at collocation points
prior:    gentle pull of log S, log T toward report values. σ is in log units, so σ = 0.5 lets a
          parameter move about ×1.6 before the penalty reaches 1.
"""
from __future__ import annotations

import torch
from torch import Tensor


def data_loss(h_pred_n: Tensor, h_obs_n: Tensor) -> Tensor:
    return torch.mean((h_pred_n - h_obs_n) ** 2)


def censored_loss(h_pred_n: Tensor, bottom_n: Tensor) -> Tensor:
    if h_pred_n.numel() == 0:
        return h_pred_n.sum()
    return torch.mean(torch.relu(h_pred_n - bottom_n) ** 2)


def pde_loss(residual: Tensor) -> Tensor:
    return torch.mean(residual**2)


def prior_loss(log_S: Tensor, log_T: Tensor, S_prior: float, T_prior: float, sigma_log: float) -> Tensor:
    s, t = torch.log(torch.tensor(S_prior)), torch.log(torch.tensor(T_prior))
    return (((log_S - s) ** 2).sum() + ((log_T - t) ** 2).sum()) / sigma_log**2
