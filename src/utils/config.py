"""Config loading, project paths and seeding.

Configs are YAML files in configs/. A config may list `defaults: [other.yaml, ...]`; those are loaded
first (paths relative to the including file) and the including file's values override them.

On Colab the data and the run outputs live on Google Drive, so both roots can be redirected with
environment variables:
    PINN_DATA_DIR  -> replaces <repo>/data
    PINN_RUNS_DIR  -> replaces <repo>/reports/runs
"""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("PINN_DATA_DIR", REPO_ROOT / "data"))


def runs_dir() -> Path:
    return Path(os.environ.get("PINN_RUNS_DIR", REPO_ROOT / "reports" / "runs"))


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: `override` merged into `base`, recursing into nested dicts."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a YAML config, resolving `defaults:` includes, then apply `overrides`."""
    path = Path(path)
    raw = load_yaml(path)
    cfg: dict[str, Any] = {}
    for inc in raw.pop("defaults", []) or []:
        cfg = deep_merge(cfg, load_config(path.parent / inc))
    cfg = deep_merge(cfg, raw)
    return deep_merge(cfg, overrides or {})


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch. `deterministic` also forces deterministic cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
