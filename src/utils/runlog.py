"""One folder per experiment run: reports/runs/<run_id>/ with the config, environment info and metrics.

    run = start_run("m2_caseC", cfg)
    ...
    run.log_metrics({"S_rel_err": 0.04})
    run.dir / "model.pt"          # checkpoints go here too
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .config import REPO_ROOT, runs_dir


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return env


@dataclass
class RunLog:
    dir: Path

    def log_metrics(self, metrics: dict, name: str = "metrics.json") -> None:
        """Merge `metrics` into <run>/<name> (later calls add or overwrite keys)."""
        path = self.dir / name
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(metrics)
        path.write_text(json.dumps(current, indent=2, default=float))


def start_run(name: str, cfg: dict, root: str | Path | None = None) -> RunLog:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(root) if root else runs_dir()
    run_dir = root / f"{stamp}_{name}"
    suffix = 1
    while run_dir.exists():          # two runs started in the same second
        run_dir = root / f"{stamp}_{name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = {"name": name, "started": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(), **_environment()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return RunLog(run_dir)
