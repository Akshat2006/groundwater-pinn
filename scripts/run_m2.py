"""M2 run matrix: single-zone PINN on synthetic aquifers (PLAN.md §6.4).

    python scripts/run_m2.py run1                 # the gate: C, 60 wells, monthly, clean, restricted q_net
    python scripts/run_m2.py run1 run2 run3 ...
    python scripts/run_m2.py run1 --steps 2000    # quick look (overrides the joint-stage length)
    python scripts/run_m2.py run1 --set training.weighting.type=fixed --set training.weighting.init=100 --tag w100

Each run writes reports/runs/<stamp>_m2_<run>/ (config, metrics.json, history.csv, model.pt).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xarray as xr  # noqa: E402

from src.training.trainer import train_zone  # noqa: E402
from src.utils.config import data_dir, deep_merge, load_config  # noqa: E402
from src.utils.runlog import start_run  # noqa: E402
from synthetic.pinn_data import score, zone_from_synthetic  # noqa: E402
from synthetic.sample_wells import sample_wells  # noqa: E402

S_INIT, T_INIT = 0.20, 3000.0          # deliberately wrong starting values (truth: 0.12, 1500)

RUNS = {
    "run1": dict(case="C", n_wells=60, schedule="monthly", note="GATE (M2-A): well-posed, restricted q_net"),
    "run2": dict(case="C", n_wells=60, schedule="monthly", q_mode="full", note="q_net decision: full (brief's) form"),
    "run3": dict(case="A", n_wells=30, schedule="quarterly", note="realistic sampling, clean"),
    "run4": dict(case="A", n_wells=30, schedule="quarterly", noise_sd=0.3, note="0.3 m measurement noise"),
    "run5": dict(case="A", n_wells=30, schedule="quarterly", offset_sd=2.0, note="2 m per-well offset"),
    "run6": dict(case="A", n_wells=30, schedule="quarterly", r_factor=0.8, note="recharge told 20% too low"),
    "run7": dict(case="A", n_wells=30, schedule="quarterly", r_factor=1.2, note="recharge told 20% too high"),
    "run8": dict(case="A", n_wells=15, schedule="quarterly", note="only 15 wells"),
    "run9": dict(case="D", n_wells=30, schedule="quarterly", note="mismatched truth (reported, not gated)"),
    "run10": dict(case="D", n_wells=30, schedule="quarterly", r_factor=0.8, note="mismatched truth, R x0.8"),
}


def parse_sets(items: list[str]) -> dict:
    """['a.b=1', 'c=x'] -> {'a': {'b': 1}, 'c': 'x'} (values parsed as YAML)."""
    import yaml
    out: dict = {}
    for item in items or []:
        key, value = item.split("=", 1)
        d = out
        *parents, leaf = key.split(".")
        for k in parents:
            d = d.setdefault(k, {})
        d[leaf] = yaml.safe_load(value)
    return out


def run(name: str, overrides: dict, seed: int = 0, tag: str = "") -> dict:
    spec = RUNS[name]
    cfg = load_config(ROOT / "configs" / "pinn.yaml")
    if "q_mode" in spec:
        cfg["q_net"]["mode"] = spec["q_mode"]
    cfg = deep_merge(cfg, overrides)
    ds = xr.open_dataset(data_dir() / "synthetic" / f"case_{spec['case']}" / "truth.nc").load()
    wells = sample_wells(ds, spec["n_wells"], spec["schedule"], seed=seed, noise_sd=spec.get("noise_sd", 0.0),
                         offset_sd=spec.get("offset_sd", 0.0))
    r_factor = spec.get("r_factor", 1.0)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zone = zone_from_synthetic(ds, wells, S_INIT, T_INIT, r_factor=r_factor,
                               smooth_days=cfg["training"]["recharge_smooth_days"], device=device)
    log = start_run(f"m2_{name}" + (f"_{tag}" if tag else ""), {"run": name, "tag": tag, "spec": spec, "S_init": S_INIT, "T_init": T_INIT, "seed": seed, **cfg})
    print(f"=== {name}: {spec['note']}  ->  {log.dir.name}", flush=True)
    res = train_zone(cfg, zone, seed=seed, run_dir=log.dir, device=device)
    metrics = {"run": name, "tag": tag, "note": spec["note"], "seconds": res.seconds, **score(res.model, ds, wells, r_factor)}
    log.log_metrics(metrics)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=1), flush=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", choices=list(RUNS))
    ap.add_argument("--steps", type=int, help="override the joint-stage length (quick tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. training.lr=5e-4")
    ap.add_argument("--tag", default="", help="label added to the run folder name")
    args = ap.parse_args()
    overrides = parse_sets(args.set)
    if args.steps:
        overrides = deep_merge(overrides, {"training": {"stages": {"joint": args.steps}}})
    for name in args.runs:
        run(name, overrides, args.seed, args.tag)


if __name__ == "__main__":
    main()
