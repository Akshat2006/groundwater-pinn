"""Predict a stage (val or test) with already-trained real-data PINNs, without retraining.

    python scripts/predict_real.py --stage test                # every run in reports/runs/*_real_*
    python scripts/predict_real.py --stage test --match w10    # only runs whose folder name contains "w10"
    python scripts/predict_real.py --stage test --match "_tbc$"   # --match is a regular expression

Training runs score validation only (run_real.py --stage-only val), so the test wells stay unseen while
settings are chosen. Once the settings are final, this adds the test predictions to each run's
predictions.parquet (replacing any earlier rows of that stage); then run scripts/check_m6.py --stage test.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.tasks import shallow_levels, target_rows, with_layout  # noqa: E402
from src.training.real_data import load_real_run, predict_heads  # noqa: E402
from src.utils.config import data_dir, runs_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["val", "test"])
    ap.add_argument("--match", default="")
    args = ap.parse_args()
    proc = data_dir() / "processed"
    static = xr.open_dataset(proc / "static_1km.nc").load()
    levels = shallow_levels(pd.read_parquet(proc / "master.parquet"))
    splits = pd.read_parquet(proc / "splits.parquet")
    assign = pd.read_parquet(proc / "zone_assign.parquet")

    for d in sorted(runs_dir().glob("*_real_*")):
        if not re.search(args.match, d.name) or not (d / "model.pt").exists():
            continue
        model, cfg = load_real_run(d, static)
        df = with_layout(levels, splits, cfg["layout"])
        zone = cfg["zone"]
        if zone is not None or cfg.get("shared"):
            a = assign[(assign.method == cfg["method"]) & (assign.layout == cfg["layout"]) & (assign.k == cfg["k"])]
            df["zone_map"] = df.well_id.map(a.set_index("well_id").zone_map)
        offsets = (pd.read_csv(d / "well_offsets.csv", index_col=0).offset_m
                   if (d / "well_offsets.csv").exists() else pd.Series(dtype=float))
        offsets.index = offsets.index.astype(str)
        new = []
        for task in ("A", "B", "C"):
            tgt = target_rows(df, task, args.stage)
            if zone is not None:
                tgt = tgt[tgt.zone_map == zone]
            if tgt.empty:
                continue
            p = predict_heads(model, tgt.x_m.values, tgt.y_m.values, tgt.date)
            if task == "A" and len(offsets):
                p = p + tgt.well_id.map(offsets).fillna(0.0).to_numpy()
            new.append(tgt[["well_id", "date", "head_m", "x_m", "y_m"]].assign(
                task=task, stage=args.stage, pred=p, zone=tgt.zone_map if cfg.get("shared") else (zone or 1)))
        old = pd.read_parquet(d / "predictions.parquet") if (d / "predictions.parquet").exists() else pd.DataFrame()
        if len(old):
            old = old[old.stage != args.stage]
        out = pd.concat([old, *new], ignore_index=True)
        out.to_parquet(d / "predictions.parquet")
        print(f"{d.name}: {sum(len(n) for n in new)} {args.stage} predictions")


if __name__ == "__main__":
    main()
