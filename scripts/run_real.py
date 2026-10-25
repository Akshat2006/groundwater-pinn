"""M6/M7: train the PINN on real data for one zone, all zones, or the whole region, and predict held-out wells.

    python scripts/run_real.py --regional --seed 0                 # M6: one PINN for the whole area (k = 1)
    python scripts/run_real.py --zone all --seed 0                 # M7: one PINN per zone (zoning from configs)
    python scripts/run_real.py --shared --seed 0                   # E20: one network, S, T and pumping per zone
    python scripts/run_real.py --zone 2 --steps 500 --tag smoke    # quick check
    options: --layout 0  --method kmeans --k 5  --set training.weighting.init=1  --stage-only val

Each run folder (reports/runs/<stamp>_real_<...>/) holds config, history, model, predictions.parquet
(every validation and test reading of the zone: tasks A, B, C), pumping_yearly.csv and metrics.json.
Task A predictions include the well's learned offset (it is a monitored well); B and C do not.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import summary  # noqa: E402
from src.evaluation.tasks import shallow_levels, target_rows, training_rows, with_layout  # noqa: E402
from src.training.real_data import predict_heads, t_days, well_offsets_m, zone_inputs  # noqa: E402
from src.training.trainer import train_zone  # noqa: E402
from src.utils.config import data_dir, deep_merge, load_config  # noqa: E402
from src.utils.runlog import start_run  # noqa: E402


def parse_sets(items):
    import yaml
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        d = out
        *parents, leaf = key.split(".")
        for k in parents:
            d = d.setdefault(k, {})
        d[leaf] = yaml.safe_load(value)
    return out


@torch.no_grad()
def yearly_pumping(model, cells: np.ndarray, static: xr.Dataset, years: range) -> pd.DataFrame:
    """Zone-mean pumping (m/yr) per calendar year, from mid-month samples at every zone cell."""
    dev = next(model.parameters()).device
    rows, cols = np.nonzero(cells)
    xs, ys = static.x.values[cols], static.y.values[rows]
    out = []
    for yr in years:
        vals = []
        for mo in range(1, 13):
            t = t_days([pd.Timestamp(yr, mo, 15)])[0]
            xyt = model.norm.to_norm(torch.as_tensor(xs, dtype=torch.float32, device=dev),
                                     torch.as_tensor(ys, dtype=torch.float32, device=dev),
                                     torch.full((len(xs),), t, dtype=torch.float32, device=dev))
            vals.append(float(model.pumping(xyt).mean()))
        out.append({"year": yr, "pumping_m_per_yr": float(np.mean(vals) * 365.25),
                    "paddy_share": float(np.sum(vals[5:9]) / np.sum(vals))})
    return pd.DataFrame(out)


def run_zone(zone_id, zone_cells, df, static, rain, priors, cfg, args, label, zone_maps=None):
    """zone_maps: the zone map over the region, for a shared zoned model (zone_id None, zone_cells = whole area)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = training_rows(df)
    if zone_id is not None:
        train = train[train.zone_map == zone_id]
    zone = zone_inputs(train, zone_cells, static, rain, priors, cfg, device=device, zone_maps=zone_maps)
    log = start_run(f"real_{label}" + (f"_{args.tag}" if args.tag else ""),
                    {"label": label, "zone": zone_id, "layout": args.layout, "method": args.method, "k": args.k,
                     "shared": zone_maps is not None, "seed": args.seed, "n_readings": int(len(zone.head_m)),
                     "n_wells": int(zone.n_wells), "T_fixed": zone.T_init, "S_init": zone.S_init, **cfg})
    T_txt = ", ".join(f"{v:.0f}" for v in np.atleast_1d(zone.T_init))
    print(f"=== {label}: {zone.n_wells} wells, {len(zone.head_m)} readings, T fixed at {T_txt} m2/day "
          f"-> {log.dir.name}", flush=True)
    res = train_zone(cfg, zone, seed=args.seed, run_dir=log.dir, device=device)
    model = res.model.eval()

    wells = pd.Index(train[train.head_m.notna()].well_id.unique())
    offsets = well_offsets_m(model, wells)
    preds = []
    for stage in ([args.stage_only] if args.stage_only else ["val", "test"]):
        for task in ("A", "B", "C"):
            tgt = target_rows(df, task, stage)
            if zone_id is not None:
                tgt = tgt[tgt.zone_map == zone_id]
            if tgt.empty:
                continue
            p = predict_heads(model, tgt.x_m.values, tgt.y_m.values, tgt.date)
            if task == "A" and len(offsets):
                p = p + tgt.well_id.map(offsets).fillna(0.0).to_numpy()
            preds.append(tgt[["well_id", "date", "head_m", "x_m", "y_m"]].assign(
                task=task, stage=stage, pred=p, zone=tgt.zone_map if zone_maps is not None else (zone_id or 1)))
    preds = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    preds.to_parquet(log.dir / "predictions.parquet")
    pump = yearly_pumping(model, zone_cells, static, range(2005, 2025))
    pump.to_csv(log.dir / "pumping_yearly.csv", index=False)
    if zone_maps is not None:                 # each zone's own pumping estimate
        ids = sorted(int(z) for z in np.unique(zone_maps[zone_cells]) if z > 0)
        pd.concat([yearly_pumping(model, (zone_maps == z) & zone_cells, static, range(2005, 2025)).assign(zone=z)
                   for z in ids]).to_csv(log.dir / "pumping_yearly_zones.csv", index=False)
    if len(offsets):
        offsets.rename("offset_m").to_csv(log.dir / "well_offsets.csv")
    metrics = {"label": label, "zone": zone_id, "seed": args.seed, "S": res.info["S"], "T_fixed": res.info["T"],
               "seconds": res.seconds, "n_wells": int(zone.n_wells), "n_readings": int(len(zone.head_m)),
               "train_rmse_m": float(res.history.data_rmse_m.iloc[-1]),
               "pumping_2005_2018_m_per_yr": float(pump[pump.year <= 2018].pumping_m_per_yr.mean()),
               "offset_sd_m": float(offsets.std()) if len(offsets) else None}
    if zone.canal is not None:                # area-mean canal recharge over the zone, m/yr
        metrics["canal_recharge_m_per_yr"] = float(zone.canal.values[0].cpu().numpy()[zone_cells].mean() * 365.25)
    if zone_maps is not None:
        metrics.update(S_zones=res.info["S_zones"], T_zones=res.info["T_zones"])
    for (stage, task), g in (preds.groupby(["stage", "task"]) if len(preds) else []):
        metrics[f"{stage}_{task}"] = summary(g, "pred")
    log.log_metrics(metrics)
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=1), flush=True)
    return metrics


def main() -> None:
    zcfg = load_config(ROOT / "configs" / "zoning.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default=None, help="zone number, or 'all'")
    ap.add_argument("--regional", action="store_true", help="one PINN for the whole area (k = 1)")
    ap.add_argument("--shared", action="store_true", help="E20: one network for the whole area, S, T, pumping per zone")
    ap.add_argument("--layout", type=int, default=0)
    ap.add_argument("--method", default=zcfg["main"]["method"])
    ap.add_argument("--k", default=zcfg["main"]["k"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, help="override the joint-stage length (quick checks)")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--stage-only", choices=["val", "test"])
    args = ap.parse_args()

    proc = data_dir() / "processed"
    cfg = load_config(ROOT / "configs" / "pinn.yaml")
    cfg = deep_merge(cfg, load_config(ROOT / "configs" / "pinn_real.yaml"))
    cfg = deep_merge(cfg, parse_sets(args.set))
    if args.steps:
        cfg = deep_merge(cfg, {"training": {"stages": {"data_only": min(3000, args.steps), "physics_frozen": min(5000, args.steps),
                                                       "joint": args.steps, "lbfgs": 0}}})
    priors = load_config(ROOT / "configs" / "priors.yaml")
    priors["rif"] = load_config(ROOT / "configs" / "study_area.yaml")["recharge"]["rif"]
    static = xr.open_dataset(proc / "static_1km.nc").load()
    rain = xr.open_dataset(proc / "rain_daily.nc").rain_mm.load()
    zones = xr.open_dataset(proc / "zones.nc").load()
    mask = zones["mask"].values.astype(bool)
    master = pd.read_parquet(proc / "master.parquet")
    splits = pd.read_parquet(proc / "splits.parquet")
    df = with_layout(shallow_levels(master), splits, args.layout)

    if args.regional:
        args.k, args.method = 1, "none"
        run_zone(None, mask, df, static, rain, priors, cfg, args, f"L{args.layout}_regional_s{args.seed}")
        return
    summ = json.loads((proc / "zoning_summary.json").read_text(encoding="utf-8"))
    k = summ["layouts"][str(args.layout)]["methods"][args.method]["chosen_k"] if args.k == "chosen" else int(args.k)
    assign = pd.read_parquet(proc / "zone_assign.parquet")
    a = assign[(assign.method == args.method) & (assign.layout == args.layout) & (assign.k == k)].set_index("well_id")
    df["zone_map"] = df.well_id.map(a.zone_map)
    zmap = zones.zone.sel({"method": args.method, "layout": args.layout, "k": k}).values
    if args.shared:
        run_zone(None, mask, df, static, rain, priors, cfg, args, f"L{args.layout}_{args.method}k{k}_shared_s{args.seed}",
                 zone_maps=zmap)
        return
    ids = sorted(int(z) for z in np.unique(zmap[mask]) if z > 0)
    todo = ids if args.zone in (None, "all") else [int(args.zone)]
    for z in todo:
        run_zone(z, (zmap == z) & mask, df, static, rain, priors, cfg, args,
                 f"L{args.layout}_{args.method}k{k}_z{z}_s{args.seed}")


if __name__ == "__main__":
    main()
