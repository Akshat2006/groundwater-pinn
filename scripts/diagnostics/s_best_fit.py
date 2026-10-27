"""Why is S at its lower limit? For a trained run, the S that best balances the equation given its own head field:
S* = sum(h_t * (lateral + R - Q)) / sum(h_t^2), split into long-term (yearly means) and seasonal parts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(r"C:/Users/HP/pinn")
sys.path.insert(0, str(ROOT))
from src.evaluation.tasks import shallow_levels, training_rows, with_layout  # noqa: E402
from src.training.real_data import load_real_run, t_days, zone_inputs  # noqa: E402
from src.utils.config import load_config  # noqa: E402

run = sorted((ROOT / "reports/runs").glob(sys.argv[1]))[-1]
proc = ROOT / "data/processed"
with xr.open_dataset(proc / "static_1km.nc") as f:
    static = f.load()
with xr.open_dataset(proc / "rain_daily.nc") as f:
    rain = f.rain_mm.load()
with xr.open_dataset(proc / "zones.nc") as f:
    zones = f.load()
model, cfg = load_real_run(run, static)
priors = load_config(ROOT / "configs/priors.yaml")
priors["rif"] = load_config(ROOT / "configs/study_area.yaml")["recharge"]["rif"]
mask = zones["mask"].values.astype(bool)
df = with_layout(shallow_levels(pd.read_parquet(proc / "master.parquet")), pd.read_parquet(proc / "splits.parquet"), cfg["layout"])
zmap = zones.zone.sel({"method": cfg["method"], "layout": cfg["layout"], "k": cfg["k"]}).values
zone = zone_inputs(training_rows(df), mask, static, rain, priors, cfg, zone_maps=zmap)

rng = np.random.default_rng(0)
rows, cols = np.nonzero(mask)
pick = rng.choice(len(rows), 1500, replace=False)
xs, ys = static.x.values[cols[pick]], static.y.values[rows[pick]]
dates = pd.date_range("2005-01-15", "2018-12-15", freq="MS") + pd.Timedelta(days=14)
X = np.tile(xs, len(dates)); Y = np.tile(ys, len(dates)); Tt = np.repeat(t_days(dates), len(xs))
xyt = model.norm.to_norm(*(torch.as_tensor(v, dtype=torch.float32) for v in (X, Y, Tt)))
x_m, y_m, t_d = model.norm.to_phys(xyt)
R = zone.recharge(x_m, y_m, t_d) + (zone.canal(x_m, y_m) if zone.canal is not None else 0)
res, terms = model.residual(xyt.clone(), R.detach(), return_terms=True)
S_pt = model.S_T_at(xyt)[0].detach().numpy().ravel()
ht = terms["h_t"].detach().numpy().ravel() * 365.25                      # m/yr
lat = terms["lateral"].detach().numpy().ravel() * 365.25                 # includes T*lap(net) only
lat_base = (model.S_T_at(xyt)[1] * model.lap_base(x_m, y_m)).detach().numpy().ravel() * 365.25
Rv = R.detach().numpy().ravel() * 365.25
Q = terms["Q"].detach().numpy().ravel() * 365.25
src = lat + lat_base + Rv - Q                                             # what S*h_t should equal
d = pd.DataFrame({"cell": np.tile(np.arange(len(xs)), len(dates)), "year": np.repeat(dates.year, len(xs)),
                  "ht": ht, "src": src, "R": Rv, "Q": Q, "lat": lat + lat_base})
yr = d.groupby(["cell", "year"])[["ht", "src"]].transform("mean")
seas = d[["ht", "src"]] - yr


def s_star(h, s):
    return float((h * s).sum() / (h * h).sum())


print(run.name, "S used (mean):", round(float(S_pt.mean()), 3))
print("RMS m/yr: h_t %.2f | R %.3f  Q %.3f  lateral %.3f | mean h_t %.3f" % (
    np.sqrt((ht**2).mean()), Rv.mean(), Q.mean(), np.sqrt((d.lat**2).mean()), ht.mean()))
print("RMS seasonal h_t %.2f m/yr, seasonal source %.3f m/yr; yearly-mean h_t %.2f, source %.3f" % (
    np.sqrt((seas.ht**2).mean()), np.sqrt((seas.src**2).mean()), np.sqrt((yr.ht**2).mean()), np.sqrt((yr.src**2).mean())))
print("best S overall %.4f | from seasonal part %.4f | from yearly means %.4f" % (
    s_star(d.ht, d.src), s_star(seas.ht, seas.src), s_star(yr.ht, yr.src)))
amp = d.assign(m=np.repeat(dates.month, len(xs))).groupby("m")[["ht", "R", "Q"]].mean()
print("monthly means (m/yr):\n", amp.round(2).T.to_string())
