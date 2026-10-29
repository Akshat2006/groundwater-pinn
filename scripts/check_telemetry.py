"""Independent check of the forecast maps against telemetry wells the model never saw.

    python scripts/check_telemetry.py --export exports/forecast-v1  -> reports/telemetry_check.md

The forecast model (forecast-v1) was trained on CGWB manual readings up to 2024-01. The telemetry (DWLR) wells
were never used for training or for any choice. Rules, fixed before looking at the result:
- months 2024-02 .. 2025-11 (after the last reading used; IMD rain is observed in all of them);
- telemetry: monthly mean depth per well, months with >= 10 days of data;
- shallow aquifer only: a well counts if its median depth is at most the 99th percentile of the shallow
  training readings (telemetry has no depth class; deeper sensors tap a different aquifer);
- the model value is the map cell that contains the well (ensemble mean depth and the 90% range);
- baseline ("nearest well, no change"): the last manual reading up to 2024-01 of the nearest shallow CGWB well,
  held constant. This is what you would know at that place without the model;
- reported: typical error (RMSE), share of wells where the map beats the baseline, 90% range coverage, and the
  change from 2024 to 2025 (same months both years) measured vs predicted.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import data_dir  # noqa: E402

START, END = "2024-02", "2025-11"


def rmse(e) -> float:
    return float(np.sqrt(np.nanmean(np.square(e))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="exports/forecast-v1")
    args = ap.parse_args()
    proc = data_dir() / "processed"

    m = pd.read_parquet(proc / "master.parquet")
    sh = m[(m.depth_class == "shallow") & m.usable & m.dtw_m.notna()]
    depth_cap = float(sh[sh.split_time == "train"].dtw_m.quantile(0.99))

    tel = pd.read_parquet(proc / "telemetry_daily.parquet")
    tel = tel[(tel.date >= START) & (tel.date < pd.Timestamp(END) + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1))]
    tel["month"] = tel.date.dt.to_period("M")
    mon = (tel.groupby(["well_id", "month"])
           .agg(obs=("dtw_m", "mean"), days=("dtw_m", "size"), lat=("lat", "first"), lon=("lon", "first"))
           .reset_index())
    mon = mon[mon.days >= 10]
    med = mon.groupby("well_id").obs.median()
    n_all = mon.well_id.nunique()
    mon = mon[mon.well_id.map(med) <= depth_cap].copy()

    # model value at the cell containing each well
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    mon["x"], mon["y"] = to_utm.transform(mon.lon.values, mon.lat.values)
    ds = xr.open_dataset(Path(args.export) / "water_level_monthly.nc")
    t = pd.DatetimeIndex(mon.month.dt.to_timestamp()) + pd.Timedelta(days=14)
    pts = dict(time=xr.DataArray(t, dims="p"), x=xr.DataArray(mon.x.values, dims="p"),
               y=xr.DataArray(mon.y.values, dims="p"))
    mon["model"] = ds.depth_m.sel(**pts, method="nearest").values
    mon["range90"] = ds.range90_m.sel(**pts, method="nearest").values
    mon = mon[mon.model.notna()]                                 # outside the shown map

    # baseline: last reading up to 2024-01 of the nearest shallow CGWB well
    last = (sh[sh.date < "2024-02-01"].sort_values("date").groupby("well_id")
            .agg(x=("x_m", "first"), y=("y_m", "first"), dtw=("dtw_m", "last")))
    _, idx = cKDTree(last[["x", "y"]].values).query(mon[["x", "y"]].values)
    mon["baseline"] = last.dtw.values[idx]

    mon["e_model"] = mon.model - mon.obs
    mon["e_base"] = mon.baseline - mon.obs
    per = mon.groupby("well_id").agg(model=("e_model", rmse), base=("e_base", rmse), n=("obs", "size"))

    # change from 2024 to 2025, same calendar months both years
    mon["yr"], mon["mo"] = mon.month.dt.year, mon.month.dt.month
    both = mon.pivot_table(index=["well_id", "mo"], columns="yr", values=["obs", "model"]).dropna()
    ch = pd.DataFrame({"obs": both[("obs", 2025)] - both[("obs", 2024)],
                       "model": both[("model", 2025)] - both[("model", 2024)]}).groupby("well_id").mean()

    inside = (mon.e_model.abs() <= mon.range90)
    res = {
        "wells_with_data": n_all, "wells_used": int(per.shape[0]), "well_months": int(len(mon)),
        "depth_cap_m": depth_cap,
        "rmse_model": rmse(mon.e_model), "rmse_base": rmse(mon.e_base),
        "mae_model": float(mon.e_model.abs().mean()), "mae_base": float(mon.e_base.abs().mean()),
        "bias_model": float(mon.e_model.mean()),
        "wells_better": float((per.model < per.base).mean()), "coverage": float(inside.mean()),
        "change_wells": int(len(ch)), "change_obs_median": float(ch.obs.median()),
        "change_model_median": float(ch.model.median()),
        "change_mae_model": float((ch.model - ch.obs).abs().mean()), "change_mae_base": float(ch.obs.abs().mean()),
        "change_sign_agree": float((np.sign(ch.model) == np.sign(ch.obs)).mean()),
    }
    for k, v in res.items():
        print(f"{k:22s} {v:.3f}" if isinstance(v, float) else f"{k:22s} {v}")

    md = [
        "# Forecast check on telemetry wells (never used by the model)",
        "",
        f"Maps: `{args.export}` (trained on manual readings up to 2024-01). Telemetry months {START} to {END}, "
        f"shallow wells only (median depth <= {depth_cap:.1f} m).",
        "",
        f"- Wells: {res['wells_used']} of {res['wells_with_data']} telemetry wells with data in the period "
        f"({res['well_months']} well-months).",
        f"- **Typical error (RMSE): map {res['rmse_model']:.2f} m vs nearest-well baseline {res['rmse_base']:.2f} m** "
        f"(average error {res['mae_model']:.2f} vs {res['mae_base']:.2f} m). The map's average bias is "
        f"{res['bias_model']:+.2f} m (+ = map deeper than measured).",
        f"- **The map beats the baseline at {res['wells_better']:.0%} of wells.**",
        f"- **90% range coverage:** the measured depth was inside depth +- range90 in {res['coverage']:.0%} of well-months.",
        f"- **Change 2024 to 2025** ({res['change_wells']} wells, same months both years): measured median "
        f"{res['change_obs_median']:+.2f} m, map {res['change_model_median']:+.2f} m; the map got the direction "
        f"right at {res['change_sign_agree']:.0%} of wells; average error of the change {res['change_mae_model']:.2f} m "
        f"vs {res['change_mae_base']:.2f} m for \"no change\".",
        "",
        "Plain words: RMSE is the typical size of the error in metres. The baseline is what you would guess at that "
        "spot without the model: the last reading of the closest CGWB well. Coverage should be near 90%.",
        "",
        "Caveats: telemetry sensors are a different network (some may sit in a deeper layer even after the depth "
        "filter), readings are preliminary, and one cell of the map is 1 km wide.",
    ]
    (ROOT / "reports" / "telemetry_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote reports/telemetry_check.md")


if __name__ == "__main__":
    main()
