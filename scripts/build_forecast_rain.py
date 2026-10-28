"""Rainfall for the forecast model: observed IMD rain to the latest published day, then the average year.

    python scripts/build_forecast_rain.py --end 2027-12-31     -> data/processed/rain_daily_forecast.nc

- 1990-2024: data/processed/rain_daily.nc (IMD final gridded rain, as used by every other run);
- full years after 2024: IMD yearwise files (imdlib.get_data), as long as IMD has published them;
- the current year: IMD real-time daily files (imdlib.get_real_data; preliminary values), day by day up to the
  last day IMD has published;
- after that, up to --end: the 2005-2024 mean rain for that day of the year (an "average rainfall" assumption).
The file's attrs record the last observed day, so the maps can label the months that use average rain.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import data_dir  # noqa: E402


def on_grid(r: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """IMD rain clipped to the reference (lat, lon) grid, missing values (-999) as NaN."""
    r = r.where(r > -900).reindex(lat=ref.lat, lon=ref.lon, method="nearest", tolerance=0.01)
    r = r.astype(np.float32).rename("rain_mm")
    r["time"] = pd.DatetimeIndex(r.time.values).normalize()
    return r.transpose("time", "lat", "lon")


def main() -> None:
    import imdlib

    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2027-12-31")
    args = ap.parse_args()
    end = pd.Timestamp(args.end)
    raw = data_dir() / "raw" / "imd_rain"
    proc = data_dir() / "processed"
    base = xr.open_dataset(proc / "rain_daily.nc").rain_mm.load()
    parts, last = [base], pd.Timestamp(base.time.values[-1])

    # whole years IMD has published since the base file
    yr = last.year + 1
    while yr < pd.Timestamp.today().year:
        try:
            if not (raw / "rain" / f"{yr}.grd").exists():
                imdlib.get_data("rain", yr, yr, fn_format="yearwise", file_dir=str(raw))
            r = on_grid(imdlib.open_data("rain", yr, yr, "yearwise", str(raw)).get_xarray()["rain"], base)
        except Exception as e:                       # not published yet
            print(f"{yr}: not available ({e!r:.80})")
            break
        parts.append(r)
        last = pd.Timestamp(r.time.values[-1])
        print(f"{yr}: yearwise, {r.sizes['time']} days")
        yr += 1

    # the current year, day by day from the real-time archive
    rt = raw / "realtime"
    rt.mkdir(parents=True, exist_ok=True)
    day, got = last + pd.Timedelta(days=1), []
    while day < min(pd.Timestamp.today().normalize(), end):
        f = rt / f"rain_ind0.25_{day:%y_%m_%d}.grd"
        if not f.exists() or f.stat().st_size == 0:
            try:
                imdlib.get_real_data("rain", f"{day:%Y-%m-%d}", f"{day:%Y-%m-%d}", file_dir=str(rt))
            except Exception as e:
                print(f"real-time rain stops at {day:%Y-%m-%d} ({e!r:.80})")
                break
        if not f.exists() or f.stat().st_size == 0:
            print(f"real-time rain stops at {day:%Y-%m-%d} (no file)")
            break
        got.append(day)
        day += pd.Timedelta(days=1)
    if got:
        r = imdlib.open_real_data("rain", f"{got[0]:%Y-%m-%d}", f"{got[-1]:%Y-%m-%d}", str(rt)).get_xarray()["rain"]
        parts.append(on_grid(r, base))
        last = got[-1]
        print(f"real-time: {len(got)} days, {got[0]:%Y-%m-%d} .. {last:%Y-%m-%d}")

    # average year after the last observed day
    clim = base.sel(time=slice("2005-01-01", "2024-12-31")).groupby("time.dayofyear").mean()
    future = pd.date_range(last + pd.Timedelta(days=1), end, freq="D")
    if len(future):
        avg = clim.sel(dayofyear=xr.DataArray(future.dayofyear.values, dims="time")).drop_vars("dayofyear")
        parts.append(avg.assign_coords(time=future).astype(np.float32).rename("rain_mm"))

    rain = xr.concat(parts, dim="time")
    rain = rain.sel(time=~rain.get_index("time").duplicated())
    assert rain.get_index("time").is_monotonic_increasing and rain.get_index("time")[-1] == end
    rain.attrs.update(base.attrs, observed_until=f"{last:%Y-%m-%d}",
                      note=f"observed IMD rain to {last:%Y-%m-%d} (real-time values after the last yearwise file are "
                           f"preliminary); 2005-2024 day-of-year mean after that, to {end:%Y-%m-%d}")
    rain.to_dataset().to_netcdf(proc / "rain_daily_forecast.nc")
    print(f"wrote rain_daily_forecast.nc: {rain.time.values[0]:.10} .. {end:%Y-%m-%d}, observed until {last:%Y-%m-%d}")


if __name__ == "__main__":
    main()
