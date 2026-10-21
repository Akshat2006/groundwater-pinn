"""IMD 0.25° daily gridded rainfall (Pai et al. 2014) via imdlib, clipped to the study area.

The rain stays on its native 0.25° grid: regridding to 1 km would add no information. Values at wells
and collocation points come from bilinear interpolation (xarray .interp / GridField).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def load_rain(root: str | Path, start_year: int, end_year: int,
              bbox: tuple[float, float, float, float], pad_deg: float = 0.5) -> xr.DataArray:
    """Daily rain (mm/day, float32) on (time, lat, lon) for west, south, east, north = bbox (+ pad)."""
    import imdlib
    west, south, east, north = bbox
    years = []
    for yr in range(start_year, end_year + 1):
        ds = imdlib.open_data("rain", yr, yr, "yearwise", str(root)).get_xarray()
        r = ds["rain"].sel(lat=slice(south - pad_deg, north + pad_deg), lon=slice(west - pad_deg, east + pad_deg))
        r = r.where(r > -900)                                     # IMD missing value is -999
        years.append(r.astype(np.float32).load())
    rain = xr.concat(years, dim="time")
    rain["time"] = rain.time.values.astype("datetime64[ns]")
    rain.name = "rain_mm"
    rain.attrs.update(units="mm/day", source="IMD 0.25 deg gridded rainfall (Pai et al. 2014), via imdlib")
    return rain


def rain_at_points(rain: xr.DataArray, lon, lat) -> xr.DataArray:
    """Daily rain series at each point, (time, point), bilinear in space.

    Cells with no data (outside India's land mask) are filled from the nearest valid neighbour first.
    """
    from scipy.ndimage import distance_transform_edt
    empty = rain.isnull().all("time").values
    _, (ii, jj) = distance_transform_edt(empty, return_indices=True)
    filled = rain.copy(data=rain.values[:, ii, jj])
    pts = xr.Dataset({"lon": ("point", np.asarray(lon, dtype=float)), "lat": ("point", np.asarray(lat, dtype=float))})
    return filled.interp(lon=pts.lon, lat=pts.lat)
