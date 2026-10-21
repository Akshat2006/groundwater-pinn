"""Readers for CGWB groundwater-level files -> tidy long tables (one row per reading).

India-WRIS CSVs (scripts/fetch_wris_gwl.py) have a 23-name header, but from `tehsil` onwards the
values are shifted two columns left of their names. Like the QGIS "CGWB Groundwater Data Downloader"
plugin, we take the front fields by position and the rest counted from the end of the row.

M0 only needs these readers for the G0 data check; M3 adds QC, head conversion and the master table.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MANUAL = "MANUAL-Water Level"
TELEMETRY = "GPRS-Water Level"
AMSL = "Manual Water Level - AMSL"   # different reference level: always dropped

# position in the row -> output column (negative = counted from the end)
_WRIS_FIELDS = {0: "well_id", 1: "station_name", 3: "lat", 4: "lon", 6: "state", 10: "acq_mode",
                -9: "dtype", -8: "value", -7: "time", -6: "unit", -3: "well_type",
                -2: "well_depth_m", -1: "aquifer_type"}
_WRIS_NCOL = 23


def read_wris_csv(path: str | Path, kind: str = "manual") -> pd.DataFrame:
    """Read one WRIS district CSV.

    kind: "manual" (quarterly CGWB readings; `dtw_m` = depth to water, m below ground),
          "telemetry" (6-hourly DWLR; raw signed values kept in `value`, sign handled in M3),
          or "all".
    """
    path = Path(path)
    raw = pd.read_csv(path, header=None, skiprows=1, dtype=str, keep_default_na=False)
    if raw.shape[1] != _WRIS_NCOL:
        raise ValueError(f"{path}: expected {_WRIS_NCOL} columns, got {raw.shape[1]}")
    df = pd.DataFrame({name: raw.iloc[:, pos] for pos, name in _WRIS_FIELDS.items()})
    df = df[df["dtype"] != AMSL]
    if kind == "manual":
        df = df[df["dtype"] == MANUAL]
    elif kind == "telemetry":
        df = df[df["dtype"] == TELEMETRY]
    elif kind != "all":
        raise ValueError(f"kind must be manual/telemetry/all, got {kind!r}")

    df = df.copy()
    for col in ("lat", "lon", "value", "well_depth_m"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["district"] = path.stem.replace("_", " ")        # file name = configured district name
    df["source"] = "wris"
    if kind == "manual":
        df = df.rename(columns={"value": "dtw_m"})
    return df.reset_index(drop=True)


def read_wris(root: str | Path, kind: str = "manual") -> pd.DataFrame:
    """Read every <root>/<state>/<district>.csv."""
    files = sorted(Path(root).glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"no WRIS CSVs under {root}")
    return pd.concat([read_wris_csv(f, kind) for f in files], ignore_index=True)


def read_idp(path: str | Path, districts: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Read the India Data Portal CGWB depth-to-water CSV (no station codes or metadata).

    districts: {state: [district, ...]} filter, matched case-insensitively.
    Wells are keyed by name + rounded coordinates, because some names repeat at different places.
    """
    df = pd.read_csv(path, usecols=["date", "state_name", "district_name", "station_name",
                                    "latitude", "longitude", "currentlevel"])
    df = df.rename(columns={"state_name": "state", "district_name": "district", "latitude": "lat",
                            "longitude": "lon", "currentlevel": "dtw_m", "date": "time"})
    if districts:
        keep = {(s.lower(), d.lower()) for s, ds in districts.items() for d in ds}
        key = list(zip(df["state"].str.lower(), df["district"].str.lower()))
        df = df[[k in keep for k in key]]
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["well_id"] = ("IDP:" + df["station_name"].astype(str) + ":"
                     + df["lat"].round(4).astype(str) + "," + df["lon"].round(4).astype(str))
    df["source"] = "idp"
    return df.reset_index(drop=True)
