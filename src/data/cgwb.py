"""Readers for CGWB groundwater-level files -> tidy long tables (one row per reading).

India-WRIS CSVs (scripts/fetch_wris_gwl.py) have a 23-name header, but from `tehsil` onwards the
values are shifted two columns left of their names. Like the QGIS "CGWB Groundwater Data Downloader"
plugin, we take the front fields by position and the rest counted from the end of the row.

M0 only needs these readers for the G0 data check; M3 adds QC, head conversion and the master table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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


# --- M3: one table of manual readings + telemetry ------------------------------------------------

_NEST = r"(?:^|[\s(\-_])(S|M|D|[Ss]hallow|[Mm]edium|[Dd]eep)\)?(?:[_\-]?\d+)?\s*$"
_NEST_NAMES = {"S": "S", "M": "M", "D": "D", "shallow": "S", "medium": "M", "deep": "D"}


def nest_level(names: pd.Series) -> pd.Series:
    """Nested-piezometer suffix in the station name: S (shallow), M (medium), D (deep) or <NA>."""
    s = names.astype(str).str.extract(_NEST, expand=False)
    return s.map(lambda v: _NEST_NAMES.get(v, _NEST_NAMES.get(str(v).lower())) if isinstance(v, str) else None)


def depth_class(aquifer_type: pd.Series, nest: pd.Series, well_depth_m: pd.Series, deep_depth_m: float) -> pd.Series:
    """"deep" if the well taps a confined/semi-confined or deep aquifer, else "shallow".

    The WRIS aquifer field is blank for many medium/deep nested piezometers (median depth ~200-300 m),
    so the name suffix and the well depth are used as well.
    """
    deep = (aquifer_type.isin(["Confined", "Semi-Confined"]) | nest.isin(["M", "D"])
            | (well_depth_m >= deep_depth_m))
    return pd.Series(np.where(deep, "deep", "shallow"), index=aquifer_type.index)


def assemble_manual(wris_root: str | Path, idp_path: str | Path | None = None,
                    idp_fill: dict[str, list[str]] | None = None, match_km: float = 0.2,
                    same_reading_days: int = 20) -> pd.DataFrame:
    """WRIS manual readings + India Data Portal readings for the `idp_fill` districts.

    An IDP well is the same well as a WRIS well if it lies within `match_km` and has the same nest
    suffix; it then takes the WRIS id and metadata. IDP readings within `same_reading_days` of a WRIS
    reading of the same well are dropped (WRIS wins).
    """
    w = read_wris(wris_root, "manual")
    w["date"] = w["time"].dt.normalize()
    cols = ["well_id", "station_name", "lat", "lon", "state", "district", "date", "dtw_m",
            "well_type", "well_depth_m", "aquifer_type", "source"]
    w = w[cols].copy()
    for c in ("well_type", "aquifer_type"):
        w[c] = w[c].replace("", pd.NA)
    if not (idp_path and idp_fill):
        return w.reset_index(drop=True)

    i = read_idp(idp_path, idp_fill)
    i["date"] = i["time"].dt.normalize()
    meta = w.groupby("well_id").agg(lat=("lat", "first"), lon=("lon", "first"), name=("station_name", "first"),
                                    well_type=("well_type", "first"), well_depth_m=("well_depth_m", "first"),
                                    aquifer_type=("aquifer_type", "first"))
    meta["nest"] = nest_level(meta["name"])
    iw = i.groupby("well_id").agg(lat=("lat", "first"), lon=("lon", "first"), name=("station_name", "first"))
    iw["nest"] = nest_level(iw["name"])
    from scipy.spatial import cKDTree
    k = np.cos(np.radians(float(meta.lat.mean())))
    tree = cKDTree(np.c_[meta.lat, meta.lon * k])
    pairs = tree.query_ball_point(np.c_[iw.lat, iw.lon * k], r=match_km / 111.0)
    mapping = {}
    for idp_id, cand, nest in zip(iw.index, pairs, iw.nest):
        same = [meta.index[j] for j in cand if meta.nest.iloc[j] == nest or (pd.isna(meta.nest.iloc[j]) and pd.isna(nest))]
        if len(same) == 1:
            mapping[idp_id] = same[0]
    i["idp_id"] = i["well_id"]
    i["well_id"] = i["well_id"].map(lambda v: mapping.get(v, v))
    for c in ("well_type", "well_depth_m", "aquifer_type"):
        i[c] = i["well_id"].map(meta[c]) if c in meta else pd.NA
    i["well_type"] = i["well_type"].astype("object")
    # drop IDP readings that duplicate a WRIS reading of the same well
    wd = w[["well_id", "date"]].rename(columns={"date": "wdate"})
    j = i.reset_index().merge(wd, on="well_id", how="left")
    near = (j["date"] - j["wdate"]).abs() <= pd.Timedelta(days=same_reading_days)
    dup_rows = set(j.loc[near, "index"])
    i = i.drop(index=list(dup_rows))
    out = pd.concat([w, i[cols]], ignore_index=True)
    out.attrs["idp_matched_wells"] = len(mapping)
    out.attrs["idp_dropped_duplicates"] = len(dup_rows)
    return out


def telemetry_daily(wris_root: str | Path, max_dtw_m: float = 150.0, spike_m: float = 5.0) -> pd.DataFrame:
    """6-hourly DWLR telemetry -> daily median depth to water, after basic cleaning.

    Telemetry values are negative below ground, so dtw = -value. Readings outside (0, max_dtw_m] are
    dropped, then any day more than `spike_m` from the well's 31-day rolling median.
    """
    t = read_wris(wris_root, "telemetry")
    t = t.assign(dtw_m=-t["value"], date=t["time"].dt.normalize())
    n_raw = len(t)
    t = t[(t.dtw_m > 0) & (t.dtw_m <= max_dtw_m)]
    d = (t.groupby(["well_id", "date"]).agg(dtw_m=("dtw_m", "median"), n_obs=("dtw_m", "size"),
                                             lat=("lat", "first"), lon=("lon", "first"), district=("district", "first"))
         .reset_index())
    roll = d.groupby("well_id")["dtw_m"].transform(lambda s: s.rolling(31, center=True, min_periods=5).median())
    d = d[~((d.dtw_m - roll).abs() > spike_m)]
    d.attrs["n_raw"] = n_raw
    return d.reset_index(drop=True)
