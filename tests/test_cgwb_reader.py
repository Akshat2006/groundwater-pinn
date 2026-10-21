import pandas as pd
import pytest

from src.data.cgwb import read_wris, read_wris_csv
from src.utils.config import data_dir

HEADER = ("stationCode,stationName,stationType,latitude,longitude,agencyName,state,district,majorBasin,"
          "tributary,dataAcquisitionMode,stationStatus,tehsil,block,village,datatype_code,description,"
          "data_value,data_time,unit,well_type,well_depth,well_aquifer_type\n")
# Real WRIS rows: values from `tehsil` on sit two columns left of their names.
ROWS = [
    'W1,"Doraha, Pz",Ground Water,30.80,76.02,CGWB,Punjab,LUDHIANA,,,Manual,Active,-,GGZ,MANUAL-Water Level,21.5,2010-05-14 00:00:00,m,,,Tube Well,60.0,Unconfined\n',
    "W1,Doraha Pz,Ground Water,30.80,76.02,CGWB,Punjab,LUDHIANA,,,Manual,Active,-,GGZ,MANUAL-Water Level,18.2,2010-11-10 00:00:00,m,,,Tube Well,60.0,Unconfined\n",
    "W1,Doraha Pz,Ground Water,30.80,76.02,CGWB,Punjab,LUDHIANA,,,Manual,Active,-,GGZ,Manual Water Level - AMSL,-240.1,2010-11-10 00:00:00,m,,,Tube Well,60.0,Unconfined\n",
    "T1,Gadauli,Ground Water,30.38,77.16,CGWB,Punjab,LUDHIANA,,,Telemetric,Active,-,GGZ,GPRS-Water Level,-32.96,2025-01-20 06:00:00,m,,,,,\n",
]


@pytest.fixture
def wris_dir(tmp_path):
    d = tmp_path / "Punjab"
    d.mkdir()
    (d / "Fatehgarh_Sahib.csv").write_text(HEADER + "".join(ROWS), encoding="utf-8")
    return tmp_path


def test_manual_rows_parsed_from_shifted_columns(wris_dir):
    df = read_wris(wris_dir)
    assert list(df["dtw_m"]) == [21.5, 18.2]                 # AMSL and telemetry rows dropped
    r = df.iloc[0]
    assert (r.well_id, r.lat, r.lon, r.well_type, r.well_depth_m, r.aquifer_type) == \
           ("W1", 30.80, 76.02, "Tube Well", 60.0, "Unconfined")
    assert r.station_name == "Doraha, Pz"                     # quoted comma kept in one field
    assert r.district == "Fatehgarh Sahib"
    assert r.time == pd.Timestamp("2010-05-14")


def test_telemetry_kind(wris_dir):
    df = read_wris_csv(wris_dir / "Punjab" / "Fatehgarh_Sahib.csv", kind="telemetry")
    assert list(df["value"]) == [-32.96] and df["well_type"].iloc[0] == ""


@pytest.mark.data
def test_real_wris_files_parse():
    root = data_dir() / "raw" / "cgwb_levels" / "wris"
    if not root.exists():
        pytest.skip("WRIS downloads not present")
    df = read_wris(root)
    assert df["well_id"].nunique() > 1000
    assert df["time"].notna().all()
    assert df["lat"].between(28, 32.5).all() and df["lon"].between(74, 78.5).all()
