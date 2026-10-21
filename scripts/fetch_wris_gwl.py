"""Download CGWB groundwater-level data (manual + telemetry) from India-WRIS, one file per district.

Uses the same public "Ground Water Level" dataset download as the India-WRIS website
(and the QGIS plugin "CGWB Groundwater Data Downloader"). Must be run from an Indian network.

Usage:  python scripts/fetch_wris_gwl.py --out data/raw/cgwb_levels/wris
"""
import argparse
import json
import time
import urllib.parse
from datetime import date
from pathlib import Path

import requests

BASE = "https://indiawris.gov.in"
DISTRICTS = {
    "Punjab": ["Ludhiana", "Patiala", "Sangrur", "Fatehgarh Sahib", "Moga", "Barnala", "Jalandhar", "Kapurthala"],
    "Haryana": ["Karnal", "Kaithal", "Panipat", "Kurukshetra", "Yamunanagar", "Ambala", "Sonipat", "Jind"],
}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (academic research download; groundwater PINN project)"
    s.get(f"{BASE}/dataSet/", timeout=60)  # obtain session cookie
    s.headers.update({"Accept": "application/json, text/plain, */*", "Content-Type": "application/json",
                      "Origin": BASE, "Referer": f"{BASE}/dataSet/"})
    return s


def wris_district_names(s: requests.Session, state: str) -> list[str]:
    states = s.post(f"{BASE}/masterState/StateList", json={"datasetcode": "GWATERLVL"}, timeout=60).json()["data"]
    code = next(x.get("statecode") or x.get("stateCode") for x in states
                if (x.get("state") or x.get("stateName") or x.get("statename", "")).lower() == state.lower())
    dists = s.post(f"{BASE}/masterDistrict/getDistrictbyState",
                   json={"statecode": code, "datasetcode": "GWATERLVL"}, timeout=60).json()["data"]
    return [d.get("districtName") or d.get("districtname") or d.get("district") for d in dists]


def download(s, state, district, start, end, path: Path) -> int:
    url = f"{BASE}/Dataset/{urllib.parse.quote('Ground Water Level')}"
    params = dict(stateName=state, districtName=district, agencyName="CGWB", startdate=start, enddate=end,
                  download="true", page="0", size="3000000")
    r = s.post(url, params=params, headers={**s.headers, "accept": "text/csv"}, timeout=600)
    r.raise_for_status()
    path.write_bytes(r.content)
    return r.content.count(b"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/cgwb_levels/wris")
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--pause", type=float, default=5.0, help="seconds between requests (be polite)")
    a = ap.parse_args()
    out = Path(a.out)
    s = make_session()
    log = {"downloaded_on": str(date.today()), "start": a.start, "end": a.end, "files": {}}
    for state, wanted in DISTRICTS.items():
        available = wris_district_names(s, state)
        lookup = {n.lower().replace(" ", ""): n for n in available}
        for d in wanted:
            name = lookup.get(d.lower().replace(" ", ""))
            if name is None:
                print(f"!! {state}/{d}: not in WRIS list -> {sorted(available)}")
                log["files"][f"{state}/{d}"] = "NOT FOUND"
                continue
            path = out / state / f"{d.replace(' ', '_')}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > 1000:
                print(f"== {state}/{d}: exists, skipping"); continue
            n = download(s, state, name, a.start, a.end, path)
            print(f"ok {state}/{d}: {n:,} rows"); log["files"][f"{state}/{d}"] = n
            time.sleep(a.pause)
    (out / "download_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
