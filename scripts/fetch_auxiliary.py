"""Download the automatic (non-manual) datasets for the Punjab–Haryana belt.

    python scripts/fetch_auxiliary.py all          # or: dem | worldcover | canals | boundaries | imd | bhuvan

Sources:
  dem         Copernicus DEM GLO-30 tiles (AWS open data)          -> data/raw/dem/
  worldcover  ESA WorldCover 2021 10 m land cover (fallback for Bhuvan) -> data/raw/worldcover/
  canals      OpenStreetMap waterway=canal via Overpass API         -> data/raw/canals/osm/
  boundaries  GADM 4.1 India level-2 (districts)                    -> data/raw/boundaries/
  imd         IMD 0.25° daily gridded rainfall via imdlib           -> data/raw/imd_rain/
  bhuvan      Bhuvan (NRSC) LULC 50K 2005-06/2011-12/2015-16, PB+HR, via public WMS -> data/raw/bhuvan_lulc/
Existing files are skipped, so the script can be re-run safely.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests

RAW = Path("data/raw")
BBOX = dict(south=28.4, west=74.5, north=32.0, east=78.0)  # belt wells + ~0.4° buffer
IMD_YEARS = (1990, 2024)


def fetch(url: str, dest: Path, retries: int = 3) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    return "404"
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
                return f"ok {dest.stat().st_size / 1e6:.1f} MB"
        except requests.RequestException as e:
            print(f"   retry {attempt + 1}: {e}")
            time.sleep(10)
    return "FAILED"


def dem():
    for lat in range(int(BBOX["south"]), int(BBOX["north"])):
        for lon in range(int(BBOX["west"]), int(BBOX["east"])):
            name = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
            url = f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
            print("dem", name, fetch(url, RAW / "dem" / f"{name}.tif"))


def worldcover():
    for lat in (27, 30):
        for lon in (72, 75):
            name = f"ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map.tif"
            url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}"
            print("worldcover", name, fetch(url, RAW / "worldcover" / name))


def canals():
    dest = RAW / "canals" / "osm" / "canals_overpass.json"
    if dest.exists():
        print("canals exists"); return
    b = BBOX
    query = (f'[out:json][timeout:300];'
             f'(way["waterway"="canal"]({b["south"]},{b["west"]},{b["north"]},{b["east"]}););out geom;')
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=600,
                      headers={"User-Agent": "groundwater-pinn academic project"})
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    print("canals", len(r.json().get("elements", [])), "ways")


def boundaries():
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json.zip"
    print("boundaries", fetch(url, RAW / "boundaries" / "gadm41_IND_2.json.zip"))


def imd():
    import imdlib
    out = RAW / "imd_rain"
    out.mkdir(parents=True, exist_ok=True)
    # imdpune.gov.in drops connections after a few rapid downloads, so pace requests and back off on timeouts.
    for y in range(IMD_YEARS[0], IMD_YEARS[1] + 1):
        f = out / "rain" / f"{y}.grd"
        if f.exists() and f.stat().st_size > 1_000_000:
            continue
        for attempt in range(6):
            try:
                imdlib.get_data("rain", y, y, fn_format="yearwise", file_dir=str(out))
                print("imd", y, "ok", flush=True)
                break
            except Exception as e:
                wait = 60 * (attempt + 1)
                print(f"   imd {y} attempt {attempt + 1} failed ({type(e).__name__}); waiting {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"!! imd {y} gave up", flush=True)
        time.sleep(20)


BHUVAN_WMS = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
BHUVAN_LAYERS = [f"lulc:{st}_LULC50K_{yr}" for st in ("PB", "HR") for yr in ("0506", "1112", "1516")]
BHUVAN_RES_DEG = 0.001  # about 100 m pixels; aggregated to 1 km land-use fractions in M3


def bhuvan():
    """Bhuvan LULC 50K (NRSC) via the public WMS. WFS (vector download) is disabled on Bhuvan, so we
    fetch rendered PNG tiles with antialiasing off. Class codes are recovered in M3 from the exact
    fill colours in each layer's SLD style, which is saved alongside the tiles."""
    out = RAW / "bhuvan_lulc" / "wms_lulc50k"
    out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (academic research; groundwater PINN project)"
    tile_deg = 1.2  # 1200 x 1200 px tiles at 0.001°
    meta = {"service": BHUVAN_WMS, "res_deg": BHUVAN_RES_DEG, "crs": "EPSG:4326", "tiles": {}}
    for layer in BHUVAN_LAYERS:
        lname = layer.split(":")[1]
        sld = out / f"{lname}_style.sld"
        if not sld.exists():
            r = s.get(BHUVAN_WMS, params=dict(service="WMS", version="1.1.1", request="GetStyles", layers=layer), timeout=120)
            r.raise_for_status()
            sld.write_bytes(r.content)
        lat = BBOX["south"]
        while lat < BBOX["north"] - 1e-9:
            lon = BBOX["west"]
            while lon < BBOX["east"] - 1e-9:
                n, e = min(lat + tile_deg, BBOX["north"]), min(lon + tile_deg, BBOX["east"])
                w_px, h_px = round((e - lon) / BHUVAN_RES_DEG), round((n - lat) / BHUVAN_RES_DEG)
                f = out / f"{lname}_{lat:.1f}N_{lon:.1f}E.png"
                meta["tiles"][f.name] = dict(layer=layer, bbox=[lon, lat, e, n], width=w_px, height=h_px)
                if not (f.exists() and f.stat().st_size > 0):
                    params = dict(service="WMS", version="1.1.1", request="GetMap", layers=layer, styles="",
                                  srs="EPSG:4326", bbox=f"{lon},{lat},{e},{n}", width=w_px, height=h_px,
                                  format="image/png", transparent="true", format_options="antialias:none")
                    for attempt in range(4):
                        try:
                            r = s.get(BHUVAN_WMS, params=params, timeout=300)
                            if r.headers.get("content-type", "").startswith("image/png"):
                                f.write_bytes(r.content); break
                            print(f"   {f.name}: not an image ({r.headers.get('content-type')})", flush=True)
                        except requests.RequestException as ex:
                            print(f"   {f.name}: {type(ex).__name__}", flush=True)
                        time.sleep(15 * (attempt + 1))
                    time.sleep(2)  # be polite to Bhuvan
                lon += tile_deg
            lat += tile_deg
        print("bhuvan", layer, "done", flush=True)
    (out / "tiles.json").write_text(json.dumps(meta, indent=2))


def main():
    jobs = dict(dem=dem, worldcover=worldcover, canals=canals, boundaries=boundaries, imd=imd, bhuvan=bhuvan)
    which = sys.argv[1:] or ["all"]
    selected = list(jobs) if "all" in which else which
    for name in selected:
        print(f"=== {name}")
        try:
            jobs[name]()
        except Exception as e:  # keep going; report at the end
            print(f"!! {name} failed: {e}")
    log = RAW / "auxiliary_download_log.json"
    prev = json.loads(log.read_text()) if log.exists() else {}
    prev.update({n: str(date.today()) for n in selected})
    log.write_text(json.dumps(prev, indent=2))


if __name__ == "__main__":
    main()
