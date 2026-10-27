"""District paddy (rice) share for the pumping network: data/reference/paddy_share_district.csv.

    python scripts/build_paddy.py

Rice is the main groundwater user in this belt (flooded fields June-September). Source: district-wise crop
area statistics (Directorate of Economics & Statistics, via India Data Portal: data/raw/crop_stats/idp_crop_wise_apy.csv).
Share = kharif rice area / district area on the 1 km grid, median over the training years (agricultural years
2005-06 to 2018-19). The median absorbs boundary changes (Barnala split from Sangrur in 2006) and one bad value
(Ludhiana 2004-05). One static value per district: pumping changes over time are left to the per-zone trend.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import data_dir  # noqa: E402

YEARS = [f"{y}-{y + 1}" for y in range(2005, 2019)]


def main() -> None:
    with xr.open_dataset(data_dir() / "processed" / "static_1km.nc") as f:
        static = f.load()
    names = dict(item.split("=") for item in static.attrs["district_names"].split(";"))
    ids, cells = np.unique(static.district_id.values, return_counts=True)
    res = float(static.attrs.get("res_m", 1000.0))
    area_ha = {names[str(i)]: n * res**2 / 1e4 for i, n in zip(ids, cells) if i > 0}

    crops = pd.read_csv(data_dir() / "raw" / "crop_stats" / "idp_crop_wise_apy.csv")
    rice = crops[crops.state_name.isin(["Punjab", "Haryana"]) & (crops.crop_name == "Rice") & crops.year.isin(YEARS)]
    rice = rice.groupby(["district_name", "year"]).area.sum()
    rows = []
    for i, name in sorted(names.items(), key=lambda kv: int(kv[0])):
        yearly = rice.loc[name] if name in rice.index.get_level_values(0) else pd.Series(dtype=float)
        rows.append({"district_id": int(i), "district": name, "rice_area_ha_median": float(yearly.median()),
                     "n_years": int(yearly.notna().sum()), "district_area_ha": area_ha[name],
                     "paddy_share": float(yearly.median() / area_ha[name])})
    out = pd.DataFrame(rows)
    path = ROOT / "data" / "reference" / "paddy_share_district.csv"
    out.round(4).to_csv(path, index=False)
    print(out.round(3).to_string(index=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
