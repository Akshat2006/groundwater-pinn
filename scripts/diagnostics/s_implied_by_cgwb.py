"""What S would CGWB's own water balance imply, given the observed well declines? Per district:
deficit = (extraction - extractable recharge) / area  [CGWB 2020], decline = median well trend 2005-2018 [m/yr].
S_implied = deficit / decline (ignores lateral flow across district borders)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(r"C:/Users/HP/pinn")
sys.path.insert(0, str(ROOT))
from src.evaluation.tasks import shallow_levels  # noqa: E402

with xr.open_dataset(ROOT / "data/processed/static_1km.nc") as f:
    static = f.load()
lv = shallow_levels(pd.read_parquet(ROOT / "data/processed/master.parquet"))
lv = lv[(lv.date.dt.year >= 2005) & (lv.date.dt.year <= 2018) & lv.head_m.notna()]
# trend per well from the same-season (pre-monsoon, Apr-Jun) readings, wells with >= 8 years
pre = lv[lv.date.dt.month.isin([4, 5, 6])]
rows = []
for w, g in pre.groupby("well_id"):
    if g.date.dt.year.nunique() >= 8:
        yr = g.date.dt.year + g.date.dt.dayofyear / 365.25
        rows.append({"well_id": w, "x_m": g.x_m.iloc[0], "y_m": g.y_m.iloc[0], "trend": np.polyfit(yr, g.head_m, 1)[0]})
tw = pd.DataFrame(rows)
xi = np.clip(np.searchsorted(static.x.values, tw.x_m), 0, len(static.x) - 1)
yi = np.argmin(np.abs(static.y.values[None, :] - tw.y_m.values[:, None]), axis=1)
tw["district_id"] = static.district_id.values[yi, xi]
dec = tw[tw.district_id > 0].groupby("district_id").agg(decline=("trend", "median"), wells=("trend", "size"))

cg = pd.read_csv(ROOT / "data/reference/cgwb_district_2020.csv")
cells = pd.Series(static.district_id.values[static.district_id.values > 0]).value_counts()
cg["area_ha"] = cg.district_id.map(cells) * 100.0
cg["E"] = cg.extraction_ham / cg.area_ha
cg["A"] = cg.extractable_ham / cg.area_ha
cg["deficit"] = cg.E - cg.A
cg = cg.join(dec, on="district_id")
cg["S_implied"] = cg.deficit / -cg.decline
print(cg[["district", "state", "E", "A", "deficit", "decline", "wells", "S_implied"]].round(3).to_string(index=False))
tot = cg.deficit.mul(cg.area_ha).sum() / cg.area_ha.sum()
d = (cg.decline * cg.area_ha).sum() / cg.area_ha.sum()
print(f"belt: deficit {tot:.3f} m/yr, decline {d:.3f} m/yr, implied S {tot / -d:.2f}")
