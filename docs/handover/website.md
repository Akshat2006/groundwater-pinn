# Groundwater map website: guide for the website teammate

## What you're building
An interactive map website that shows our model's **groundwater depth maps** for the Punjab–Haryana study area (16 districts), month by month from 2005 to the end of 2027: model estimates up to January 2024 (the last well readings used) and a **forecast** after that.

The model is already trained. Its results are exported as ordinary map files, so **you don't need any AI, Python or PyTorch code.** You only display files.

## What you get
One zip from the project lead: `map_files_forecast-v1.zip` (also attached to the project's GitHub Release `forecast-v1`). It replaces the earlier `map_files_model-v1.0.zip`: same files and names, just more months (to 2027-12). If you already built with the old zip, swap the `data/` folder; no code change.

| File | What it is |
|---|---|
| `geotiff/depth_YYYY-MM.tif` | Depth to water below ground [m] for that month (276 files, 2005-01 … 2027-12). 1 km cells. Blank (NaN) outside the shown area. CRS EPSG:32643 (UTM 43N) |
| `geotiff/range90_YYYY-MM.tif` | **Uncertainty [m]** for that month: the likely range. Show it as "depth ± this" (e.g. "24 m ± 12 m"). It is calibrated: on wells the model never saw, the true depth fell inside this range about 84% of the time in years with readings and about 70% for years ahead; on 160 independent telemetry wells in 2024–2025 it held 87% of the time. Label it "likely range (90%)" |
| `geotiff/spread_YYYY-MM.tif` | Optional extra: how much the model's 5 versions disagree (about 1 m). Not needed for the site; if you show it, label it "model disagreement (relative)", not "error" |
| `study_area.geojson` | Outline of the study area (WGS84) |
| `districts.geojson` | The 16 districts, with names |
| `zones.geojson` | The model's 4 zones (areas that behave alike) |
| `wells.geojson` | Monitoring wells (name, district, type) |
| `not_shown.geojson` | Areas left blank on purpose (hilly foothills, or places where the model is unreliable) |
| `metadata.json` | Dates, units, grid, **accuracy numbers**, the **disclaimer texts** to show on the page, and `time` → `kind`: for every month, `"estimate"` or `"forecast"` |
| `water_level_monthly.nc` | All of the above in one NetCDF file (only if you prefer Python; not needed for the site) |
| `preview.png` | What two of the maps should look like |

## Features
**Must have**
1. Map of the whole study area by default (OpenStreetMap background + depth layer + district outlines).
2. **Month/year slider** (and a play button to animate 2005 → 2027). Mark the forecast part of the slider (after 2024-01) clearly, e.g. a different colour, and show an **Estimate** / **Forecast** badge next to the month, taken from `metadata.json` → `time` → `kind`.
3. **Colour legend** in metres ("depth to water below ground"; deeper = darker).
4. **Hover/click** a spot → shows depth in metres with its likely range (from `range90`), district, and month.
5. **Draw an area** (rectangle or circle) → zoom to it and show its average depth, and its change from 2005 to the chosen month.
6. Toggle between **depth** and **uncertainty** (the `range90` layer).
7. Toggle for **wells** (dots) and **zones** (outlines).

**Required labels (examiners will check these)**
- Show the **accuracy line** `metadata.json` → `accuracy` → `text` on the page, word for word (in these files: "Typical error: about 9 m at places without a monitoring well in years with readings, and about 10 m for 1-5 years ahead (measured on the 2020-2024 test of the same model recipe).").
- Show the texts in `metadata.json` → `disclaimers` (e.g. in an "About this map" box). Key points: model estimate, not live data; about 1 km detail; months after 2024-01 are a forecast, using real rainfall up to the date in `time` → `rain_observed_until` and average rainfall after it.
- Wells are visible by default, so it's clear the colours between them are estimated.
- `not_shown` areas are grey with the label "not shown: outside the model's reliable range".
- Call months after 2024-01 a "forecast". Never call the map live or real-time data, and never say pumping was measured.

## Suggested stack (simple, free, no server)
- **Leaflet** (map) + **georaster** and **georaster-layer-for-leaflet** (show the GeoTIFFs and read the value under the mouse; they handle the UTM projection) + **Leaflet.draw** (drawing areas) + **Turf.js** (area statistics).
- Plain HTML/JS, or React if you prefer.
- Host free on **GitHub Pages** or **Netlify**.
- Put the site in **your own GitHub repo** (e.g. `groundwater-map`). Don't push to the main project repo.
- Size: about 130 MB for all months (the `.nc` file isn't needed on the site). Fine for GitHub Pages. Load each month's file only when the slider reaches it.

## Steps
1. Unzip the zip into your site's `data/` folder.
2. Get one month (e.g. `depth_2018-11.tif`) showing on a Leaflet map with a legend.
3. Add the slider, hover values, area drawing, toggles and the About box.
4. Send the project lead the live link.

## Updates
The model will be re-run from time to time with new rainfall and well readings. Each update comes as a new zip with the **same file names** and more months, so keep all data in one `data/` folder that can be swapped, and read the list of months from `metadata.json` → `dates` (don't hard-code 2027).

## Questions
Ask the project lead. Don't change the numbers or the colours' meaning: the map must show exactly what the model produced.
