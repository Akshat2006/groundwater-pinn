# Groundwater map website: guide for the website teammate

## What you're building
An interactive map website that shows our model's **groundwater depth maps** for the Punjab–Haryana study area (16 districts), month by month from 2005 to 2024.

The model is already trained. Its results are exported as ordinary map files, so **you don't need any AI, Python or PyTorch code.** You only display files.

## What you get
A zip from the project lead (first a **draft** made from an early model, later the **final** one, `model-v1.0`, with **the same file names**, so your site keeps working when you swap them).

| File | What it is |
|---|---|
| `geotiff/depth_YYYY-MM.tif` | Depth to water below ground [m] for that month (240 files, 2005-01 … 2024-12). 1 km cells. Blank (NaN) outside the shown area. CRS EPSG:32643 (UTM 43N) |
| `geotiff/spread_YYYY-MM.tif` | Uncertainty [m] for that month (how much the model's 5 versions disagree). Zero in the draft (one version only) |
| `study_area.geojson` | Outline of the study area (WGS84) |
| `districts.geojson` | The 16 districts, with names |
| `zones.geojson` | The model's 4 zones (areas that behave alike) |
| `wells.geojson` | Monitoring wells (name, district, type) |
| `not_shown.geojson` | Areas left blank on purpose (hilly foothills, or places where the model is unreliable) |
| `metadata.json` | Dates, units, grid, **accuracy numbers**, and the **disclaimer texts** to show on the page |
| `water_level_monthly.nc` | All of the above in one NetCDF file (only if you prefer Python; not needed for the site) |
| `preview.png` | What two of the maps should look like |

## Features
**Must have**
1. Map of the whole study area by default (OpenStreetMap background + depth layer + district outlines).
2. **Month/year slider** (and a play button to animate 2005 → 2024).
3. **Colour legend** in metres ("depth to water below ground"; deeper = darker).
4. **Hover/click** a spot → shows depth in metres, district, and month.
5. **Draw an area** (rectangle or circle) → zoom to it and show its average depth, and its change from 2005 to the chosen month.
6. Toggle between **depth** and **uncertainty**.
7. Toggle for **wells** (dots) and **zones** (outlines).

**Honesty rules (required, examiners will check these)**
- Show the **accuracy line** from `metadata.json` → `accuracy` on the page, e.g. "Typical error: about 2 m near monitoring wells, about 8 m far from them."
- Show the texts in `metadata.json` → `disclaimers` (e.g. in an "About this map" box). Key points: model estimate, not live data; about 1 km detail; years after 2018 are a hindcast.
- Wells are visible by default, so it's clear the colours between them are estimated.
- `not_shown` areas are grey with the label "not shown: outside the model's reliable range".
- Never call it a live or real-time forecast, and never say pumping was measured.

## Suggested stack (simple, free, no server)
- **Leaflet** (map) + **georaster** and **georaster-layer-for-leaflet** (show the GeoTIFFs and read the value under the mouse; they handle the UTM projection) + **Leaflet.draw** (drawing areas) + **Turf.js** (area statistics).
- Plain HTML/JS, or React if you prefer.
- Host free on **GitHub Pages** or **Netlify**.
- Put the site in **your own GitHub repo** (e.g. `groundwater-map`). Don't push to the main project repo.
- Size: about 50 MB for all months. Fine for GitHub Pages. Load each month's file only when the slider reaches it.

## Steps
1. Unzip the draft into your site's `data/` folder.
2. Get one month (e.g. `depth_2018-11.tif`) showing on a Leaflet map with a legend.
3. Add the slider, hover values, area drawing, toggles and the About box.
4. When the final zip `model-v1.0` arrives: replace the `data/` folder. Nothing else changes.
5. Send the project lead the live link.

## Questions
Ask the project lead. Don't change the numbers or the colours' meaning: the map must show exactly what the model produced.
