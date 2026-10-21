# G0 data check: well density and continuity (2026-10-17)

**Source:**
- `data/raw/cgwb_levels/idp/cgwb-changes-in-depth-to-water-level.csv` (India Data Portal, CGWB, 550,850 rows, all India).
- The companion file `state-groundwater-boards-...csv` has only about 1,500 Punjab rows and none for Haryana; it is largely unused.

**Columns available:**
- date, state, district, station_name, latitude, longitude, basin, source
- `currentlevel` (depth to water, m below ground)
- `level_diff` (change from the previous reading; we recompute this ourselves)

**Not available:** well type, well depth, dry-well flags, ground reference level.

## Coverage
- Punjab and Haryana data runs **2013–2021**, with a trickle of 2022 rows for Punjab only. The file's overall maximum is March 2023.
- Stations reporting per year: about 470–910 in 2013–2018, then about 290–400 in 2019–2021.
- Readings cluster in Jan, Apr–May, Aug and Oct–Dec (the CGWB quarterly rounds). Exact dates vary from year to year.

## Candidate regions

A **usable** well here means one with at least N years that have both a pre-monsoon reading (Mar–Jun) and a post-monsoon reading (Oct–Dec).

| Region | Wells (any) | Usable, N ≥ 5 (all years) |
|---|---|---|
| Central Punjab (Ludhiana, Patiala, Sangrur, Fatehgarh Sahib, Moga, Barnala) | 300 | 57 |
| Haryana belt (Karnal, Kurukshetra, Kaithal, Panipat) | 307 | 51 |
| Western UP (Meerut, Muzaffarnagar, Baghpat; no Shamli data) | 36 | 7 |
| **Chosen: 16-district Punjab–Haryana belt** | **950** | **205** |

Within the chosen belt, counting **training years 2013–2018 only**:

| N | Usable wells |
|---|---|
| ≥ 3 | 330 |
| **≥ 4** | **242 (Tier-1 zoning wells)** |
| ≥ 5 | 179 |
| ≥ 6 | 98 |

## Data quality notes
- 654 of 10,898 consecutive-reading changes (6%) exceed 5 m. Some are real seasonal swings, but many are single-reading spikes, so M3 needs a spike filter.
- Median depth to water across usable wells is 18.7 m (10th–90th percentile 3.9–35.2 m). Two wells have readings above 60 m, which need checking.
- 15 Punjab and 9 Haryana station names map to more than one coordinate. In M3, key wells by name + coordinates.

## Decision (first version, superseded by the addendum below)
- **Region:** the 16-district Punjab–Haryana belt.
- **Period:** 2013–2021 (train 2013–18, validate 2019, test 2020–21).
- **Wells:** Tier 1 (242) for zoning; Tier 2 (all wells after QC) for training and evaluation.
- **Zone sweep:** k = 1, 3, 5 (8 only if every zone keeps ≥12 wells).
- **More data being sought:** QGIS plugin / India-WRIS for pre-2013 and 2022–24, plus a CGWB request for metadata. See PLAN.md Section 9.

*The counts were produced with throwaway scan scripts. M0 will turn them into `notebooks/00_density_scan.ipynb` so they can be reproduced.*

---

## Addendum (same day): India-WRIS download

`scripts/fetch_wris_gwl.py` downloaded CGWB data for all 16 districts from India-WRIS, 1990–2025. Files are in `data/raw/cgwb_levels/wris/`.

| Measure | IDP file | **WRIS download** |
|---|---|---|
| Years | 2013–2021 | **1990–2024** (manual readings) |
| Manual wells | 950 | **1,105** |
| Wells with ≥10 years of pre- and post-monsoon readings | 0 | **166** (66 have ≥20 years) |
| Tier-1 candidates (≥5 years in 2005–2018) | n/a | **238** (+ Karnal from IDP) |
| Well type / depth / aquifer type | none | type and depth for about 93% of wells; aquifer type for 72% |
| Telemetry (6-hourly) | none | **248 wells**, mostly 2022–2025 |
| Test-period wells | ~295/yr (2020–21) | 296 / 293 / **786** / 376 / 194 (2020–2024) |

**Caveats:**
- **Karnal is sparse in WRIS** (2019–2022 only), so IDP fills 2013–2021 for Karnal.
- **Sign convention:** telemetry values are negative (below ground), manual readings are positive.
- **AMSL readings:** 35 "AMSL" readings use a different reference level and must be dropped.

**Revised decision:**
- **Period:** train 2005–2018, validate 2019, test 2020–2024; telemetry 2022–2025 as a bonus within-season test.
- **Wells:** main analysis uses unconfined/unknown-aquifer wells.

---

## Reproduced in M0 (2026-10-17): `notebooks/00_density_scan.ipynb`

The notebook recomputes everything from the raw WRIS files and saves the tables to `reports/tables/g0_*.csv`. **From now on, the notebook is the reference.**

| Measure | Earlier scan | Notebook |
|---|---|---|
| Tier-1 wells (≥5 usable years in 2005–2018) | 238 | **238** ✓ |
| Wells with ≥10 usable years (1990–2024) | 166 | **166** ✓ |
| Manual wells | 1,105 | **1,139** |

The earlier scan was throwaway code with a slightly different filter. The extra wells are mostly the 130 wells that only start in 2023–24 (new piezometers and unknown-type wells), which matter for the test period only.

**Tier 1 by region:**

| Region | Wells | Tier 1 |
|---|---|---|
| Central Punjab | 385 | 71 |
| Doaba (Jalandhar, Kapurthala) | 158 | 40 |
| Haryana belt | 596 | 127 |

**Two findings that matter later:**

1. **Only 193 Tier-1 wells are in the main analysis.** The other 45 tap confined or semi-confined aquifers and are excluded (PLAN.md §1). By type, Tier 1 is 157 tube wells and 81 dug wells.
   - 193 is still enough for k = 1, 3 and 5.
   - k = 8 is borderline once the test and validation blocks are removed (about 140 training wells, which needs ≥12 per zone). Decide in M4.
2. **The network changes in 2012.**
   - Dug wells (median depth 15 m) hold steady at about 90 per year from 2000 to 2014.
   - Tube wells (median depth 75 m) jump from about 85 (2011) to about 290 (2012), then to 400–570 (2014–2018).
   - So the training period mixes a shallow-dominated network (2005–2011) with a deeper one (2012–2018).
   - Each well keeps its own time series, so this does not bias per-well fits. It could, however, make zones reflect *well type* rather than aquifer behaviour.
   - **M4 check:** cross-tabulate zone against well type. If zones are mostly well-type groups, add well depth as a covariate or zone each type separately (ask first).

**Dry readings:**
- WRIS has no dry flag.
- 435 readings (1.6%) are within 1 m of the recorded well bottom. These are possibly dry or near-dry and will be treated as censored in M3.
- 7 readings are ≤ 0 m and 8 are > 100 m; these go to QC.

**Karnal:**
- WRIS has 58 wells, but only for 2019–2022.
- The IDP file has 104 Karnal wells for 2013–2021 (852 readings). These go into the merge in M3.
