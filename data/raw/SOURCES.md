# Raw data sources (do not edit files in `data/raw/`)

All files were downloaded on 2026-10-17 unless noted. Cite every source in the report/paper.

| Folder | What | Source / URL | How | Licence / terms |
|---|---|---|---|---|
| `cgwb_levels/idp/` | CGWB + state board depth to water, 2013–2023 (all India) | India Data Portal: ckandev.indiadataportal.com/dataset/groundwater | manual (user) | ODC-BY |
| `cgwb_levels/wris/<state>/<district>.csv` | CGWB manual (1990–2024) + telemetry (2022–25) levels, well type/depth/aquifer, 16 districts | India-WRIS "Ground Water Level" dataset download (indiawris.gov.in) | `scripts/fetch_wris_gwl.py` (needs Indian network) | Govt of India open data; cite CGWB / India-WRIS |
| `crop_stats/idp_crop_wise_apy.csv` | District crop area/production/yield 1997-98 to 2022-23 (DES APY) | India Data Portal, dataset "area-production-yield-apy" | auto (curl) | ODC-BY; source DES, MoA&FW |
| `crop_stats/idp_cropwise_apy_upag.csv` | Same, UPAG series 2013-14 to 2024-25 | India Data Portal, same dataset | auto (curl) | ODC-BY |
| `cgwb_reports/assessment/*_blockwise.pdf` | Dynamic Ground Water Resources (block-level extraction, recharge, stage): Punjab and Haryana 2020 + 2022 + 2024 | cgwb.gov.in (publication 1542 = Punjab 2024, 1613 = Haryana 2024; old_website GWR-2022 state reports) | auto (curl) | CGWB publication |
| `cgwb_reports/assessment/*_2023_summary.pdf` | 2-page 2023 categorisation summaries | cgwb.gov.in (publications 1237, 1218) | auto | CGWB |
| `cgwb_reports/naquim/` | NAQUIM aquifer mapping reports, all 16 districts (Panipat and Sonipat via the NCR Haryana Vol II report) | cgwb.gov.in/old_website/AQM/NAQUIM_REPORT/... and cgwbpnm uploads | auto (curl) | CGWB |
| `dem/` | Copernicus DEM GLO-30 tiles N28–N31, E074–E077 | copernicus-dem-30m.s3.amazonaws.com | `scripts/fetch_auxiliary.py dem` | Copernicus DEM licence (free; attribution "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA") |
| `worldcover/` | ESA WorldCover 2021 10 m land cover (fallback for Bhuvan) | esa-worldcover.s3.eu-central-1.amazonaws.com | `scripts/fetch_auxiliary.py worldcover` | CC BY 4.0 |
| `canals/osm/` | OpenStreetMap waterway=canal, belt bbox | Overpass API | `scripts/fetch_auxiliary.py canals` | ODbL (© OpenStreetMap contributors) |
| `cgwb_reports/gec2015/` | GEC-2015 methodology report + detailed implementation guidelines | cgwb.gov.in (publication 340) | auto (curl) | CGWB |
| `bhuvan_lulc/wms_lulc50k/` | Bhuvan LULC 50K 2005-06, 2011-12, 2015-16 (PB, HR) as rendered WMS tiles (~100 m) + SLD colour legends | bhuvan-vec2.nrsc.gov.in/bhuvan/wms (public WMS; WFS disabled) | `scripts/fetch_auxiliary.py bhuvan` | © NRSC/ISRO; cite Bhuvan LULC 50K |
| `boundaries/` | GADM 4.1 India level-2 (districts) | geodata.ucdavis.edu/gadm | `scripts/fetch_auxiliary.py boundaries` | GADM: free for academic/non-commercial use |
| `imd_rain/` | IMD 0.25° daily gridded rainfall 1990–2024 | IMD Pune via `imdlib` | `scripts/fetch_auxiliary.py imd` | IMD; cite Pai et al. (2014). All 35 years on disk (1996 and 1998 needed several retries) |

**Still manual:** nothing required. Everything was fetched automatically on 2026-10-17.
