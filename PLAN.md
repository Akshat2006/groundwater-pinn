# PLAN.md: Zoned PINN for Groundwater Forecasting

> **Status:** approved 2026-10-17. **M0 done 2026-10-17** (environment, tests, G0 notebook, literature check). **M1 done 2026-10-17** (FD solver passes all analytical checks; synthetic cases A–D in `data/synthetic/`; `reports/m1_solver.md`). Next: M2.

## Context

CLAUDE.md is the project brief. It covers zoned physics-informed neural networks (PINNs) for groundwater levels in an alluvial Indo-Gangetic aquifer, with unmetered pumping (Q) treated as a hidden variable. The repo (`C:\Users\HP\pinn`) is currently empty. This plan turns the brief's milestones M0–M10 into tasks you can pick up one at a time.

**Your answers:**
- Language: **Python**.
- Study area: **you asked me to choose** (Section 1).
- Timeline: **no timeline wanted**, so the plan focuses on implementation. Effort is given only as a size, and a minimum viable version (MVP) is marked.
- Compute: **own/college GPU + Google Colab**.
- Data: **nothing downloaded yet**.

**Solo project.** You are doing all three jobs: data/GIS, modelling and evaluation. Milestones run one after another, except that manual downloads fill the gaps while models train.

**Solo work order:**
1. **M0.** ✅ Done: G0, region, WRIS 1990–2025 data, conda env `pinn` (GTX 1050 works), 29 tests passing, `reports/lit_check.md`.
2. **M1 → M2** (synthetic). M1 ✅ done (`reports/m1_solver.md`). While these run, do the remaining downloads in small chunks.
3. **M3, minimal first:** CGWB levels + IMD rain + DEM + static land use. Add canals and seasonal land use later.
4. **M5 splits + RF baseline → M4 zoning + zoning preview.** This is the decision checkpoint.
5. **M6 → M7 → M8 (MVP experiments) → M10.**
6. Stretch items only after that.

**Solo time-savers:**
- **WRIS is scripted.** Groundwater levels come from `scripts/fetch_wris_gwl.py` (one command), so no manual clicking is needed.
- **Use ESA WorldCover if Bhuvan is slow.** Use it for static farmland fraction, with the crop calendar for seasonality.
- **Canals:** OSM only. WRIS canal layers are optional.

**If you fall behind, cut in this order:**
1. E20, E21
2. M9 scenarios
3. E13
4. LSTMs (E4, E6)
5. Interface loss (E14)
6. Bhuvan seasonal land use (use WorldCover instead)
7. Canal seepage (state it as a limitation)
8. M2 runs: keep only runs 1, 2, 6 and 9

**Never cut:**
- the synthetic gate (M2 run 1)
- the zoning preview
- E5 vs E7 with 3 layouts
- the Section 13 deliverables

**Your time vs mine:** I can write most of the code. Your time goes on:
- downloads
- checking that results make physical sense
- understanding each module well enough to explain it in a viva
- the write-up

Plan to read and run each milestone's code yourself before moving on.

Effort sizes: **S** = a day or two of focused work, **M** = about a week, **L** = more than a week. These are rough. Data wrangling (M3) is usually underestimated.

**Expectations (honest version).**
- The 5% gain over the regional PINN is a *realistic target, not a guarantee*. The odds are better than even for zoned vs regional PINN, about a toss-up against random forest/kriging, and hard against simple seasonal forecasts.
- An **early zoning preview** in M4 gives a warning signal long before the final review.
- The project is built so that most marks come from deliverables under our control. Section 13 lists them.

---

## 1. Study area and period (decided after the G0 data check, 2026-10-17)

**Study area: the Punjab–Haryana alluvial belt, 16 contiguous districts** (about 28.8–31.6°N, 75.0–77.5°E):
- **Punjab:** Ludhiana, Patiala, Sangrur, Fatehgarh Sahib, Moga, Barnala, Jalandhar, Kapurthala
- **Haryana:** Karnal, Kaithal, Panipat, Kurukshetra, Yamunanagar, Ambala, Sonipat, Jind

Why this area:
- **Central Punjab alone failed G0.** It has only 57 long-record wells, and the Haryana belt alone has 51.
- **The combined belt passes.** It is all thick, flat alluvium, where the single-layer equation is most defensible.
- **Strong signals.** It includes some of India's most over-exploited blocks, and pumping follows the regular paddy/wheat calendar.
- **Good contrast for zoning.** It mixes canal-fed and tubewell-heavy areas, so zones should differ.

**Data sources** (details in `reports/g0_density.md`):

| Source | Years | Wells | Extras |
|---|---|---|---|
| **India-WRIS CGWB download** (main source; `scripts/fetch_wris_gwl.py` → `data/raw/cgwb_levels/wris/`) | **1990–2024** manual readings | **1,139 manual** + **248 telemetry** wells (6-hourly, mostly 2022–2025) | **well type** (733 tube, 261 dug, 36 piezometer, 14 bore, 95 unknown), **well depth**, **aquifer type** (689 unconfined, 69 confined, 39 semi-confined) |
| India Data Portal CSV (`data/raw/cgwb_levels/idp/`) | 2013–2021 | 950 | none. Used to **fill Karnal**, where WRIS only has 2019–2022, and to cross-check WRIS |

Manual wells reporting per year: about 140–215 (1990–2011), 380–735 (2012–2018), 470 (2019), about 295 (2020–21), **786 (2022)**, 376 (2023), 194 (2024).

**Period:**
- Train: **2005–2018.** This matches the start of Bhuvan land use and the original brief. Pre-2005 readings are used only for long-term trend features of long-record wells.
- Validation: **2019.**
- Temporal test: **2020–2024**, a 5-year test with a dense 2022 round.
- **Telemetry wells (2022–2025)** are a bonus test set. They show whether the model gets the *within-season* shape right between quarterly readings, which is a direct check of the physics.

**Physics-consistent well selection:**
- Main analysis uses wells tapping the **unconfined** aquifer, plus wells with unknown aquifer type. This is the layer the single-layer equation describes.
- Confined and semi-confined wells (108) are excluded and reported separately.
- Well type (tube/dug/piezometer) is kept as a column and checked in the error analysis.

**Two well tiers:**
- **Tier 1, the zoning wells:** wells with ≥5 years in 2005–2018 that have both pre-monsoon (Mar–Jun) and post-monsoon (Oct–Dec) readings. WRIS alone gives 238, of which **193 are unconfined/unknown** and so in the main analysis. Adding IDP Karnal gives more. The network shifts from dug to tube wells in 2012; check in M4 that zones are not just well-type groups (`reports/g0_density.md`). Only these are used to compute behaviour features and run k-means. 166 wells have ≥10 such years over 1990–2024, so the trend features are solid.
- **Tier 2, all wells:** about 1,100 or more, after QC and dropping wells with fewer than 3 readings. Every reading goes into the PINN's data loss and into evaluation. Each well's zone comes from the zone map, exactly like an unmonitored place.

**Zone-count sweep:** k = 1, 3 and 5 are required. k = 8 is run if every zone keeps ≥12 Tier-1 training wells, which is likely with about 240 Tier-1 wells.

**Known data quirks for M3:**
- Telemetry values are negative (metres below ground), while manual readings are positive.
- 35 "AMSL" readings use a different reference level; drop them.
- 3 negative and 8 over-100 m manual values need QC.
- Spike filter as in Section 2.7.
- Key wells by WRIS station code. Match IDP wells by name and coordinates.

---

## 2. Check of the brief: problems found and suggested fixes

Every item marked **[needs OK]** changes or sharpens a decision in CLAUDE.md. Approving this plan approves them. Tell me which ones to reject.

### 2.1 Pumping Q is not identifiable as currently specified (most serious)
**Identifiable** means the data can pin down one unique answer.

The equation is S·∂h/∂t − T∇²h = R − Q. The brief's `q_net` takes **(x, y, t, land use, season)**. A network with free x, y, t inputs can draw any field it likes. So for *any* S and T, some Q fits the data exactly: Q = R − S·h_t + T∇²h. The data alone then cannot fix S, T or Q, and the network simply soaks up the leftover error.

Two more trade-offs make this harder:
- **Q versus R errors.** Any error in computed recharge R goes straight into Q. If rainfall infiltration is overestimated by 20%, pumping comes out roughly 20% too high to compensate.
- **S versus R scaling.** S is only identified *through* R. Dividing the equation by S shows the data only "sees" (R − Q)/S and T/S. If R is scaled up, S scales up with it.

**Fixes:**
- **[needs OK] Restricted Q.** Default form: Q = softplus(q_net(land-use intensity, crop-season features, slow trend term)), times a learnable scale per zone. This has **no raw x, y, t inputs**, so Q can vary in space only through land use. Both versions (restricted and the brief's unrestricted q_net) are tested on synthetic data in M2, and **whichever recovers the true Q is adopted**. The evidence decides.
- **Prior on S and T.** Add a gentle "prior" penalty on log S and log T that pulls them toward the CGWB report values. A **prior** is a starting belief that the data can override, but only if it has strong evidence.
- **[needs OK] Q means net pumping.** Report Q as **net abstraction**: pumping minus the irrigation water that seeps back down. Irrigation return flow is not in R, so the model cannot tell the two apart. Say so in the paper.
- **Recharge sensitivity.** Vary the rainfall-infiltration factor by ±25% and report how much S and Q move. This is an honest way to show the trade-off.
- **External sanity check (not ground truth).** Compare inferred Q totals with the **CGWB block-level extraction estimates** from the Dynamic Ground Water Resource Assessment reports.

### 2.2 Transmissivity T will probably be weakly determined in the real data (physics check)
**Diffusivity** is how fast a change in head spreads sideways. With typical alluvial values (T ≈ 1,500 m²/day, S ≈ 0.12), diffusivity is about 12,500 m²/day. A pressure change therefore spreads only about **1 km in one season**, about 2 km in a year and about 7 km in 10 years. CGWB wells are often 5–10 km apart.

So at seasonal scale each location behaves almost like a **bucket**: its level change ≈ (R − Q)·Δt / S. Sideways flow (the T term) is small.

Consequences:
- **T may be recovered poorly.** The plan treats T as weakly constrained, uses the report prior and reports T with wide uncertainty bands.
- **Prediction at unmonitored places** will come mostly from the R and Q(land-use) maps and the zone's S, not from sideways flow. That is still a valid physics-based prediction, but the paper should say so.
- **Interface conditions between zones matter less** than the brief implies, at least at seasonal scale. So they stay a stretch goal.

**[needs OK] M2 acceptance criteria are split:**
- **M2-A, well-posed synthetic case:** dense wells, monthly data, and pumping clusters strong enough to create visible cones. This case must hit the brief's "S, T, Q within ~10%".
- **M2-B, realistic synthetic case:** sparse wells, quarterly data. This case is **reported, not gated**. A poor T here is a finding, not a bug.

### 2.3 Zone assignment and zone domains
- **Zones may be scattered.** k-means on behaviour features can produce zones that are scattered patches rather than regions. **k-means** is an algorithm that sorts wells into k groups of similar ones. A PINN needs a *region* where it enforces physics and samples collocation points, and XPINN interfaces need neighbouring regions.
- **[needs OK] Fix:** keep k-means on behaviour, then:
  1. Train a classifier that predicts zone from **static features only**: location, land use, elevation, canal distance, aquifer type.
  2. Apply it to a 1 km grid and smooth the result with a majority filter. This gives the **zone map**.
  3. **Zone polygons from that map** become each PINN's domain.
  4. Wells are trained in their **map zone**, so training and prediction use exactly the same rule.

  Report how often the map zone agrees with the behaviour zone; this is the "assignment accuracy" in M4. If the map stays badly scattered, fall back to clustering with a contiguity constraint. That would be a method change, so I will ask first.
- **Leakage risk.** Zoning features must come from **training wells and training years only**, and held-out wells get their zone from the map only. **Leakage** means test information sneaking into training, which makes results look better than they really are. Unit tests enforce this.
- **Zone count versus data.** A zone-count sweep up to k = 8 at 10–15 wells per zone needs **≥150 training wells**. G0 found 242 Tier-1 wells, about 170 after the hold-outs. So k = 8 is borderline and is run only if every zone keeps ≥12 wells.

### 2.4 PINN training stability
- **Scaling is the most common PINN bug.** Heads are about 200–250 m with ±3 m seasonal swings, coordinates are tens of km, and time is decades. All inputs are normalised to [-1, 1] and h is standardised. The **PDE residual** (how badly the network breaks the physics equation at a point) is computed in physical units via the chain rule and then divided by a typical flux, so its size is about 1. A unit test catches scaling mistakes.
- **[needs OK] `h_net` takes only (x, y, t).** The brief allows "[+ forcings]". If rainfall is fed to `h_net`, its time derivative becomes ill-defined and noisy. Rainfall acts through R in the PDE instead.
- **Staged training as in the brief:**
  1. Data only.
  2. Add physics, with S and T frozen at their prior values.
  3. Train everything jointly.

  Plus adaptive loss weighting, using the "learning-rate annealing" rule from Wang, Teng & Perdikaris (2021): it rebalances the losses so no single one dominates. Plus Adam (a standard optimiser) and then an optional L-BFGS polish (a second optimiser that is precise near the end of training).
- **Fallback if the PINN will not converge:** a **differentiable finite-difference inverse model**. We reuse the M1 grid solver, written in PyTorch, and fit S, T and the Q parameters by gradient descent through the simulation. It has the same physics and unknowns and is usually far more stable. This is a method change, so I will ask before switching.

### 2.5 Sparse quarterly data
- **The rainfall lag feature breaks down.** At quarterly resolution the lag can only be 0, 3, 6 or 9 months, so almost every well gets the same lag. **Fix:** compute the **monsoon response** instead: post-monsoon minus pre-monsoon rise per 100 mm of monsoon rain, plus the cumulative-rain window (1–6 months) that best explains each level change. **[needs OK]**, small change.
- **Robust features.** Amplitude = mean (Nov − May). Trend = **Theil–Sen slope** of the May series. Theil–Sen is a trend line that is not thrown off by a few bad readings.
- **This is where the physics helps.** Between quarterly readings, the monthly R and the seasonal Q pattern shape the curve. This is a genuine advantage over pure data models and is worth highlighting.
- **Telemetry data.** NWDP and CGWB DWLR telemetry, if any exists in the area, is used only as an extra check. It is a stretch goal.

### 2.6 Evaluation design gaps
- **[needs OK] Three prediction tasks.** These must be defined, because the baselines cannot all do all tasks:
  - **Task A, temporal forecast:** monitored wells, test years.
  - **Task B, spatial prediction:** held-out blocks, training years. Held-out wells have no history, so an LSTM that relies on a well's own past cannot be used.
  - **Task C, both:** held-out blocks in test years. This is the hardest and most realistic task.
- **Repeat the hold-out.** One 15–20% block hold-out is noisy. Repeat with 3–5 different random block layouts if compute allows (stretch). The MVP uses 1.
- **Uncertainty.** Deep ensembles are usually over-confident. **[needs OK] Fix:** add a measurement-noise term to the ensemble spread, then calibrate one scaling factor on the *validation* blocks so that 90% intervals really contain about 90% of values. **Calibrated** means the stated uncertainty matches reality.
- **What forecasting means here.** The PINN forecasts *given* rainfall. Test-period forecasts use observed rainfall, which is a hindcast. Baselines get the same rainfall, so the comparison is fair, but the paper must state it.
- **Better time extrapolation.** Temporal extrapolation with a coordinate network is weak. An optional alternative: run the M1 grid solver forward from the PINN's last state, using the learned S, T and Q. The same code is used in M9.

### 2.7 Data issues
- **Elevation error versus head gradients.** SRTM has several metres of vertical error, while Punjab's head gradient is only about 0.3 m/km. Elevation error would distort ∇h and therefore T.
  - **[needs OK]** Use **Copernicus DEM GLO-30**, which is more accurate and freely downloadable, with SRTM as the fallback.
  - Smooth the elevation to about 1 km and use CGWB reference levels if they exist.
  - A constant per-well elevation error does not affect ∂h/∂t, only T. This is another reason T is uncertain.
- **Dry wells are censored readings.** Use a one-sided penalty: predicted head must be ≤ the head at the well bottom. This needs well depth, which is often missing. Where it is missing, drop the dry readings and count them.
  - **G0 update:** the WRIS download supplies **well depth** (for about 93% of wells) and well type. But no explicit dry flag has been seen yet. M3 checks whether dry readings appear as a code or as gaps. The censored loss is switched on only if dry readings can be identified.
  - A gap after a well's level has been falling may be an unrecorded dry reading. This is noted as a limitation.
  - **Spike QC:** flag any reading that jumps >5 m and returns within the next reading (for example Samrala 2018: 14.4 → 18.95 → 15.7 m). Also flag readings >3 robust standard deviations from the well's own seasonal pattern. Flagged readings are excluded and counted.
- **Bhuvan land use may only be viewable online, not downloadable as raw files.** Fallback: **ESA WorldCover** for static farmland fraction, plus the crop calendar for seasonality.
- **India-WRIS canal layers may not be downloadable.** Fallback: OpenStreetMap canals (`waterway=canal`). Canal seepage then uses GEC-2015 norm values and is held fixed (not learned, because learning it would make Q less identifiable). Test ±50% sensitivity.
- **Unconfined aquifer.** T = K × saturated thickness, and thickness shrinks as levels fall. A constant T per zone is an approximation; state it as a limitation.

### 2.8 Plan review: ranked problems and where each fix lives

| Rank | Problem | Severity | Fixable? | Fix (and where in this plan) |
|---|---|---|---|---|
| 1 | At this scale sideways flow is weak, so the PDE acts like a per-location "bucket" and the physics may add little | Critical | **Limitation**: can be measured and framed, not removed | Bucket-model baseline E18 and T-off ablation E19 (Section 7); frame the claims around zone S, land-use-linked Q and spatial prediction (Section 13) |
| 2 | Q can't be uniquely identified on real data, and the synthetic test used Q in exactly the model's assumed form (an "inverse crime": testing on data built with the model's own assumptions) | Critical | **Limitation**, mitigated | Synthetic Case D with mismatched Q structure and wrong R (Section 6); real-data Q reported only as zone/season totals with recharge-sensitivity ranges and a CGWB cross-check |
| 3 | Data may not support the design: unverified sources, dry dug wells, a switch from dug wells to piezometers that may tap different aquifer layers | High | Fixable if G0 passes | Expanded G0 gate (M0): continuity, well-type mix over time, depth spread, dry fraction; use one consistent well type |
| 4 | Few test wells (about 30–50), so a 5–10% difference could be luck | High | Partly | Repeated block hold-outs (3 layouts) moved into the MVP; bootstrap confidence ranges; primary metric fixed in advance (Section 7) |
| 5 | More than 1,000 PINN trainings, which won't fit the compute | High | Fixable | Compute budget (Section 12): 3 seeds for sweeps and ablations, vmap training of several seeds at once, early stopping |
| 6 | A coordinate PINN extrapolates poorly in time, and test "forecasts" use observed rain (a hindcast) | High | Fixable | Default forecast = FD solver run forward from the PINN's last state; a true forecast with climatological rain, E21 |
| 7 | One network per zone splits the data thinly and leaves borders mismatched | Med–High | Fixable | Variant E20: one network over the whole region with S, T and Q settings per zone. **[needs OK]** to compare it against per-zone PINNs |
| 8 | The MVP was too large | Med–High | Fixable | Leaner MVP (Section 11) |
| 9 | Seed ensembles miss assumption errors (recharge) | Medium | Partly | Ensemble members also vary the recharge factor; calibrate on validation |
| 10 | Elevation error swamps head gradients | Medium | Partly | Data loss on per-well level *changes* (anomalies) as well as absolute heads; Copernicus DEM; T reported as uncertain |
| 11 | Novelty not verified | Medium | Fixable | Literature search in M0 |
| 12 | Censoring needs well depth; pre-monsoon dates spread over Mar–May; irrigation return flow | Low | Fixable | Drop and count dry readings without depth; use actual dates; Q = net abstraction |

---

## 3. Repository layout (repo root = `C:\Users\HP\pinn`)

```
pinn/
├── CLAUDE.md, PLAN.md, README.md
├── environment.yml            # conda env (conda-forge: geopandas, rasterio; PyTorch+CUDA)
├── pyproject.toml             # makes `src` importable (pip install -e .), pytest config
├── configs/
│   ├── study_area.yaml        # districts, CRS (EPSG:32643), period, splits
│   ├── synthetic/case_A.yaml, case_B.yaml, case_C.yaml   # true S,T,R,Q, grid, wells, noise
│   ├── zoning.yaml            # features, k range, min zone size, grid resolution
│   ├── pinn.yaml              # architecture, loss weights, stages, collocation, ensemble
│   ├── baselines.yaml         # kriging variogram, RF/GBM, LSTM settings
│   └── experiments/*.yaml     # one file per experiment in Section 7
├── data/ raw/ interim/ processed/          # raw = manual downloads, never edited
├── src/
│   ├── utils/
│   │   ├── config.py          # load/merge YAML, set all seeds
│   │   ├── scalers.py         # min-max / standard scalers with save/load (JSON)
│   │   ├── geo.py             # lat/lon <-> UTM 43N, grids, polygon point sampling
│   │   └── runlog.py          # create reports/runs/<run_id>/ with config copy + metrics.json
│   ├── data/
│   │   ├── cgwb.py            # read WRIS Excel / IDP CSV -> tidy long table, station metadata
│   │   ├── imd.py             # download/read IMD 0.25° daily rain (imdlib) -> xarray
│   │   ├── landuse.py         # Bhuvan LULC (or WorldCover) -> ag/kharif/rabi fractions per 1 km cell
│   │   ├── dem.py             # Copernicus/SRTM tiles -> mosaic, smoothed elevation at wells/grid
│   │   ├── canals.py          # WRIS or OSM canals -> distance + density rasters
│   │   ├── boundaries.py      # district/block polygons, study-area mask
│   │   └── merge.py           # build master well-time table + gridded forcing cube
│   ├── features/
│   │   ├── head.py            # head = elevation − depth to water; dry/censor flags; QC
│   │   ├── rainfall.py        # rainfall windows, monsoon totals per well/season
│   │   └── behaviour.py       # zoning features: amplitude, Theil–Sen trend, monsoon response
│   ├── zoning/
│   │   ├── cluster.py         # standardise, k-means, elbow+silhouette k selection, merge small zones
│   │   ├── assign.py          # static-feature classifier, smoothed zone map, zone polygons
│   │   └── report.py          # zone map figure, sizes, assignment accuracy
│   ├── physics/
│   │   ├── pde.py             # autograd derivatives + PDE residual in physical units
│   │   ├── recharge.py        # R = RIF·rain + canal seepage; interpolator R(x,y,t)
│   │   ├── fd_solver.py       # finite-difference 2D solver (numpy + torch versions); used by M1, M9, fallback
│   │   └── interface.py       # head/flux matching between neighbouring zones (stretch)
│   ├── models/
│   │   ├── h_net.py           # MLP (x,y,t) -> head
│   │   ├── q_net.py           # small MLP (land use, season, trend) -> Q ≥ 0
│   │   ├── pinn.py            # ZonePINN: h_net + q_net + log_S + log_T + scalers
│   │   └── baselines/
│   │       ├── naive.py       # persistence, seasonal average
│   │       ├── kriging.py     # ordinary kriging per time step (pykrige)
│   │       ├── trees.py       # random forest / LightGBM with location, rain, land use
│   │       └── lstm.py        # regional LSTM (Task A), optional zoned LSTM
│   ├── training/
│   │   ├── losses.py          # data, censored, PDE, prior, interface losses
│   │   ├── weighting.py       # adaptive loss weights (gradient-norm rule)
│   │   ├── collocation.py     # sample/resample points inside zone polygon × time
│   │   ├── trainer.py         # staged training loop, Adam -> L-BFGS, checkpoints
│   │   └── ensemble.py        # run N seeds per zone, aggregate mean/intervals
│   ├── evaluation/
│   │   ├── splits.py          # spatial block splitter + temporal split
│   │   ├── metrics.py         # RMSE, MAE, NSE, fraction improved, Wilcoxon, coverage, plausibility
│   │   ├── calibration.py     # noise term + validation scaling of intervals
│   │   ├── distance.py        # error vs distance to nearest training well
│   │   └── tables.py          # assemble results tables (CSV + LaTeX)
│   └── scenarios/
│       └── whatif.py          # forward runs with modified Q/R using learned S,T (+ ensemble spread)
├── synthetic/
│   ├── generate.py            # build synthetic aquifer from config using src/physics/fd_solver.py
│   ├── sample_wells.py        # sparse wells, quarterly/monthly sampling, noise, offsets, censoring
│   └── analytical.py          # exact solutions used by tests
├── scripts/                   # thin command-line entry points (one per milestone step), small addition to the brief's layout
├── notebooks/                 # exploration only (G0 density scan lives here)
├── tests/                     # see Section 8
└── reports/ figures/ runs/ tables/
```

---

## 4. Key interfaces

### 4.1 Master table: `data/processed/master.parquet` (one row per well × observation)

| Column | Type | Meaning |
|---|---|---|
| well_id | str | CGWB station code |
| district, block | str | admin units |
| lat, lon | float | WGS84 |
| x_m, y_m | float | UTM 43N metres |
| date | datetime | observation date |
| season | category | jan / pre (Mar–May) / aug / post (Nov) |
| dtw_m | float | depth to water below ground (as recorded) |
| is_dry | bool | dry well (censored) |
| well_depth_m | float, nullable | for censoring bound |
| well_type | category | dug / piezometer / tubewell |
| aquifer_type | category, nullable | from reports |
| ground_elev_m | float | smoothed DEM at well |
| head_m | float, nullable | ground_elev_m − dtw_m (NaN if dry) |
| rain_prev_season_mm, rain_monsoon_mm | float | rainfall windows |
| recharge_prev_m | float | R accumulated since previous reading |
| lu_ag_frac, lu_kharif_frac, lu_rabi_frac | float | land-use intensities (1 km cell) |
| canal_dist_km, canal_density | float | canal proximity |
| qc_flag | str | ok / outlier / duplicate / suspect_jump |
| split_space | category | train / val / test (block split) |
| split_time | category | train / val / test |
| zone_behaviour, zone_map | int, nullable | k-means zone (train wells) / map zone (all) |

Companion files:
- `data/processed/forcing.nc` (xarray): monthly R(x, y, t) on a 1 km grid, plus static land use, elevation and canal layers.
- `data/processed/zone_map.tif` and `zones.gpkg`.

### 4.2 Function signatures (Python, PyTorch)

```python
# src/physics/pde.py
@dataclass
class PhysScales:            # converts normalised network units back to physical units
    x_half_m: float; y_half_m: float; t_half_days: float
    h_mean_m: float; h_std_m: float; flux_scale_m_per_day: float

def derivatives(h_norm: Tensor, xyt_norm: Tensor, sc: PhysScales) -> dict[str, Tensor]:
    """Return physical h_t [m/day], h_xx, h_yy [1/m] via autograd (xyt_norm.requires_grad=True)."""

def pde_residual(h_norm: Tensor, xyt_norm: Tensor, S: Tensor, T: Tensor,
                 R: Tensor, Q: Tensor, sc: PhysScales) -> Tensor:
    """(S*h_t - T*(h_xx+h_yy) - R + Q) / sc.flux_scale_m_per_day ; R, Q in m/day."""

# src/physics/recharge.py
def compute_recharge(rain_mm_day: xr.DataArray, rif: float,
                     canal_seepage_m_day: xr.DataArray | None = None) -> xr.DataArray: ...
class RechargeField:                       # fast lookup R(x, y, t) for collocation points
    def __call__(self, xyt_phys: Tensor) -> Tensor: ...

# src/physics/fd_solver.py
def solve_fd(grid: Grid, S: np.ndarray, T: np.ndarray, R: Callable, Q: Callable,
             h0: np.ndarray, bc: BoundarySpec, dt_days: float, n_steps: int,
             return_budget: bool = True) -> FDResult   # heads + per-step water budget

# src/models/h_net.py
class HNet(nn.Module):
    def __init__(self, in_dim=3, width=48, depth=7, act="tanh"): ...
    def forward(self, xyt_norm: Tensor) -> Tensor: ...          # -> h_norm, shape (N,1)

# src/models/q_net.py
class QNet(nn.Module):
    def __init__(self, n_feat: int, width=16, depth=3, q_scale_m_day=1e-3,
                 mode: Literal["restricted", "full"] = "restricted"): ...
    def forward(self, feats: Tensor) -> Tensor: ...             # -> Q >= 0 [m/day] (softplus)

# src/models/pinn.py
class ZonePINN(nn.Module):
    def __init__(self, cfg, scales: PhysScales, S_prior: float, T_prior: float): ...
    S: Tensor  # property, exp(log_S)
    T: Tensor  # property, exp(log_T)
    def head(self, xyt_phys: Tensor) -> Tensor: ...             # metres
    def pumping(self, xyt_phys: Tensor, feats: Tensor) -> Tensor: ...
    def residual(self, xyt_phys: Tensor, feats: Tensor, R: Tensor) -> Tensor: ...

# src/training/trainer.py
def train_zone(cfg: dict, zone: ZoneData, recharge: RechargeField,
               seed: int, run_dir: Path) -> TrainResult: ...    # stages, checkpoints, logs

# src/zoning/cluster.py
def behaviour_features(df_train: pd.DataFrame, rain: xr.DataArray) -> pd.DataFrame: ...
def choose_k(X: np.ndarray, k_range=range(2, 13), seed=0) -> KSelection: ...  # inertia+silhouette
def cluster_wells(X: np.ndarray, k: int, seed=0) -> np.ndarray: ...
def merge_small_zones(labels, xy, min_size=12) -> np.ndarray: ...

# src/zoning/assign.py
def fit_zone_classifier(static_train: pd.DataFrame, labels: np.ndarray) -> Classifier: ...
def build_zone_map(clf, grid_static: xr.Dataset, smooth_cells=3) -> xr.DataArray: ...
def assign_zone(xy: np.ndarray, zone_map: xr.DataArray) -> np.ndarray: ...
def zone_polygons(zone_map: xr.DataArray) -> dict[int, shapely.Polygon]: ...

# src/evaluation/splits.py
def spatial_block_split(wells: gpd.GeoDataFrame, block_km=20.0, test_frac=0.175,
                        val_frac=0.10, seed=0, min_train_per_zone=10) -> pd.Series: ...
def temporal_split(df: pd.DataFrame, train_start="2005-01-01", train_end="2018-12-31",
                   val_end="2019-12-31") -> pd.Series: ...  # test = 2020-2024

# src/evaluation/metrics.py
def rmse(y, yhat) -> float; def mae(y, yhat) -> float; def nse(y, yhat) -> float
def per_well(df, pred_col) -> pd.DataFrame
def fraction_improved(err_a: pd.Series, err_b: pd.Series) -> float
def paired_wilcoxon(err_a, err_b) -> tuple[float, float]       # statistic, p-value
def coverage(y, lo, hi) -> float; def interval_width(lo, hi) -> float
def plausibility_rate(pred: pd.DataFrame, dry_pumped_mask: pd.Series) -> float
def error_vs_distance(df, train_xy, bins_km=(0, 5, 10, 20, 40)) -> pd.DataFrame
```

---

## 5. Milestones

Each milestone lists its tasks, inputs and outputs, done criteria, size and role. **MVP** marks the minimum version to finish if time runs short.

### M0 Setup and G0 data gate (S) **MVP**
**Tasks:**
- Create the repo layout from Section 3, `environment.yml` (Python 3.11, PyTorch with CUDA, geopandas, rasterio, xarray, netCDF4, pyproj, shapely, imdlib, scikit-learn, lightgbm, pykrige, pytest, pyyaml, matplotlib) and `pyproject.toml`.
- Set up a Colab notebook that clones the repo and installs it, with checkpoints saved to Drive.
- Seed/config utility, run logger, `.gitignore` (excludes `data/raw`), `git init`.
- **G0:** `notebooks/00_density_scan.ipynb` reads the India Data Portal CSV and reports, per candidate region:
  - usable wells (Section 1 criterion)
  - **well-type mix per year** (dug vs piezometer vs tubewell)
  - **depth spread**
  - **dry fraction**
  - longest continuous record per well
- **Literature check:** search Google Scholar for "zoned/clustered PINN groundwater" and "PINN unknown pumping / source term inversion groundwater". Record the closest papers in `reports/lit_check.md` and adjust the novelty wording.

**Out:** a working environment, `reports/g0_density.md` (study-area and well-type decision for you to confirm) and `reports/lit_check.md`.

**Done:**
- `pytest` runs (empty suite passes) on both the laptop and Colab.
- `torch.cuda.is_available()` is True on the GPU machine.
- The G0 table exists.
- One consistent well type (or a documented mix) is chosen. *Done: the WRIS metadata allows the main analysis to use unconfined/unknown-aquifer wells, with well type kept for the error analysis (Section 1).*

### M1 Synthetic aquifer generator (M) **MVP**
**Tasks:**
- `src/physics/fd_solver.py`: cell-centred grid with backward-Euler time stepping. This method stays stable for any time step. It uses a sparse solve (scipy, factorised once) and supports fixed-head and no-flow boundaries.
- A water budget computed at every step.
- `synthetic/generate.py`, `sample_wells.py` and `analytical.py` (Section 6).

**Out:** `data/synthetic/case_*/truth.nc` (full h, R and Q fields) and `wells.parquet` (in the same schema as the master table).

**Done:**
- The analytical tests in Section 6.3 pass.
- The water balance closes to < 1e-8 relative error every step.
- Grid refinement shows about 4× smaller error per halving of Δx (second-order convergence, meaning the error shrinks roughly with Δx², as expected for this scheme).

### M2 Single-zone PINN on synthetic data (L) **MVP**
**Tasks:**
- Implement `pde.py`, `h_net.py`, `q_net.py`, `pinn.py`, `losses.py`, `weighting.py`, `collocation.py` and `trainer.py`.
- Run the synthetic matrix from Section 6.4, including both q_net modes.

**Out:** `reports/m2_synthetic.md` with the recovered S, T and Q against the truth, a degradation-with-noise table and identifiability plots (how S and Q shift when R is wrong).

**Done:**
- **M2-A (well-posed):** S and T within 10%, and total pumping volume per season within 10%. Also report the field-level relative Q error.
- **M2-B (realistic):** results are reported with an honest reading.
- **M2-D (mismatched structure, Case D):** reported, not gated. This shows how far wrong the model's assumptions about Q and R can be before its estimates break, and it is the most honest figure for the paper.

If M2-A fails after the fixes in the risk register, **stop and discuss**. Do not move to real data.

### M3 Real data pipeline (L) **MVP** *(downloads can start now, in parallel with M1–M2)*
**Tasks:**
- Loaders in `src/data/*`, head conversion and QC, dry flags, rainfall windows, land-use fractions, elevation, canals, the merged master table and `forcing.nc`.
- Extract S, T and recharge-factor priors by hand from CGWB reports into `configs/priors.yaml`.

**Out:** `master.parquet`, `forcing.nc` and `reports/m3_data_gaps.md` (wells per year and season, dry fraction, missing land-use or canal coverage, and a map).

**Done:**
- The table covers all chosen districts.
- Gaps are documented.
- Schema checks pass.
- A head-versus-time plot for 10 random wells looks sane (no 20 m jumps).

### M4 Zoning (M) **MVP**
**Tasks:**
- Behaviour features on training wells and training years only.
- k selection over 2–12 using the elbow and the silhouette:
  - **Elbow:** the k after which adding zones stops helping much.
  - **Silhouette:** a −1 to 1 score of how well separated the groups are.
- Merge zones with fewer than 12 wells.
- Static-feature classifier, smoothed zone map and polygons.
- Assign the held-out wells.

**Out:** `zone_map.tif`, `zones.gpkg`, zone labels in the master table and `reports/m4_zoning.md`.

**Early zoning preview (decision checkpoint):** this needs the M5 splits and the RF baseline, so run it right after those.
- On **validation blocks only**, compare:
  - an unzoned random forest against a zoned one (one forest per zone, or zone as an input), and
  - an unzoned bucket model against a zoned one (S·Δh = (R − Q)·Δt, with S and the Q scale fitted per zone).
- Output: `reports/zoning_preview.md`, the % change in RMSE.

| Preview result | Meaning | What we do |
|---|---|---|
| Zoned beats unzoned by ≥5% | Good sign for the main claim | Proceed |
| 0–5% | Small gain | Try other k values and the E20 variant; check assignment accuracy |
| No gain | Zoning doesn't help here | Discuss before M7 (reframe or revisit zoning features) |

This costs a few hours and gives the earliest honest signal on the 5% target.

**Done:**
- Every zone has ≥10–15 training wells.
- The map is saved.
- Assignment accuracy (map zone versus behaviour zone, cross-validated) is reported.
- The zoning preview report exists.

### M5 Baselines and splits (M) **MVP**, except the LSTM, which is optional
**Tasks:**
- `splits.py` (blocks of about 20 km, 15–20% test, 10% validation).
- `metrics.py`.
- Baselines: persistence and seasonal average (Task A), ordinary kriging per date (Task B), RF/LightGBM (Tasks A, B, C), regional LSTM (Task A).

**Out:** `reports/tables/baselines.csv` and a plain-language metric explainer in `reports/metrics_explained.md`.

**Done:** a baseline table for every task and baseline combination that is valid.

### M6 Regional (unzoned) PINN on real data (M) **MVP**
**Tasks:**
- One PINN over the whole study area: the same code as M7 with k = 1.
- Tune the stage lengths and loss weights on the validation blocks only.

**Done:**
- Training is stable across 3 seeds: no NaNs, and the PDE loss falls by more than 10×.
- S and T are in a physically plausible range.
- Metrics are produced for Tasks A, B and C.

### M7 Zoned PINN, Q-net and ensembles (L) **MVP** = no interface loss, 5 seeds
**Tasks:**
- Train a ZonePINN per zone.
- Variant **E20** (if approved): one network over the whole region with per-zone `log_S`, `log_T` and Q scale.
- Run an ensemble of 5 seeds (10 if compute allows). The members also vary the recharge infiltration factor (×0.8, ×1.0, ×1.2), so the ranges include recharge uncertainty.
- **Forecasts (Tasks A/C):** run the FD solver forward from the PINN's last training-period state, using each member's learned S, T and Q. Compare this with direct PINN extrapolation.
- Calibrate intervals on validation (`calibration.py`).
- Produce S, T and Q maps.
- Physical plausibility check.
- Compare Q with the CGWB block extraction figures.
- Stretch: an XPINN-style interface loss.

**Done:**
- A full metrics table.
- Coverage of the 90% intervals.
- Plausibility rate.
- S, T and Q maps with ensemble spread in `reports/figures/`.

### M8 Experiments (M–L) **MVP** = k sweep, no-physics, no-Q, error vs distance
**Tasks:** everything in Section 7.

**Done:** results tables and figures are in `reports/`, each with a 3-line plain-language reading.

### M9 Scenarios (M) *stretch*
**Tasks:**
- `whatif.py`: the FD solver with learned S and T per zone, starting from the last PINN state.
- Scenarios: −20% paddy pumping, a dry monsoon (−30% rain), and expanded canal seepage.
- Run each scenario once per ensemble member to get a spread.

**Done:** example what-if curves with uncertainty bands, labelled as **projections**.

### M10 Write-up support (M)
**Tasks:**
- Generate figures and tables for the slides and the IEEE paper: study map, zone map, synthetic recovery, main results table, error vs distance, S, T and Q maps, calibration plot and scenarios.
- Write a limitations list drawn from Section 2.

---

## 6. Synthetic aquifer plan (M1–M2)

### 6.1 Grid and setup
- **Domain:** 60 km × 60 km, Δx = 1 km (60 × 60 cells).
- **Time:** Δt = 1 day, 1 year of spin-up (discarded) followed by 10 years.
- **Boundaries:** west edge at fixed head 220 m (a "river"), east edge at fixed head 238 m (a regional gradient of about 0.3 m/km), and no flow across the north and south edges.
- **Initial state:** a linear plane between the two edges, then spin-up.

### 6.2 True values

| Case | S | T (m²/day) | R | Q | Purpose |
|---|---|---|---|---|---|
| **A** homogeneous | 0.12 | 1,500 | 0.22 × synthetic daily rain (about 650 mm/yr, 75% in Jul–Sep), ±15% N–S gradient → about 0.14 m/yr | Q = q0 · a(x,y) · s(month) · (1+0.02·yr). a = smooth "farmland intensity" field in [0.3, 1]. s = 1.0 Jun–Sep (paddy), 0.5 Nov–Mar (wheat), 0.2 otherwise, normalised to mean 1. q0 sets the domain-mean Q at 0.25 m/yr, giving a decline of roughly 0.8–0.9 m/yr | main recovery test |
| **B** two zones | W: 0.08 / E: 0.16 | W: 800 / E: 2,500 | as A | as A | zoning sanity check (M4/M7 on synthetic) |
| **C** well-posed | 0.12 | 1,500 | as A, plus one canal strip (0.3 m/yr within 2 km) | as A, plus 3 pumping clusters at 3× intensity (visible cones) | makes T identifiable; this is the M2-A gate |
| **D** mismatched (anti inverse crime) | 0.12 | 1,500 | true R includes irrigation return flow and canal seepage the model is *not* told about. The model gets R ×1.0 / ×0.8 / ×1.2 | true Q = A's form + random 5–10 km patches (±30%) + 10 village-scale hot spots. The model's land-use input is a *coarsened, noisy* version of a(x,y) | an honest test: is Q still roughly right when the model's assumptions are wrong? |

The farmland field a(x,y) is also given to `q_net` as its "land-use" input, just as Bhuvan data will be on the real data.

### 6.3 Analytical tests for the solver (M1)
1. **Sine-mode decay.** h = h₀ + A·sin(πx/L)·sin(πy/L)·exp(−2π²(T/S)t/L²), with fixed-head edges and R = Q = 0. Checks accuracy and second-order convergence.
2. **Uniform source in a closed box.** No-flow edges and constant R − Q, so h(t) = h₀ + (R − Q)·t/S exactly. Checks the units and the sign convention.
3. **Water balance every step.** Storage change = (R − Q)·area·Δt + boundary inflow, to < 1e-8 relative error.
4. *(optional)* **Theis drawdown** for a single pumping well, away from the boundaries. Qualitative check only.

### 6.4 Run matrix for M2

**Wells:** 15, 30 and 60, placed by Poisson-disc sampling (random but evenly spread).

**Sampling:** monthly, or quarterly CGWB-style (Jan 15, May 15, Aug 15, Nov 15).

**Noise levels to test:**
- Gaussian measurement noise σ = 0, 0.1, 0.3 and 0.5 m.
- A per-well constant offset (simulating elevation error) with σ = 0 and 2 m.
- 10% missing readings.
- R misspecified ×0.8 and ×1.2.
- *(optional)* 5% censored dry readings.

**MVP runs** (about 8):
1. C-60-monthly-clean, restricted q_net. **This is the gate.**
2. C-60-monthly-clean, full q_net. **This is the q_net decision.**
3. A-30-quarterly-clean.
4. A-30-quarterly at σ = 0.3.
5. A-30-quarterly with a 2 m offset.
6. A-30-quarterly with R ×0.8.
7. A-30-quarterly with R ×1.2.
8. A-15-quarterly-clean.
9. D-30-quarterly, R ×1.0.
10. D-30-quarterly, R ×0.8. *(Case D: reported, not gated)*

**Metrics:**
- Relative error of S and T.
- Seasonal total-Q error.
- Relative L2 error of the Q field (overall size of the error across the whole map, as a fraction of the true Q).
- Head RMSE at hidden wells.

---

## 7. Experiment list (all on real data unless marked)

**Tasks:** A = temporal forecast at monitored wells; B = held-out spatial blocks; C = both.

| # | Experiment | Tasks | MVP? |
|---|---|---|---|
| E1 | Persistence, seasonal average | A | ✔ |
| E2 | Ordinary kriging per date | B | ✔ |
| E3 | RF / LightGBM (location, rainfall, land use, canals, elevation) | A B C | ✔ |
| E4 | Regional LSTM (no physics, no zoning) | A | optional |
| E5 | Single regional PINN, k = 1. **Main comparison** | A B C | ✔ |
| E6 | Zoned LSTM | A | optional |
| E7 | **Ours:** zoned PINN, latent Q, ensemble | A B C | ✔ |
| E8 | Zone-count sweep, k = 1, 3, 5 (+ 8 if every zone keeps ≥12 Tier-1 wells, + chosen k): accuracy vs k | B C | ✔ |
| E9 | Ablation, no physics: same networks, PDE weight = 0 | A B C | ✔ |
| E10 | Ablation, no Q: q_net removed (Q = 0) | A B C | ✔ |
| E11 | Ablation, no zoning = E5 (reuse) | n/a | ✔ |
| E12 | q_net restricted vs full (real data, after M2's synthetic verdict) | B C | optional |
| E13 | No canal seepage / recharge factor ±25% sensitivity | B | optional |
| E14 | With vs without interface loss | B C | stretch |
| E15 | Error vs distance to nearest training well (bins 0–5, 5–10, 10–20, 20+ km) for E2, E3, E5, E7 | B | ✔ |
| E16 | Repeated block hold-outs: mean ± spread | B C | ✔ for 3 layouts (5 is stretch) |
| E17 | Synthetic Case B: does zoning recover the two true zones and their S/T? | synthetic | optional |
| E18 | **Bucket-model baseline**: S·dh/dt = R − Q(land use, season), no sideways flow, zoned | A B C | ✔ |
| E19 | Ablation, T term off: the PINN with T∇²h removed | B C | ✔ |
| E20 | One network over the whole region with per-zone S/T/Q settings, vs per-zone PINNs. **[needs OK]** | A B C | optional |
| E21 | True forecast: test period driven by *climatological* rain (the long-term average for each month) instead of observed rain | A C | optional |

**Fixed before any test results are seen** (so nobody can cherry-pick):
- **Primary result:** the per-well RMSE change of E7 vs E5 on Task B. Averaged over the 3 block layouts, with a 95% **bootstrap confidence range**: resample the test wells many times and see how much the result moves.
- Everything else is a secondary result.

**Reported for every model:**
- **RMSE:** the typical error in metres; big misses count extra.
- **MAE:** the plain average error in metres.
- **NSE:** 1 = perfect, 0 = no better than always guessing the average, below 0 = worse than that.
- **Fraction of wells improved** over E5.
- **Wilcoxon signed-rank test** across wells: checks whether the per-well improvements are consistent rather than luck. The p-value is the chance of seeing an improvement this consistent if there were really no difference. Smaller is stronger evidence.

**PINN-only metrics:**
- **Coverage of the 90% intervals:** it should be about 90%.
- **Plausibility rate:** how often the model predicts rising levels in dry, heavily pumped seasons, which should almost never happen.
- **S and T** compared with the CGWB report ranges.
- **Q totals** compared with the CGWB block extraction figures.

---

## 8. Test plan (`tests/`, run with `pytest`)

| File | What it checks |
|---|---|
| `test_fd_solver.py` | the analytical cases in 6.3; water balance < 1e-8 each step; fixed-head edges hold; zero forcing at steady state stays put; the numpy and torch solvers agree |
| `test_pde_residual.py` | **(critical)** feed exact analytical h (sine-decay; a polynomial with known derivatives) as torch functions, **through the scalers**, and check residual ≈ 0, or equal to the known value, to 1e-5; sign test: raising Q lowers h_t; unit test: m/day in, m/day out |
| `test_recharge.py` | ΣR = RIF × Σrain (mass conserved through regridding and time aggregation); mm → m conversion |
| `test_scalers.py` | normalise → denormalise round trip; save/load equality |
| `test_head.py` | head = elevation − depth to water; dry readings produce NaN head and is_dry = True; negative depth to water (flowing well) flagged |
| `test_qnet.py` | output ≥ 0 for random inputs; restricted mode ignores x, y, t |
| `test_zoning.py` | merge_small_zones guarantees the minimum size; deterministic given a seed; **no leakage** (features change if test-period rows change → test fails) |
| `test_splits.py` | train and test wells are disjoint; test blocks don't overlap train blocks; test fraction within 15–20%; every zone keeps ≥10 train wells; deterministic |
| `test_metrics.py` | NSE(perfect) = 1, NSE(mean) = 0; coverage on constructed data; fraction improved on a toy case |
| `test_smoke_train.py` | 300 iterations on a tiny synthetic case: total loss falls, no NaNs, parameters stay finite (CPU, < 1 min) |
| `test_master_schema.py` | the master table has the required columns, types and value ranges (depth to water 0–150 m, lat/lon inside the study area) |

---

## 9. Data acquisition checklist (for you)

Before you start:
- Keep original filenames.
- Log each download (URL, date, notes) in `data/raw/SOURCES.md`.
- India-WRIS works best from an Indian IP or network.

| # | What | Where | Put in | Priority |
|---|---|---|---|---|
| 1 | **India Data Portal groundwater-level CSV** (all India, 22,965 wells), plus its codebook | indiadataportal.com → search "groundwater level" (CGWB) | `data/raw/cgwb_levels/idp/` | ✅ **Done** (2013–2021 for Punjab/Haryana) |
| 2 | **Longer history and metadata: India-WRIS CGWB levels 1990–2025** for the 16 belt districts (manual + telemetry, with well type, depth and aquifer) | `python scripts/fetch_wris_gwl.py` (same public download as the WRIS site and QGIS plugin; needs an Indian network) | `data/raw/cgwb_levels/wris/<state>/<district>.csv` | ✅ **Done 2026-10-17.** Karnal is sparse in WRIS, so it is filled from IDP |
| 2b | *(Optional, not planned)* data request to CGWB NWR Chandigarh | Draft in `docs/cgwb_data_request.md` | `data/raw/cgwb_levels/cgwb_request/` | Skipped by decision. Only revisit if Karnal or confined-aquifer data becomes critical |
| 3 | Well/station metadata (type, depth, aquifer) | Included in the WRIS download (item 2) | (in the WRIS files) | ✅ Done |
| 4 | CGWB NAQUIM aquifer reports (S/Sy, T, thickness) | cgwb.gov.in | `data/raw/cgwb_reports/naquim/` | ✅ All 16 districts (Panipat and Sonipat via the NCR Haryana Vol II report; plus a 2024 Panipat block-level plan) |
| 5 | GEC-2015 methodology report + detailed guidelines (infiltration factors, canal seepage, return flow) | cgwb.gov.in (publication 340) | `data/raw/cgwb_reports/gec2015/` | ✅ Done |
| 6 | Dynamic Ground Water Resource Assessment, **block-level** (extraction, recharge, stage) | cgwb.gov.in | `data/raw/cgwb_reports/assessment/` | ✅ Punjab and Haryana 2020, 2022, 2024 (plus 2023 summaries) |
| 6b | **District crop area by year (rice etc.)**, which drives year-to-year pumping | India Data Portal APY (DES) | `data/raw/crop_stats/` | ✅ Done: all 16 districts, 1997-98 to 2024-25 |
| 7 | Bhuvan LULC 50K (NRSC), Punjab + Haryana, 2005-06 / 2011-12 / 2015-16 | Bhuvan public WMS (`scripts/fetch_auxiliary.py bhuvan`). WFS/vector download is disabled, so rendered tiles are converted to class codes in M3 using the official SLD colours | `data/raw/bhuvan_lulc/wms_lulc50k/` | ✅ Auto. Note: 50K classes have a single 'Crop land' class (no kharif/rabi split), so seasonality comes from district crop statistics (item 6b) |
| 8 | Canal network | OpenStreetMap (auto). WRIS canal layers optional | `data/raw/canals/osm/` | ✅ Auto |
| 9 | District boundaries | GADM 4.1 (auto) | `data/raw/boundaries/` | ✅ Auto |
| 10 | Telemetry (6-hourly) | Included in the WRIS download (248 wells) | (in the WRIS files) | ✅ Done |

**Downloaded automatically by code** (`scripts/fetch_auxiliary.py`, `scripts/fetch_wris_gwl.py`; sources and licences in `data/raw/SOURCES.md`):
- IMD rainfall 1990–2024 → `data/raw/imd_rain/`
- Copernicus DEM GLO-30 tiles → `data/raw/dem/`
- OpenStreetMap canals → `data/raw/canals/osm/`
- ESA WorldCover 2021 (fallback) → `data/raw/worldcover/`
- GADM district boundaries → `data/raw/boundaries/`

---

## 10. Risk register

| Risk | Early warning sign | Fallback |
|---|---|---|
| PINN won't converge | PDE loss flat or rising while data loss falls; S or T hit their bounds; NaNs | Check scaling with `test_pde_residual`. Then: longer data-only stage, lower learning rate, gradient-norm weighting, smaller zone domain, L-BFGS polish. Last resort: the **differentiable FD inverse model** (asks you first) |
| Q not recovered on synthetic data | M2 run 1 or 2 has total-Q error > 10%; Q spread across seeds is large; Q mirrors R | Restrict Q further (zone scalar × land use × fixed crop calendar). Report only seasonal or zone totals. If still failing, drop the Q claim and report the net source (R − Q) only. This is still publishable |
| T unidentifiable | T posterior ≈ prior; T varies by more than 2× across seeds with the same head fit | Fix T at the report prior, learn S and Q, show T sensitivity. Expected on real data (Section 2.2) |
| Too few usable wells | **Happened at G0:** central Punjab had only 57 long-record wells | **Done:** expanded to the 16-district belt (242 Tier-1 wells) and two well tiers. WRIS 1990–2024 download added long histories and metadata; limit k ≤ 5 if zones get thin |
| Short test period | **Resolved:** the WRIS download gives a 2020–2024 test period plus 248 telemetry wells | Watch the 2020–21 rounds (only about 295 wells); report per-year metrics |
| Downloads blocked | WRIS timeouts, needs an Indian IP | IDP CSV as the main source; QGIS plugin from an Indian network; download from an Indian network (college Wi-Fi) |
| Bhuvan not downloadable | Only a WMS map view is offered | ESA WorldCover static farmland fraction plus the crop calendar |
| Elevation errors distort heads | Neighbouring wells show implausible head jumps > 5 m; noisy ∇h | Copernicus DEM, 1 km smoothing, CGWB reference levels; treat T as uncertain |
| Zones are scattered or poorly assigned | Many small polygons; assignment accuracy < 60% | Stronger smoothing; add coordinates to the classifier; contiguity-constrained clustering (asks you first) |
| Zoned doesn't beat regional | E7 ≈ E5 on validation | Check wells per zone, then the k sweep. Report honestly; a negative result is still reportable |
| Intervals badly calibrated | Validation coverage < 75% or > 98% | Noise term plus a validation scaling factor; more seeds |
| Leakage inflates results | Suspiciously high scores on test blocks; zone features using test years | Leakage tests in Section 8; fit scalers on training data only |
| Many dry or censored wells | Dry fraction > 20% in G0 | Censored loss (needs well depth), or restrict to piezometers |
| Colab session dies | Lost runs | Checkpoint every N iterations to Drive; one zone per job; use the college GPU for sweeps |
| Scope overrun | M3 still unfinished when the M2 results are done | Fall back to the MVP set (M0–M8 items marked **MVP**) |
| Zoning gives no gain | Zoning preview (M4) shows < 2% gain on validation | Try other k values, E20, better static features; else reframe the claims around spatial prediction plus S/Q maps (Section 13) |
| Physics adds nothing | E7 ≈ E18 (bucket) and E19 ≈ E7 | Report it as a finding about alluvial scale; the claims shift to zone parameters and pumping inference |
| Compute overrun | One zone × one seed takes more than 30 min on the GPU | Smaller networks, fewer collocation points, vmap seeds, 3 seeds for sweeps |

---

## 11. Minimum viable version (MVP), leaner

**MVP:**
- M0 (including G0 and the literature check), M1.
- M2: runs 1, 2, 3, 4, 6 and 9.
- M3 with these data sources:
  - CGWB levels
  - IMD rainfall
  - DEM
  - Static land use
  - Canals: OSM is fine.
- M4 with the zoning preview.
- M5: persistence, seasonal average, kriging, RF.
- M6.
- M7: per-zone PINN, 5 members (seeds × recharge factor), no interface loss.
- M8: E8 (k sweep), E9 (no physics), E10 (no Q), E15 (error vs distance), E16 (3 layouts), E18 (bucket).
- M10.

**Stretch, in priority order:**
1. E20
2. E19
3. E21
4. M9 scenarios
5. E13
6. LSTMs (E4, E6)
7. Interface loss (E14)
8. 5 layouts
9. Telemetry data

## 12. Compute budget

**Rule of thumb:** prototype everything on synthetic data on the GPU; use Colab only for overflow.

| What | Seeds/members | Layouts |
|---|---|---|
| Final model (E7) and main comparison (E5) | 5 (or 10) | 3 |
| k sweep, ablations, variants | 3 | 1 (the best layout is *not* chosen; it's layout #1, fixed in advance) |

**Speed-ups:**
- Train seeds in parallel on one GPU with `torch.func` (vmap).
- Stop early on the validation loss.
- Networks of width ≤ 48.
- Measure one zone × one seed in M6 and update this table.

## 13. Marks-safe deliverables and claims

**Deliverables that don't depend on beating anything** (build them so they look good in the slides):
1. **A working end-to-end pipeline:** raw CGWB files → master table → zones → trained models → maps. It is reproducible from configs.
2. **Synthetic validation:** a figure showing the model recovers known S, T and Q (Cases A and C), and an honest one showing where it breaks (Case D).
3. **A fair evaluation:** 6+ baselines, spatial block hold-outs, repeated layouts, confidence ranges.
4. **Maps:** zones, S, T and net pumping with uncertainty, plus an error-vs-distance plot showing where new monitoring wells would help most.
5. **An honest limitations section** drawn from Sections 2 and 2.8.

**Claims**

| Safe to claim | Do not claim |
|---|---|
| Spatial prediction at unmonitored places, tested with block hold-outs | "We measured / recovered pumping" on real data |
| Zone-specific aquifer parameters, compared with CGWB ranges | "Physics greatly improves forecasts" (unless E18/E19 show it) |
| Net abstraction estimates as ranges, cross-checked with CGWB block data | A guaranteed 5–10% gain |
| Synthetic proof the method works when its assumptions hold | That T is accurately mapped |

**Odds of the 5% target (honest)**

| Comparison | Odds of ≥5% gain |
|---|---|
| Zoned vs regional PINN (main) | Better than even |
| PINN vs RF/kriging | Toss-up |
| PINN vs seasonal-average forecast | Hard |

## 14. Verification of this plan once coding begins
- Each milestone's "Done" items are checked by a script in `scripts/` that writes a short `reports/mX_*.md`. `pytest` must stay green.
- No milestone starts before the previous one passes. In particular, **no real-data modelling before M2-A passes**. Data downloads and the M3 loaders may run in parallel.
