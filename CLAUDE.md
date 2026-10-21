# CLAUDE.md — Zoned Physics-Informed Neural Networks for Groundwater Forecasting

> **For Claude Code:** This file is the project brief and build plan for a student team's Experiential Learning (EL) project. Read it fully before writing code.
>
> **Rules**
> - **Build in the milestone order in Section 9.** Each milestone has acceptance criteria. Don't start the next one until the current one passes.
> - **Get it working on synthetic data first.** Never move to real data until the PINN recovers known parameters on a synthetic aquifer (Milestone 2).
> - **Keep decisions.** Decisions in Sections 2–7 were made deliberately. Don't swap methods without asking.
> - **Explain metrics in plain words.** The project lead says they don't have a statistics background, so explain any metric or statistical idea simply when reporting results.
> - **Never overclaim.** The "5–10% improvement" is a target, not a known result. Pumping estimates are inferences with no ground truth.
> - **Don't bypass data access restrictions.** Some sources need manual download (Section 5). Ask the user to place files in `data/raw/`.
> - **Language.** See Section 8.1. This is an open decision to confirm with the user.

---

## 1. Project summary

| Item | Value |
|---|---|
| **Title** | Zoned Physics-Informed Neural Networks for Groundwater Level Forecasting in Sparsely Monitored Aquifers |
| **SDG** | SDG 6 (Clean Water and Sanitation); secondary SDG 13 (Climate Action) |
| **Type** | Software only, research-oriented. No hardware. |
| **Core idea** | Group wells into zones that behave alike. Train one physics-informed neural network (PINN) per zone. Each one learns water levels from well data while obeying the groundwater flow equation, and infers unmetered pumping as a hidden variable. |
| **Framing** | Not "a better forecaster" but a *physics-based digital model of the aquifer*. It can predict where there are no wells, separate pumping from rainfall, and simulate "what if" scenarios. |
| **Team name options** | AquiferX, Team Terraflow |
| **Deliverables** | Working model + evaluation, EL presentation, IEEE-format conference paper |

---

## 2. Problem statement (final wording)

> India extracts over 245 billion cubic metres of groundwater a year, and more than one in ten assessment units are already over-exploited. Yet we track it through just 23,000 wells, read four times a year, which tells us what's already been lost, not what's coming next. Between those wells, we're blind. The biggest cause of decline, farm pumping, isn't measured at all. And today's forecasting models either ignore how groundwater physically behaves, or treat an entire region as if it were the same everywhere. So depletion is discovered only after wells run dry. What's missing is a way to forecast groundwater ahead of time, everywhere, including the places no well can see.

**Verified statistics** (CGWB Dynamic Ground Water Resources Assessment 2024, via PIB: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2089039)
- Annual extraction **245.64 BCM**; extractable resource 406.19 BCM; stage of extraction **60.47%**
- Of **6,746** assessment units: **751 over-exploited (11.1%)**, 206 critical, 711 semi-critical, 4,951 safe, 127 saline

**What exists vs what's missing:** India has strong *monitoring* (CGWB wells) and *annual assessment* (block-level water balance, GEC-2015 method). It has no routine, location-level *forecasting*, especially between wells.

---

## 3. Novelty (to our knowledge — full Google Scholar check pending)

1. **Zoning + PINN combined.** Zoning studies used plain LSTMs, and groundwater PINN studies used one model per domain. These two research threads haven't been combined.
2. **Unmetered pumping inferred as a hidden variable.** Existing groundwater PINNs assume pumping is known or leave it out.
3. **A real, sparse Indian well network** rather than synthetic test cases.
4. **Capabilities a plain forecaster lacks:** prediction at unmonitored locations, separating pumping from rainfall, "what if" scenario simulation, and an uncertainty map showing where new monitoring wells would help most.

**Closest existing work to cite and differentiate**
- **XPINN** (Jagtap & Karniadakis, 2020) and **cPINN** (Jagtap, Kharazmi & Karniadakis, 2020): domain decomposition, a network per subdomain, joined by interface conditions.
  - We differ by drawing zones from *how wells behave* rather than geometry, and by inferring pumping.
  - Consider adding XPINN-style interface conditions between neighbouring zones.
- **Not federated learning.** Our data is public and centralised, zone models are independent, and nothing is aggregated.

---

## 4. Literature (verified findings)

- **Sun, Hu, Sun & Cao (2023)**, *J. Hydrology: Regional Studies* 50, 101577. Beijing, 54 wells.
  - LSTM RMSE 0.70 m vs physics-based model 1.96 m, but the LSTM's 9-year trend was the *opposite* of the physics model's.
  - Hybrids improved 67–77% of wells; adding a rainfall lag raised NSE by about 0.1.
  - Future work they name: embed governing equations inside the model, i.e. a PINN.
- **Secci, Godoy & Gómez-Hernández (2024)**, *Computers & Geosciences* 182, 105494.
  - Transient unconfined-flow PINN with two networks (head + water table) and staged training.
  - 7×20 tanh, Adam, 25–35k collocation points.
  - NSE 0.87–0.99 vs MODFLOW; a plain NN failed on the same sparse data.
  - Synthetic only. Code: https://github.com/godoyva/Unconflow-PINN (MATLAB).
- **Zhang et al. (2022)**, GW-PINN, *Advances in Water Resources* 165, 104243. Core method reference. [VERIFY results]
- **Shadab et al. (2021)**, arXiv:2112.13792. Recovered hydraulic conductivity with a PINN (inverse problem).
- **Regional zoning study (2024)**, *J. Hydrology: Regional Studies*, ScienceDirect S2214581824003604. [VERIFY authors/title]
  - Single model: 67.6% of wells had RMSE > 2.5 m.
  - Zoned: 23.9%.
- **Raissi, Perdikaris & Karniadakis (2019)**, *J. Comput. Phys.* 378. The original PINN paper.
- **Wang, Yu & Perdikaris (2022)**, *J. Comput. Phys.* 449. Why PINNs fail to train (loss imbalance). Read before debugging.
- **Bayesian PINN calibration evidence.** A naive Bayesian PINN had only 23–38% real coverage for a nominal 95% interval; a careful one reached 94–95%. Uncertainty must be checked, not assumed.
- **Review of 178 PINN groundwater papers** (*Environmental Earth Sciences*, DOI 10.1007/s12665-026-12866-9). The field is moving from synthetic to heterogeneous cases; training stability is the main bottleneck.

Caveat: most of the supporting forecasting evidence is from outside India. Testing on Indian data is part of the contribution.

---

## 5. Data

| Data | Status | Source / access |
|---|---|---|
| Groundwater levels (depth to water) | Measured, mostly quarterly (Jan, Mar–May, Aug, Nov) | CGWB / India-WRIS (indiawris.gov.in). **Manual download** by state → district → year → season (Excel). Community QGIS plugin "CGWB Groundwater Data Downloader" works best from an Indian IP. India Data Portal CSV (22,965 wells, ODC-BY). NWDP has 6-hourly telemetry CSVs for some states |
| Rainfall | Measured, 0.25° daily, 1901–present | IMD via `imddata` (pip) or `imdlib` |
| Land use / irrigation | Measured, **proxy** for pumping | Bhuvan (ISRO/NRSC) LULC rasters. Manual download |
| Elevation | Measured | SRTM via `elevation` package or USGS EarthExplorer |
| Canal command areas | Measured | India-WRIS layers (needed for canal seepage recharge) |
| Aquifer type, S/T priors | Coarse, PDF reports | CGWB district/aquifer mapping reports |
| Recharge R | **Computed** | rainfall × infiltration coefficient (+ canal seepage) |
| S, T | **Inferred** | Learnable PINN parameters, initialised from reports |
| Pumping Q | **Inferred, no ground truth** | Q-network inside the PINN |

**Critical preprocessing**
- CGWB gives *depth to water*. The equation needs *hydraulic head*: head = ground elevation − depth to water.
- Flag dry wells separately. A dry well means the level is below the well bottom (censored), not missing.
- Expect data wrangling to take most of the timeline.

**Study area:** an **alluvial** region of the Indo-Gangetic plains (Punjab, Haryana or western UP). The simple porous-media equation fits alluvium much better than fractured hard rock. Ask the user to confirm the exact districts before downloading.

---

## 6. Method

### 6.1 Governing equation (depth-averaged 2D transient flow)
**S · ∂h/∂t = ∇·(T∇h) + R − Q**, which with constant T per zone becomes **S · ∂h/∂t = T(∂²h/∂x² + ∂²h/∂y²) + R − Q**

| Symbol | Meaning | Units | Role |
|---|---|---|---|
| h(x,y,t) | Hydraulic head | m | Network output |
| S | Storativity / specific yield | – | Learned (log-parameterised, init from reports) |
| T | Transmissivity | m²/day | Learned (log-parameterised) |
| R | Recharge | m/day | Computed input |
| Q | Pumping per unit area | m/day | Learned via Q-net (non-negative output) |

### 6.2 Zoning
- **Behaviour features per well:**
  - seasonal amplitude
  - long-term trend slope
  - rainfall-response lag (the lag with the highest cross-correlation)
  - optionally well depth and aquifer type
- Standardise features, run k-means, choose k by elbow + silhouette over k = 2–12.
- **Minimum 10–15 wells per zone.** Merge smaller clusters into the nearest one.
- Map zones and check spatial coherence.
- **Unmonitored locations have no history**, so assign their zone using only static features (location / nearest wells, land use, elevation, aquifer type). The evaluation must use this same process for held-out wells.

### 6.3 Per-zone PINN
- `h_net`: MLP (x, y, t [+ forcings]) → h. Start from tanh, roughly 6–8 layers × 20–64 units.
- `q_net`: small MLP (x, y, t, land-use intensity, season) → Q ≥ 0 (softplus).
- `log_S`, `log_T`: trainable scalars.
- **Losses:**
  - data loss at wells
  - PDE residual at collocation points (autograd derivatives)
  - boundary/initial losses
  - optional interface loss matching h and flux at borders with neighbouring zones (XPINN-style)
- **Training:** normalise all inputs/outputs and keep the scalers. Use staged training: data first, then add physics, then joint. Use adaptive loss weighting, Adam then optional L-BFGS, and periodically resample collocation points.
- **Ensemble:** 5–10 seeds per zone for uncertainty.

### 6.4 Known weaknesses and how the design handles them
1. **Pumping isn't uniquely identifiable** (recharge/pumping/storativity trade off). R is computed rather than learned, Q is constrained (non-negative, tied to land use and season), Q recovery is validated on a synthetic benchmark, and the spread of Q across the ensemble is reported.
2. **Zone assignment for unmonitored locations.** Use static features only and include this in the evaluation.
3. **Easy test from nearby wells.** Use a **spatial block hold-out** and report error vs distance to the nearest training well.
4. **Unfair LSTM comparison.** Add **kriging** and **random forest / gradient boosting** spatial baselines.
5. **Simplified physics.** Choose an alluvial area and state the single-layer limitation.
6. **Recharge isn't just rain.** Add canal seepage using canal command areas.
7. **Sparse quarterly data.** Forecast seasonal-scale changes and use telemetry wells if available.
8. **Less data per zone.** Run a zone-count experiment (k = 1, 3, 5, 8) and report accuracy vs k.
9. **Zones don't exchange water at borders.** Use interface conditions, or state it as a limitation.
10. **"What if" scenarios are projections.** Run them by plugging the learned S and T into a forward solver (PINN forward mode or finite differences) with modified Q or R, and present them with uncertainty.

---

## 7. Evaluation

**Hold-out:** spatial blocks covering 15–20% of wells, plus a later time period for temporal extrapolation.

**Baselines**
1. Persistence (next season = this season) and seasonal average
2. Kriging (spatial interpolation)
3. Random forest / gradient boosting with location, rainfall and land use
4. Regional LSTM (no physics, no zoning)
5. Single regional PINN (physics, no zoning). **This is the main comparison.**
6. Zoned LSTM (optional; isolates the effect of physics)
7. **Ours:** zoned PINN with latent Q

**Metrics** (explain each in plain words when reporting)
- **RMSE / MAE:** typical error in metres
- **NSE:** 1 = perfect, 0 = no better than the average, below 0 = worse than the average
- **Fraction of wells improved**, plus a paired test across wells
- **Coverage:** how often the true value falls inside the stated 90% range; it should be close to 90%
- **Physical plausibility:** how often a model predicts rising levels in dry, heavily pumped periods
- **Recovered S and T** compared with CGWB report ranges
- **Q recovery error** on the synthetic benchmark

**Targets (not guaranteed)**
- Roughly 5–10% lower RMSE than the single regional PINN on held-out blocks
- Improvement on at least 67–77% of wells
- 90% coverage within about 5 points
- Synthetic Q recovered within about 10%

A negative result is still reportable.

---

## 8. Engineering setup

### 8.1 Language (confirm with the user)
The user's saved preference says to write code in C++. This project's ecosystem is Python-first: PyTorch/DeepXDE, rasterio, geopandas, xarray, `imddata`, scikit-learn, QGIS tooling. **Recommended:** Python for the pipeline and models. If C++ is required, use LibTorch for the PINN core and keep data prep in Python. **Ask the user before starting Milestone 0.**

### 8.2 Suggested repository layout
```
groundwater-pinn/
├── CLAUDE.md
├── README.md
├── environment.yml / requirements.txt
├── configs/            # study area, zones, model + training hyperparameters (YAML)
├── data/
│   ├── raw/            # manual downloads (CGWB, Bhuvan) — not committed
│   ├── interim/
│   └── processed/      # master well-time table, zone assignments
├── src/
│   ├── data/           # loaders: cgwb.py, imd.py, landuse.py, dem.py; merge.py
│   ├── features/       # head conversion, rainfall windows, zoning features
│   ├── zoning/         # clustering, k selection, zone assignment for unmonitored points
│   ├── physics/        # PDE residual, recharge model, boundary/interface terms
│   ├── models/         # h_net, q_net, PINN wrapper, baselines (persistence, kriging, RF, LSTM)
│   ├── training/       # staged training, loss weighting, ensembles
│   ├── evaluation/     # spatial block splits, metrics, calibration, plausibility checks
│   └── scenarios/      # forward solver for what-if runs
├── synthetic/          # synthetic aquifer generator with known S, T, Q
├── notebooks/          # exploration only
├── tests/
└── reports/figures/
```

### 8.3 Conventions
- Configuration in YAML, with fixed random seeds
- Save scalers alongside models
- Log every experiment (config + metrics) to `reports/`
- Unit-test the PDE residual against an analytical solution

---

## 9. Milestones (build in this order)

**M0: Setup**
- Confirm language and study area with the user
- Create the repo layout and environment
- *Done when:* the environment installs and an empty test suite runs.

**M1: Synthetic aquifer generator**
- Finite-difference solver for the 2D equation with known S, T, R, Q
- Sample "wells" at sparse locations and quarterly times
- *Done when:* the solver conserves water balance and matches an analytical case.

**M2: Single-zone PINN on synthetic data**
- Fit h, recover S and T, recover Q
- *Done when:* S and T are within about 10% of true values and Q is within about 10% on clean synthetic data. Then report how this degrades with added noise.

**M3: Real data pipeline**
- Loaders for CGWB, IMD, Bhuvan, SRTM and canal layers
- Head conversion, cleaning, dry-well flags, master table
- *Done when:* the master table covers the chosen districts with documented gaps.

**M4: Zoning**
- Features, clustering, k selection, zone map, static-feature zone assignment
- *Done when:* every zone has at least 10–15 wells, the zone map is saved, and assignment accuracy on known wells is reported.

**M5: Baselines**
- Persistence, seasonal average, kriging, RF/GBM, regional LSTM
- Spatial block splits
- *Done when:* baseline metrics table produced.

**M6: Regional PINN (unzoned) on real data**
- *Done when:* it trains stably and metrics are produced.

**M7: Zoned PINN + Q-net + ensembles**
- Optional interface conditions
- *Done when:* full metrics, calibration, plausibility check, S/T/Q maps.

**M8: Experiments**
- Zone-count sweep, ablations (no physics, no zoning, no Q), error vs distance to nearest well
- *Done when:* results tables and figures are in `reports/`.

**M9: Scenarios (stretch)**
- Forward runs with modified Q or R using the learned S and T
- *Done when:* example what-if curves with uncertainty.

**M10: Write-up support**
- Figures and tables for the slides and the IEEE paper

---

## 10. Stakeholders (for README / paper)

- **Primary:** CGWB and State Groundwater Boards (early warning, monitoring-well placement), irrigation/agriculture departments (pre-season advice), researchers.
- **Secondary:** NGOs, urban utilities, farmers (via extension officers).
- **Position:** the tool complements CGWB's annual assessment rather than replacing it. Training is offline; queries are near-instant.

---

## 11. Glossary

- **Hydraulic head:** water-table elevation (m)
- **Depth to water:** what CGWB records; distance from ground level down to the water table
- **Storativity (S):** water released per unit drop in head
- **Transmissivity (T):** ease of sideways flow
- **Recharge (R):** water reaching the water table
- **Q:** pumping
- **PINN:** neural network trained on data loss + physics loss
- **Collocation points:** random points where the physics is enforced; no measurement needed there
- **Inverse problem:** recovering unknown parameters from observations
- **Kriging:** standard method for estimating values between measured points
- **Spatial block hold-out:** hiding whole areas, not random wells, to test genuine extrapolation
- **Deep ensemble:** several models trained from different starts; their spread gives uncertainty
