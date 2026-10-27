# M6/M7 working notes: real-data PINN batches

Validation data only (layout 0, seed 0). The test wells have not been scored.

## Batch 1 (2026-10-24, `notebooks/colab_real_batch1.ipynb`)

Full table: `reports/m6_real_pinn_val.md`.

**1. Storage S collapses, and forecasts blow up with it.**

- S starts at 0.12. As soon as it is allowed to learn, it falls in every run.
- Regional default: 0.023. Zones: 0.0002 / 0.007 / 0.0002 / 0.018. Data weight 100: 0.0002.
- The tighter the fit to the readings, the lower S goes. Small zones and a high data weight fit tightest.
- Runs whose S fell below about 0.001 forecast 2019 badly: zone 1 18.6 m, zone 3 10.4 m, regional w100 22.1 m, against 4.1 m for the regional default.
- This is the same mechanism as in the M2 noise runs (run4). The network fits reading noise, which makes the water level jump about in time. The cheapest way to keep the physics satisfied is then to shrink S, which switches the time part of the equation off.
- Values like 0.0002 are confined-aquifer values, not specific yields of a water-table aquifer. The CGWB reports give 0.072–0.15 (priors.yaml).

**2. At places with no well, interpolating depth to water beats interpolating head.**

Validation Task B, inverse-distance weighting:
- head: 10.0 m
- depth to water, with the target's elevation added back: 6.9 m

A Gaussian-process stand-in for a smooth network surface gives the same picture on all 3 layouts:
- head: 12.8 / 7.2 / 8.7 m
- depth: 9.6 / 6.7 / 8.2 m

Elevation smoothed at 5–10 km keeps most of the gain:
- mean 8.46–8.47 m, against 9.56 m with no terrain
- at 20 km: 9.65 m

Not the explanation:
- the local terrain residual (elevation minus smoothed elevation) explains none of the PINN's errors (R² ≈ 0);
- a residual correction by inverse-distance weighting helps only a little (9.38 → 9.19 m).

**3. Zoned is worse than regional with these settings, mostly because of item 1.**

| Task | Zoned beats regional on |
|---|---|
| A | 35% of wells |
| B | 27% of wells |

In zones 2 and 4, where S stayed above 0.005, the gap is small.

**4. Baseline added for fairness: depth kriging** (krige depth to water, add the target's elevation back; this is how CGWB draws water-level maps).

| Stage | Task | Depth kriging | Previous best baseline |
|---|---|---|---|
| Validation | B | 8.08 m | LightGBM 8.38 m |
| Test | B | 8.75 m | LightGBM 8.63 m (still best) |

Means over the 3 layouts. This raises the bar for the PINN.

**5. Pre-set choice rule on batch 1 alone:** `off1` (weaker well corrections) scores 2.80 against 3.33 for the default. It has the best Task A (2.41 m, but persistence scores 1.90 m) and the worst Task B (12.2 m).

**6. d3 (synthetic, 30 wells read monthly).**
- Seasonal pumping error is 8%, against 49% with quarterly readings (d1). Total pumping −7%.
- About 30 frequently read wells *can* resolve seasonal pumping, so seasonal estimates on real data would need the telemetry wells.
- S is −15% and T −18% here. T was learned in this run; it is fixed on real data.

## Changes for batch 2 (`notebooks/colab_real_batch2.ipynb`)

- `training.S_range: [0.072, 0.15]`: S is squashed into the CGWB report range. If S sits at the lower edge, report it as "the data push S below the report range".
- `h_net.terrain_sigma_km: 10`: the network models head minus 10 km-smoothed elevation. The PDE still acts on the full head: T·∇²(base) enters as a known source, so the physics is unchanged; only the network's default between wells changes.
- Variants, chosen by the same pre-set rule:
  - S range widened to 0.02–0.25 (`tbw`)
  - no terrain (`sr`)
  - weaker well corrections (`tb_off1`)

## Batch 2 (2026-10-25, `notebooks/colab_real_batch2.ipynb`)

All 8 runs finished. Validation, layout 0, seed 0. Score = A / 1.90 + B / 7.94 (each task relative to its best baseline: persistence, LightGBM).

| Run | S | val A | val B | val C | Score |
|---|---|---|---|---|---|
| regional `tb_off1` (terrain, S range, weak corrections) | 0.072 | 2.03 | 9.51 | 11.31 | **2.27** |
| regional `tb` (terrain, S range) | 0.072 | 2.85 | 10.46 | 10.78 | 2.82 |
| regional `sr` (S range, no terrain) | 0.072 | 3.30 | 9.08 | 10.86 | 2.88 |
| regional `tbw` (S range 0.02–0.25) | 0.020 | 3.95 | 10.69 | 9.92 | 3.42 |
| zoned `tb` (4 zone networks) | 0.072 ×4 | 2.73 | 19.18 | 12.47 | 3.85 |
| *best earlier: regional `off1`* | 0.075 | 2.41 | 12.16 | 13.54 | 2.80 |

**1. Keeping S in the report range fixes the forecasts.** Every run with the range sits at its lower edge (0.072), and the widened run sits at its lower edge (0.020). So the data still push S below the CGWB range; report it that way. But forecasts are much better with the range: Task A 2.85 m (`tb`) against 3.95 m (`tbw`) and 4.08 m (batch 1 default).

**2. Pre-set rule picks `tb_off1`** (score 2.27). Its Task A (2.03 m) is level with the seasonal-trend baseline (2.04 m) but behind persistence (1.90 m). Task B is 9.51 m against 7.94 m for LightGBM and 8.08 m for depth kriging. Its per-well corrections spread 7.0 m. It is now the default in `configs/pinn_real.yaml`.

**3. Terrain helped Task A but not Task B in the PINN** (B: `tb` 10.46 m vs `sr` 9.08 m), unlike the Gaussian-process stand-in. Not explained yet.

**4. One network per zone fails at places with no well.**
- Task A: zoned 2.73 m vs regional `tb` 2.85 m; zoned better on 53–71% of wells per zone.
- Task B: zoned 19.18 m vs regional `tb` 10.46 m. Zones 3 and 4 are 15 m too low on average (zone 3: 26 validation wells, errors up to 39 m). Their per-well corrections are small (0.3–1.6 m), so the network surface itself is wrong. The regional network is wrong in the same places, but by about half as much: a zone network sees 73–269 wells and extrapolates further.
- This is risk 7 in PLAN.md §2.8 ("one network per zone splits the data thinly"). The planned fix is E20.

## Batch 3 (`notebooks/colab_real_batch3.ipynb`)

- **E20, "shared" zoned model** (`run_real.py --shared`): one water-level network for the whole region (all 648 training wells); S, T and pumping per zone (zone weights enter the pumping network; each zone has its own pumping trend; T fixed per zone from the reports as before).
- Zones 1–4 re-run with `tb_off1`, so per-zone and shared use the chosen setting.
- Final regional `tb_off1` runs: seeds 1–4 on layout 0, seeds 0–4 on layouts 1 and 2.

**Rule fixed before batch 3 results:** per-zone (E7) vs shared (E20), both `tb_off1`, layout 0, seed 0: the lower score A / 1.90 + B / 7.94 becomes "ours" and gets the final runs (5 seeds × 3 layouts). The other design is reported on validation as a design comparison. The primary result (zoned vs regional PINN, Task B, test, 3 layouts) is unchanged.

### Batch 3 revised: canal seepage added, final runs moved to batch 4

Recharge was rain only, although CLAUDE.md §6.4 item 6 asks for canal seepage.

**Evidence that it is missing:**
- Validation Task B wells within 2 km of a canal: the regional `tb_off1` PINN is 6.1 m too low on average (29 wells). Farther away: 0.6 m (31 wells).
- Training wells' per-well corrections: +1.7 m near canals, −0.95 m far.
- Long-term trend: −0.34 m/yr near canals, −0.43 m/yr far.

**How it is added (`canal_seepage` in `configs/pinn_real.yaml`, `real_data.canal_recharge`):**
- GEC-2015 formula: recharge = wetted area × seepage factor × running days.
- Table 9 norm: unlined canals in normal soils, 17.5 ham/day per million m² of wetted area. Assumed: wetted perimeter 10 m, 180 running days a year.
- OSM canal density, spread with a 2 km Gaussian.
- Result: area mean about 0.025 m/yr (rain recharge about 0.14 m/yr); 0.066 m/yr within 2 km of a canal.

**Size check on training wells only:** a 20–50 year finite-difference run with this source gives a near/far bump of 2.0–4.1 m, the same size as the 2.7 m gap in the training corrections. So strength 1 (the norm) is used without tuning. Its correlation with individual wells' corrections is weak (0.10–0.17): it explains the group difference, not each well.

**Runs:**
- regional and shared with strength 1 (`tbc`);
- regional at half and double strength (`tbc05`, `tbc2`), as sensitivity only.

**Rule fixed before the results:** canals are kept if regional `tbc` scores lower than regional `tb_off1`. The zoned design is then chosen as before, with canals if kept. Final runs (5 seeds × 3 layouts) follow in batch 4 with the settings decided here.

## Post-processing checks for places with no well (2026-10-25, validation, layout 0)

**Residual kriging** (krige each training well's correction to the unseen places and add it). Variants fixed before looking; scratch script, no retraining.

| Run | No correction (B / C) | Krige learned offsets | Krige mean training residual |
|---|---|---|---|
| regional `tb_off1` | 9.51 / 11.31 | 9.39 / 11.18 | 9.48 / 11.29 |
| regional `tb` | 10.46 / 10.78 | 10.46 / 10.77 | 10.50 / 10.82 |
| per-zone `tb` | 19.18 / 12.47 | 19.19 / 12.46 | 19.22 / 12.50 |

Gain at most 1.3%. The corrections are not smooth in space at well spacing (as in M4: neighbour correlation ~0.1), so kriging cannot carry them to new places. Kept only as an optional secondary row in the final tables (same step for every PINN, so it does not change zoned vs regional).

**Averaging the PINN with a baseline** (equal weights, no tuning), Task B: with depth kriging 8.38, with LightGBM 8.20, against LightGBM alone 7.94. Errors of the PINN and the baselines are strongly correlated (0.63–0.76): all methods miss the *same* wells. Most of the Task B error is local variation that none of the inputs explain (water levels differ by 31 m between B wells), so no post-processing closes the gap to LightGBM. Not adopted.

## Real batch 3 (validation, layout 0, seed 0)

Typical error (RMSE) in metres. "Wells improved" = share of wells whose own error is smaller than with the regional PINN of the same setting; p = Wilcoxon paired test across wells (below 0.05 = unlikely to be luck).

| Model | A (monitored, 2019) | B (no well) | C (no well, 2019) | Score |
|---|---|---|---|---|
| regional `tb_off1` (no canals) | 2.03 | 9.51 | 11.31 | 2.27 |
| regional `tbc` (canals, GEC norm) | 2.03 | 9.12 | 10.78 | **2.22** |
| regional `tbc05` / `tbc2` (half / double canals, sensitivity) | 2.02 / 2.03 | 8.92 / 7.56 | 10.89 / 9.61 | not chosen from |
| per-zone networks `tb_off1` | 2.09 | 11.21 | 12.06 | 2.51 |
| **shared zoned** `tb_off1` | 2.15 | 8.43 | 11.01 | 2.19 |
| **shared zoned** `tbc` | 2.16 | **8.00** | 10.27 | **2.15** |
| baselines: persistence / LightGBM / trend + kriging | 1.90 | 7.94 | 8.57 | |

**Rules applied (fixed before the runs):**
- Canals kept: regional `tbc` 2.22 < `tb_off1` 2.27. `canal_seepage.strength` default is now 1.
- Zoned design: shared (2.15) beats per-zone (2.51). Shared `tbc` goes to the final runs.

**Zoned (shared) vs regional, both with canals:**
- Task B: 8.00 vs 9.12 m (12% lower); 75% of wells improved, p < 0.001.
- Task C: 10.27 vs 10.78 m (5% lower); 76% of wells improved, p = 0.045.
- Task A: 2.16 vs 2.03 m (6% higher); 52% of wells improved, p = 0.87 (no consistent difference per well).

Sharing one water-level network fixed the per-zone failure at unseen places (B 11.2 → 8.4 m), as expected from PLAN.md risk #7.

**Against the baselines (shared `tbc`):** Task B 8.00 vs LightGBM 7.94 (level in RMSE, but LightGBM better on 60% of wells, p = 0.14), beats depth kriging (8.08) and kriging (9.59). Task C: behind trend + kriging (8.57).

**Learned parameters:** S stays at the lower bound 0.072 in 3 of 4 zones (zone 4: 0.076–0.087). Pumping 0.20–0.23 m/yr over 2005–2018; canal recharge adds 0.028 m/yr on average.

**Caveats:** one seed, one layout, 60 validation wells in task B. Double canal seepage scored best of all on B (7.56), which suggests the GEC-norm assumptions may be on the low side; it is reported as sensitivity, not adopted.

**Next:** batch 4 (`notebooks/colab_real_batch4.ipynb`): regional and shared `tbc`, 5 seeds × 3 layouts; then the test set is scored once (`predict_real.py --stage test`, `check_m6.py --stage test --ref-tag tbc`).

## Batch 4: final runs, validation, and the test rules (written before the test set was scored)

28 runs on three people's Colab accounts (`real4_L{0,1,2}_results.zip`), plus layout 0 / seed 0 from batch 3. Every run completed all 48,000 steps. Final models: **shared zoned `tbc`** and **regional `tbc`**, 5 seeds × 3 block layouts each. No test predictions existed before this section was committed.

**Validation (2019), mean of 5 seeds, RMSE in metres** (`reports/m6_real_pinn_val.md`):

| Layout | A zoned / regional | B zoned / regional | C zoned / regional | B: wells improved, p |
|---|---|---|---|---|
| 0 | 2.01 / 1.97 | **8.13 / 9.15** (−11%) | 9.99 / 10.80 | 73%, p < 0.001 |
| 1 | 2.11 / 2.13 | **6.98 / 7.41** (−6%) | 7.12 / 7.30 | 61%, p = 0.003 |
| 2 | 2.04 / 2.00 | 8.31 / 8.34 (−0.4%) | 5.62 / 5.43 | 45%, p = 0.13 |

The zoned advantage at no-well places (B) holds in layouts 0 and 1 and vanishes in layout 2. This is the expected size of the effect on test: smaller than the single-seed batch 3 figure (−12%).

**Test rules (fixed now, applied once):**
- The test period (2020–24) is predicted once for the 30 `tbc` runs (`predict_real.py --stage test --match "_tbc$"`) and scored with `check_m6.py --stage test --ref-tag tbc`. Nothing is retrained or re-tuned afterwards.
- Headline per task: the **mean over the 3 layouts** of the 5-seed-ensemble RMSE, for zoned and regional; the % difference is computed from those means. Wells improved and Wilcoxon p are reported per layout, not pooled (the same wells recur across layouts).
- Target from CLAUDE.md: 5–10% lower RMSE than the regional PINN on held-out blocks (task B), ≥ 67% of wells improved. It is reported as met or not met either way.
- Baselines (persistence, seasonal trend, kriging, depth kriging, LightGBM, trend + kriging) are reported alongside, on the same readings.
- The website maps (`export_maps.py --version model-v1.0`) use the layout-0 shared `tbc` 5-seed ensemble, chosen now regardless of the test result.
