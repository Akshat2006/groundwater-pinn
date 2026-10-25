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
