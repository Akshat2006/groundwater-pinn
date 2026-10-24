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
