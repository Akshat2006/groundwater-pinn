# Batch 5: pumping v2 and M8 experiments (rules written before any result)

Written 2026-10-27, after the test set was scored once for model v1.0 (`reports/m6_notes.md`) and after the M7 report (`reports/m7_report.md`). **No batch 5 run had been trained when this was written.**

## Why

The M7 report showed that v1.0's pumping is almost flat across districts: 0.20-0.25 m/yr everywhere. Its rank correlation with CGWB's 2020 district extraction is only 0.31, and its Punjab/Haryana ratio is 1.01× against CGWB's 2.2×. S sits at its lower limit (0.072) in 83% of zones, and T is fixed from reports. So no parameter of the flow equation is identified from the data.

Diagnosis, from district figures only (no model runs):
- The pumping network's only land-use input is cropland share. That share is 80-90% almost everywhere, so it cannot tell districts apart.
- District paddy (rice) share does track CGWB extraction: rank correlation 0.76 across the 16 districts (`data/reference/paddy_share_district.csv`, training-year median).
- Caveat: CGWB's own extraction estimates are partly built from crop areas, so agreement with CGWB after adding paddy is partly by construction. It shows consistency, not independent proof.

## Variants (zoned shared model, canals on, everything else as v1.0)

| Tag | Change |
|---|---|
| `pq` | paddy share as a second pumping input (`q_net.paddy_input=true`) |
| `pqs` | `pq` + S fixed at the GEC-2015 norm 0.12 (`training.fix_S=true`), so S and T both come from reports and pumping is the one learned term |

Layouts 0, 1, 2; seed 0: 6 runs. Validation only (`--stage-only val`).

## Decision rule

**Accuracy gate.** For each run, score = A / 1.90 + B / 7.94: validation RMSE over all of the run's validation readings, the same score as batches 2-3.
- The variant's mean over the 3 layouts must be at most 2% above the reference.
- The reference is the v1.0 zoned `tbc` runs: the mean single-run score over 5 seeds, then over the 3 layouts, which is **2.172**.
- So the gate is a score of **2.215** or below.
- Why 2%: one seed's 3-layout mean varies by about 1.5% (seed spread 0.089 / 0.028 / 0.014 per layout).

**Pumping gate.** Both of these must hold:
- Rank correlation with CGWB 2020 extraction per area, across the 16 districts, is 0.5 or above. The model's district pumping is the 2017-2020 mean, averaged over the variant's 3 runs.
- The Punjab/Haryana pumping ratio is 1.3 or above.

**Choice.**
- If `pqs` passes both gates, it becomes v2: the cleaner design, with S and T from reports and pumping learned.
- Otherwise `pq`, if it passes both.
- Otherwise v1.0 stays, and the attempt is reported as a negative finding.

**If a v2 is chosen:**
- Seeds 1-4 × 3 layouts are trained with the same settings.
- Its test score is reported as a **follow-up experiment, after the test set had already been opened once**.
- The v1.0 headline (zoned vs regional, Task B, −8.1%) stays the primary result.
- The website maps move to v2 only if its Task B validation score is no worse than v1.0's.

## M8 experiments (same batch, reported whatever the outcome; nothing is chosen from them)

All runs are the zoned shared model with v1.0 settings, seed 0, layouts 0, 1 and 2. They are compared with v1.0 zoned `tbc` on validation.

| Tag | Question | Change |
|---|---|---|
| `nophys` | Does the physics help? | No PDE loss at all: 49,000 data-only steps (same total), no L-BFGS. A neural network with the terrain base and well corrections |
| `noq` | Does inferred pumping help? | Pumping switched off (Q = 0) |
| `ksweep` with k = 3 and k = 8 | How many zones? | k-means zone count. k = 1 is the regional model, k = 5 is v1.0 |

Error against distance to the nearest training well is computed on the laptop from existing predictions, with no training.
