# Why storage (S) cannot be learned from these data (2026-10-27)

Diagnostics only: no training, no test data.

## 1. What the trained model implies

v1.0 zoned model, layout 0, seed 0, 2005-2018, 1,500 random cells × 168 months. Take the model's own water-level changes (h_t) and the water sources in the equation (lateral flow + recharge − pumping). The S that best balances S·h_t against the sources is:

| Part of the signal | Best-fitting S |
|---|---|
| everything | 0.061 |
| seasonal ups and downs | 0.058 |
| yearly means (long-term decline) | 0.071 |

All three are below the CGWB report range (0.072-0.15), which is why S sits at its lower limit.

The sizes involved:
- The water level moves by 2.9 m/yr (root-mean-square), mostly seasonally, with a mean decline of 0.50 m/yr.
- The sources are small: recharge 0.18 m/yr (rain × 0.22 plus canals), pumping 0.23 m/yr, lateral flow 0.09 m/yr.

Water levels only fix the ratio (sources) / S. Double recharge, pumping and S together, and the levels are identical. S can therefore only be learned if one source is known in absolute size. Rain-only recharge sets that size, and it is too small: it leaves out irrigation return flow, which CGWB counts as recharge. On the synthetic aquifer (M2) recharge was known exactly, and S was recovered within 10%.

## 2. Could CGWB's district figures supply the missing scale? (option "B", checked before building it)

The idea was to anchor the model's pumping to CGWB 2020 district extraction, so that S could be learned. First check: what S does CGWB's own balance imply, given the observed declines?

- Belt-wide CGWB deficit (extraction − extractable recharge) = 0.37 m/yr of water.
- The median pre-monsoon well trend over 2005-2018 (wells with ≥ 8 years) = −0.29 m/yr.
- Implied S ≈ 1.25. Using the model's area-mean decline of 0.50 m/yr instead gives about 0.7.

Alluvium cannot hold more than about 0.25-0.3. With S = 0.12, CGWB's deficit would mean declines of about 3 m/yr; the wells show 0.3-0.5 m/yr. Per-district values scatter widely, because few wells have long pre-monsoon records, but the belt-wide gap is a factor of 5-10.

**So the three sources disagree:**
- the water levels;
- CGWB's extraction and recharge estimates;
- the report storage value of 0.12.

They cannot all be right at once. Possible reasons:
- CGWB overestimates extraction (estimated from well counts × norms) or underestimates recharge;
- substantial inflow from outside the shallow aquifer;
- both.

Anchoring the model to CGWB extraction would force S to physically impossible values, or the model would have to ignore the anchor. **Option B was therefore not built.**

## Consequence for the project

- S is not identifiable from quarterly water levels with the recharge we can compute. This is a finding, not a bug (CLAUDE.md §6.4 item 1).
- The honest design is S and T from reports and pumping learned: variant `pqs` in batch 5.
- Its pumping is then net pumping on the scale set by the rain recharge. It is about a quarter to a third of CGWB's gross extraction. The gap is explained by return flow and by the CGWB balance discrepancy above.
