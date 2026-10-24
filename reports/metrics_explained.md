# What the numbers mean

Every score in the reports compares a model's predicted water level with the measured one at real CGWB wells that the model did not learn from.

## Error sizes (in metres)

| Measure | Plain meaning | Example |
|---|---|---|
| **RMSE** (root mean square error) | The typical size of an error. Big misses count extra, because errors are squared before averaging. | RMSE 3 m: predictions are typically about 3 m off, and a few are much further off. |
| **MAE** (mean absolute error) | The plain average size of the error. Always at most the RMSE; a big gap between them means a few large misses. | MAE 2 m, RMSE 3 m: most errors are around 2 m, with some large ones. |
| **Median well RMSE** | Each well gets its own RMSE. This is the middle one when wells are sorted. A few bad wells cannot dominate it. | 1.9 m: half the wells have a typical error below 1.9 m. |

## Skill scores

| Measure | Plain meaning |
|---|---|
| **NSE** (Nash–Sutcliffe efficiency), per well | 1 = perfect. 0 = no better than always predicting the well's own average level over the period. Below 0 = worse than that. It is harsh when a well has only a few readings that barely change: a constant 1 m error then gives a very negative NSE. |
| **Fraction of wells improved** | The share of wells where our model's error is smaller than the comparison model's. 70% means 7 in 10 wells got better. |
| **Wilcoxon signed-rank test (p-value)** | Are the per-well improvements consistent, or could they be luck? The p-value is the chance of seeing improvements this consistent if both models were really equally good. Below 0.05 is usually taken as convincing. |
| **Bootstrap confidence range** | Re-draw the test wells at random many times and recompute the result. The middle 95% of those results is the range the true improvement probably lies in. |

## Uncertainty

| Measure | Plain meaning |
|---|---|
| **Coverage of a 90% range** | How often the measured value falls inside the model's stated 90% range. It should be close to 90%. Much lower = the model is over-confident; much higher = its ranges are wider than needed. |
| **Range width** | How wide the stated ranges are. Narrow ranges with good coverage are best. |

## Physics checks

| Measure | Plain meaning |
|---|---|
| **Plausibility rate** | How often the model predicts rising water levels during dry, heavily pumped months, which should almost never happen. |
| **Storage (S) and flow ease (T)** | Compared with the ranges in CGWB aquifer reports. S is the water released per metre of level drop; T is how easily water moves sideways. |
| **Pumping (Q)** | Estimated net groundwater abstraction per zone and year, compared with CGWB block-level extraction figures. Both are estimates; neither is a meter reading. |

## The three tasks

| Task | Question | Why it matters |
|---|---|---|
| **A** | How well can we forecast later years at wells we already monitor? | Early warning at existing wells |
| **B** | How well can we estimate levels in areas with no wells? | Filling the gaps between wells |
| **C** | Both at once: later years, no wells nearby | The realistic use for unmonitored villages |

**Hindcast:** test-year predictions use the rainfall that actually fell, for every model. They show how well a model explains what happened, not how well it would have predicted it months ahead.
