# Literature check (M0), 2026-10-17

**How this was done:**
- Web searches (not Google Scholar itself) for:
  - zoned/clustered PINN groundwater
  - PINN unknown pumping / source inversion
  - PINN with Indian field data
  - XPINN domain decomposition for aquifers
- Mostly abstracts and search snippets were read; several publisher pages blocked full text.
- This is **a first pass, not an exhaustive review.** Before the paper is written, repeat the queries at the end in Google Scholar and read the closest papers in full.

## What already exists

### 1. Zoning wells, then training a separate *data-driven* model per zone (no physics)

The idea of zoning to improve forecasts is established, so our zoning step is not new by itself.

| Paper | What it does | Relevance |
|---|---|---|
| Zhou et al., *J. Hydrology: Regional Studies* 56, 102011 (Dec 2024), Beijing Plain ([DOAJ](https://doaj.org/article/b7339ee9d91543a6b11999d2a3ca50ae)) | GRA-CLU clustering into 6 "dynamic types", then one LSTM per zone. RMSE down 31.5–59.8%, NSE up 5.7–50% vs unzoned | Probably the "regional zoning study (2024)" marked [VERIFY] in CLAUDE.md §4. **But** the brief quotes 67.6% → 23.9% of wells with RMSE > 2.5 m, and that figure did not appear in the abstract. Check the full paper before citing those numbers |
| Clustering-integrated spatiotemporal deep learning, *J. Hydrology: Regional Studies* (2026), Zhuoshui River basin, Taiwan ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214581826006683)) | k-means on 120 wells, with cluster-specific normalisation and LSTM-type models | Same idea (k-means + per-cluster deep model), again without physics |
| Northern France, *HESS* 29, 841 (2025) ([HESS](https://hess.copernicus.org/articles/29/841/2025/hess-29-841-2025.html)) | Models trained per groundwater-level "type" (clustered on spectral properties) beat one model trained on everything | More support that zoning helps data-driven models |

### 2. PINNs for groundwater: mostly synthetic, forward, or inverting K only

| Paper | What it does | Relevance |
|---|---|---|
| Secci, Godoy & Gómez-Hernández, *Computers & Geosciences* 182 (2023/24) ([author version](https://jgomez.webs.upv.es/wordpress/wp-content/uploads/2023/11/Author_version_PINN.pdf)) | Transient unconfined PINN, synthetic data | Already in the brief |
| Virupaksha et al., *Water Resources Research* (2026) ([Wiley](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025WR040754)) | Transient unconfined PINN under dynamic conditions; notes that most prior work is steady-state and confined | Full text blocked. **Read it**: it may have pumping cases |
| Coastal aquifer PINN, northern Oman, *Hydrological Sciences Journal* (2026) ([T&F](https://www.tandfonline.com/doi/full/10.1080/02626667.2026.2653056)) | Regional PINN surrogate for MODFLOW/MT3DMS, trained on synthetic model output; pumping is a **known input** | Closest "regional PINN"; it does not infer pumping |
| KLE-PINN, *J. Hydrology* (2024) ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022169424010990)) | Inverse PINN for the conductivity field from heads, using pumping-test data | Inverts K, not pumping |
| Shadab et al., arXiv:2112.13792 | Steady unconfined PINN; recovers K | Already in the brief |
| Khan et al., *Frontiers in Water* (2026) ([Frontiers](https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2026.1787659/full)) | Physics-informed *Bayesian* NN for recharge in Oman; water balance as a loss penalty; no pumping term, no well data | Shows "physics-informed" is used loosely; useful contrast |
| GW-PINN, *Advances in Water Resources* (July 2022) ([WHU page](https://jszy.whu.edu.cn/zhuyan1/en/lwcg/444123/content/66594.htm)) | "GW-PINN: a deep learning algorithm for solving groundwater flow equations" | **Partly verified:** title, journal and date confirmed (Wuhan University group). Authors and volume/article number (brief says Zhang et al., 165, 104243) not yet confirmed |

### 3. Domain decomposition (one network per subdomain) in subsurface flow

- **Layered unsaturated soils**, arXiv:2202.11274 ([arXiv](https://arxiv.org/html/2202.11274v1)): one NN per soil layer, with an inverse problem for the surface flux. Subdomains come from *geometry* (layers), not from well behaviour.
- **Porous-media PINN review**, 2026 ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S266654682600073X)): XPINNs reduce memory per network; interface design adds complexity.
- **Groundwater PINN review**, "evidence, limits, and a roadmap" ([ResearchGate](https://www.researchgate.net/publication/401189290_Physics-informed_neural_networks_for_groundwater_evidence_limits_and_a_roadmap)): proposes splitting aquifers into subregions (PINNs where heterogeneous and data-rich, conventional solvers elsewhere). This is a **proposal, not a clustering method**. It is the closest statement of our idea, so we cite it and position ourselves as one concrete implementation.

### 4. Our study area / India

- **Bist-Doab, Punjab**, *Hydrogeology Journal* (2026) ([Springer](https://link.springer.com/article/10.1007/s10040-026-03191-3)): MODFLOW plus ML (SVR, RF, ANN) to *correct the residuals* of a physical model. **The Bist Doab includes our Jalandhar and Kapurthala districts.**
  - It is a hybrid, but not a PINN, and it does not infer pumping.
  - Cite it and compare with it: it is the strongest local "physics + ML" precedent.
- **South-central Punjab**, *Physics and Chemistry of the Earth* (2024) ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1474706524000615)): RF/Bagging models forecast level changes, with RF best.
  - Supports using RF as a strong baseline (E3).
- **Quality-controlled India groundwater levels with specific yield**, *Scientific Data* (2025) ([Nature](https://www.nature.com/articles/s41597-025-05899-5)): it might provide **specific-yield priors** for S, alongside the NAQUIM reports.
  - Not yet read (the page needs a login redirect). Check it in M3.
- **No PINN study using Indian field well data was found.**

## What we did *not* find
1. **No PINN with zones drawn from well behaviour.** Zoning + deep learning exists only with LSTMs; PINN subdomains come from geometry (layers, regular splits).
2. **No PINN that infers unmetered pumping as an unknown field from real well observations.** Pumping is a known input, or the unknown is K or a contaminant source.
3. **No groundwater PINN on Indian (or Indo-Gangetic) field data.**

## Adjusted novelty wording (use in the paper and slides instead of CLAUDE.md §3; "to our knowledge" kept)
> To our knowledge, zoning wells by behaviour has so far been paired only with data-driven models (e.g. LSTMs), and groundwater PINNs either use one network per domain or split domains by geometry. We combine the two: zones learned from well behaviour, each with its own physics-informed model.
>
> We also treat net groundwater abstraction as a hidden, land-use-constrained source term inferred from sparse field observations. Previous groundwater PINNs take pumping as a known input or invert hydraulic conductivity only.
>
> We test this on a real, sparsely monitored Indian alluvial aquifer, with spatial block hold-outs.

**Wording to avoid:**
- "first ever"
- "zoning is new" (it is not)
- anything implying pumping is *measured* (see PLAN.md §13)

## Queries to repeat in Google Scholar (before writing the paper)
- `"physics-informed" groundwater cluster OR zonation OR zoning`
- `"physics-informed" groundwater pumping OR abstraction OR extraction inverse`
- `"physics-informed" groundwater India OR Punjab OR Haryana OR "Ganga"`
- `XPINN OR "domain decomposition" groundwater aquifer`
- `"differentiable" groundwater model pumping inversion` (the M2 fallback method may also have precedents)
