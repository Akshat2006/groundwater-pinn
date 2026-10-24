# Zoned PINNs for groundwater level forecasting

**Zoned physics-informed neural networks (PINNs) for groundwater forecasting in a sparsely monitored aquifer** (Punjab–Haryana alluvial belt, India).

- Wells are grouped into zones that behave alike.
- Each zone gets a PINN that fits the water levels, follows the groundwater flow equation, and infers unmetered pumping as a hidden term.

**Key documents:**
- [CLAUDE.md](CLAUDE.md): the project brief.
- [PLAN.md](PLAN.md): the build plan (milestones M0–M10).
- [data/raw/SOURCES.md](data/raw/SOURCES.md): where every dataset came from.

## Setup (laptop / college GPU)

```bash
conda env create -f environment.yml
conda activate pinn
pip install -e .
python scripts/check_env.py      # versions + whether the GPU really runs
python -m pytest -q              # all tests should pass
```

**Notes:**
- PyTorch is installed from the CUDA 12.6 wheels, which still support older GPUs such as the GTX 1050.
- `requirements-lock.txt` lists the exact versions that were tested (2026-10-17).

## Setup (Google Colab)

1. On the laptop, run `python scripts/pack_for_colab.py`, then upload `pinn_code.zip` to `MyDrive/pinn/` on Google Drive.
2. Open `notebooks/colab_setup.ipynb` in Colab with a GPU runtime and run all cells.

Run outputs are saved to `MyDrive/pinn/runs/`, so nothing is lost if the session dies.

## Data

The raw data (about 2 GB) is not in git. It is re-created by:

```bash
python scripts/fetch_wris_gwl.py         # CGWB water levels via India-WRIS (needs an Indian network)
python scripts/fetch_auxiliary.py all    # DEM, land use, canals, boundaries, IMD rainfall, Bhuvan LULC
```

The India Data Portal CSV and the CGWB PDF reports were downloaded by hand or with curl (see `data/raw/SOURCES.md`).

## Layout

```
configs/      YAML configs (study area, models, experiments)
src/          library code: data, features, zoning, physics, models, training, evaluation, scenarios
synthetic/    synthetic aquifer with known S, T, Q (M1–M2)
scripts/      command-line entry points (downloads, env check, Colab packing)
notebooks/    00_density_scan (G0 data check), colab_setup
tests/        pytest suite
reports/      results, tables, figures, run logs
```

## Status

| Milestone | Status |
|---|---|
| M0 setup + G0 data check + literature check | done (2026-10-17) |
| M1 synthetic aquifer (finite-difference solver) | done (2026-10-17), see `reports/m1_solver.md` |
| M2 single-zone PINN on synthetic data | gate passed on 1 of 3 seeds; diagnostics done: a 5,000-step settling stage (now default) rescues the realistic case (S −14%, T −17%); re-check batch `notebooks/colab_m2_recheck.ipynb` pending (`reports/m2_synthetic.md`) |
| M3 real-data pipeline | built (2026-10-22): `python scripts/build_master.py`, report `reports/m3_data_gaps.md` |
| M4 zoning | done (2026-10-24): `scripts/build_zones.py`, `check_m4.py`, `zoning_preview.py`; report `reports/m4_zoning.md`. 4 k-means zones picked by the early zoning preview on validation blocks (+3.6% on the pre-set measure, +17.6% on change over the years) |
| M5 baselines | done (2026-10-24): `scripts/run_baselines.py`, report `reports/m5_baselines.md`. Best: task A persistence 3.2 m; tasks B/C LightGBM 8.6 / 10.8 m |
