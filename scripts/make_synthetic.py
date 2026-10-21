"""Build the synthetic aquifers (M1) and their well designs.

    python scripts/make_synthetic.py            # all cases: A B C D
    python scripts/make_synthetic.py A C

Writes data/synthetic/case_X/truth.nc and wells_<design>.parquet (the clean designs listed in each case
config; noisy variants for M2 are made on the fly with synthetic.sample_wells).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import data_dir, load_config  # noqa: E402
from synthetic.generate import build_case, save_truth  # noqa: E402
from synthetic.sample_wells import sample_wells  # noqa: E402


def main(cases: list[str]) -> None:
    for case in cases:
        t0 = time.time()
        cfg = load_config(ROOT / "configs" / "synthetic" / f"case_{case}.yaml")
        ds = build_case(cfg)
        out = data_dir() / "synthetic" / f"case_{case}"
        save_truth(ds, out / "truth.nc")
        for name, d in (cfg.get("designs") or {}).items():
            wells = sample_wells(ds, d["n_wells"], d["schedule"], seed=cfg["seed"])
            wells.to_parquet(out / f"wells_{name}.parquet", index=False)
        print(f"case {case}: {out}  ({time.time() - t0:.0f}s, max budget error "
              f"{float(ds.budget_rel_error.max()):.1e})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B", "C", "D"])
