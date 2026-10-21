"""M3: build the master well table and gridded inputs from data/raw (see src/data/merge.py).

    python scripts/build_master.py        -> data/processed/{master,wells,telemetry_daily}.parquet,
                                             static_1km.nc, rain_daily.nc
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.merge import build_all  # noqa: E402
from src.utils.config import data_dir, load_config  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "study_area.yaml")
    summary = build_all(cfg, data_dir() / "raw", data_dir() / "processed")
    (data_dir() / "processed" / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
