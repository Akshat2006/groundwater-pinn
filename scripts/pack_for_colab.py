"""Zip the code (and data/processed, if any) for Google Colab.

    python scripts/pack_for_colab.py        -> pinn_code.zip; upload it to MyDrive/pinn/

Raw downloads, environments and run outputs are left out; Colab only needs the processed data.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}   # anywhere
SKIP_PREFIXES = [("data", "raw"), ("data", "interim"), ("data", "synthetic"), ("reports", "runs")]
SKIP_SUFFIXES = {".pyc", ".pt", ".ckpt", ".log", ".zip", ".docx"}   # .docx: personal documents in the repo root


def main() -> None:
    out = ROOT / "pinn_code.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT)
            if p.is_dir() or SKIP_DIRS & set(rel.parts[:-1]) or p.suffix in SKIP_SUFFIXES:
                continue
            if any(rel.parts[:2] == pre for pre in SKIP_PREFIXES) or "egg-info" in str(rel):
                continue
            z.write(p, rel)
            n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
