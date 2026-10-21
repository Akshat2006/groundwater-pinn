"""Print what this machine can run: versions, GPU, and whether CUDA kernels actually execute.

    python scripts/check_env.py
"""
import importlib
import platform
import sys

PACKAGES = ["numpy", "scipy", "pandas", "xarray", "geopandas", "rasterio", "pyproj", "shapely",
            "imdlib", "sklearn", "lightgbm", "pykrige", "matplotlib", "torch"]


def main() -> int:
    print(f"python   {sys.version.split()[0]}  ({platform.platform()})")
    missing = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"{name:<10} {getattr(mod, '__version__', '?')}")
        except ImportError:
            missing.append(name)
            print(f"{name:<10} MISSING")

    import torch
    print(f"\ntorch CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA GPU: not available (CPU only)")
    else:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"CUDA GPU: {name}, compute capability {cap[0]}.{cap[1]}, {mem:.1f} GB")
        try:   # some builds detect old GPUs but ship no kernels for them
            a = torch.randn(512, 512, device="cuda")
            torch.tanh(a @ a).sum().item()
            print("CUDA kernels: OK")
        except RuntimeError as e:
            print(f"CUDA kernels: FAILED ({e})")
            return 1
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
