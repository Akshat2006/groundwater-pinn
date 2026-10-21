"""Checks that the environment has everything later milestones need."""
import importlib

import pytest

MODULES = ["numpy", "scipy", "pandas", "pyarrow", "xarray", "netCDF4", "geopandas", "rasterio", "pyproj",
           "shapely", "imdlib", "sklearn", "lightgbm", "pykrige", "yaml", "matplotlib", "torch"]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_utm43n_transform():
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True).transform(75.0, 30.0)
    # 75°E is the zone's central meridian, which gets the 500 km "false easting"
    assert abs(x - 500_000) < 1e-3 and 3_310_000 < y < 3_330_000


def test_torch_autograd_second_derivative():
    """PINNs need second derivatives through autograd: d²/dx² of sin(x) = -sin(x)."""
    import torch
    x = torch.linspace(0, 3, 50, dtype=torch.float64, requires_grad=True)
    y = torch.sin(x)
    (dy,) = torch.autograd.grad(y.sum(), x, create_graph=True)
    (d2y,) = torch.autograd.grad(dy.sum(), x)
    assert torch.allclose(d2y, -torch.sin(x).detach(), atol=1e-12)


@pytest.mark.gpu
def test_cuda_kernels_run():
    """Not just is_available(): older GPUs can be detected but have no compiled kernels."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    a = torch.randn(256, 256, device="cuda")
    x = torch.randn(256, device="cuda", requires_grad=True)
    torch.tanh(a @ x).sum().backward()
    assert torch.isfinite(x.grad).all()
