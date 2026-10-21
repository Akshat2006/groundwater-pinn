import json

import numpy as np
import pytest
import yaml

from src.utils.config import REPO_ROOT, deep_merge, load_config, set_seed
from src.utils.runlog import start_run


def test_deep_merge_overrides_nested_without_mutating():
    base = {"a": 1, "net": {"width": 32, "depth": 6}}
    out = deep_merge(base, {"net": {"width": 48}, "b": 2})
    assert out == {"a": 1, "b": 2, "net": {"width": 48, "depth": 6}}
    assert base["net"]["width"] == 32


def test_load_config_resolves_defaults(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"lr": 1e-3, "net": {"width": 32, "depth": 6}}))
    (tmp_path / "run.yaml").write_text(yaml.safe_dump({"defaults": ["base.yaml"], "net": {"width": 64}}))
    cfg = load_config(tmp_path / "run.yaml", overrides={"lr": 5e-4})
    assert cfg == {"lr": 5e-4, "net": {"width": 64, "depth": 6}}


def test_study_area_config_is_valid():
    cfg = load_config(REPO_ROOT / "configs" / "study_area.yaml")
    assert sum(len(d) for d in cfg["districts"].values()) == 16
    p = cfg["period"]
    assert p["train"][1] < p["val"][0] <= p["val"][1] < p["test"][0]


def test_set_seed_is_reproducible():
    set_seed(7); a = np.random.rand(3)
    set_seed(7); b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)


def test_set_seed_torch():
    torch = pytest.importorskip("torch")
    set_seed(3); a = torch.rand(3)
    set_seed(3); b = torch.rand(3)
    assert torch.equal(a, b)


def test_start_run_writes_config_meta_and_metrics(tmp_path):
    run = start_run("unit", {"x": 1}, root=tmp_path)
    run.log_metrics({"rmse": 0.5})
    run.log_metrics({"nse": 0.9})
    assert yaml.safe_load((run.dir / "config.yaml").read_text()) == {"x": 1}
    assert json.loads((run.dir / "meta.json").read_text())["name"] == "unit"
    assert json.loads((run.dir / "metrics.json").read_text()) == {"rmse": 0.5, "nse": 0.9}
    assert start_run("unit", {}, root=tmp_path).dir != run.dir
