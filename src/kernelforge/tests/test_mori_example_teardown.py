"""The mori example's worker teardown ignores distributed runtime failures and surfaces anything else."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "mori_ep_dispatch_combine"


@pytest.fixture
def driver(monkeypatch):
    torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", torch.distributed)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", torch.multiprocessing)
    monkeypatch.syspath_prepend(str(_EXAMPLE_DIR))
    spec = importlib.util.spec_from_file_location("mori_example_driver_under_test", _EXAMPLE_DIR / "driver.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.dist.is_initialized.return_value = True
    return module


@pytest.mark.parametrize("failing_call", ["barrier", "destroy_process_group"])
def test_teardown_surfaces_an_unexpected_failure(driver, failing_call):
    getattr(driver.dist, failing_call).side_effect = ValueError("teardown bug")

    with pytest.raises(ValueError, match="^teardown bug$"):
        driver._dist_teardown(None, healthy=True)


def test_teardown_continues_past_a_runtime_failure_in_each_step(driver):
    calls = []

    def barrier():
        calls.append("barrier")
        raise RuntimeError("peer died")

    def destroy_process_group():
        calls.append("destroy")
        raise RuntimeError("already gone")

    driver.dist.barrier.side_effect = barrier
    driver.dist.destroy_process_group.side_effect = destroy_process_group

    driver._dist_teardown(None, healthy=True)

    assert calls == ["barrier", "destroy"]
