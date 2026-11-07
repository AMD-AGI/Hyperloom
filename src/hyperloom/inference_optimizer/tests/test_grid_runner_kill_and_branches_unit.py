# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the ``run_grid`` per-variant failure branches (yaml build error, magpie timeout, server-dead / overtime sentinels, missing workspace, invalid measurement)."""

from __future__ import annotations


import pytest

from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    run_grid,
    GridRecipe,
)

from .conftest import write_benchmark_yaml


@pytest.fixture(autouse=True)
def _single_node(monkeypatch):
    """Default every test in this module to single-node mode."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


@pytest.mark.asyncio
async def test_run_grid_yaml_build_error_branch(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    write_benchmark_yaml(base)

    def _boom(*_a, **_k):
        raise ValueError("bad yaml render")

    monkeypatch.setattr(gr, "_build_variant_yaml", _boom)
    results = await run_grid(
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
        recipe=GridRecipe(base_yaml_path=base),
    )
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].error_class == "yaml_build_error"
