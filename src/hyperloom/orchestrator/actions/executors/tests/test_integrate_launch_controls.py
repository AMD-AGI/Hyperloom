# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round's own removal controls reach the bench it is graded on."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _integrate_bench as integrate_bench
from hyperloom.orchestrator.actions.executors import _ray_serving
from hyperloom.orchestrator.actions.executors._grid_runner import VariantResult
from hyperloom.orchestrator.actions.executors._integrate_results import _PatchBench
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor


def _fake_grid(monkeypatch, config_path: Path) -> list:
    grids: list = []

    async def fake_run_grid(*, grid, **_kwargs):
        grids.append(grid)
        return [VariantResult(name=grid[0].name, extra_server_args="", extra_envs={}, status="failed")]

    monkeypatch.setattr(integrate_bench, "run_grid", fake_run_grid)
    monkeypatch.setattr(integrate_bench, "prepare_grid_benchmark_config", lambda *_args, **_kwargs: config_path)
    monkeypatch.setattr(_ray_serving, "maybe_serving_lease", lambda **_kwargs: None)
    return grids


@pytest.mark.asyncio
async def test_the_bench_launches_without_what_the_round_removes(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text("benchmark: {}\n", encoding="utf-8")
    grids = _fake_grid(monkeypatch, config_path)

    bench, _ = await IntegratePatchExecutor(session_dir=tmp_path)._bench_patch(
        _PatchBench(
            params={
                "config_path": str(config_path),
                "framework": "vllm",
                "base_extra_args": "--enable-foo --max-num-seqs 64",
                "base_unset_envs": ["BASE_OFF"],
            },
            output_root=tmp_path / "out",
            specialist_task_id="t-spec",
            launch_controls={"remove_args": ["--enable-foo"], "unset_envs": ["VLLM_FOO"]},
        ),
        extra_server_args_applied="",
        extra_envs_applied={},
    )

    (variant,) = grids[0]
    assert (variant.remove_args, variant.unset_envs) == (["--enable-foo"], ["VLLM_FOO"])
    assert bench["effective_config"] == {
        "extra_envs": {},
        "extra_server_args": "--max-num-seqs 64",
        "remove_args": ["--enable-foo"],
        "unset_envs": ["BASE_OFF", "VLLM_FOO"],
        "args_mode": "append",
    }
