# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round's own removal controls reach the bench it is graded on."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors import _integrate_bench as integrate_bench
from hyperloom.orchestrator.actions.executors import _ray_serving
from hyperloom.orchestrator.actions.executors._grid_runner import VariantResult
from hyperloom.orchestrator.actions.executors._integrate_results import _PatchBench
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.shared_state import EnablementRound
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.orchestrator.tests._helpers import integrate_extra


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


@pytest.mark.asyncio
async def test_a_removal_only_round_benches_without_the_removed_flag(tmp_path: Path, monkeypatch):
    session_dir = tmp_path / "session"
    workspace = session_dir / "runs" / "specialist" / "t-spec"
    workspace.mkdir(parents=True)
    (workspace / "specialist_done.json").write_text(json.dumps({"proposal_set": [], "patches_written": []}))
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text("benchmark: {}\n", encoding="utf-8")
    grids = _fake_grid(monkeypatch, config_path)
    params = {
        "specialist_task_id": "t-spec",
        "config_path": str(config_path),
        "framework": "vllm",
        "base_extra_args": "--enable-foo --max-num-seqs 64",
        "remove_args": ["--enable-foo"],
    }
    task = Task(
        task_id="t-int",
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key="t-int",
        requires_lanes=(),
    )
    ctx = RunnerContext(task=task, lease=None, extra=integrate_extra(params))
    ctx.extra["shared_state"] = SimpleNamespace(
        phase="FRAMEWORK_AGENT",
        enablement=EnablementRound(),
        framework="vllm",
        benchmark_mode="synthetic",
        current_best={"tput": 100.0},
        get_specialist_patch_verdict=lambda _sid: "approve",
        save=lambda _path: None,
    )

    result = await IntegratePatchExecutor(session_dir=session_dir)(ctx)

    assert result["status"] != "no_patches", result.get("reason")
    assert [variant.remove_args for grid in grids for variant in grid] == [["--enable-foo"]]
    assert result["bench_result"]["effective_config"]["extra_server_args"] == "--max-num-seqs 64"
