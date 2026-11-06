# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A round's own removal controls reach the bench it is graded on."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _integrate_bench as integrate_bench
from hyperloom.orchestrator.actions.executors import _ray_serving
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridRecipe,
    GridVariant,
    VariantResult,
    _build_variant_yaml,
)
from hyperloom.orchestrator.actions.executors._launch_evidence import RequestedLaunch, build_launch_evidence
from hyperloom.orchestrator.actions.executors._workload_envs import LaunchLayer
from hyperloom.orchestrator.actions.executors._integrate_results import _PatchBench
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.loop._integrate_promotion import _integrate_keep_variant
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.shared_state import EnablementRound, SharedState, inject_stack_base_params
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.orchestrator.tests._helpers import integrate_extra


_INHERITED_ARGS = "--tensor-parallel-size 8"
_BASELINE_VLLM = Path(__file__).parents[4] / "inference_optimizer" / "assets" / "configs" / "baseline_vllm.yaml"


def _baseline_config(tmp_path: Path) -> Path:
    """The shipped vLLM baseline with server args the recipe itself inherits, as an operator's --server-args leaves."""
    cfg = yaml.safe_load(_BASELINE_VLLM.read_text(encoding="utf-8"))
    cfg["benchmark"].setdefault("envs", {})["EXTRA_VLLM_ARGS"] = _INHERITED_ARGS
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _launched_args(recipe: GridRecipe, variant: GridVariant, slot: Path) -> tuple[Path, str]:
    path = _build_variant_yaml(recipe, variant, output_subdir=slot)
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]["envs"].get("EXTRA_VLLM_ARGS", "")


def _fake_grid(monkeypatch, config_path: Path, *, tput: float | None = None) -> list:
    grids: list = []

    async def fake_run_grid(*, grid, recipe, output_root, **_kwargs):
        grids.append(grid)
        slot = output_root / grid[0].name
        launched, _ = _launched_args(recipe, grid[0], slot)
        return [
            VariantResult(
                name=grid[0].name,
                extra_server_args="",
                extra_envs={},
                status="failed" if tput is None else "succeeded",
                output_throughput=tput,
                launch_evidence=build_launch_evidence(
                    RequestedLaunch(launched, "vllm"), actual_server_log=None, slot=slot
                ),
            )
        ]

    monkeypatch.setattr(integrate_bench, "run_grid", fake_run_grid)
    monkeypatch.setattr(integrate_bench, "prepare_grid_benchmark_config", lambda *_args, **_kwargs: config_path)
    monkeypatch.setattr(_ray_serving, "maybe_serving_lease", lambda **_kwargs: None)
    return grids


@pytest.mark.asyncio
async def test_the_bench_launches_without_what_the_round_removes(tmp_path: Path, monkeypatch):
    config_path = _baseline_config(tmp_path)
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


_BASE_ARGS = "--enable-foo --max-num-seqs 64"


async def _removal_only_round(tmp_path: Path, monkeypatch, *, tput: float | None = None) -> tuple[Task, dict, list]:
    session_dir = tmp_path / "session"
    workspace = session_dir / "runs" / "specialist" / "t-spec"
    workspace.mkdir(parents=True)
    (workspace / "specialist_done.json").write_text(json.dumps({"proposal_set": [], "patches_written": []}))
    config_path = _baseline_config(tmp_path)
    grids = _fake_grid(monkeypatch, config_path, tput=tput)
    params = {
        "specialist_task_id": "t-spec",
        "config_path": str(config_path),
        "framework": "vllm",
        "base_tput": 100.0,
        "base_extra_args": _BASE_ARGS,
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
        current_best={"tput": 100.0, "extra_server_args": _BASE_ARGS},
        get_specialist_patch_verdict=lambda _sid: "approve",
        save=lambda _path: None,
    )
    return task, await IntegratePatchExecutor(session_dir=session_dir)(ctx), grids


@pytest.mark.asyncio
async def test_a_removal_only_round_benches_without_the_removed_flag(tmp_path: Path, monkeypatch):
    _, result, grids = await _removal_only_round(tmp_path, monkeypatch)

    assert result["status"] != "no_patches", result.get("reason")
    assert [variant.remove_args for grid in grids for variant in grid] == [["--enable-foo"]]
    assert result["bench_result"]["effective_config"]["extra_server_args"] == "--max-num-seqs 64"


@pytest.mark.asyncio
@pytest.mark.parametrize("replayed", [False, True], ids=["live", "resume-replay"])
async def test_a_removal_only_keep_lifts_a_launch_without_the_removed_flag(tmp_path: Path, monkeypatch, replayed):
    task, result, _ = await _removal_only_round(tmp_path, monkeypatch, tput=120.0)
    assert result["status"] == "kept", result.get("reason")
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=100.0, current_best={"tput": 100.0, "extra_server_args": _BASE_ARGS}, model_path="/models/m"
    )

    variant = _integrate_keep_variant(result, None if replayed else task, result["bench_result"], 120.0, "")

    assert coord.writeback.lift_to_current_best("integrate_patch", 120.0, variant) is True
    params: dict = {}
    inject_stack_base_params(params, coord.shared_state)
    next_base = LaunchLayer(
        extra_server_args=params.get("base_extra_args", ""),
        remove_args=params.get("base_remove_args"),
        unset_envs=params.get("base_unset_envs"),
        args_mode=params.get("base_args_mode", "append"),
    )
    _, next_round = _launched_args(
        GridRecipe(base_yaml_path=Path(result["bench_result"]["materialized_config"]), base=next_base),
        GridVariant(name="next"),
        tmp_path / "next",
    )
    assert next_round == f"{_INHERITED_ARGS} --max-num-seqs 64"
    assert coord.writeback.current_best_launch_config()["remove_args"] == ["--enable-foo"]
