# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integrate kernel-request handler + report runner + e2e tests."""

from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.actions.executors.report import ReportExecutor
from hyperloom.orchestrator.roles import (
    MockBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.tests._session_base import freeze_session_base
from hyperloom.orchestrator.loop.sub_agent_runner import (
    SubAgentRunner,
)
from hyperloom.orchestrator.bus.resource_lock import (
    SqliteLeaseBackend,
)
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.bus.message_bus import Message

from .conftest import seed_kernel_keep
from hyperloom.orchestrator.kernel import _integrate_payload as integrate_payload
from hyperloom.orchestrator.kernel import _integrate_rebaseline as integrate_rebaseline


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    # Stub the interpreter resolver so the unit test never spawns a real probe.
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")
    from hyperloom.orchestrator.actions.executors import _benchmark_interpreter

    monkeypatch.setattr(
        _benchmark_interpreter,
        "_resolve_magpie_python",
        lambda _env=None: "/usr/bin/python3",
    )
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_silent() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {n: MockBackend(silent, name=n) for n in ("orchestration", "critic")}


def _write_baseline_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/x",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1},
            "benchmark_script": "sglang_mi300x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _fake_workspace(slot: Path, *, tput: float = 800.0, accuracy: float | None = None) -> Path:
    workspace = slot / "benchmark_sglang_smoke"
    workspace.mkdir(parents=True, exist_ok=True)
    if accuracy is not None:
        # Mimic the lm-eval output the serving GSM8K run leaves behind.
        (workspace / "results_gsm8k.json").write_text(
            json.dumps({"results": {"gsm8k": {"exact_match,strict-match": accuracy}}}),
            encoding="utf-8",
        )
    (workspace / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "sglang",
                "model": "/path/models/Qwen-Qwen3-8B",
                "throughput": {
                    "request_throughput": tput / 256,
                    "output_throughput": tput,
                    "total_token_throughput": tput * 2,
                    "completed_requests": 80,
                    "duration_seconds": 25.0,
                },
                "latency": {
                    "ttft": {"mean_ms": 140, "p99_ms": 158},
                    "e2el": {"mean_ms": 2500, "p99_ms": 2580},
                },
            }
        )
    )
    return workspace


def _write_patch_pair(
    tmp_path: Path,
    *,
    suffix: str = ".py",
    original: str = "def kernel():\n    return 'original'\n",
    optimized: str = "def kernel():\n    return 'optimized'\n",
) -> tuple[Path, Path]:
    target = tmp_path / f"kernel{suffix}"
    patch_file = tmp_path / f"optimized_kernel{suffix}"
    target.write_text(original, encoding="utf-8")
    patch_file.write_text(optimized, encoding="utf-8")
    return target, patch_file


@pytest.fixture
def graded_integrate_case(native_benchmark, session_dir, tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "5")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", "1")
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)
    state = SharedState.load_or_init(session_dir)
    state.framework = "sglang"
    state.benchmark_mode = "agentx"
    state.agentx_accuracy = False
    state.baseline_tput = 100.0
    state.baseline_accuracy = 0.80
    state.baseline_perf = {
        "workload_identity": "fixed-workload",
        "output_throughput": 100.0,
        "input_throughput": 900.0,
        "total_throughput": 1000.0,
        "e2e_norm_intvty_p90": 100.0,
        "e2e_norm_intvty_p50": 100.0,
        "duration_seconds": 900.0,
        "request_error_rate": 0.0,
    }
    state.current_best = {"action": "baseline", "tput": 100.0, **state.baseline_perf}
    state.save(session_dir)
    payload = {
        "base_tput": 100.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_graded",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "allow_unknown_target": True,
        "skip_rebuild": True,
    }
    measurement = {
        "status": "succeeded",
        "framework": "sglang",
        "output_throughput": 110.0,
        "input_throughput": 990.0,
        "total_token_throughput": 1100.0,
        "e2e_norm_intvty_p90": 110.0,
        "e2e_norm_intvty_p50": 110.0,
        "duration_seconds": 900.0,
        "request_error_rate": 0.0,
        "tpot_p90_ms": 10.0,
        "completed_requests": 80,
        "valid_measurement": True,
        "benchmark_backend": "magpie_agentx",
        "diagnostic_only": False,
        "workload_identity": "fixed-workload",
        "accuracy": 0.80,
        "accuracy_task": "gsm8k",
        "accuracy_metric": "exact_match,strict-match",
    }

    async def _measure(ctx):
        assert ctx.task.kind == "baseline"
        assert ctx.task.params["defer_accuracy_until_after_measure"] is True
        assert ctx.task.params["post_measure_accuracy_min_tput"] == 101.0
        assert ctx.task.params["post_measure_accuracy_keep_policy"] == {
            "base_tput": 100.0,
            "keep_threshold_pct": 1.0,
            "stack_incremental_keep_threshold_pct": 0.5,
        }
        assert target.read_text(encoding="utf-8") == patch_file.read_text(encoding="utf-8")
        workspace = Path(ctx.task.params["output_dir"]) / "benchmark_sglang_e2e"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "server.log").write_text(f"INFO importing {target.name}\n", encoding="utf-8")
        if "input_throughput" in measurement and "total_token_throughput" in measurement:
            measurement["input_throughput"] = measurement["total_token_throughput"] - measurement["output_throughput"]
        measurement["workspace"] = str(workspace)
        measurement["report_path"] = str(workspace / "benchmark_report.json")
        Path(measurement["report_path"]).write_text(json.dumps(measurement), encoding="utf-8")
        return measurement

    executor = AsyncMock(side_effect=_measure)
    monkeypatch.setattr(BaselineExecutor, "__call__", executor)
    yield state, payload, measurement, target
    executor.assert_awaited_once()


# integrate_handler


@pytest.fixture
def integrate_recipe_case(session_dir, tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.baseline.double_run_requested", lambda params: False)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session_dir))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SERVER_ARGS", "--disable-cuda-graph --mem-fraction-static 0.7")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", json.dumps({"OPERATOR_DROP": "1", "OPERATOR_KEEP": "1"}))
    base_yaml = tmp_path / "recipe.yaml"
    _write_baseline_yaml(base_yaml)
    cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    cfg["benchmark"]["envs"].update(
        EXTRA_SGLANG_ARGS="--disable-cuda-graph --cuda-graph-max-bs 8 --page-size 16",
        YAML_DROP="1",
        YAML_KEEP="1",
        SHARED="yaml",
        REENABLE="yaml",
    )
    base_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    state = SharedState.load_or_init(session_dir)
    state.framework = "sglang"
    state.benchmark_mode = "synthetic"
    state.baseline_tput = 100.0
    state.baseline_config_path = str(base_yaml)
    state.baseline_perf = {
        "output_throughput": 100.0,
        "input_throughput": 900.0,
        "total_throughput": 1000.0,
        "e2e_norm_intvty_p90": 100.0,
        "e2e_norm_intvty_p50": 100.0,
        "duration_seconds": 900.0,
        "request_error_rate": 0.0,
    }
    state.base_launch_digest = freeze_session_base(
        session_dir,
        args=("--disable-cuda-graph", "--max-running-requests", "64"),
        env={"REFERENCE_DROP": "1", "REFERENCE_KEEP": "1"},
    ).digest()
    state.current_best = {
        "action": "integrate",
        "tput": 100.0,
        **state.baseline_perf,
        "extra_server_args": "--page-size 32 --cuda-graph-max-bs 16",
        "extra_envs": {"CURRENT_ONLY": "1", "SHARED": "state"},
        "remove_args": ["--disable-cuda-graph", "--cuda-graph-max-bs 8"],
        "unset_envs": ["YAML_DROP", "REFERENCE_DROP", "OPERATOR_DROP", "REENABLE"],
        "args_mode": "replace",
    }
    state.save(session_dir)
    benchmarks = []

    async def _measure(executor, _attempt, *, config_path, output_dir, **_kwargs):
        benchmark = yaml.safe_load(config_path.read_text(encoding="utf-8"))["benchmark"]
        benchmarks.append(benchmark)
        tput = 100.0 if benchmark["envs"].get("LEG") == "A" else 110.0
        return {
            "status": "succeeded",
            "framework": "sglang",
            "output_throughput": tput,
            "input_throughput": tput * 9,
            "total_token_throughput": tput * 10,
            "e2e_norm_intvty_p90": 100.0,
            "e2e_norm_intvty_p50": 100.0,
            "duration_seconds": 900.0,
            "request_error_rate": 0.0,
            "completed_requests": 80,
            "valid_measurement": True,
        }

    monkeypatch.setattr(BaselineExecutor, "_run_single_benchmark", _measure)
    return state, base_yaml, benchmarks


@pytest.mark.asyncio
@pytest.mark.parametrize("args_mode", ["append", "replace"])
@pytest.mark.parametrize("controls_source", ["current-best", "request"])
async def test_integrate_handler_materializes_recipe_controls(
    session_dir, integrate_recipe_case, args_mode, controls_source
):
    state, base_yaml, benchmarks = integrate_recipe_case
    state.current_best["args_mode"] = args_mode
    controls = {key: deepcopy(state.current_best[key]) for key in ("remove_args", "unset_envs", "args_mode")}
    payload = {
        "kernel_id": "gemm_recipe",
        "source": "forge_gemm_tuning",
        "mode": "env_only",
        "extra_server_args": "",
        "extra_envs": {"CANDIDATE_ONLY": "1", "SHARED": "candidate", "REENABLE": "candidate"},
    }
    if controls_source == "request":
        payload.update(controls, extra_server_args="--page-size 64 --cuda-graph-max-bs 16")
        state.current_best.update(
            remove_args=["--page-size"],
            unset_envs=["CURRENT_ONLY"],
            args_mode="replace" if args_mode == "append" else "append",
        )
    state.save(session_dir)
    original_yaml = base_yaml.read_bytes()

    result = await krh.integrate_handler(payload, session_dir=session_dir)

    assert result["status"] == "ok"
    assert len(benchmarks) == 1
    envs = benchmarks[0]["envs"]
    args = envs["EXTRA_SGLANG_ARGS"]
    assert "--disable-cuda-graph" not in args
    assert "--cuda-graph-max-bs 8" not in args
    assert "--cuda-graph-max-bs 16" in args
    assert f"--page-size {64 if controls_source == 'request' else 32}" in args
    for inherited in ("--page-size 16", "--max-running-requests 64", "--mem-fraction-static 0.7"):
        assert (inherited in args) is (args_mode == "append")
    for key in ("YAML_DROP", "REFERENCE_DROP", "OPERATOR_DROP"):
        assert key not in envs
    for key in ("YAML_KEEP", "REFERENCE_KEEP", "OPERATOR_KEEP", "CURRENT_ONLY", "CANDIDATE_ONLY"):
        assert envs[key] == "1"
    assert envs["SHARED"] == "candidate"
    if controls_source == "request":
        assert "REENABLE" not in envs
    else:
        assert envs["REENABLE"] == "candidate"
    assert benchmarks[0]["benchmark_script"] == "sglang_mi300x.sh"
    assert base_yaml.read_bytes() == original_yaml


@pytest.mark.asyncio
async def test_integrate_handler_explicit_empty_controls_keep_inherited_recipe(session_dir, integrate_recipe_case):
    _state, _base_yaml, benchmarks = integrate_recipe_case

    result = await krh.integrate_handler(
        {
            "kernel_id": "gemm_empty_controls",
            "source": "forge_gemm_tuning",
            "mode": "env_only",
            "extra_envs": {"CANDIDATE_ONLY": "1"},
            "remove_args": [],
            "unset_envs": [],
            "args_mode": "",
        },
        session_dir=session_dir,
    )

    assert result["status"] == "ok"
    assert len(benchmarks) == 1
    envs = benchmarks[0]["envs"]
    for arg in (
        "--disable-cuda-graph",
        "--cuda-graph-max-bs 8",
        "--max-running-requests 64",
        "--mem-fraction-static 0.7",
    ):
        assert arg in envs["EXTRA_SGLANG_ARGS"]
    assert "--page-size 32" in envs["EXTRA_SGLANG_ARGS"]
    for key in ("YAML_DROP", "REFERENCE_DROP", "OPERATOR_DROP", "CURRENT_ONLY", "CANDIDATE_ONLY"):
        assert envs[key] == "1"


@pytest.mark.asyncio
async def test_integrate_handler_requested_unset_wins_over_current_and_explicit_envs(
    session_dir, integrate_recipe_case
):
    state, _base_yaml, benchmarks = integrate_recipe_case
    state.current_best["extra_envs"]["REENABLE"] = "state"
    state.save(session_dir)

    result = await krh.integrate_handler(
        {
            "kernel_id": "gemm_unset_current_env",
            "source": "forge_gemm_tuning",
            "mode": "env_only",
            "unset_envs": ["CURRENT_ONLY", "REENABLE"],
            "extra_envs": {"REENABLE": "candidate", "CANDIDATE_ONLY": "1"},
        },
        session_dir=session_dir,
    )

    assert result["status"] == "ok"
    assert len(benchmarks) == 1
    envs = benchmarks[0]["envs"]
    assert "CURRENT_ONLY" not in envs
    assert "REENABLE" not in envs
    assert envs["CANDIDATE_ONLY"] == "1"
    assert envs["SHARED"] == "state"


@pytest.mark.asyncio
async def test_integrate_handler_unsetting_last_current_env_still_materializes_recipe(
    session_dir, integrate_recipe_case
):
    state, _base_yaml, benchmarks = integrate_recipe_case
    state.current_best.update(
        extra_server_args="",
        extra_envs={"CURRENT_ONLY": "1"},
        remove_args=[],
        unset_envs=[],
        args_mode="append",
    )
    state.save(session_dir)

    result = await krh.integrate_handler(
        {
            "kernel_id": "gemm_unset_last_env",
            "source": "forge_gemm_tuning",
            "mode": "env_only",
            "unset_envs": ["CURRENT_ONLY"],
        },
        session_dir=session_dir,
    )

    assert result["status"] == "ok"
    assert len(benchmarks) == 1
    assert "CURRENT_ONLY" not in benchmarks[0]["envs"]
    assert result["new_tput"] == 110.0
    assert result["extra_envs"] == {}
    assert result["apply_result"]["reason"] == "env_only_no_patch_applied"


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_reference_controls", [False, True], ids=["frozen-controls", "empty-controls"])
async def test_gemm_paired_materializes_each_frozen_recipe_controls(
    session_dir, integrate_recipe_case, monkeypatch, empty_reference_controls
):
    from hyperloom.orchestrator.phases.kernel import KernelPhase

    state, base_yaml, benchmarks = integrate_recipe_case
    monkeypatch.setenv("HYPERLOOM_GEMM_PAIRED_PAIRS", "2")
    reference = {
        "tput": 100.0,
        "extra_server_args": "--page-size 8",
        "extra_envs": {"LEG": "A", "REFERENCE_ONLY": "1"},
        "remove_args": [] if empty_reference_controls else ["--disable-cuda-graph"],
        "unset_envs": [] if empty_reference_controls else ["YAML_DROP"],
        "args_mode": "" if empty_reference_controls else "append",
    }
    candidate = {
        "tput": 110.0,
        "extra_server_args": "--page-size 64 --disable-cuda-graph --cuda-graph-max-bs 16",
        "extra_envs": {"LEG": "B", "CANDIDATE_ONLY": "1", "REENABLE": "candidate"},
        "remove_args": ["--disable-cuda-graph"],
        "unset_envs": ["YAML_DROP", "REFERENCE_DROP", "OPERATOR_DROP", "REENABLE"],
        "args_mode": "replace",
    }
    state.current_best.update(
        extra_server_args="--page-size 128",
        extra_envs={"LIVE_ONLY": "1"},
        remove_args=["--page-size"],
        unset_envs=["REFERENCE_ONLY", "CANDIDATE_ONLY"],
        args_mode="replace",
    )
    state.save(session_dir)
    frozen = deepcopy((reference, candidate, state.current_best))
    payloads = []
    integrate = krh.integrate_handler

    async def _integrate(payload, **kwargs):
        payloads.append(deepcopy(payload))
        return await integrate(payload, **kwargs)

    monkeypatch.setattr(krh, "integrate_handler", _integrate)
    verdict = await KernelPhase._confirm_gemm_gain_paired(
        SimpleNamespace(session_dir=session_dir),
        reference,
        candidate,
        config_path=str(base_yaml),
    )

    assert verdict is not None
    assert verdict.median_delta_pct == 10.0 and verdict.reason == "candidate_faster"
    assert len(payloads) == len(benchmarks) == 4
    for idx, (payload, benchmark) in enumerate(zip(payloads, benchmarks)):
        recipe = reference if idx % 2 == 0 else candidate
        for key in ("remove_args", "unset_envs", "args_mode"):
            assert payload.get(key) == recipe[key]
        envs = benchmark["envs"]
        args = envs["EXTRA_SGLANG_ARGS"]
        assert "LIVE_ONLY" not in envs
        assert "--page-size 128" not in args
        assert envs["LEG"] == ("A" if idx % 2 == 0 else "B")
        if idx % 2 == 0:
            assert "--page-size 8" in args
            assert "--mem-fraction-static 0.7" in args
            assert ("--disable-cuda-graph" in args) is empty_reference_controls
            assert ("YAML_DROP" in envs) is empty_reference_controls
            assert envs["REFERENCE_ONLY"] == "1"
            assert "CANDIDATE_ONLY" not in envs
        else:
            assert "--page-size 64" in args
            assert "--cuda-graph-max-bs 16" in args
            for inherited in ("--disable-cuda-graph", "--page-size 16", "--mem-fraction-static 0.7"):
                assert inherited not in args
            for key in ("YAML_DROP", "REFERENCE_DROP", "OPERATOR_DROP", "REFERENCE_ONLY", "REENABLE"):
                assert key not in envs
            assert envs["CANDIDATE_ONLY"] == "1"
    assert (reference, candidate, state.current_best) == frozen
    assert SharedState.load_or_init(session_dir).current_best == frozen[2]


@pytest.mark.asyncio
@pytest.mark.parametrize("agentx_env", [True, False], ids=["env", "persisted-mode"])
@pytest.mark.parametrize(
    "output,total,intvty,decision,reason",
    [
        pytest.param(110.0, 900.0, 90.0, "REVERT", "p50_gain_below_threshold,p90_regressed", id="median-regresses"),
        pytest.param(110.0, 1100.0, 100.0, "REVERT", "p50_gain_below_threshold", id="median-flat"),
        pytest.param(110.0, 1100.0, 102.99, "REVERT", "p50_gain_below_threshold", id="median-just-below-the-bar"),
        pytest.param(110.0, 1100.0, 103.0, "KEEP", None, id="median-exactly-at-the-bar"),
        pytest.param(110.0, 1000.0, 110.0, "KEEP", None, id="median-clears-the-bar"),
        pytest.param(90.0, 1000.0, 110.0, "REVERT", "output_regressed", id="output-guard-breach"),
        pytest.param(110.0, 949.9, 110.0, "KEEP", None, id="total-no-longer-participates"),
    ],
)
async def test_integrate_handler_grades_full_e2e_measurement(
    session_dir, graded_integrate_case, monkeypatch, agentx_env, output, total, intvty, decision, reason
):
    _, payload, measurement, target = graded_integrate_case
    if not agentx_env:
        monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    measurement.update(
        output_throughput=output,
        total_token_throughput=total,
        e2e_norm_intvty_p90=intvty,
        e2e_norm_intvty_p50=intvty,
    )

    result = await krh.integrate_handler(payload, session_dir=session_dir)

    assert result["status"] == "ok"
    assert result["decision"] == decision
    assert result["graded_objective"] == "e2e_norm_intvty_p50"
    assert result["gain_pct"] == pytest.approx(intvty - 100.0)
    assert ("accuracy_gate" in result) is (decision == "KEEP")
    assert result.get("decision_reason") == reason
    assert result["base_tput"] == 100.0
    assert result["new_tput"] == output
    assert result["bench_result"] == measurement
    expected_source = Path(payload["patch_path"]).read_text(encoding="utf-8")
    if result["decision"] != "KEEP":
        expected_source = "def kernel():\n    return 'original'\n"
        assert result["revert_result"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == expected_source


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_from", ["candidate", "reference"])
@pytest.mark.parametrize("missing_axis", ["total", "intvty"])
@pytest.mark.parametrize(
    "output,stack",
    [
        pytest.param(200.0, False, id="large-output-gain"),
        pytest.param(50.0, False, id="output-regression"),
        pytest.param(100.75, True, id="stack-output-gain"),
    ],
)
async def test_integrate_handler_requires_requested_axes(
    session_dir, graded_integrate_case, missing_from, missing_axis, output, stack
):
    state, payload, measurement, target = graded_integrate_case
    measurement["output_throughput"] = output
    if stack:
        state.current_best["action"] = "integrate"
        state.optimization_stack = [dict(state.current_best)]
    incomplete = measurement if missing_from == "candidate" else state.current_best
    total_key = "total_token_throughput" if missing_from == "candidate" else "total_throughput"
    for axis in (
        ("input_throughput", total_key) if missing_axis == "total" else ("e2e_norm_intvty_p90", "e2e_norm_intvty_p50")
    ):
        incomplete.pop(axis)
    state.save(session_dir)

    result = await krh.integrate_handler(payload, session_dir=session_dir)

    assert result["status"] == "ok"
    assert result["decision"] == "NEEDS_REVIEW"
    assert result["graded_objective"] == "output_throughput"
    assert result["gain_pct"] == pytest.approx(output - 100.0)
    assert result["base_tput"] == 100.0
    assert result["new_tput"] == output
    assert result["bench_result"] == measurement
    assert "accuracy_gate" not in result
    assert result.get("decision_reason") != "stack_positive_increment"
    assert result["revert_result"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'original'\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_output", [True, False], ids=["explicit-output", "synthetic"])
@pytest.mark.parametrize("with_axes", [True, False], ids=["complete-axes", "output-only"])
@pytest.mark.parametrize(
    "output,decision",
    [(110.0, "KEEP"), (101.0, "NEEDS_REVIEW"), (98.0, "REVERT")],
)
async def test_integrate_handler_preserves_output_grading_and_threshold(
    session_dir, graded_integrate_case, monkeypatch, explicit_output, with_axes, output, decision
):
    state, payload, measurement, _ = graded_integrate_case
    if explicit_output:
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    else:
        monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
        state.benchmark_mode = ""
        state.save(session_dir)
    measurement.update(output_throughput=output, total_token_throughput=900.0, e2e_norm_intvty_p90=90.0)
    if not with_axes:
        for axis in ("input_throughput", "total_token_throughput", "e2e_norm_intvty_p90"):
            measurement.pop(axis)
        for axis in ("input_throughput", "total_throughput", "e2e_norm_intvty_p90"):
            state.current_best.pop(axis)
        state.save(session_dir)

    result = await krh.integrate_handler(payload, session_dir=session_dir)

    assert result["decision"] == decision
    assert result["graded_objective"] == "output_throughput"
    assert result["gain_pct"] == pytest.approx(output - 100.0)
    assert result["base_tput"] == 100.0
    assert result["new_tput"] == output
    assert result["bench_result"] == measurement


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output,total,intvty,decision",
    [
        pytest.param(100.75, 1350.0, 135.0, "REVERT", id="median-regresses"),
        pytest.param(100.1, 1507.5, 150.75, "REVERT", id="median-below-the-bar"),
        pytest.param(100.1, 1500.0, 154.5, "KEEP", id="median-exactly-at-the-bar"),
        pytest.param(100.1, 1560.0, 160.0, "KEEP", id="median-clears-the-bar"),
        pytest.param(94.0, 1500.0, 165.0, "REVERT", id="output-guard-breach"),
        pytest.param(100.1, 1424.9, 165.0, "KEEP", id="total-no-longer-participates"),
    ],
)
async def test_integrate_handler_grades_stack_increment_on_live_intvty_anchor(
    session_dir, graded_integrate_case, output, total, intvty, decision
):
    state, payload, measurement, _ = graded_integrate_case
    state.current_best.update(
        action="integrate",
        total_throughput=1500.0,
        input_throughput=1400.0,
        e2e_norm_intvty_p90=150.0,
        e2e_norm_intvty_p50=150.0,
    )
    state.optimization_stack = [dict(state.current_best)]
    state.save(session_dir)
    measurement.update(
        output_throughput=output,
        total_token_throughput=total,
        e2e_norm_intvty_p90=intvty,
        e2e_norm_intvty_p50=intvty,
    )

    result = await krh.integrate_handler(payload, session_dir=session_dir)

    assert result["decision"] == decision
    expected_gain = (intvty - 150.0) / 150.0 * 100.0
    assert result["gain_pct"] == pytest.approx(expected_gain)
    assert result["graded_objective"] == "e2e_norm_intvty_p50"
    assert result.get("decision_reason") != "stack_positive_increment"
    assert "stack_incremental_keep_threshold_pct" not in result
    assert ("accuracy_gate" in result) is (decision == "KEEP")
    if decision != "KEEP":
        assert result["revert_result"]["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["accuracy", "valid_measurement"])
async def test_integrate_handler_intvty_win_still_requires_valid_e2e_evidence(session_dir, graded_integrate_case, gate):
    _, payload, measurement, target = graded_integrate_case
    measurement[gate] = 0.60 if gate == "accuracy" else False

    result = await krh.integrate_handler(payload, session_dir=session_dir)

    assert result["decision"] == "REVERT"
    assert result["revert_result"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'original'\n"
    if gate == "accuracy":
        assert result["decision_reason"] == "accuracy_regression"
        assert result["accuracy_pass"] is False
        assert result["gain_pct"] == pytest.approx(10.0)
        assert result["bench_result"] == measurement
    else:
        assert result["error_class"] == "bench_exception"
        assert result["rebaseline_detail"] == measurement


@pytest.mark.asyncio
async def test_integrate_retries_once_after_stale_aiter_baton(tmp_path, monkeypatch):
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "[aiter] waiting for baton release at /root/.aiter/build/pa_ragged/lock\n",
        encoding="utf-8",
    )
    sweeps = iter(
        [
            {"scanned": 0, "deleted": 0, "errors": 0},
            {"scanned": 1, "deleted": 1, "errors": 0},
        ]
    )
    monkeypatch.setattr(integrate_rebaseline, "_sweep_integrate_aiter_locks", lambda **_kwargs: next(sweeps))
    calls = 0

    async def _executor(_ctx):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": "failed", "error_class": "timeout"}
        return {"status": "succeeded", "output_throughput": 100.0}

    result = await integrate_rebaseline._run_integrate_rebaseline_with_lock_retry(
        _executor,
        object(),
        workspace=tmp_path,
        reason="test integrate",
    )

    assert calls == 2
    assert result["status"] == "succeeded"
    assert result["stale_jit_lock_retry"]["retry_succeeded"] is True


@pytest.mark.asyncio
async def test_integrate_does_not_delete_or_retry_live_baton_owner(tmp_path, monkeypatch):
    (tmp_path / "server.log").write_text(
        "[aiter] waiting for baton release at /root/.aiter/build/pa_ragged/lock\n",
        encoding="utf-8",
    )
    sweeps = iter(
        [
            {"scanned": 0, "deleted": 0, "errors": 0},
            {
                "scanned": 0,
                "deleted": 0,
                "errors": 0,
                "skipped_live": True,
            },
        ]
    )
    monkeypatch.setattr(integrate_rebaseline, "_sweep_integrate_aiter_locks", lambda **_kwargs: next(sweeps))
    calls = 0

    async def _executor(_ctx):
        nonlocal calls
        calls += 1
        return {"status": "failed", "error_class": "timeout"}

    result = await integrate_rebaseline._run_integrate_rebaseline_with_lock_retry(
        _executor,
        object(),
        workspace=tmp_path,
        reason="test integrate",
    )

    assert calls == 1
    assert result["error_class"] == "stale_jit_lock"
    assert result["stale_jit_lock"]["retry_attempted"] is False


@pytest.mark.asyncio
async def test_integrate_retries_once_after_aiter_jit_registry_mismatch(tmp_path, monkeypatch):
    dropped: list[dict] = []
    extra_envs = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "/tmp/merged.csv"}

    def _drop(envs=None, *, backup_dir=None, also_modules=()):
        dropped.append({"envs": envs, "backup_dir": backup_dir, "also_modules": also_modules})
        return {"action": "invalidate"}

    monkeypatch.setattr(
        integrate_rebaseline, "_sweep_integrate_aiter_locks", lambda **_kwargs: {"scanned": 0, "deleted": 0}
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._aiter_jit.drop_serving_so_for_envs",
        _drop,
    )
    calls = 0

    async def _executor(_ctx):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "status": "failed",
                "error_class": "aiter_jit_registry_mismatch",
                "error": "kernel 'k' is not present in the compiled registry",
            }
        return {"status": "succeeded", "output_throughput": 100.0}

    ctx = types.SimpleNamespace(
        extra={},
        task=types.SimpleNamespace(params={"extra_envs": extra_envs}),
    )
    result = await integrate_rebaseline._run_integrate_rebaseline_with_lock_retry(
        _executor,
        ctx,
        workspace=tmp_path,
        reason="test integrate registry",
    )

    assert calls == 2
    assert dropped
    assert dropped[0]["envs"] == extra_envs
    assert result["status"] == "succeeded"
    assert result["aiter_jit_registry_mismatch_retry"]["retry_succeeded"] is True


@pytest.mark.asyncio
async def test_integrate_registry_retry_does_not_restamp_unrelated_retry_failure(tmp_path, monkeypatch):
    (tmp_path / "server.log").write_text(
        "kernel 'k' is not present in the compiled registry\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        integrate_rebaseline, "_sweep_integrate_aiter_locks", lambda **_kwargs: {"scanned": 0, "deleted": 0}
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._aiter_jit.drop_serving_so_for_envs",
        lambda *args, **kwargs: {"action": "invalidate"},
    )
    calls = 0

    async def _executor(_ctx):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "status": "failed",
                "error_class": "aiter_jit_registry_mismatch",
                "error": "kernel 'k' is not present in the compiled registry",
            }
        (tmp_path / "server.log").write_text(
            "kernel 'k' is not present in the compiled registry\nHIP out of memory\n",
            encoding="utf-8",
        )
        return {"status": "failed", "error_class": "oom", "error": "HIP out of memory"}

    result = await integrate_rebaseline._run_integrate_rebaseline_with_lock_retry(
        _executor,
        types.SimpleNamespace(
            extra={},
            task=types.SimpleNamespace(
                params={"extra_envs": {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "/tmp/merged.csv"}}
            ),
        ),
        workspace=tmp_path,
        reason="test integrate registry restamp",
    )

    assert calls == 2
    assert result["error_class"] == "oom"


def test_resolve_integrate_payload_fills_source_when_patch_path_present(
    session_dir,
    tmp_path,
):
    """A queued KEEP integrated by kernel_id and patch_path takes its source_file from the queued record."""
    state = SharedState.load_or_init(session_dir)
    patch_path = tmp_path / "k001_opt.cu"
    source_path = tmp_path / "gemm_moe_ck2stages.cu"
    patch_path.write_text("// optimized\n", encoding="utf-8")
    source_path.write_text("// original\n", encoding="utf-8")
    seed_kernel_keep(state, "k001", source_file=str(source_path), artifact=str(patch_path))
    state.save(session_dir)

    payload = integrate_payload._fill_integrate_defaults_from_state(
        {"kernel_id": "k001", "patch_path": str(patch_path)}, session_dir=session_dir
    )
    resolved, err = integrate_payload._resolve_integrate_payload(payload, session_dir=session_dir)

    assert err is None
    assert resolved["patch_path"] == str(patch_path)
    assert resolved["source_file"] == str(source_path)


@pytest.mark.asyncio
async def test_integrate_handler_rejects_rebaseline_that_exited_nonzero(session_dir, tmp_path):
    """A non-zero re-baseline cannot promote, however good its throughput looks."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        _fake_workspace(Path(cmd[out_idx + 1]), tput=900.0)
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="cleanup failed")

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_nonzero",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["decision"] != "KEEP"


# integrate_handler accuracy gate
def _accuracy_payload(base_yaml: Path, target: Path, patch_file: Path, kernel_id: str) -> dict:
    return {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": kernel_id,
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }


def _seed_baseline_accuracy(session_dir: Path, accuracy: float) -> None:
    state = SharedState.load_or_init(session_dir)
    state.baseline_accuracy = accuracy
    state.save(session_dir)


def _runner(*, tput: float, accuracy: float | None):
    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        _fake_workspace(Path(cmd[out_idx + 1]), tput=tput, accuracy=accuracy)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    return _fake_run


@pytest.mark.asyncio
async def test_integrate_handler_skips_accuracy_gate_on_throughput_loss(session_dir, tmp_path):
    """A candidate that lost throughput is never graded: no verdict is spent on it."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=700.0, accuracy=0.10),
    ):
        res = await krh.integrate_handler(
            _accuracy_payload(base_yaml, target, patch_file, "k_acc_slow"),
            session_dir=session_dir,
        )

    assert res["decision"] == "REVERT"
    assert "accuracy_gate" not in res
    assert "accuracy_pass" not in res


def _applyback_payload(base_yaml: Path, target: Path, patch_file: Path, kernel_id: str) -> dict:
    payload = _accuracy_payload(base_yaml, target, patch_file, kernel_id)
    payload["artifact_kind"] = "framework_applyback"
    payload["integration_validation_status"] = "pending"
    return payload


@pytest.mark.asyncio
async def test_applyback_losing_throughput_leaves_the_verdict_unsettled(session_dir, tmp_path):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=700.0, accuracy=0.79),
    ):
        res = await krh.integrate_handler(
            _applyback_payload(base_yaml, target, patch_file, "k_ab_slow"),
            session_dir=session_dir,
        )

    assert res["decision"] == "REVERT"
    assert "integration_validation_status" not in res
    assert "validation_tier" not in res


async def _integrate_with_accuracy_round(session_dir, tmp_path, monkeypatch, *, tput: float, kernel_id: str):
    """Integrate a kernel re-baselined at ``tput`` against a base of 800 tok/s, the session running the D9 round."""
    from hyperloom.orchestrator.actions.executors.benchmark_run import accuracy
    from hyperloom.orchestrator.actions.executors.benchmark_run.accuracy import AccuracyRoundOutcome

    calls = []

    async def run_candidate_accuracy_round(slot, launch, state, *, session):
        calls.append((slot, launch, session))
        return AccuracyRoundOutcome(slot / "accuracy", "")

    monkeypatch.setattr(accuracy, "accuracy_round_due", lambda state: True)
    monkeypatch.setattr(accuracy, "run_candidate_accuracy_round", run_candidate_accuracy_round)
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=tput, accuracy=0.79),
    ):
        res = await krh.integrate_handler(
            _accuracy_payload(base_yaml, target, patch_file, kernel_id), session_dir=session_dir
        )
    return res, calls


@pytest.mark.asyncio
async def test_a_kernel_below_the_threshold_runs_no_accuracy_round(session_dir, tmp_path, monkeypatch) -> None:
    res, calls = await _integrate_with_accuracy_round(
        session_dir, tmp_path, monkeypatch, tput=805.0, kernel_id="k_acc_slow"
    )

    assert calls == []
    assert res["decision"] != "KEEP"


@pytest.mark.asyncio
async def test_integrate_handler_invalid_rebaseline_is_retryable_fault(
    session_dir,
    tmp_path,
):
    """A failed re-baseline must route through the fault retry budget."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    def _fake_run(cmd, *args, **kwargs):
        # Zero-throughput workspace -> is_valid_measurement False.
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=0.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_fault",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    # error_class here is deliberately NOT in the fault whitelist, proving the status-based check saves the patch.
    assert res["status"] == "failed"
    assert res["decision"] == "REVERT"
    assert res["error"] == "re-baseline did not succeed"
    assert isinstance(res["error_class"], str) and res["error_class"]
    from hyperloom.orchestrator.state import shared_state as _ss

    assert res["error_class"] not in _ss._INTEGRATE_FAULT_ERROR_CLASSES

    # The envelope must be treated as a retryable fault, not a terminal REVERT.
    state = SharedState()
    entry = state.record_kernel_integrate_result(res)
    assert entry is not None
    assert entry["last_was_fault"] is True
    assert entry.get("retryable") is True
    assert "rejected" not in entry
    assert "k_fault" not in state.rejected_kernel_ids


@pytest.mark.asyncio
async def test_integrate_handler_reverts_applied_source_on_non_keep(
    session_dir,
    tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target = tmp_path / "kernel.py"
    patch_file = tmp_path / "optimized_kernel.py"
    target.write_text("def kernel():\n    return 'original'\n", encoding="utf-8")
    patch_file.write_text("def kernel():\n    return 'optimized'\n", encoding="utf-8")

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=700.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_bad",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["decision"] == "REVERT"
    assert res["apply_result"]["status"] == "ok"
    assert res["revert_result"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'original'\n"


@pytest.mark.asyncio
async def test_integrate_handler_fails_when_patch_inputs_missing(
    session_dir,
    tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)

    res = await krh.integrate_handler(
        {
            "base_tput": 800.0,
            "config_path": str(base_yaml),
            "kernel_id": "k_missing",
        },
        session_dir=session_dir,
    )

    assert res["status"] == "failed"
    assert res["decision"] == "REVERT"
    assert res["error_class"] == "missing_integration_inputs"
    assert "patch_path" in res["missing"]
    assert "target_file/source_file" in res["missing"]


@pytest.mark.asyncio
async def test_integrate_handler_rejects_text_patch_artifact(session_dir, tmp_path):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target = tmp_path / "kernel.py"
    patch_file = tmp_path / "optimized.txt"
    target.write_text("def kernel():\n    return 'original'\n", encoding="utf-8")
    patch_file.write_text("```python\ndef kernel():\n    return 'optimized'\n```\n", encoding="utf-8")

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_text",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "failed"
    assert res["decision"] == "REVERT"
    assert "complete source file" in res["apply_result"]["error"]
    assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'original'\n"


@pytest.mark.asyncio
async def test_integrate_handler_rejects_incompatible_standalone_cpp(
    session_dir,
    tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target = tmp_path / "kernel.cu"
    patch_file = tmp_path / "optimized.cu"
    target.write_text(
        "namespace aiter {\nvoid add_rmsnorm() {}\nvoid rmsnorm() {}\n}\n",
        encoding="utf-8",
    )
    patch_file.write_text(
        "#include <torch/extension.h>\n"
        "__global__ void optimized_kernel() {}\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}\n",
        encoding="utf-8",
    )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_standalone",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "failed"
    assert "PYBIND11" in res["apply_result"]["error"]
    assert "add_rmsnorm" in target.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_integrate_handler_rejects_zero_base_tput(session_dir):
    res = await krh.integrate_handler({"base_tput": 0}, session_dir=session_dir)
    assert res["status"] == "failed"
    assert "base_tput" in res["error"]


def test_integrate_registered_under_two_aliases():
    assert krh.KERNEL_REQUEST_HANDLERS["integrate"] is krh.KERNEL_REQUEST_HANDLERS["apply_patch"]


# Coordinator wiring of integrate request


# ReportExecutor
@pytest.mark.asyncio
@pytest.mark.parametrize("record_alert", [False, True], ids=["no-alert", "recorded-alert"])
async def test_report_executor_writes_md_and_json(session_dir, record_alert):
    """Run the report runner against seeded state + bus events; both files parse."""
    state = SharedState(
        session_id=session_dir.name,
        model_name="Qwen-Qwen3-8B",
        model_path="/path/models/Qwen-Qwen3-8B",
        cumulative_gain_validated=12.5,
        current_best={
            "action": "backends",
            "tput": 900.0,
            "ttft_mean_ms": 130.0,
            "e2el_mean_ms": 2400.0,
            "workspace": "/x/y/z",
        },
        max_minutes=120,
        stop_reason="target_reached",
    )
    state.save(session_dir)

    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        await c.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
            ),
        )
        # The real baseline action would have set this on completion, and PRELUDE would have handed over to
        # FRAMEWORK_AGENT, where explore is proposable.
        c.shared_state.baseline_tput = 800.0
        c.shared_state.phase = "FRAMEWORK_AGENT"
        await c.router.handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                # Admission refuses an explore with nothing to measure.
                payload={
                    "action_name": "explore",
                    "predicted_gain_pct": 5.0,
                    "params": {
                        "grid": [{"name": "chunked_prefill", "extra_server_args": "--chunked-prefill-size 4096"}]
                    },
                },
            ),
        )
        if record_alert:
            await c.bus.append_and_seq(
                Message.new(
                    from_agent="coordinator",
                    to_agent="orchestration",
                    topic="alert",
                    payload={"severity": "warning", "summary": "Recorded report alert"},
                )
            )
        c.shared_state.save(session_dir)
    finally:
        await c.stop()

    db = SqliteConnection(tmp_path_helper(session_dir))
    locks = SqliteLeaseBackend(db)
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr, session_dir=session_dir)
    sub.register_executor("report", ReportExecutor())

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="report",
            params={},
            idempotency_key="rep-1",
        ),
    )
    res = await sub.run_task(task)
    db.close()
    assert res.state == "succeeded"

    md = Path(res.result["md_path"])
    js = Path(res.result["json_path"])
    assert md.exists()
    assert js.exists()
    summary = json.loads(js.read_text())
    assert summary["session_id"] == session_dir.name
    assert summary["baseline_tput"] == 800.0
    assert summary["cumulative_gain_validated"] == 12.5
    assert summary["stop_reason"] == "target_reached"
    assert summary["event_counts_by_topic"].get("proposal", 0) == 2
    assert summary["event_counts_by_topic"].get("alert", 0) == int(record_alert)
    alerts = [item for item in summary["highlights"] if item["topic"] == "alert"]
    assert len(alerts) == int(record_alert)
    if record_alert:
        assert alerts[0]["summary"] == "sev=warning Recorded report alert"
    md_text = md.read_text()
    assert ("Recorded report alert" in md_text) is record_alert
    assert session_dir.name in md_text
    assert "## Throughput" in md_text
    assert "12.50%" in md_text
    assert "target_reached" in md_text


def tmp_path_helper(session_dir: Path) -> Path:
    """Point ReportExecutor's SqliteConnection at the session's DB."""
    return session_dir / "storage" / "coordinator.db"


def test_single_node_cpp_itfs_patch_takes_the_rebuild_verdict_of_its_backup(monkeypatch) -> None:
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    backup = {"is_cpp_itfs": True, "build_dir": "/aiter/jit/build", "module_names": ["module_custom"]}
    seen: list[dict] = []

    def verify(cpp_itfs_backup):
        seen.append(cpp_itfs_backup)
        return {"verified": False, "status": "stale_cpp_itfs_cache"}

    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)
    monkeypatch.setattr(integrate_rebaseline.patch_jit_cache, "verify_cpp_itfs_rebuilt", verify)

    result = integrate_rebaseline._cpp_itfs_rebuild_check({"cpp_itfs_cache_backup": backup})

    assert result == {"verified": False, "status": "stale_cpp_itfs_cache"}
    assert seen == [backup]
