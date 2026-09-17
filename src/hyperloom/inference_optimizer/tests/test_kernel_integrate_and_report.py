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
from hyperloom.orchestrator.actions.executors import (
    ReportExecutor,
)
from hyperloom.orchestrator.roles import (
    MockBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.loop.sub_agent_runner import (
    SubAgentRunner,
)
from hyperloom.orchestrator.bus.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from hyperloom.orchestrator.state.task_registry import TaskRegistry
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.bus.storage import SqliteConnection


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    # Point HYPERLOOM_KERNEL_AGENT_ROOT at the kernel-agent tree so the handler resolves apply_kernel_patch.py.
    kernel_agent_root = Path(__file__).resolve().parents[4] / "src" / "hyperloom" / "agents" / "kernel"
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(kernel_agent_root))
    # Stub the interpreter resolver so the unit test never spawns a real probe.
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")
    from hyperloom.orchestrator.actions.executors import _benchmark_interpreter

    monkeypatch.setattr(
        _benchmark_interpreter,
        "_resolve_magpie_python",
        lambda: "/usr/bin/python3",
    )
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_silent() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {n: MockBackend(silent, name=n) for n in ("orchestration", "critic", "robustness")}


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
def graded_integrate_case(session_dir, tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "5")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", "1")
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)
    state = SharedState.load_or_init(session_dir)
    state.framework = "sglang"
    state.benchmark_mode = "agentx"
    state.baseline_tput = 100.0
    state.baseline_accuracy = 0.80
    state.baseline_perf = {
        "output_throughput": 100.0,
        "input_throughput": 900.0,
        "total_throughput": 1000.0,
        "e2e_norm_intvty_p90": 100.0,
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
        "tpot_p90_ms": 10.0,
        "completed_requests": 80,
        "submission_valid": True,
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
@pytest.mark.asyncio
async def test_integrate_handler_materializes_persisted_agentx_mode(session_dir, tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)
    state = SharedState.load_or_init(session_dir)
    state.framework = "sglang"
    state.benchmark_mode = "agentx"
    state.baseline_double_run = False
    state.baseline_tput = 100.0
    state.baseline_accuracy = 0.80
    state.baseline_perf = {
        "output_throughput": 100.0,
        "input_throughput": 900.0,
        "total_throughput": 1000.0,
        "e2e_norm_intvty_p90": 100.0,
    }
    state.current_best = {"action": "baseline", "tput": 100.0, **state.baseline_perf}
    state.save(session_dir)
    seen = {}
    measurement = {
        "status": "succeeded",
        "framework": "sglang",
        "output_throughput": 110.0,
        "input_throughput": 990.0,
        "total_token_throughput": 1100.0,
        "e2e_norm_intvty_p90": 110.0,
        "completed_requests": 80,
        "submission_valid": True,
        "accuracy": 0.80,
        "accuracy_task": "gsm8k",
        "accuracy_metric": "exact_match,strict-match",
    }

    async def _measure(executor, *, config_path, output_dir, **_kwargs):
        seen["shared_state"] = executor.shared_state
        seen["benchmark"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))["benchmark"]
        workspace = output_dir / "benchmark_sglang_e2e"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "server.log").write_text(f"INFO importing {target.name}\n", encoding="utf-8")
        measurement["workspace"] = str(workspace)
        measurement["report_path"] = str(workspace / "benchmark_report.json")
        Path(measurement["report_path"]).write_text(json.dumps(measurement), encoding="utf-8")
        return measurement

    monkeypatch.setattr(BaselineExecutor, "_run_single_benchmark", _measure)
    result = await krh.integrate_handler(
        {
            "base_tput": 100.0,
            "config_path": str(base_yaml),
            "kernel_id": "k_persisted_agentx",
            "patch_path": str(patch_file),
            "target_file": str(target),
            "allow_unknown_target": True,
            "skip_rebuild": True,
        },
        session_dir=session_dir,
    )

    assert seen["benchmark"]["benchmark_script"] == "aiperf_client.sh"
    assert seen["shared_state"] is not None
    assert seen["shared_state"].benchmark_mode == "agentx"
    assert yaml.safe_load(base_yaml.read_text(encoding="utf-8"))["benchmark"]["benchmark_script"] == "sglang_mi300x.sh"
    assert result["status"] == "ok"
    assert result["decision"] == "KEEP"
    assert result["graded_objective"] == "e2e_norm_intvty_p90"
    assert result["bench_result"] == measurement
    assert result["gain_pct"] == pytest.approx(10.0)


@pytest.fixture
def integrate_recipe_case(session_dir, tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
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
    state.benchmark_mode = "agentx"
    state.baseline_double_run = False
    state.baseline_tput = 100.0
    state.baseline_config_path = str(base_yaml)
    state.baseline_perf = {
        "output_throughput": 100.0,
        "input_throughput": 900.0,
        "total_throughput": 1000.0,
        "e2e_norm_intvty_p90": 100.0,
    }
    state.reference_server_args = "--disable-cuda-graph --max-running-requests 64"
    state.reference_envs = {"REFERENCE_DROP": "1", "REFERENCE_KEEP": "1"}
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

    async def _measure(executor, *, config_path, output_dir, **_kwargs):
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
            "completed_requests": 80,
            "submission_valid": True,
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

    assert result["status"] == "ok", result
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
    assert benchmarks[0]["benchmark_script"] == "aiperf_client.sh"
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

    assert result["status"] == "ok", result
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

    assert result["status"] == "ok", result
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

    assert result["status"] == "ok", result
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
        budget_minutes=1,
    )

    assert verdict is not None
    assert verdict.pairs == [(100.0, 110.0), (100.0, 110.0)]
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
@pytest.mark.parametrize(
    "output,total,intvty,stack,accuracy_outcome,decision",
    [
        pytest.param(90.0, 1100.0, 110.0, False, "pass", "KEEP", id="intvty-win-output-drop"),
        pytest.param(100.1, 1500.0, 153.0, True, "pass", "KEEP", id="stack-exact-2-percent-floor"),
        pytest.param(100.1, 1500.0, 154.5, True, "pass", "KEEP", id="stack-above-agentx-floor"),
        pytest.param(90.0, 1100.0, 110.0, False, "missing", "NEEDS_REVIEW", id="accuracy-missing"),
        pytest.param(90.0, 1100.0, 110.0, False, "regressed", "REVERT", id="accuracy-regressed"),
        pytest.param(90.0, 1100.0, 110.0, False, "failed", "NEEDS_REVIEW", id="accuracy-failed"),
    ],
)
async def test_integrate_handler_double_run_schedules_accuracy_on_graded_keep(
    session_dir, tmp_path, monkeypatch, output, total, intvty, stack, accuracy_outcome, decision
):
    from hyperloom.orchestrator.actions.executors import _server_lifecycle
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "5")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "0")
    monkeypatch.setenv("RUN_EVAL", "true")
    # AgentX currently falls back to one round; admit its script only in this
    # test to exercise deferred double-run scheduling, not enable AgentX reuse.
    monkeypatch.setattr(
        _server_lifecycle,
        "MAGPIE_BUILTIN_SCRIPTS",
        _server_lifecycle.MAGPIE_BUILTIN_SCRIPTS | {"aiperf_client.sh"},
    )
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)
    state = SharedState.load_or_init(session_dir)
    state.framework = "sglang"
    state.benchmark_mode = "agentx"
    state.baseline_double_run = True
    state.baseline_tput = 100.0
    state.baseline_accuracy = 0.80
    state.baseline_perf = {
        "output_throughput": 100.0,
        "input_throughput": 900.0,
        "total_throughput": 1000.0,
        "e2e_norm_intvty_p90": 100.0,
    }
    state.current_best = {"action": "baseline", "tput": 100.0, **state.baseline_perf}
    if stack:
        state.current_best.update(
            action="integrate", total_throughput=1500.0, input_throughput=1400.0, e2e_norm_intvty_p90=150.0
        )
        state.optimization_stack = [dict(state.current_best)]
    state.save(session_dir)
    rounds = []
    measured = {}

    async def _benchmark(executor, *, config_path, output_dir, **_kwargs):
        assert executor.shared_state.baseline_double_run is True
        assert target.read_text(encoding="utf-8") == patch_file.read_text(encoding="utf-8")
        bench = yaml.safe_load(config_path.read_text(encoding="utf-8"))["benchmark"]
        rounds.append((output_dir.name, bench))
        run_eval = bench["envs"]["RUN_EVAL"] == "true"
        if run_eval:
            assert output_dir.name == "accuracy_round"
            assert len(rounds) == 3
        round_output, round_total, round_intvty, round_tpot = {
            "warmup_round": (50.0, 500.0, 50.0, 20.0),
            "measure_round": (output, total, intvty, 10.0),
            "accuracy_round": (999.0, 9999.0, 1.0, 1000.0),
        }[output_dir.name]
        workspace = output_dir / "benchmark_sglang_e2e"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "server.log").write_text(f"INFO importing {target.name}\n", encoding="utf-8")
        result = {
            "status": "succeeded",
            "framework": "sglang",
            "output_throughput": round_output,
            "input_throughput": round_total - round_output,
            "total_token_throughput": round_total,
            "e2e_norm_intvty_p90": round_intvty,
            "tpot_p90_ms": round_tpot,
            "completed_requests": 80,
            "submission_valid": True,
            "workspace": str(workspace),
            "report_path": str(workspace / "benchmark_report.json"),
        }
        if run_eval and accuracy_outcome in {"pass", "regressed"}:
            score = 0.80 if accuracy_outcome == "pass" else 0.60
            accuracy_file = workspace / "results_gsm8k.json"
            accuracy_file.write_text(
                json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score}}}), encoding="utf-8"
            )
            result.update(
                accuracy=score,
                accuracy_task="gsm8k",
                accuracy_metric="exact_match,strict-match",
                accuracy_source=str(accuracy_file),
            )
        elif run_eval and accuracy_outcome == "failed":
            result.update(status="failed", error_class="subprocess_nonzero")
        Path(result["report_path"]).write_text(json.dumps(result), encoding="utf-8")
        if output_dir.name == "measure_round":
            measured.update(result)
        return result

    monkeypatch.setattr(BaselineExecutor, "_run_single_benchmark", _benchmark)
    result = await krh.integrate_handler(
        {
            "base_tput": 100.0,
            "config_path": str(base_yaml),
            "kernel_id": "k_deferred_accuracy",
            "patch_path": str(patch_file),
            "target_file": str(target),
            "allow_unknown_target": True,
            "skip_rebuild": True,
        },
        session_dir=session_dir,
    )

    assert result["status"] == "ok"
    assert [name for name, _ in rounds] == ["warmup_round", "measure_round", "accuracy_round"]
    assert [bench["envs"]["RUN_EVAL"] for _, bench in rounds] == ["false", "false", "true"]
    assert [bench["server_lifecycle"]["cleanup"] for _, bench in rounds] == [False, False, True]
    assert len({bench["envs"]["PORT"] for _, bench in rounds}) == 1
    assert len({bench["server_lifecycle"]["pid_dir"] for _, bench in rounds}) == 1
    assert result["decision"] == decision
    assert result["graded_objective"] == "e2e_norm_intvty_p90"
    reference_intvty = 150.0 if stack else 100.0
    assert result["gain_pct"] == pytest.approx((intvty - reference_intvty) / reference_intvty * 100.0)
    assert result.get("decision_reason") != "stack_positive_increment"
    assert result["new_tput"] == output
    assert "accuracy" not in measured
    assert all(result["bench_result"][key] == value for key, value in measured.items())
    stage = result["bench_result"]["accuracy_stage"]
    assert stage["status"] == ("failed" if accuracy_outcome == "failed" else "succeeded")
    assert "accuracy_round" in Path(stage["workspace"]).parts
    if decision == "KEEP":
        assert result["accuracy_pass"] is True
        assert result["accuracy"] == pytest.approx(0.80)
        assert target.read_text(encoding="utf-8") == patch_file.read_text(encoding="utf-8")
    else:
        assert result["decision_reason"] == (
            "accuracy_regression" if accuracy_outcome == "regressed" else "accuracy_evidence_missing"
        )
        assert result["revert_result"]["status"] == "ok"
        assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'original'\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("agentx_env", [True, False], ids=["env", "persisted-mode"])
@pytest.mark.parametrize(
    "output,total,intvty,decision",
    [
        pytest.param(110.0, 900.0, 90.0, "REVERT", id="both-axes-regress"),
        pytest.param(110.0, 1100.0, 90.0, "NEEDS_REVIEW", id="interactivity-tradeoff-recorded"),
        pytest.param(110.0, 1100.0, 100.0, "NEEDS_REVIEW", id="flat-interactivity-recorded"),
        pytest.param(110.0, 1000.0, 110.0, "KEEP", id="intvty-win"),
        pytest.param(90.0, 1000.0, 110.0, "KEEP", id="output-loss-intvty-win"),
        pytest.param(90.0, 950.0, 102.0, "KEEP", id="exact-floor-and-throughput-guard"),
        pytest.param(110.0, 949.9, 110.0, "NEEDS_REVIEW", id="throughput-guard-breach"),
        pytest.param(110.0, 1100.0, 101.99, "NEEDS_REVIEW", id="below-agentx-floor"),
    ],
)
async def test_integrate_handler_grades_full_e2e_measurement(
    session_dir, graded_integrate_case, monkeypatch, agentx_env, output, total, intvty, decision
):
    _, payload, measurement, target = graded_integrate_case
    if not agentx_env:
        monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    measurement.update(output_throughput=output, total_token_throughput=total, e2e_norm_intvty_p90=intvty)

    result = await krh.integrate_handler(payload, session_dir=session_dir)

    assert result["status"] == "ok"
    assert result["decision"] == decision
    assert result["graded_objective"] == "e2e_norm_intvty_p90"
    assert result["gain_pct"] == pytest.approx(intvty - 100.0)
    assert ("accuracy_gate" in result) is (decision == "KEEP")
    if decision == "REVERT":
        assert result["decision_reason"] == "intvty_regression"
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
    for axis in ("input_throughput", total_key) if missing_axis == "total" else ("e2e_norm_intvty_p90",):
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
        monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
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
        pytest.param(100.75, 1350.0, 135.0, "REVERT", id="stack-both-axes-regress"),
        pytest.param(100.1, 1507.5, 150.75, "NEEDS_REVIEW", id="stack-output-floor-does-not-apply"),
        pytest.param(100.1, 1507.35, 152.985, "NEEDS_REVIEW", id="stack-under-agentx-floor"),
        pytest.param(100.1, 1500.0, 153.0, "KEEP", id="stack-exact-agentx-floor"),
        pytest.param(100.1, 1500.0, 154.5, "KEEP", id="stack-positive-increment"),
        pytest.param(100.75, 1507.5, 135.0, "NEEDS_REVIEW", id="stack-interactivity-tradeoff-recorded"),
        pytest.param(100.1, 1424.9, 165.0, "NEEDS_REVIEW", id="stack-throughput-guard-breach"),
    ],
)
async def test_integrate_handler_grades_stack_increment_on_live_intvty_anchor(
    session_dir, graded_integrate_case, output, total, intvty, decision
):
    state, payload, measurement, _ = graded_integrate_case
    state.current_best.update(
        action="integrate", total_throughput=1500.0, input_throughput=1400.0, e2e_norm_intvty_p90=150.0
    )
    state.optimization_stack = [dict(state.current_best)]
    state.save(session_dir)
    measurement.update(output_throughput=output, total_token_throughput=total, e2e_norm_intvty_p90=intvty)

    result = await krh.integrate_handler(payload, session_dir=session_dir)

    assert result["decision"] == decision
    expected_gain = (intvty - 150.0) / 150.0 * 100.0
    assert result["gain_pct"] == pytest.approx(expected_gain)
    assert result["graded_objective"] == "e2e_norm_intvty_p90"
    assert result.get("decision_reason") != "stack_positive_increment"
    assert "stack_incremental_keep_threshold_pct" not in result
    assert ("accuracy_gate" in result) is (decision == "KEEP")
    if decision != "KEEP":
        assert result["revert_result"]["status"] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["accuracy", "submission_valid"])
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
    monkeypatch.setattr(
        krh,
        "_sweep_integrate_aiter_locks",
        lambda **_kwargs: next(sweeps),
    )
    calls = 0

    async def _executor(_ctx):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": "failed", "error_class": "timeout"}
        return {"status": "succeeded", "output_throughput": 100.0}

    result = await krh._run_integrate_rebaseline_with_lock_retry(
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
    monkeypatch.setattr(
        krh,
        "_sweep_integrate_aiter_locks",
        lambda **_kwargs: next(sweeps),
    )
    calls = 0

    async def _executor(_ctx):
        nonlocal calls
        calls += 1
        return {"status": "failed", "error_class": "timeout"}

    result = await krh._run_integrate_rebaseline_with_lock_retry(
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

    def _drop(envs=None, *, backup_dir=None):
        dropped.append({"envs": envs, "backup_dir": backup_dir})
        return {"action": "invalidate"}

    monkeypatch.setattr(krh, "_sweep_integrate_aiter_locks", lambda **_kwargs: {"scanned": 0, "deleted": 0})
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
    result = await krh._run_integrate_rebaseline_with_lock_retry(
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

    monkeypatch.setattr(krh, "_sweep_integrate_aiter_locks", lambda **_kwargs: {"scanned": 0, "deleted": 0})
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

    result = await krh._run_integrate_rebaseline_with_lock_retry(
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
    """Queued KEEPs may pass patch_path while relying on kernel_opt_attempts for source_file."""
    state = SharedState.load_or_init(session_dir)
    patch_path = tmp_path / "k001_opt.cu"
    source_path = tmp_path / "gemm_moe_ck2stages.cu"
    patch_path.write_text("// optimized\n", encoding="utf-8")
    source_path.write_text("// original\n", encoding="utf-8")
    state.last_kernel_opt = {
        "kernel_id": "k004",
        "best_artifact_path": "/tmp/k004_opt.cu",
        "source_file": "/tmp/rmsnorm.cu",
    }
    state.kernel_opt_attempts = {
        "k001": {
            "last_decision": "KEEP",
            "last_artifact_path": str(patch_path),
            "last_source_file": str(source_path),
        },
    }
    state.save(session_dir)

    resolved, err = krh._resolve_integrate_payload(
        {"kernel_id": "k001", "patch_path": str(patch_path)},
        session_dir=session_dir,
    )

    assert err is None
    assert resolved["patch_path"] == str(patch_path)
    assert resolved["source_file"] == str(source_path)


@pytest.mark.asyncio
async def test_integrate_handler_keep_decision(session_dir, tmp_path):
    """re-baseline returns 900 vs base 800 → KEEP."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_abc",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "KEEP"
    assert res["kernel_id"] == "k_abc"
    assert res["patch_path"] == str(patch_file)
    assert res["base_tput"] == 800.0
    assert res["new_tput"] == 900.0
    assert res["gain_pct"] == pytest.approx((900 - 800) / 800 * 100)
    assert "report_path" in res
    assert "workspace" in res


@pytest.mark.asyncio
async def test_integrate_handler_keeps_positive_stack_increment(
    session_dir,
    tmp_path,
):
    """When a kernel stack already exists, a positive incremental gain should KEEP."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    state = SharedState.load_or_init(session_dir)
    state.baseline_tput = 90.0
    state.current_best = {
        "action": "integrate",
        "kernel_id": "k004",
        "tput": 100.0,
    }
    state.optimization_stack = [
        {
            "action": "integrate",
            "kernel_id": "k004",
            "tput": 100.0,
        }
    ]
    state.save(session_dir)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=100.75)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput": 100.0,
        "config_path": str(base_yaml),
        "kernel_id": "k001",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "KEEP"
    assert res["gain_pct"] == pytest.approx(0.75)
    assert res["decision_reason"] == "stack_positive_increment"
    assert res["stack_incremental_gain_pct"] == pytest.approx(0.75)
    assert res["stack_incremental_keep_threshold_pct"] == pytest.approx(0.5)
    assert res["revert_result"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_integrate_handler_rejects_stack_increment_under_noise_floor(
    session_dir,
    tmp_path,
):
    """A sub-0.5% stack increment should remain NEEDS_REVIEW, not KEEP."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    state = SharedState.load_or_init(session_dir)
    state.baseline_tput = 90.0
    state.current_best = {
        "action": "integrate",
        "kernel_id": "k004",
        "tput": 100.0,
    }
    state.optimization_stack = [
        {
            "action": "integrate",
            "kernel_id": "k004",
            "tput": 100.0,
        }
    ]
    state.save(session_dir)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=100.49)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput": 100.0,
        "config_path": str(base_yaml),
        "kernel_id": "k001",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "NEEDS_REVIEW"
    assert res["gain_pct"] == pytest.approx(0.49)
    assert "decision_reason" not in res
    assert res["revert_result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_integrate_handler_keeps_exact_stack_increment_noise_floor(
    session_dir,
    tmp_path,
):
    """A +0.5% stack increment should KEEP at the configured noise floor."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    state = SharedState.load_or_init(session_dir)
    state.baseline_tput = 90.0
    state.current_best = {
        "action": "integrate",
        "kernel_id": "k004",
        "tput": 100.0,
    }
    state.optimization_stack = [
        {
            "action": "integrate",
            "kernel_id": "k004",
            "tput": 100.0,
        }
    ]
    state.save(session_dir)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=100.5)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput": 100.0,
        "config_path": str(base_yaml),
        "kernel_id": "k001",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "KEEP"
    assert res["stack_incremental_gain_pct"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_integrate_handler_accepts_valid_rebaseline_with_wrapper_warning(session_dir, tmp_path):
    """Valid throughput should drive KEEP even if Magpie reports success=false."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        workspace = _fake_workspace(slot, tput=900.0)
        report_path = workspace / "benchmark_report.json"
        data = json.loads(report_path.read_text())
        data["success"] = False
        report_path.write_text(json.dumps(data))
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_warn",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "KEEP"
    assert res["new_tput"] == 900.0


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


@pytest.mark.asyncio
async def test_integrate_handler_revert_decision(session_dir, tmp_path):
    """re-baseline returns 700 vs base 800 → REVERT."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)

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

    target, patch_file = _write_patch_pair(tmp_path)
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
    assert res["gain_pct"] < -1


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
async def test_integrate_handler_keeps_when_accuracy_holds(session_dir, tmp_path):
    """A throughput win whose accuracy stays within tolerance still KEEPs."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=900.0, accuracy=0.79),
    ):
        res = await krh.integrate_handler(
            _accuracy_payload(base_yaml, target, patch_file, "k_acc_ok"),
            session_dir=session_dir,
        )

    assert res["decision"] == "KEEP"
    assert res["accuracy_pass"] is True
    assert res["accuracy"] == pytest.approx(0.79)
    assert res["baseline_accuracy"] == pytest.approx(0.80)
    assert "decision_reason" not in res


@pytest.mark.asyncio
async def test_integrate_handler_reverts_on_accuracy_regression(session_dir, tmp_path):
    """A throughput win that loses accuracy beyond tolerance must REVERT."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=900.0, accuracy=0.60),
    ):
        res = await krh.integrate_handler(
            _accuracy_payload(base_yaml, target, patch_file, "k_acc_bad"),
            session_dir=session_dir,
        )

    assert res["gain_pct"] > 1, "the patch must be a throughput win for this to be a real gate test"
    assert res["decision"] == "REVERT"
    assert res["accuracy_pass"] is False
    assert res["decision_reason"] == "accuracy_regression"
    # A non-KEEP must restore the pristine source.
    assert res["revert_result"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'original'\n"


@pytest.mark.asyncio
async def test_integrate_handler_missing_accuracy_blocks_keep_when_baseline_known(session_dir, tmp_path):
    """A known baseline accuracy but no measured score is an evidence gap, not a regression."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=900.0, accuracy=None),
    ):
        res = await krh.integrate_handler(
            _accuracy_payload(base_yaml, target, patch_file, "k_acc_missing"),
            session_dir=session_dir,
        )

    assert res["decision"] == "NEEDS_REVIEW"
    assert res["accuracy_pass"] is None
    assert res["decision_reason"] == "accuracy_evidence_missing"


@pytest.mark.asyncio
async def test_integrate_handler_without_baseline_accuracy_keeps_on_throughput(session_dir, tmp_path):
    """No baseline accuracy (eval-less setup) degrades to a throughput-only KEEP."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=900.0, accuracy=None),
    ):
        res = await krh.integrate_handler(
            _accuracy_payload(base_yaml, target, patch_file, "k_acc_nobase"),
            session_dir=session_dir,
        )

    assert res["decision"] == "KEEP"
    assert res["accuracy_gate"]["degraded"] is True


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


@pytest.mark.asyncio
async def test_integrate_handler_accuracy_gate_opt_out(session_dir, tmp_path, monkeypatch):
    """``INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY=0`` restores throughput-only KEEP."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", "0")
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=900.0, accuracy=None),
    ):
        res = await krh.integrate_handler(
            _accuracy_payload(base_yaml, target, patch_file, "k_acc_optout"),
            session_dir=session_dir,
        )

    assert res["decision"] == "KEEP"


def _applyback_payload(base_yaml: Path, target: Path, patch_file: Path, kernel_id: str) -> dict:
    payload = _accuracy_payload(base_yaml, target, patch_file, kernel_id)
    payload["artifact_kind"] = "framework_applyback"
    payload["integration_validation_status"] = "pending"
    return payload


@pytest.mark.asyncio
async def test_applyback_keep_stamps_the_serving_validation_tier(session_dir, tmp_path):
    """Serving accuracy is what settles a reference-only artifact."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=900.0, accuracy=0.79),
    ):
        res = await krh.integrate_handler(
            _applyback_payload(base_yaml, target, patch_file, "k_ab_ok"),
            session_dir=session_dir,
        )

    assert res["decision"] == "KEEP"
    assert res["accuracy_pass"] is True
    assert res["artifact_kind"] == "framework_applyback"
    assert res["integration_validation_status"] == "passed"
    assert res["validation_tier"] == "integrate_e2e_accuracy"


@pytest.mark.asyncio
async def test_applyback_cannot_opt_out_of_the_accuracy_gate(session_dir, tmp_path, monkeypatch):
    """The operator opt-out cannot waive the only end-to-end evidence there is."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_REQUIRE_KERNEL_ACCURACY", "0")
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=900.0, accuracy=None),
    ):
        res = await krh.integrate_handler(
            _applyback_payload(base_yaml, target, patch_file, "k_ab_optout"),
            session_dir=session_dir,
        )

    assert res["gain_pct"] > 1, "the patch must be a throughput win for this to be a real gate test"
    assert res["decision"] == "NEEDS_REVIEW"
    assert res["decision_reason"] == "accuracy_evidence_missing"
    assert "integration_validation_status" not in res
    assert res["revert_result"]["status"] == "ok"


@pytest.mark.asyncio
async def test_applyback_without_baseline_accuracy_cannot_degrade_to_throughput(session_dir, tmp_path):
    """An eval-less setup degrades for a normal patch, but never for this one."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=900.0, accuracy=None),
    ):
        res = await krh.integrate_handler(
            _applyback_payload(base_yaml, target, patch_file, "k_ab_nobase"),
            session_dir=session_dir,
        )

    assert res["decision"] == "NEEDS_REVIEW"
    assert res["accuracy_gate"]["degraded"] is True
    assert res["accuracy_gate"]["blocked"] is True
    assert res["decision_reason"] == "accuracy_evidence_missing"


@pytest.mark.asyncio
async def test_applyback_accuracy_regression_reuses_the_shared_verdict(session_dir, tmp_path):
    """A regression keeps the existing reason vocabulary; artifact_kind separates it."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=900.0, accuracy=0.60),
    ):
        res = await krh.integrate_handler(
            _applyback_payload(base_yaml, target, patch_file, "k_ab_regress"),
            session_dir=session_dir,
        )

    assert res["decision"] == "REVERT"
    assert res["decision_reason"] == "accuracy_regression"
    assert res["artifact_kind"] == "framework_applyback"
    assert "integration_validation_status" not in res
    assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'original'\n"


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


def _runner_with_server_log(*, tput: float, accuracy: float | None, server_log: str):
    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        workspace = _fake_workspace(Path(cmd[out_idx + 1]), tput=tput, accuracy=accuracy)
        (workspace / "server.log").write_text(server_log, encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    return _fake_run


@pytest.mark.asyncio
async def test_multi_file_patch_records_per_file_import_evidence(session_dir, tmp_path):
    """Every file the patch wrote is graded, not just the primary target."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)
    payload = _applyback_payload(base_yaml, target, patch_file, "k_ab_multi")
    payload["patch_write_paths"] = [str(target), "vllm/flydsl_gemm.py"]

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner_with_server_log(
            tput=900.0,
            accuracy=0.79,
            server_log=f"INFO importing {target.stem}.py\nINFO importing flydsl_gemm.py\n",
        ),
    ):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["decision"] == "KEEP"
    assert res["source_import_confirmed"] is True
    assert res["source_import_evidence"] == {
        str(target): True,
        "vllm/flydsl_gemm.py": True,
    }


@pytest.mark.asyncio
async def test_multi_file_patch_that_never_loaded_loses_its_keep(session_dir, tmp_path):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)
    payload = _applyback_payload(base_yaml, target, patch_file, "k_ab_unloaded")
    payload["patch_write_paths"] = [str(target), "vllm/flydsl_gemm.py"]

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner_with_server_log(
            tput=900.0,
            accuracy=0.79,
            server_log="INFO serving an unrelated model\n",
        ),
    ):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["decision"] == "NEEDS_REVIEW"
    assert res["decision_reason"] == "source_not_confirmed_imported"
    assert res["source_import_confirmed"] is False
    # The downgrade replaces the KEEP, so the verdict stays unsettled.
    assert "integration_validation_status" not in res


@pytest.mark.asyncio
async def test_partly_traced_multi_file_patch_keeps_and_only_annotates(session_dir, tmp_path):
    """Lazy or folded-in modules must not cost a patch its KEEP."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)
    payload = _applyback_payload(base_yaml, target, patch_file, "k_ab_partial")
    payload["patch_write_paths"] = [str(target), "vllm/flydsl_gemm.py"]

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner_with_server_log(
            tput=900.0,
            accuracy=0.79,
            server_log=f"INFO importing {target.stem}.py\n",
        ),
    ):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["decision"] == "KEEP"
    assert res["integration_validation_status"] == "passed"
    assert "source_import_confirmed" not in res
    assert res["source_import_evidence"]["vllm/flydsl_gemm.py"] is False


@pytest.mark.asyncio
async def test_import_evidence_never_substitutes_for_accuracy(session_dir, tmp_path):
    """Fully-confirmed imports still cannot carry a patch past the accuracy gate."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)
    payload = _applyback_payload(base_yaml, target, patch_file, "k_ab_acc_over_import")
    payload["patch_write_paths"] = [str(target), "vllm/flydsl_gemm.py"]

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner_with_server_log(
            tput=900.0,
            accuracy=0.60,
            server_log=f"INFO importing {target.stem}.py\nINFO importing flydsl_gemm.py\n",
        ),
    ):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["source_import_confirmed"] is True
    assert res["decision"] == "REVERT"
    assert res["decision_reason"] == "accuracy_regression"


@pytest.mark.asyncio
async def test_single_file_patch_import_behaviour_is_unchanged(session_dir, tmp_path):
    """Without a declared write set the check still grades the one target."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner_with_server_log(
            tput=900.0,
            accuracy=0.79,
            server_log="INFO serving an unrelated model\n",
        ),
    ):
        res = await krh.integrate_handler(
            _accuracy_payload(base_yaml, target, patch_file, "k_single_import"),
            session_dir=session_dir,
        )

    assert res["decision"] == "NEEDS_REVIEW"
    assert res["decision_reason"] == "source_not_confirmed_imported"
    assert "source_import_evidence" not in res


@pytest.mark.asyncio
async def test_integrate_accuracy_verdict_lands_in_attempt_ledger(session_dir, tmp_path):
    """The attempt ledger must carry the accuracy evidence for post-hoc audit."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    _seed_baseline_accuracy(session_dir, 0.80)
    target, patch_file = _write_patch_pair(tmp_path)

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=_runner(tput=900.0, accuracy=0.60),
    ):
        res = await krh.integrate_handler(
            _accuracy_payload(base_yaml, target, patch_file, "k_acc_ledger"),
            session_dir=session_dir,
        )

    entry = SharedState().record_kernel_integrate_result(res)
    assert entry is not None
    attempt = entry["attempts"][-1]
    assert attempt["accuracy"] == pytest.approx(0.60)
    assert attempt["accuracy_pass"] is False
    assert attempt["decision_reason"] == "accuracy_regression"


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
async def test_integrate_handler_resolves_patch_and_target_from_state(
    session_dir,
    tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)
    state = SharedState(
        session_id=session_dir.name,
        last_kernel_opt={
            "kernel_id": "k006",
            "best_artifact_path": str(patch_file),
        },
        last_trace_analyze={
            "hot_kernels_top15": [
                {
                    "kernel_id": "k006",
                    "source_file": str(target),
                    "reusable_native_kernel": True,
                }
            ],
        },
    )
    state.save(session_dir)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k006",
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["status"] == "ok"
    assert res["decision"] == "KEEP"
    assert res["patch_path"] == str(patch_file)
    assert res["target_file"] == str(target)
    assert res["apply_result"]["status"] == "ok"
    assert res["finalize_result"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "def kernel():\n    return 'optimized'\n"


@pytest.mark.asyncio
async def test_integrate_handler_accepts_runtime_jit_deferred_apply(
    session_dir,
    tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    apply_result = {
        "status": "ok",
        "manifest_path": str(manifest),
        "target_file": str(target),
        "rebuild": {
            "status": "deferred",
            "mode": "runtime_jit",
        },
    }
    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k006",
        "patch_path": str(patch_file),
        "target_file": str(target),
    }
    with (
        patch.object(
            krh,
            "_maybe_apply_kernel_patch",
            return_value=apply_result,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=_fake_run,
        ),
    ):
        result = await krh.integrate_handler(payload, session_dir=session_dir)

    assert result["status"] == "ok"
    assert result["decision"] == "KEEP"
    assert result["apply_result"]["rebuild"] == {
        "status": "deferred",
        "mode": "runtime_jit",
    }


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
async def test_integrate_handler_injects_extra_server_args(
    session_dir,
    tmp_path,
):
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    seen: dict[str, object] = {}

    def _fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        seen["envs"] = cfg["benchmark"]["envs"]
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    target, patch_file = _write_patch_pair(tmp_path)
    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_good",
        "extra_server_args": "--cuda-graph-max-bs 8",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)

    assert res["decision"] == "KEEP"
    # extra_server_args preserved verbatim; watchdog timeout auto-appended.
    sglang_args = seen["envs"]["EXTRA_SGLANG_ARGS"]
    assert "--cuda-graph-max-bs 8" in sglang_args
    assert "--watchdog-timeout" in sglang_args


@pytest.mark.asyncio
async def test_integrate_handler_needs_review_when_within_threshold(
    session_dir,
    tmp_path,
):
    """re-baseline returns 805 (+0.625%) vs base 800 → NEEDS_REVIEW."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=805.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    payload = {
        "base_tput": 800.0,
        "config_path": str(base_yaml),
        "kernel_id": "k_review",
        "patch_path": str(patch_file),
        "target_file": str(target),
        "skip_rebuild": True,
    }
    with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
        res = await krh.integrate_handler(payload, session_dir=session_dir)
    assert res["decision"] == "NEEDS_REVIEW"


@pytest.mark.asyncio
async def test_integrate_handler_rejects_zero_base_tput(session_dir):
    res = await krh.integrate_handler({"base_tput": 0}, session_dir=session_dir)
    assert res["status"] == "failed"
    assert "base_tput" in res["error"]


def test_integrate_registered_under_two_aliases():
    assert krh.has_handler("integrate")
    assert krh.has_handler("apply_patch")
    assert krh.get_handler("integrate") is krh.get_handler("apply_patch")


# Coordinator wiring of integrate request
@pytest.mark.asyncio
async def test_coordinator_integrate_request_emits_keep_response(session_dir, tmp_path):
    """REQUEST{kind=integrate} → handler runs → RESPONSE carries KEEP/REVERT."""
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/profile.trace.json.gz",
            "reusable_native_kernel_ids": ["k1"],
        }
        c.shared_state.save(session_dir)
        with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
            await c._handle_intent(
                "orchestration",
                Intent(
                    type=IntentType.REQUEST,
                    payload={
                        "target_agent": "kernel_agent",
                        "kind": "integrate",
                        "params": {
                            "base_tput": 800.0,
                            "config_path": str(base_yaml),
                            "kernel_id": "k1",
                            "patch_path": str(patch_file),
                            "target_file": str(target),
                            "skip_rebuild": True,
                        },
                    },
                ),
            )
        responses = sorted(
            await c.bus.tail(topic="response", to_agent="orchestration"),
            key=lambda msg: msg.seq,
        )
        assert responses
        r = responses[0]
        assert r.payload["kind"] == "integrate_done"
        assert r.payload["status"] == "ok"
        result = r.payload["result"]
        assert result["decision"] == "KEEP"
        assert result["new_tput"] == 900.0
        assert c.shared_state.current_best["action"] == "integrate"
        assert c.shared_state.current_best["variant_name"] == "k1"
        assert any(
            item.get("action") == "integrate" and item.get("kernel_id") == "k1"
            for item in c.shared_state.optimization_stack
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_stops_repeating_same_kernel_integrate_after_cap(
    session_dir,
    tmp_path,
    monkeypatch,
):
    # Pin the legacy integrate dispatch cap (retire same kernel after N attempts) by opting out of the honest-E2E
    # path, which widens the cap.
    monkeypatch.setenv("HL_HONEST_E2E", "0")
    base_yaml = tmp_path / "base.yaml"
    _write_baseline_yaml(base_yaml)
    target, patch_file = _write_patch_pair(tmp_path)
    run_calls = 0

    def _fake_run(cmd, *args, **kwargs):
        nonlocal run_calls
        run_calls += 1
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=805.0)  # below KEEP threshold
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    c = Coordinator(session_dir, backends=_backends_silent())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/profile.trace.json.gz",
            "reusable_native_kernel_ids": ["k_repeat"],
        }
        c.shared_state.save(session_dir)
        payload = {
            "target_agent": "kernel_agent",
            "kind": "integrate",
            "params": {
                "base_tput": 800.0,
                "config_path": str(base_yaml),
                "kernel_id": "k_repeat",
                "patch_path": str(patch_file),
                "target_file": str(target),
                "skip_rebuild": True,
            },
        }
        with patch("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", side_effect=_fake_run):
            for _ in range(4):
                await c._handle_intent(
                    "orchestration",
                    Intent(type=IntentType.REQUEST, payload=payload),
                )

        responses = sorted(
            await c.bus.tail(topic="response", to_agent="orchestration"),
            key=lambda msg: msg.seq,
        )
        integrate_results = [r.payload["result"] for r in responses if r.payload.get("kind") == "integrate_done"]
        assert len(integrate_results) == 4
        # First 3 attempts run the integrate path; the 4th is short-circuited.
        assert run_calls > 0, "first 3 attempts must spawn subprocess"
        assert run_calls % 3 == 0, (
            f"first 3 attempts should contribute equal subprocess counts; "
            f"got {run_calls} (4th attempt should contribute 0)"
        )
        assert [r["decision"] for r in integrate_results[:3]] == [
            "NEEDS_REVIEW",
            "NEEDS_REVIEW",
            "NEEDS_REVIEW",
        ]
        assert integrate_results[-1]["status"] == "skipped"
        assert integrate_results[-1]["decision"] == "REVERT"
        assert integrate_results[-1]["error_class"] == "kernel_patch_rejected"

        saved = SharedState.load_or_init(session_dir)
        assert saved.rejected_kernel_patches
        assert saved.rejected_kernel_patches[0]["kernel_id"] == "k_repeat"
    finally:
        await c.stop()


# ReportExecutor
@pytest.mark.asyncio
async def test_report_executor_writes_md_and_json(session_dir):
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
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
            ),
        )
        # The real baseline action would have set this on completion; explore requires baseline_tput > 0
        # (execution_order) to be proposable next.
        c.shared_state.baseline_tput = 800.0
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "explore", "predicted_gain_pct": 5.0},
            ),
        )
        await c._handle_intent(
            "robustness",
            Intent(
                type=IntentType.ALERT,
                payload={"severity": "low", "summary": "noise"},
            ),
        )
        c.shared_state.save(session_dir)
    finally:
        await c.stop()

    db = SqliteConnection(tmp_path_helper(session_dir))
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    sub.register_executor("report", ReportExecutor())

    task = await tr.create(
        kind="report",
        params={"session_dir": str(session_dir)},
        idempotency_key="rep-1",
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
    assert summary["event_counts_by_topic"].get("proposal", 0) >= 2
    assert summary["event_counts_by_topic"].get("alert", 0) >= 1
    md_text = md.read_text()
    assert session_dir.name in md_text
    assert "## Throughput" in md_text
    assert "12.50%" in md_text
    assert "target_reached" in md_text


def tmp_path_helper(session_dir: Path) -> Path:
    """Point ReportExecutor's SqliteConnection at the session's DB."""
    return session_dir / "storage" / "coordinator.db"


@pytest.mark.asyncio
async def test_report_executor_failed_when_session_dir_unresolvable(tmp_path, monkeypatch):
    """An unresolvable session_dir yields a structured failure, not a crash."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "noses"))
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    sub.register_executor("report", ReportExecutor())
    task = await tr.create(
        kind="report",
        params={},
        idempotency_key="rep-fail-1",
    )
    res = await sub.run_task(task)
    db.close()
    assert res.state == "succeeded"
    assert res.result["status"] == "failed"
    assert "session_dir" in res.result.get("error", "")
