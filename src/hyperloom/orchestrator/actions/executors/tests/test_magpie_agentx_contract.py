# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native AgentX contracts through real CLI, materialization and result boundaries."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from hyperloom.common.perf_metric import constraint_veto_reason, graded_axes_of
from hyperloom.inference_optimizer.cli.benchmark_config import bind_benchmark, configure_benchmark
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.orchestrator.actions.executors._grid_runner import GridVariant, _build_variant_yaml
from hyperloom.orchestrator.actions.executors._workload_envs import materialize_benchmark_config
from hyperloom.orchestrator.actions.executors.benchmark_result import (
    extract_benchmark_measurement,
    is_valid_measurement,
)
from hyperloom.orchestrator.actions.executors.magpie_agentx import bind_source_evidence, enable_profile, load_benchmark


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    monkeypatch.setenv(BENCHMARK_CONFIG_ENV, "")
    path = tmp_path / "workload.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "Qwen/Qwen3-0.6B",
                    "precision": "bf16",
                    "runner_type": "mi355x",
                    "docker_image": "example/sglang:pinned",
                    "agentx": {"enabled": True, "mode": "canonical"},
                    "envs": {"TP": "1", "EP_SIZE": "1", "CONC": "8"},
                    "profiler": {"torch_profiler": {"enabled": False, "num_steps": 5, "num_profiles": 3}},
                }
            }
        )
    )
    return path


def test_yaml_binds_workload_and_is_snapshotted(source, tmp_path):
    args = _build_parser().parse_args(["optimize", "--benchmark-config", str(source)])
    configure_benchmark(args)
    assert (args.framework, str(args.model), args.tp, args.ep, args.conc) == ("sglang", "Qwen/Qwen3-0.6B", 1, 1, 8)
    assert args.enable_conc_sweep is False
    session = tmp_path / "session"
    session.mkdir()
    bind_benchmark(session, args)
    assert (session / "benchmark.yaml").read_text() == source.read_text()


def test_agentx_switch_accepts_explicit_yaml(source, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    args = _build_parser().parse_args(["optimize", "--benchmark-config", str(source)])
    configure_benchmark(args)
    assert args.conc == 8


def test_materializer_does_not_turn_native_workload_into_synthetic(source, tmp_path, monkeypatch):
    monkeypatch.setenv("ISL", "1")
    monkeypatch.setenv("OSL", "1")
    monkeypatch.setenv("CONC", "64")
    rendered = materialize_benchmark_config(
        source,
        tmp_path / "round",
        gpu_type="mi300x",
        extra_server_args="--disable-cuda-graph",
        extra_envs={"SGLANG_TEST": "1"},
    )
    bench = load_benchmark(rendered)
    assert bench["runner_type"] == "mi355x"
    assert "benchmark_script" not in bench
    assert bench["envs"] == {"TP": "1", "EP_SIZE": "1", "CONC": "8", "RUN_EVAL": "false"}
    launch = bench["agentx"]["launch_overrides"]
    assert launch["append_args"] == ["--disable-cuda-graph"]
    assert launch["env"] == {"SGLANG_TEST": "1"}


def test_grid_composes_accepted_launch_then_candidate(source, tmp_path):
    base = materialize_benchmark_config(
        source, tmp_path / "base", extra_server_args="--chunked-prefill-size 4096", extra_envs={"SGLANG_TEST": "1"}
    )
    variant = GridVariant("candidate", "--chunked-prefill-size 2048", {"SGLANG_NEXT": "1"}, unset_envs=["SGLANG_TEST"])
    candidate = _build_variant_yaml(base, "", variant, output_subdir=tmp_path / "candidate")
    launch = load_benchmark(candidate)["agentx"]["launch_overrides"]
    assert launch["append_args"] == ["--chunked-prefill-size", "2048"]
    assert launch["env"] == {"SGLANG_NEXT": "1"}
    assert launch["unset_env"] == ["SGLANG_TEST"]
    assert load_benchmark(base)["agentx"]["launch_overrides"]["env"] == {"SGLANG_TEST": "1"}


@pytest.mark.parametrize("appended,recipe", [(True, False), (False, True), (True, True)])
def test_grid_ablation_removes_the_option_from_its_owning_layers(source, tmp_path, report, appended, recipe):
    from hyperloom.inference_optimizer.agentx.report import native_launch_evidence
    from hyperloom.orchestrator.actions.executors._launch_evidence import persist_launch_evidence

    option = "--cuda-graph-max-bs"
    base = materialize_benchmark_config(source, tmp_path / "base")
    receipt = report["agentx_metrics"]["server_launch"]
    receipt["base_argv"] = list(receipt["effective_argv"]) + ([option, "128"] if recipe else [])
    persist_launch_evidence(native_launch_evidence(extract_benchmark_measurement(report), base), slot=base.parent)
    candidate = _build_variant_yaml(
        base,
        f"{option}=256" if appended else "",
        GridVariant("ablate", "--mem-fraction-static 0.85", {}, remove_args=[option]),
        output_subdir=tmp_path / "candidate",
    )
    overrides = load_benchmark(candidate)["agentx"]["launch_overrides"]
    assert overrides["append_args"] == ["--mem-fraction-static", "0.85"]
    assert overrides["remove_args"] == ([option] if recipe else [])


def test_recipe_removal_survives_readding_and_ablating_an_override(source, tmp_path):
    base = materialize_benchmark_config(
        source, tmp_path / "base", remove_args=["--cuda-graph-max-bs"], extra_server_args="--cuda-graph-max-bs 256"
    )
    candidate = _build_variant_yaml(
        base,
        "",
        GridVariant("ablate", "", {}, remove_args=["--cuda-graph-max-bs"]),
        output_subdir=tmp_path / "candidate",
    )
    overrides = load_benchmark(candidate)["agentx"]["launch_overrides"]
    assert overrides["append_args"] == []
    assert overrides["remove_args"] == ["--cuda-graph-max-bs"]


def test_source_evidence_is_taken_after_application_and_survives_materialization(source, tmp_path):
    root = tmp_path / "framework"
    root.mkdir()
    module = root / "kernel.py"
    module.write_text("new kernel")
    patch = tmp_path / "candidate.patch"
    patch.write_text(
        "diff --git a/kernel.py b/kernel.py\n--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-old\n+new kernel\n"
    )
    rendered = materialize_benchmark_config(source, tmp_path / "run")
    bind_source_evidence(rendered, [{"framework_root": str(root), "patch_file": str(patch)}])
    evidence = load_benchmark(rendered)["agentx"]["launch_overrides"]["source_files"]
    assert str(module) in evidence
    next_round = materialize_benchmark_config(rendered, tmp_path / "next", extra_server_args="--disable-cuda-graph")
    assert load_benchmark(next_round)["agentx"]["launch_overrides"]["source_files"] == evidence


@pytest.fixture
def report():
    return {
        "scenario": "agentx",
        "success": True,
        "benchmark_valid": True,
        "publishable": True,
        "framework": "sglang",
        "model": "Qwen/Qwen3-0.6B",
        "agentx_metrics": {
            "mode": "canonical",
            "execution_owner": "magpie",
            "server_launch": {
                "effective_argv": ["python3", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen3-0.6B"],
                "effective_env": {},
            },
            "scenario_type": "agentic",
            "dataset": "fixed-corpus",
            "recipe": {"tp": 1, "ep": 1, "conc": 8, "model": "Qwen/Qwen3-0.6B"},
            "requests": {"total": 100, "successful": 100, "error_rate": 0.0},
            "throughput": {
                "output_tokens_per_second": 50.0,
                "total_tokens_per_second": 100.0,
                "duration_seconds": 3600,
            },
            "latency_seconds": {"e2e_norm_intvty": {"p50": 25.0, "p90": 10.0}, "ttft": {"p50": 0.2}},
        },
    }


@pytest.mark.parametrize(
    "watts,started,veto",
    [
        (800.0, 1000.0, ""),
        (950.0, 1000.0, "power_budget_exceeded"),
        (None, 1000.0, "power_unmeasured"),
        (800.0, 500.0, "power_unmeasured"),
    ],
)
def test_native_power_reaches_baseline_and_keep_budgets(report, tmp_path, watts, started, veto):
    from hyperloom.orchestrator.measurement.baseline_budget import baseline_budget_stop_reason

    workspace = tmp_path / "benchmark_sglang_round"
    workspace.mkdir()
    (tmp_path / "gpu_power.json").write_text(
        json.dumps(
            {
                "started_unix": started,
                "avg_power_w": watts,
                "serving_gpus": [0],
                "per_gpu": {"0": {"avg_power_w": watts}},
            }
        )
    )
    report["gpu_monitor"] = {"sample_count": 10, "power_watts": {"avg": 100.0}}
    measurement = extract_benchmark_measurement(report, workspace=workspace, subprocess_started_unix=1000.0)
    state = SimpleNamespace(agentx_backend="magpie", baseline_perf=graded_axes_of(measurement), power_budget_w=900.0)
    assert constraint_veto_reason(measurement, state) == veto
    assert baseline_budget_stop_reason(state, measurement) == ("baseline_over_power_budget" if veto else "")
    if watts is not None and started == 1000.0:
        assert measurement["gpu_power_avg_w"] == watts
        assert measurement["gpu_power_by_gpu_w"] == {"0": watts}


def test_public_metrics_and_identity_reach_keep_guards(report):
    measured = extract_benchmark_measurement(report)
    assert measured["e2e_norm_intvty_p50"] == 25
    assert measured["ttft_p50_ms"] == 200
    state = SimpleNamespace(agentx_backend="magpie", baseline_perf=graded_axes_of(measured))
    assert constraint_veto_reason(measured, state) == ""
    changed = deepcopy(report)
    changed["agentx_metrics"]["recipe"]["conc"] = 16
    assert constraint_veto_reason(extract_benchmark_measurement(changed), state) == "agentx_workload_mismatch"


@pytest.mark.parametrize("key", ["success", "benchmark_valid", "publishable"])
def test_rejected_report_cannot_be_salvaged(report, key, tmp_path):
    report[key] = False
    (tmp_path / "results.json").write_text('{"output_throughput": 9000, "completed": 100}')
    measured = extract_benchmark_measurement(report, workspace=tmp_path)
    assert not is_valid_measurement(measured)
    assert measured["output_throughput"] == 50


def test_profile_uses_same_workload_and_is_never_keep_evidence(source, tmp_path, report, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")
    monkeypatch.setenv("HYPERLOOM_SGLANG_SHAPE_MODE", "patched")
    monkeypatch.delenv("PROFILE_OSL", raising=False)
    monkeypatch.delenv("HYPERLOOM_PROFILE_MAX_ITERS", raising=False)
    monkeypatch.delenv("HYPERLOOM_PROFILE_DELAY_ITERS", raising=False)
    rendered = materialize_benchmark_config(source, tmp_path / "profile")
    enable_profile(rendered)
    bench = load_benchmark(rendered)
    torch_profiler = bench["profiler"]["torch_profiler"]
    assert torch_profiler["enabled"] is True
    assert torch_profiler["num_profiles"] == 3
    assert torch_profiler["num_steps"] == 5
    assert "start_seconds" not in torch_profiler
    assert "PROFILE_EXTRA_BODY" not in bench["envs"]
    assert bench["model"] == "Qwen/Qwen3-0.6B"
    assert str(bench["envs"]["CONC"]) == "8"
    assert "OSL" not in bench["envs"]
    assert load_benchmark(source)["profiler"]["torch_profiler"] == {
        "enabled": False,
        "num_steps": 5,
        "num_profiles": 3,
    }
    report["agentx_metrics"]["diagnostic_only"] = True
    measured = extract_benchmark_measurement(report)
    assert measured["round_succeeded"] is True
    assert not is_valid_measurement(measured)
    assert constraint_veto_reason(measured, SimpleNamespace()) == "agentx_evidence_invalid"


@pytest.mark.parametrize("checkout", ["default", "yaml", "environment"])
def test_native_baseline_runs_through_the_real_executor(source, tmp_path, monkeypatch, report, checkout):
    import asyncio
    import subprocess

    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
    from hyperloom.orchestrator.state.shared_state import SharedState

    monkeypatch.setenv(BENCHMARK_CONFIG_ENV, str(source))
    monkeypatch.setenv("INFERENCEX_PATH", "/synthetic/InferenceX")
    monkeypatch.delenv("MAGPIE_INFERENCEX_PATH", raising=False)
    if checkout == "yaml":
        config = yaml.safe_load(source.read_text())
        config["benchmark"]["inferencex_path"] = "/native/InferenceX"
        source.write_text(yaml.safe_dump(config))
    elif checkout == "environment":
        monkeypatch.setenv("MAGPIE_INFERENCEX_PATH", "/native/InferenceX")
    report["agentx_metrics"]["server_launch"] = {
        "effective_argv": [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "Qwen/Qwen3-0.6B",
            "--disable-cuda-graph",
        ],
        "effective_env": {},
    }
    observed = {}

    def run(cmd, **kwargs):
        config = Path(cmd[cmd.index("--benchmark-config") + 1])
        observed.update(load_benchmark(config))
        assert kwargs["env"].get("MAGPIE_INFERENCEX_PATH") == (
            "/native/InferenceX" if checkout == "environment" else None
        )
        assert observed.get("inferencex_path") == ("/native/InferenceX" if checkout == "yaml" else None)
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        workspace = slot / "benchmark_sglang_contract"
        workspace.mkdir(parents=True)
        (workspace / "benchmark_report.json").write_text(json.dumps(report))
        return subprocess.CompletedProcess(cmd, 0, "complete", "")

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", run)
    state = SharedState(benchmark_mode="agentx", agentx_backend="magpie", enablement_mode="off")
    ctx = SimpleNamespace(
        task=SimpleNamespace(
            task_id="native",
            params={
                "output_dir": str(tmp_path / "round"),
                "timeout_sec": 10,
                "model_path": "Qwen/Qwen3-0.6B",
                "extra_server_args": "--disable-cuda-graph",
            },
        ),
        extra={"shared_state": state},
    )
    executor = BaselineExecutor(magpie_python="python3", default_config_path=source, session_dir=tmp_path)
    result = asyncio.run(executor(ctx))
    assert result["status"] == "succeeded", result
    assert result["valid_measurement"] is True
    assert result["launch_evidence"]["magpie_server_launch"] == report["agentx_metrics"]["server_launch"]
    assert observed["agentx"]["launch_overrides"]["append_args"] == ["--disable-cuda-graph"]
    assert "PROFILE" not in observed["envs"]


@pytest.mark.parametrize(
    "report", [None, {}, {"success": True, "throughput": {"output_throughput": 9000, "completed": 100}}]
)
def test_native_missing_report_cannot_salvage_synthetic_results(source, tmp_path, report):
    (tmp_path / "results.json").write_text('{"output_throughput": 9000, "completed": 100}')
    measured = extract_benchmark_measurement(report, workspace=tmp_path, config_path=source)
    assert measured["benchmark_backend"] == "magpie_agentx"
    assert not is_valid_measurement(measured)


@pytest.mark.asyncio
@pytest.mark.parametrize("throughput,interactivity,expected", [(60.0, 30.0, "kept"), (40.0, 20.0, "reverted")])
async def test_applied_source_is_measured_and_transactionally_kept_or_reverted(
    source, report, tmp_path, monkeypatch, throughput, interactivity, expected
):
    import hashlib
    import subprocess

    from hyperloom.orchestrator.actions.executors import _grid_runner, _ray_serving
    from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
    from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
    from hyperloom.orchestrator.state.shared_state import SharedState
    from hyperloom.orchestrator.state.task_registry import Task
    from hyperloom.orchestrator.tests._helpers import init_git_repo, patch_integrate_patch_roots
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    patch_integrate_patch_roots(monkeypatch, tmp_path)
    monkeypatch.setenv(BENCHMARK_CONFIG_ENV, str(source))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
    monkeypatch.setattr(_ray_serving, "maybe_serving_lease", lambda **kw: None)
    session = tmp_path / "session"
    session.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    before = (repo / "src.py").read_bytes()
    specialist = runs_dir(session, "specialist", "source-test")
    specialist.mkdir(parents=True)
    patch = specialist / "candidate.patch"
    patch.write_text(
        "diff --git a/src.py b/src.py\n--- a/src.py\n+++ b/src.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
    )
    (specialist / "specialist_done.json").write_text(
        json.dumps({"patches_written": [str(patch)], "summary": "source candidate"})
    )
    anchor = extract_benchmark_measurement(report)
    state = SharedState(
        framework="sglang",
        benchmark_mode="agentx",
        agentx_backend="magpie",
        enablement_mode="off",
        current_best={**anchor, "tput": 50.0},
        baseline_perf=graded_axes_of(anchor),
        baseline_tput=50.0,
    )
    state.record_specialist_patch_verdict("source-test", "approve")
    observed = []

    def run(cmd, **kwargs):
        config = Path(cmd[cmd.index("--benchmark-config") + 1])
        launch = load_benchmark(config)["agentx"]["launch_overrides"]
        assert (repo / "src.py").read_text().endswith("return 2\n")
        assert (
            launch["source_files"][str(repo / "src.py")] == hashlib.sha256((repo / "src.py").read_bytes()).hexdigest()
        )
        observed.append(config)
        measured = deepcopy(report)
        measured["agentx_metrics"]["throughput"]["output_tokens_per_second"] = throughput
        measured["agentx_metrics"]["latency_seconds"]["e2e_norm_intvty"] = {"p50": interactivity, "p90": 10.0}
        measured["agentx_metrics"]["server_launch"]["source_files"] = launch["source_files"]
        workspace = Path(cmd[cmd.index("--output-dir") + 1]) / "benchmark_sglang_source"
        workspace.mkdir(parents=True)
        (workspace / "benchmark_report.json").write_text(json.dumps(measured))
        return subprocess.CompletedProcess(cmd, 0, "complete", "")

    monkeypatch.setattr(_grid_runner, "run_with_session_kill", run)
    task = Task(
        task_id="source-round",
        kind="integrate_patch",
        state="queued",
        idempotency_key="source-round",
        params={
            "specialist_task_id": "source-test",
            "framework_source_root": str(repo),
            "framework": "sglang",
            "config_path": str(source),
            "patches": [str(patch)],
        },
    )
    ctx = RunnerContext(task=task, lease=None, extra={"shared_state": state})
    result = await IntegratePatchExecutor(session_dir=session)(ctx)
    assert result["status"] == expected, result
    assert observed
    assert result["bench_result"]["workload_identity"] == anchor["workload_identity"]
    if expected == "kept":
        assert (repo / "src.py").read_text().endswith("return 2\n")
    else:
        assert (repo / "src.py").read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_status", ["complete", "failed", "missing"])
async def test_profile_executor_keeps_capture_windows_separate(source, report, tmp_path, monkeypatch, capture_status):
    import gzip

    from hyperloom.orchestrator.actions.executors.baseline import BenchmarkRunExecutor
    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

    monkeypatch.setenv(BENCHMARK_CONFIG_ENV, str(source))
    windows = []
    for index in range(2):
        directory = tmp_path / f"profile_{index}"
        directory.mkdir()
        files = []
        for rank in range(2):
            path = directory / f"177-TP-{rank}-DECODE.trace.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump({"traceEvents": []}, stream)
            files.append(str(path))
        windows.append({"capture_id": f"window-{index}", "status": "complete", "trace_files": files})
    report["agentx_metrics"].update(
        diagnostic_only=True, profile_capture={"status": capture_status, "profiles": windows}
    )
    report["agentx_metrics"]["recipe"]["tp"] = 2
    if capture_status == "missing":
        del report["agentx_metrics"]["profile_capture"]

    async def benchmark(self, ctx):
        return {**extract_benchmark_measurement(report), "status": "succeeded", "workspace": str(tmp_path)}

    monkeypatch.setattr(BenchmarkRunExecutor, "__call__", benchmark)
    ctx = SimpleNamespace(task=SimpleNamespace(params={"config_path": str(source)}, task_id="profile"), extra={})
    result = await ProfileExecutor(session_dir=tmp_path)(ctx)
    assert result["valid_measurement"] is False
    if capture_status != "complete":
        assert result["status"] == "failed"
        assert result["error_class"] == "profile_capture_failed"
        assert result["trace_input_ready"] is False
        return
    assert result["status"] == "succeeded", result
    assert result["primary_rank"] == 0
    assert len(result["profile_captures"]) == 2
    assert result["profile_captures"][0]["main_trace_path"] == windows[0]["trace_files"][0]
    assert result["profile_captures"][1]["main_trace_path"] == windows[1]["trace_files"][0]
    assert set(result["profile_captures"][0]["trace_files"]).isdisjoint(result["profile_captures"][1]["trace_files"])


def test_vllm_attempt_uses_python_for_magpies_module_entrypoint(source, tmp_path):
    from hyperloom.orchestrator.actions.executors.magpie_agentx import apply_native_runtime

    config = yaml.safe_load(source.read_text())
    config["benchmark"]["framework"] = "vllm"
    source.write_text(yaml.safe_dump(config))
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("python", "vllm"):
        executable = bin_dir / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    rendered = materialize_benchmark_config(source, tmp_path / "candidate")
    apply_native_runtime(rendered, {"framework_bin": str(bin_dir / "vllm")})
    assert load_benchmark(rendered)["agentx"]["launch_overrides"]["executable"] == str(bin_dir / "python")
