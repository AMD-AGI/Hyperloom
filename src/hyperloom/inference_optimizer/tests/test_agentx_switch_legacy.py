# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compatibility contracts for the environment-selected AgentX client."""

from __future__ import annotations

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _workload_envs as we

_AGENTX_ENV_KEYS = (
    "HYPERLOOM_AGENTX",
    "AGENTX_MODEL_ID",
    "AGENTX_SERVER_SCRIPT",
    "AGENTX_DATASET",
    "AGENTX_MAX_CTX",
    "AGENTX_NUM_ENTRIES",
    "AGENTX_DURATION",
    "AGENTX_WARMUP_DURATION",
    "AGENTX_NUM_WARMUP_SESSIONS",
    "AGENTX_KEEP_SERVER",
    "AIPERF_BIN",
    "WEKA_LOADER_OVERRIDE",
    "RUN_EVAL",
    "MODEL_PATH",
)


def _clear_env(monkeypatch):
    for k in _AGENTX_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def _write(path, **bench_extra):
    bench = {"framework": "vllm", "model": "/m", "envs": {}}
    bench.update(bench_extra)
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


def _materialize(src, out, **kw):
    res = we.materialize_config_with_envs(src, out, **kw)
    return yaml.safe_load(res.read_text())["benchmark"]


def test_switch_on_authoritative_overwrite(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", "vllm_custom_server.sh")
    src = _write(tmp_path / "base.yaml")
    # gpu_type pre-pins vllm_mi300x.sh; the switch must overwrite it.
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert "agentx" not in bench
    assert bench["envs"]["AGENTX_SERVER_SCRIPT"] == "vllm_custom_server.sh"
    assert "timeout_seconds" not in bench
    assert "AGENTX_PHASE_WAIT_TIMEOUT_S" not in bench["envs"]


def test_persisted_agentx_mode_switches_without_ambient_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(
        src,
        tmp_path / "out",
        gpu_type="mi300x",
        model_path="/m",
        agentx_mode=True,
    )
    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert "agentx" not in bench


def test_switch_on_injects_model_and_run_eval(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "true")
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/model/x")
    envs = bench["envs"]
    assert envs["RUN_EVAL"] == "false"
    assert envs["MODEL"] == "/model/x"


def test_switch_on_passes_agentx_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "on")
    monkeypatch.setenv("AGENTX_DATASET", "semianalysis-cc-traces-weka-with-subagents")
    monkeypatch.setenv("AGENTX_NUM_ENTRIES", "8")
    monkeypatch.setenv("AIPERF_BIN", "/venv/bin/aiperf")
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    envs = bench["envs"]
    assert envs["AGENTX_DATASET"] == "semianalysis-cc-traces-weka-with-subagents"
    assert envs["AGENTX_NUM_ENTRIES"] == "8"
    assert envs["AIPERF_BIN"] == "/venv/bin/aiperf"


def test_switch_on_materializes_workload_spec(tmp_path, monkeypatch):
    """AgentX ON must stamp a self-describing workload_spec into the recipe."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("CONC", "8")
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi355x", model_path="/models/Kimi-K3")
    spec = bench.get("workload_spec") or {}
    assert spec.get("kind") == "agentx_trace_replay"
    assert spec.get("client") == "aiperf"
    assert spec.get("scenario") == "inferencex-agentx-mvp"
    assert spec.get("corpus") == "semianalysis_cc_traces_weka_062126"
    assert spec.get("duration_s") == 3600
    assert spec.get("geak_loop_duration_s") == 900
    assert spec.get("concurrency") == 8
    placeholder = spec.get("isl_osl_placeholder") or {}
    assert placeholder.get("note")


def test_switch_forwards_weka_loader_override(tmp_path, monkeypatch):
    """Upstream's own corpus pin has no ``AGENTX_`` prefix."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("WEKA_LOADER_OVERRIDE", "semianalysis_cc_traces_weka_062126")
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert bench["envs"]["WEKA_LOADER_OVERRIDE"] == "semianalysis_cc_traces_weka_062126"


def test_runtime_overrides_honor_agentx_on(monkeypatch):
    """apply_runtime_benchmark_overrides must apply the switch, else the gpu_type-derived synthetic script silently reverts a materialize-time swap (the exact defect E1 caught: run_grid rebuilt to vllm_mi300x.sh)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    from hyperloom.orchestrator.actions.executors._benchmark_runtime import (
        apply_runtime_benchmark_overrides,
    )

    bench = {"framework": "vllm"}
    apply_runtime_benchmark_overrides(bench, model_path="/m", gpu_type="mi300x")
    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert "agentx" not in bench
    assert bench["envs"]["RUN_EVAL"] == "false"


def test_runtime_overrides_preserve_materialized_agentx_without_env(monkeypatch):
    _clear_env(monkeypatch)
    from hyperloom.orchestrator.actions.executors._benchmark_runtime import (
        apply_runtime_benchmark_overrides,
    )

    bench = {"framework": "vllm", "benchmark_script": "aiperf_client.sh"}
    apply_runtime_benchmark_overrides(bench, model_path="/m", gpu_type="mi300x")
    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert "agentx" not in bench


def test_switch_on_injects_framework_for_delegation(tmp_path, monkeypatch):
    """ON must inject ``benchmark.framework`` into ``envs.FRAMEWORK`` so aiperf_client.sh delegates to ``{framework}_{gpu}.sh``."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    for fw in ("sglang", "vllm"):
        src = _write(tmp_path / f"{fw}.yaml", framework=fw)
        bench = _materialize(src, tmp_path / f"out_{fw}", gpu_type="mi300x", model_path="/m")
        assert bench["benchmark_script"] == "aiperf_client.sh"
        assert "agentx" not in bench
        assert bench["envs"]["FRAMEWORK"] == fw


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_legacy_profile_still_patches_the_framework(tmp_path, monkeypatch, framework):
    from hyperloom.orchestrator.actions.executors import _server_patcher

    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "1")
    monkeypatch.setattr(_server_patcher, "resolve_sglang_shape_mode", lambda: "patched")
    patch_calls = []

    def patch_framework():
        patch_calls.append(framework)
        return True

    monkeypatch.setattr(we, f"ensure_{framework}_patched_for_tracelens", patch_framework)
    src = _write(
        tmp_path / "profile.yaml",
        framework=framework,
        envs={"PROFILE": "1"},
        profiler={"torch_profiler": {"enabled": True}},
    )

    bench = _materialize(src, tmp_path / "out", model_path="/models/checkpoint")

    assert patch_calls == [framework]
    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert "agentx" not in bench
    assert "harness" not in bench["workload_spec"]
    assert bench["envs"]["HYPERLOOM_TRACELENS_PATCH_STATUS"] == "ok"


def test_legacy_materialization_keeps_server_candidates_and_replay_controls(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_DURATION", "240")
    monkeypatch.setenv("AGENTX_NUM_ENTRIES", "8")
    src = _write(tmp_path / "base.yaml")

    bench = _materialize(
        src,
        tmp_path / "out",
        extra_server_args="--max-num-seqs 16",
        extra_envs={"VLLM_USE_V1": "1", "CONC": "4"},
    )

    assert "agentx" not in bench
    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert "--max-num-seqs 16" in bench["envs"]["EXTRA_VLLM_ARGS"]
    assert bench["envs"]["VLLM_USE_V1"] == "1"
    assert bench["workload_spec"]["concurrency"] == 4
    assert bench["workload_spec"]["duration_s"] == 240
    assert bench["workload_spec"]["num_entries"] == 8


@pytest.mark.parametrize("disabled", [False, "disable", {"enabled": False}])
def test_disabled_native_yaml_keeps_the_legacy_environment_contract(tmp_path, monkeypatch, disabled):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    src = _write(tmp_path / "base.yaml", agentx=disabled)

    bench = _materialize(src, tmp_path / "out", model_path="/models/checkpoint")

    assert bench["agentx"] == disabled
    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert bench["envs"]["MODEL"] == "/models/checkpoint"
    assert "harness" not in bench["workload_spec"]
