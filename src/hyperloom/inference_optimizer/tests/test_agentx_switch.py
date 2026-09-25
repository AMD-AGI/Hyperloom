# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX switch (``HYPERLOOM_AGENTX``) materialization tests."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import yaml

from hyperloom.common.env import EnvValueError
from hyperloom.inference_optimizer.agentx import native as native_agentx
from hyperloom.orchestrator.actions.executors import _workload_envs as we

_AGENTX_ENV_KEYS = (
    "HYPERLOOM_AGENTX",
    "HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT",
    "HYPERLOOM_AGENTX_GPU_COUNT",
    "AGENTX_MODEL_ID",
    "AGENTX_SERVER_SCRIPT",
    "AGENTX_MODE",
    "AGENTX_DATASET",
    "AGENTX_MAX_CTX",
    "AGENTX_NUM_ENTRIES",
    "AGENTX_WARMUP_DURATION",
    "AGENTX_NUM_WARMUP_SESSIONS",
    "AGENTX_KEEP_SERVER",
    "AIPERF_BIN",
    "WEKA_LOADER_OVERRIDE",
    "RUN_EVAL",
    "MODEL_PATH",
    "TARGET_GPU_TYPE",
)

_MODEL_ID = "acme/Test-Model"
_VLLM_LAUNCHER = "single_node/agentic/test_fp4_mi300x_vllm_mtp.sh"


@pytest.fixture(autouse=True)
def _resolved_recipe_stub(tmp_path, monkeypatch):
    """Materialization tests isolate the switch from Magpie/InferenceX I/O."""
    monkeypatch.setenv("HYPERLOOM_IMAGE", "test/agentx:image")

    def _resolve(benchmark, *, expected_gpu_count, **_kwargs):
        envs = benchmark.setdefault("envs", {})
        # Magpie persists the recipe's inner tensor parallelism.  ``setdefault``
        # lets PP/PCP tests supply an inner TP smaller than the physical count.
        envs.setdefault("TP", int(expected_gpu_count))
        expected_mask = ",".join(str(index) for index in range(int(expected_gpu_count)))
        if str(envs.get("ROCR_VISIBLE_DEVICES") or "") != expected_mask:
            raise ValueError(f"ROCR_VISIBLE_DEVICES must be exactly {expected_mask!r}")
        benchmark.setdefault("gpu_selection", {})["auto"] = False
        workload = benchmark.setdefault("workload_spec", {})
        workload["resolved_topology"] = {
            "tp": int(expected_gpu_count),
            "pp": 1,
            "pcp_size": 1,
            "ep": 1,
            "gpu_count": int(expected_gpu_count),
            "recipe_fingerprint": "a" * 64,
        }
        workload["recipe"] = {
            "name": "test-agentx-recipe",
            "recipe_fingerprint": "a" * 64,
            "image": "test/agentx:image",
        }
        return workload["resolved_topology"]

    monkeypatch.setattr(native_agentx, "resolve_native_recipe", _resolve)


def _clear_env(monkeypatch):
    for k in _AGENTX_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def _enable_native(monkeypatch, *, launcher: str = _VLLM_LAUNCHER) -> None:
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_MODEL_ID", _MODEL_ID)
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", launcher)


def _write(path, **bench_extra):
    bench = {"framework": "vllm", "model": "/m", "envs": {}}
    bench.update(bench_extra)
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


def _materialize(src, out, **kw):
    source = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    source_native = we._serialized_native_agentx_enabled((source.get("benchmark") or {}).get("agentx"))
    env_native = os.environ.get("HYPERLOOM_AGENTX", "").strip().lower() in {"1", "true", "yes", "on"}
    if source_native or env_native or kw.get("agentx_mode") is True:
        inferencex = src.parent / "ix"
        inferencex.mkdir(exist_ok=True)
        kw.setdefault("inferencex_path", str(inferencex))
    res = we.materialize_config_with_envs(src, out, **kw)
    return yaml.safe_load(res.read_text())["benchmark"]


# ── OFF path: zero regression ────────────────────────────────────────────────
def test_switch_off_keeps_synthetic_script(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert bench["benchmark_script"] == "vllm_mi300x.sh"
    assert not any(k.startswith("AGENTX") for k in bench.get("envs", {}))


def test_switch_off_no_agentx_leakage(tmp_path, monkeypatch):
    """OFF output must carry no trace of the AgentX feature."""
    _clear_env(monkeypatch)
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    dumped = yaml.safe_dump(bench).lower()
    assert "aiperf" not in dumped
    assert "agentx" not in dumped
    assert bench["benchmark_script"] == "vllm_mi300x.sh"


# ── ON path: authoritative overwrite + env injection ─────────────────────────
def test_switch_on_authoritative_overwrite(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    src = _write(tmp_path / "base.yaml")
    # gpu_type pre-pins vllm_mi300x.sh; the switch must overwrite it.
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert bench["benchmark_script"] == _VLLM_LAUNCHER
    assert bench["agentx"] == "enable"
    assert bench["model"] == _MODEL_ID
    assert bench["gpu_selection"]["auto"] is False
    assert "count" not in bench["gpu_selection"]
    assert bench["envs"]["MODEL_PATH"] == "/m"
    assert "timeout_seconds" not in bench
    assert "AGENTX_PHASE_WAIT_TIMEOUT_S" not in bench["envs"]


def test_native_outer_gpu_count_does_not_overwrite_inner_recipe_tp(
    tmp_path,
    monkeypatch,
):
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    monkeypatch.setenv("TP", "4")
    monkeypatch.setenv("HYPERLOOM_AGENTX_GPU_COUNT", "4")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    src = _write(
        tmp_path / "base.yaml",
        agentx="enable",
        envs={
            "TP": 2,
            "CONC": 8,
            "ROCR_VISIBLE_DEVICES": "0,1,2,3",
        },
    )

    bench = _materialize(
        src,
        tmp_path / "out",
        gpu_type="mi300x",
        model_path="/m",
    )

    assert bench["envs"]["TP"] == 2
    assert bench["workload_spec"]["resolved_topology"]["gpu_count"] == 4


def test_persisted_agentx_mode_switches_without_ambient_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("AGENTX_MODEL_ID", _MODEL_ID)
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", _VLLM_LAUNCHER)
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(
        src,
        tmp_path / "out",
        gpu_type="mi300x",
        model_path="/m",
        agentx_mode=True,
    )
    assert bench["benchmark_script"] == _VLLM_LAUNCHER
    assert bench["agentx"] == "enable"


def test_native_magpie_yaml_round_trips_without_duplicate_agentx_envs(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    native_config = {
        "enabled": True,
        "mode": "canonical",
        "recipe": "test-agentx-recipe",
        "selector": {"tp": 8},
        "failed_request_threshold": 0.05,
    }
    src = _write(
        tmp_path / "native.yaml",
        model=_MODEL_ID,
        benchmark_script=_VLLM_LAUNCHER,
        agentx=native_config,
    )

    bench = _materialize(
        src,
        tmp_path / "out",
        gpu_type="mi300x",
        model_path="/models/local",
        agentx_mode=True,
    )

    assert bench["agentx"] == native_config
    assert bench["model"] == _MODEL_ID
    assert bench["benchmark_script"] == _VLLM_LAUNCHER
    assert bench["envs"]["MODEL_PATH"] == "/models/local"


def test_switch_on_injects_model_and_run_eval(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/model/x")
    envs = bench["envs"]
    assert envs["RUN_EVAL"] == "false"
    assert envs["MODEL_PATH"] == "/model/x"
    assert "MODEL" not in envs
    assert bench["model"] == _MODEL_ID


def test_remote_model_id_does_not_become_a_relative_model_path(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", _VLLM_LAUNCHER)
    remote_model = "amd/GLM-5.2-MXFP4"
    src = _write(tmp_path / "base.yaml")

    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path=remote_model)

    assert bench["model"] == remote_model
    assert "MODEL_PATH" not in bench["envs"]


def test_switch_on_pins_zero_based_gpu_mask_for_native_launcher(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    monkeypatch.setenv("TP", "4")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    src = _write(tmp_path / "base.yaml")

    bench = _materialize(src, tmp_path / "out", gpu_type="mi355x", model_path="/m")

    assert bench["envs"]["TP"] == 4
    assert bench["envs"]["ROCR_VISIBLE_DEVICES"] == "0,1,2,3"


def test_switch_on_rejects_nonzero_outer_gpu_mask(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    monkeypatch.setenv("TP", "4")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    src = _write(tmp_path / "base.yaml")

    with pytest.raises(ValueError, match="ROCR_VISIBLE_DEVICES"):
        _materialize(src, tmp_path / "out", gpu_type="mi355x", model_path="/m")


def test_native_agentx_keeps_physical_mi325_runner_identity(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _enable_native(monkeypatch, launcher="single_node/agentic/glm5.2_fp8_mi325x_mtp.sh")
    monkeypatch.setenv("TARGET_GPU_TYPE", "mi325x")
    src = _write(tmp_path / "base.yaml")

    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/models/glm")

    assert bench["runner_type"] == "mi325x"
    assert bench["benchmark_script"] == "single_node/agentic/glm5.2_fp8_mi325x_mtp.sh"


def test_native_switch_rejects_legacy_corpus_and_client_overrides(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    monkeypatch.setenv("AGENTX_DATASET", "semianalysis-cc-traces-weka-with-subagents")
    monkeypatch.setenv("AGENTX_NUM_ENTRIES", "8")
    monkeypatch.setenv("AIPERF_BIN", "/venv/bin/aiperf")
    src = _write(tmp_path / "base.yaml")
    with pytest.raises(ValueError, match="corpus selection belongs"):
        _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")


def test_switch_on_materializes_workload_spec(tmp_path, monkeypatch):
    """AgentX ON must stamp a self-describing workload_spec into the recipe."""
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    monkeypatch.setenv("AGENTX_MODEL_ID", "moonshotai/Kimi-K3")
    monkeypatch.setenv("CONC", "8")
    monkeypatch.setenv("AGENTX_DURATION", "3600")
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi355x", model_path="/models/Kimi-K3")
    spec = bench.get("workload_spec") or {}
    assert spec.get("kind") == "agentx_trace_replay"
    assert spec.get("client") == "aiperf"
    assert spec.get("scenario") == "inferencex-agentx-mvp"
    assert spec.get("harness") == "magpie-native-agentx"
    assert spec.get("corpus") == "semianalysis_cc_traces_weka_062126"
    assert spec.get("duration_s") == 3600
    assert spec.get("geak_loop_duration_s") == 900
    assert spec.get("concurrency") == 8
    placeholder = spec.get("isl_osl_placeholder") or {}
    assert placeholder.get("note")


def test_home_relative_checkpoint_is_expanded_before_launch(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    src = _write(tmp_path / "base.yaml")

    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="~/checkpoint")

    assert bench["envs"]["MODEL_PATH"] == str(tmp_path / "checkpoint")


def test_home_relative_checkpoint_cannot_be_inferred_as_hf_identity(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", _VLLM_LAUNCHER)
    monkeypatch.setenv("HOME", str(tmp_path))
    src = _write(tmp_path / "base.yaml")

    with pytest.raises(ValueError, match="AGENTX_MODEL_ID"):
        _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="~/checkpoint")


def test_local_checkpoint_name_does_not_choose_the_agentx_corpus(tmp_path, monkeypatch):
    """Corpus identity comes from the recipe model, never a local basename."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_MODEL_ID", "amd/GLM-5.2-MXFP4")
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", _VLLM_LAUNCHER)
    src = _write(tmp_path / "base.yaml")

    bench = _materialize(
        src,
        tmp_path / "out",
        gpu_type="mi355x",
        model_path="/models/checkpoint",
    )

    assert bench["envs"]["MODEL_PATH"] == "/models/checkpoint"
    assert bench["workload_spec"]["canonical_corpus"] == "semianalysis_cc_traces_weka_062126"


def test_profile_compat_uses_recipe_model_for_corpus(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_MODEL_ID", "amd/GLM-5.2-MXFP4")
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", _VLLM_LAUNCHER)
    src = _write(tmp_path / "profile.yaml", profiler={"torch_profiler": {"enabled": True}})

    bench = _materialize(
        src,
        tmp_path / "out",
        gpu_type="mi355x",
        model_path="/models/checkpoint",
        allow_agentx_profile_compat=True,
    )

    assert bench["envs"]["MODEL"] == "/models/checkpoint"
    assert bench["envs"]["AGENTX_MODEL_ID"] == "amd/GLM-5.2-MXFP4"
    assert bench["workload_spec"]["canonical_corpus"] == "semianalysis_cc_traces_weka_062126"


def test_switch_off_omits_workload_spec(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert "workload_spec" not in bench


def test_native_switch_rejects_weka_loader_override(tmp_path, monkeypatch):
    """A native corpus override would not participate in recipe identity."""
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    monkeypatch.setenv("WEKA_LOADER_OVERRIDE", "semianalysis_cc_traces_weka_062126")
    src = _write(tmp_path / "base.yaml")
    with pytest.raises(ValueError, match="WEKA_LOADER_OVERRIDE"):
        _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")


def test_switch_off_does_not_leak_weka_loader_override(tmp_path, monkeypatch):
    """The synthetic path must not gain an env it never had."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("WEKA_LOADER_OVERRIDE", "semianalysis_cc_traces_weka_062126")
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert "WEKA_LOADER_OVERRIDE" not in (bench.get("envs") or {})


# ── A3: parsing ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw", ["0", "false", "no", "off", ""])
def test_switch_off_tokens_keep_the_synthetic_script(tmp_path, monkeypatch, raw):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", raw)
    src = _write(tmp_path / "base.yaml")
    bench = _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")
    assert bench["benchmark_script"] == "vllm_mi300x.sh"


def test_an_unreadable_switch_does_not_materialize_the_synthetic_workload(tmp_path, monkeypatch):
    """A typo used to read as OFF here, benchmarking the run the operator did not ask for."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "ture")
    src = _write(tmp_path / "base.yaml")
    with pytest.raises(EnvValueError, match="HYPERLOOM_AGENTX"):
        _materialize(src, tmp_path / "out", gpu_type="mi300x", model_path="/m")


def test_switch_only_serving_frameworks(tmp_path, monkeypatch):
    """Scriptable (image) frameworks must never be swapped to aiperf."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    src = _write(tmp_path / "base.yaml", framework="xdit")
    bench = _materialize(src, tmp_path / "out", model_path="/m")
    assert bench.get("benchmark_script") != "aiperf_client.sh"


def test_profile_keeps_phase_gated_compatibility_client(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    src = _write(
        tmp_path / "profile.yaml",
        envs={"PROFILE": "1"},
        profiler={"torch_profiler": {"enabled": True}},
    )
    bench = _materialize(
        src,
        tmp_path / "out",
        gpu_type="mi300x",
        model_path="/model/x",
        allow_agentx_profile_compat=True,
    )
    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert "agentx" not in bench
    assert bench["envs"]["MODEL"] == "/model/x"
    assert bench["envs"]["AGENTX_SERVER_SCRIPT"] == ""
    assert bench["workload_spec"]["harness"] == "hyperloom-profiler-compat"


@pytest.mark.parametrize(
    "extra",
    [
        {"profiler": {"system_profiler": {"enabled": True}}},
        {"profiler": {"tracelens": {"enabled": "true"}}},
        {"gap_analysis": {"enabled": True}},
    ],
)
def test_all_magpie_incompatible_profile_modes_use_compat_client(tmp_path, monkeypatch, extra):
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    src = _write(tmp_path / "profile.yaml", **extra)

    bench = _materialize(
        src,
        tmp_path / "out",
        gpu_type="mi300x",
        model_path="/model/x",
        allow_agentx_profile_compat=True,
    )

    assert bench["benchmark_script"] == "aiperf_client.sh"
    assert "agentx" not in bench
    assert bench["workload_spec"]["harness"] == "hyperloom-profiler-compat"


# ── Regression: the shared grid/baseline/profile rebuild path (E1 bug) ────────
def test_runtime_overrides_honor_agentx_on(monkeypatch):
    """apply_runtime_benchmark_overrides must apply the switch, else the gpu_type-derived synthetic script silently reverts a materialize-time swap (the exact defect E1 caught: run_grid rebuilt to vllm_mi300x.sh)."""
    _clear_env(monkeypatch)
    _enable_native(monkeypatch)
    from hyperloom.orchestrator.actions.executors._benchmark_runtime import (
        apply_runtime_benchmark_overrides,
    )

    bench = {"framework": "vllm"}
    apply_runtime_benchmark_overrides(bench, model_path="/m", gpu_type="mi300x")
    assert bench["benchmark_script"] == _VLLM_LAUNCHER
    assert bench["agentx"] == "enable"
    assert bench["envs"]["RUN_EVAL"] == "false"


def test_runtime_overrides_off_keeps_synthetic(monkeypatch):
    _clear_env(monkeypatch)  # HYPERLOOM_AGENTX cleared => OFF
    from hyperloom.orchestrator.actions.executors._benchmark_runtime import (
        apply_runtime_benchmark_overrides,
    )

    bench = {"framework": "vllm"}
    apply_runtime_benchmark_overrides(bench, model_path="/m", gpu_type="mi300x")
    assert bench["benchmark_script"] == "vllm_mi300x.sh"


def test_runtime_overrides_preserve_materialized_agentx_without_env(monkeypatch):
    _clear_env(monkeypatch)
    from hyperloom.orchestrator.actions.executors._benchmark_runtime import (
        apply_runtime_benchmark_overrides,
    )

    bench = {
        "framework": "vllm",
        "agentx": "enable",
        "benchmark_script": _VLLM_LAUNCHER,
        "envs": {
            "AGENTX_MODEL_ID": _MODEL_ID,
            "AGENTX_SERVER_SCRIPT": _VLLM_LAUNCHER,
        },
    }
    apply_runtime_benchmark_overrides(bench, model_path="/m", gpu_type="mi300x")
    assert bench["benchmark_script"] == _VLLM_LAUNCHER
    assert bench["agentx"] == "enable"


def test_sglang_agentx_keeps_native_launcher_through_runtime_rebuild(tmp_path, monkeypatch):
    """AgentX must not enter the generic SGLang runner that disables radix caching."""
    from hyperloom.orchestrator.actions.executors._benchmark_runtime import (
        apply_runtime_benchmark_overrides,
    )

    _clear_env(monkeypatch)
    launcher = "single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh"
    source = _write(
        tmp_path / "glm-agentx.yaml",
        framework="sglang",
        model="amd/GLM-5.2-MXFP4",
        runner_type="mi355x",
        agentx="enable",
        benchmark_script=launcher,
        envs={"CONC": 8},
    )
    benchmark = _materialize(source, tmp_path / "out", gpu_type="mi355x", model_path="/models/glm")

    apply_runtime_benchmark_overrides(benchmark, model_path="/models/glm", gpu_type="mi355x", conc=8)

    assert benchmark["agentx"] == "enable"
    assert benchmark["benchmark_script"] == launcher
    assert benchmark["envs"]["AGENTX_SERVER_SCRIPT"] == launcher
    assert benchmark["envs"]["CONC"] == 8
    assert "--disable-radix-cache" not in benchmark["envs"].get("EXTRA_SGLANG_ARGS", "")


def test_runtime_round_concurrency_overrides_stale_agentx_config(monkeypatch):
    _clear_env(monkeypatch)
    from hyperloom.orchestrator.actions.executors._benchmark_runtime import (
        apply_runtime_benchmark_overrides,
    )

    bench = {
        "framework": "vllm",
        "agentx": {"enabled": True, "mode": "canonical", "concurrency": 8},
        "benchmark_script": _VLLM_LAUNCHER,
        "envs": {
            "CONC": 8,
            "AGENTX_MODEL_ID": _MODEL_ID,
            "AGENTX_SERVER_SCRIPT": _VLLM_LAUNCHER,
        },
    }

    apply_runtime_benchmark_overrides(bench, model_path="/m", gpu_type="mi300x", conc=16)

    assert bench["envs"]["CONC"] == 16
    assert bench["workload_spec"]["concurrency"] == 16
    assert "concurrency" not in bench["agentx"]


# ── A2: OFF path never imports the agentx package (lazy-import guarantee) ─────
_PROBE_OFF = """
import sys
from pathlib import Path
from hyperloom.orchestrator.actions.executors import _grid_runner  # noqa: F401
from hyperloom.orchestrator.actions.executors import _workload_envs as we

assert not we.agentx_enabled()
we.materialize_config_with_envs(
    Path(sys.argv[1]), Path(sys.argv[2]), gpu_type="mi300x", model_path="/m"
)
leaked = [m for m in sys.modules if m.startswith("hyperloom.inference_optimizer.agentx")]
assert not leaked, "agentx package imported on OFF path: %r" % (leaked,)
print("OFF_OK")
"""


def test_agentx_package_not_imported_on_off_path(tmp_path):
    """A2: the default (OFF) benchmark path must never import the agentx package."""
    src = tmp_path / "base.yaml"
    out = tmp_path / "out"
    src.write_text("benchmark:\n  framework: vllm\n  model: /m\n  envs: {}\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HYPERLOOM_AGENTX", "AGENTX_", "AIPERF"))}
    r = subprocess.run(
        [sys.executable, "-c", _PROBE_OFF, str(src), str(out)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, "stdout=%s\nstderr=%s" % (r.stdout, r.stderr)
    assert "OFF_OK" in r.stdout


_PROBE_CONTRAST = """
import sys

assert not any(m.startswith("hyperloom.inference_optimizer.agentx") for m in sys.modules)
from hyperloom.inference_optimizer.agentx.runtime import maybe_prepare_agentx  # noqa: F401

assert any(m.startswith("hyperloom.inference_optimizer.agentx") for m in sys.modules)
print("CONTRAST_OK")
"""


def test_agentx_package_importable_contrast():
    """Contrast: the agentx package is real and importable (the ON _grid_runner branch does exactly this lazy import), so the OFF assertion is not vacuous."""
    r = subprocess.run(
        [sys.executable, "-c", _PROBE_CONTRAST],
        env=dict(os.environ),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, "stdout=%s\nstderr=%s" % (r.stdout, r.stderr)
    assert "CONTRAST_OK" in r.stdout


# ── Framework identity reaches the native launcher environment ───────────────
def test_switch_on_injects_framework_for_native_launcher(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    for fw in ("sglang", "vllm"):
        launcher = f"single_node/agentic/test_fp4_mi300x_{fw}_mtp.sh"
        _enable_native(monkeypatch, launcher=launcher)
        src = _write(tmp_path / f"{fw}.yaml", framework=fw)
        bench = _materialize(src, tmp_path / f"out_{fw}", gpu_type="mi300x", model_path="/m")
        assert bench["benchmark_script"] == launcher
        assert bench["envs"]["FRAMEWORK"] == fw
