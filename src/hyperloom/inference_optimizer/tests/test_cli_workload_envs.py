# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLI workload-env export regressions."""

from __future__ import annotations

import argparse
import os

import pytest
import yaml

from hyperloom.inference_optimizer.cli import (
    _export_workload_envs_for_optimize,
    _resolve_run_max_model_len,
)
from hyperloom.orchestrator.actions.executors._workload_envs import (
    FrameworkScriptMismatchError,
    materialize_config_with_envs,
)


def _ns(**kwargs) -> argparse.Namespace:
    defaults = {"conc": 8}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _write_yaml(path, framework, benchmark_script=None):
    bench = {"framework": framework, "model": "/m", "envs": {}}
    if benchmark_script:
        bench["benchmark_script"] = benchmark_script
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")


def _write_yaml_with_envs(path, framework, envs):
    bench = {"framework": framework, "model": "/m", "envs": dict(envs)}
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")


def test_framework_script_mismatch_fails_fast(tmp_path):
    """vllm framework + sglang script must raise before server boot (QRWKV-72B bug)."""
    src = tmp_path / "cfg.yaml"
    _write_yaml(src, "vllm", benchmark_script="sglang_mi300x.sh")
    with pytest.raises(FrameworkScriptMismatchError, match="framework/script mismatch"):
        materialize_config_with_envs(
            src,
            tmp_path / "out",
            model_path="/m",
            gpu_type="mi300x",
            benchmark_script="sglang_mi300x.sh",
        )


def test_framework_script_match_ok(tmp_path):
    """vllm framework derives vllm_mi300x.sh; no mismatch raised."""
    src = tmp_path / "cfg.yaml"
    _write_yaml(src, "vllm")
    out = materialize_config_with_envs(
        src,
        tmp_path / "out",
        model_path="/m",
        gpu_type="mi300x",
    )
    bench = yaml.safe_load(out.read_text())["benchmark"]
    assert bench["benchmark_script"] == "vllm_mi300x.sh"


def test_single_node_explicit_tp_overrides_stale_env(monkeypatch):
    """`optimize --tp N` must reach YAML materialization on single-node."""
    monkeypatch.setenv("TP", "8")

    _export_workload_envs_for_optimize(
        _ns(conc=64),
        nodes_resolved=1,
        tp_resolved=4,
        ep_resolved=1,
        argv=["optimize", "--tp", "4"],
    )

    assert os.environ["TP"] == "4"


def test_single_node_default_does_not_clobber_yaml_defaults(monkeypatch):
    """No explicit flag: keep single-node YAML/env resolution unchanged."""
    monkeypatch.delenv("TP", raising=False)

    _export_workload_envs_for_optimize(
        _ns(conc=8),
        nodes_resolved=1,
        tp_resolved=1,
        ep_resolved=1,
        argv=["optimize", "--model", "/m"],
    )

    assert "TP" not in os.environ


def test_multi_node_always_exports_workload_envs(monkeypatch):
    """Multi-node child workers still receive resolved workload values."""
    for key in ("TP", "CONC", "EP"):
        monkeypatch.delenv(key, raising=False)

    _export_workload_envs_for_optimize(
        _ns(conc=32),
        nodes_resolved=2,
        tp_resolved=8,
        ep_resolved=2,
        argv=["optimize", "--nodes", "2"],
    )

    assert os.environ["TP"] == "8"
    assert os.environ["CONC"] == "32"
    assert os.environ["EP"] == "2"


def test_operator_server_args_env_routes_to_vllm_args(tmp_path, monkeypatch):
    """One server-args injection point must reach vLLM YAML materialization."""
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_SERVER_ARGS",
        "--gpu-memory-utilization 0.85 --kv-cache-dtype fp8_e4m3",
    )
    src = tmp_path / "cfg.yaml"
    _write_yaml(src, "vllm")

    out = materialize_config_with_envs(src, tmp_path / "out")
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]

    assert "EXTRA_VLLM_ARGS" in envs
    assert "--gpu-memory-utilization 0.85" in envs["EXTRA_VLLM_ARGS"]
    assert "--kv-cache-dtype fp8_e4m3" in envs["EXTRA_VLLM_ARGS"]


def test_operator_server_args_dedup_vllm_single_value_flags(tmp_path, monkeypatch):
    """Operator flags should override YAML defaults without duplicate vLLM keys."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SERVER_ARGS", "--gpu-memory-utilization 0.85")
    src = tmp_path / "cfg.yaml"
    _write_yaml_with_envs(
        src,
        "vllm",
        {"EXTRA_VLLM_ARGS": "--gpu-memory-utilization 0.95 --trust-remote-code"},
    )

    out = materialize_config_with_envs(src, tmp_path / "out")
    args = yaml.safe_load(out.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]

    assert args.count("--gpu-memory-utilization") == 1
    assert "--gpu-memory-utilization 0.85" in args
    assert "--trust-remote-code" in args


def test_conc_env_ladder_materializes_as_single_baseline_and_sweep_ladder(
    tmp_path,
    monkeypatch,
):
    """CONC=4,16,128 is recognized as a ladder, not crashed by int()."""
    monkeypatch.setenv("CONC", "4,16,128")
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CONC_SWEEP_CONCS", raising=False)
    src = tmp_path / "cfg.yaml"
    _write_yaml(src, "vllm")

    out = materialize_config_with_envs(src, tmp_path / "out")
    envs = yaml.safe_load(out.read_text())["benchmark"]["envs"]

    assert envs["CONC"] == 4
    assert os.environ["INFERENCE_OPTIMIZER_CONC_SWEEP_CONCS"] == "4,16,128"


def test_explicit_max_model_len_wins_over_auto(tmp_path, monkeypatch):
    """Explicit --max-model-len / $MAX_MODEL_LEN must not be recomputed."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"max_position_embeddings": 4096}', encoding="utf-8")
    monkeypatch.delenv("MAX_MODEL_LEN", raising=False)

    value, source = _resolve_run_max_model_len(
        _ns(model=str(model), isl=1024, osl=1024, max_model_len=200000),
    )

    assert value == 200000
    assert source == "--max-model-len"


def test_env_max_model_len_wins_over_auto(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"max_position_embeddings": 4096}', encoding="utf-8")
    monkeypatch.setenv("MAX_MODEL_LEN", "200000")

    value, source = _resolve_run_max_model_len(
        _ns(model=str(model), isl=1024, osl=1024, max_model_len=None),
    )

    assert value == 200000
    assert source == "$MAX_MODEL_LEN"
