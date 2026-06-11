# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLI workload-env export regressions."""

from __future__ import annotations

import argparse
import os

import pytest
import yaml

from inference_optimizer.cli import _export_workload_envs_for_optimize
from inference_optimizer.orchestrator.action_executors._workload_envs import (
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


def test_framework_script_mismatch_fails_fast(tmp_path):
    """vllm framework + sglang script must raise before server boot (QRWKV-72B bug)."""
    src = tmp_path / "cfg.yaml"
    _write_yaml(src, "vllm", benchmark_script="sglang_mi300x.sh")
    with pytest.raises(ValueError, match="framework/script mismatch"):
        materialize_config_with_envs(
            src, tmp_path / "out", model_path="/m", gpu_type="mi300x",
            benchmark_script="sglang_mi300x.sh",
        )


def test_framework_script_match_ok(tmp_path):
    """vllm framework derives vllm_mi300x.sh; no mismatch raised."""
    src = tmp_path / "cfg.yaml"
    _write_yaml(src, "vllm")
    out = materialize_config_with_envs(
        src, tmp_path / "out", model_path="/m", gpu_type="mi300x",
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
