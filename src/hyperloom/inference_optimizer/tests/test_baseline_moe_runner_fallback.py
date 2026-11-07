# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline one-shot fallback when a pinned MoE runner backend kills the server.

Hyperloom no longer injects a default ``--moe-runner-backend`` (sglang's own
``auto`` already picks the right one). An operator pin, a reference recipe, or
an inherited grid/variant flag can still select a scheme without a runner (e.g.
Quark MXFP4 forced onto ``triton``), crashing on the first forward pass; the
executor retries once with the flag dropped.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

from ._baseline_fakes import fake_workspace, run_baseline, write_sglang_yaml

_QUARK_TRACEBACK = (
    'File ".../quark/schemes/quark_w4a4_mxfp4_moe.py", line 295, in apply_weights\n'
    "    return self.runner.run(dispatch_output, quant_info)\n"
    "AttributeError: 'QuarkW4A4MXFp4MoE' object has no attribute 'runner'\n"
)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))
    monkeypatch.delenv("HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND", raising=False)


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-moe-1", params=params)
    return SimpleNamespace(task=task, extra={})


# --- _is_moe_runner_rooted_failure -----------------------------------------
def test_moe_runner_failure_from_error_tail():
    result = {"status": "failed", "error": _QUARK_TRACEBACK}
    assert BaselineExecutor._is_moe_runner_rooted_failure(result) is True


def test_moe_runner_failure_scans_logs(tmp_path: Path):
    out = tmp_path / "task"
    ws = out / "benchmark_sglang_x"
    ws.mkdir(parents=True)
    (ws / "server.log").write_text(_QUARK_TRACEBACK, encoding="utf-8")
    result = {"status": "failed", "error": "server died during startup", "output_dir": str(out)}
    assert BaselineExecutor._is_moe_runner_rooted_failure(result) is True


def test_moe_runner_failure_negative():
    result = {"status": "failed", "error": "CUDA out of memory"}
    assert BaselineExecutor._is_moe_runner_rooted_failure(result) is False


def test_unrelated_missing_runner_attribute_is_not_a_moe_failure():
    # A missing 'runner' attribute outside the MoE path must not burn a retry.
    result = {
        "status": "failed",
        "error": (
            'File ".../bench/harness.py", line 12, in start\n'
            "    self.runner.run(request)\n"
            "AttributeError: 'BenchHarness' object has no attribute 'runner'\n"
        ),
    }
    assert BaselineExecutor._is_moe_runner_rooted_failure(result) is False


def test_moe_failure_detected_for_non_quark_scheme():
    # Detection is keyed on the MoE runner, not on Quark: sglang's int4fp8 and mxfp4 dynamic-quant MoE methods fail
    # the same way.
    result = {
        "status": "failed",
        "error": (
            'File ".../layers/quantization/quark_int4fp8_moe.py", line 446, in apply\n'
            "    return self.runner.run(dispatch_output, quant_info)\n"
            "AttributeError: 'QuarkInt4Fp8MoEMethod' object has no attribute 'runner'\n"
        ),
    }
    assert BaselineExecutor._is_moe_runner_rooted_failure(result) is True


# --- end-to-end fallback ----------------------------------------------------
def _sglang_args(cmd) -> str:
    cfg = yaml.safe_load(Path(cmd[cmd.index("--benchmark-config") + 1]).read_text())
    return str(cfg["benchmark"]["envs"].get("EXTRA_SGLANG_ARGS", ""))


def _run_baseline(tmp_path, model: str, crash_stderr, **params) -> tuple[list[str], dict]:
    """Run the baseline over a Magpie that fails with ``crash_stderr(sglang_args)`` when it returns text."""
    calls: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        sglang_args = _sglang_args(cmd)
        calls.append(sglang_args)
        stderr = crash_stderr(sglang_args)
        if stderr is not None:
            return subprocess.CompletedProcess(cmd, 1, "", stderr)
        fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": model,
            "gpu_type": "mi300x",
            "extra_envs": {"RUN_EVAL": "false"},
            **params,
        }
    )
    result = run_baseline(write_sglang_yaml(tmp_path / "base.yaml", model), tmp_path, ctx, fake_run)
    return calls, result
