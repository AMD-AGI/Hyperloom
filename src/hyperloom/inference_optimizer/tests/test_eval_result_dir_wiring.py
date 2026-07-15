# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the ``$EVAL_RESULT_DIR`` wiring (P0 accuracy-gate fix).

InferenceX ``run_lm_eval`` (benchmark_lib.sh) reads ``$EVAL_RESULT_DIR`` for
lm-eval's ``--output_path``; unset, it falls back to ``/tmp/eval_out-*`` so the
``results*.json`` escape the task workspace and the accuracy gate sees no
baseline (``baseline_accuracy=0.0`` -> throughput-only KEEP). Hyperloom only set
``$RESULT_DIR``. These tests pin that:

* the baseline / grid subprocess env exports ``$EVAL_RESULT_DIR`` mirrored from
  ``$RESULT_DIR``; and
* the accuracy parse search root is aligned to that dir, where lm-eval
  (lm_eval 0.4.9.2) writes ``<root>/<model_sanitized>/results_<ts>.json``.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._accuracy_gate import parse_eval_results
from hyperloom.orchestrator.actions.executors._grid_runner import _run_magpie
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

_GSM8K_RESULTS = {
    "results": {"gsm8k": {"exact_match,strict-match": 0.83, "alias": "gsm8k"}},
}


def _write_lm_eval_output(root: Path, *, model_dir: str = "model__sanitized") -> Path:
    """Reproduce lm_eval 0.4.9.2 --output_path layout: <root>/<model>/results_<ts>.json."""
    dest = Path(root) / model_dir
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "results_2026-07-15T10-00-00.000000.json"
    out.write_text(json.dumps(_GSM8K_RESULTS), encoding="utf-8")
    return out


# --- parse_eval_results search-root behavior against the real lm-eval layout ---


def test_parse_eval_results_finds_lm_eval_output_under_root(tmp_path):
    # lm-eval writes one directory level below --output_path; the recursive
    # ``**/results*.json`` glob must catch it from the aligned root.
    _write_lm_eval_output(tmp_path)
    out = parse_eval_results(tmp_path, framework="sglang")
    assert out.get("accuracy") == pytest.approx(0.83)
    assert out.get("task") == "gsm8k"


def test_parse_eval_results_misses_when_root_is_benchmark_subdir(tmp_path):
    # Root-cause shape: lm-eval writes under the slot (== $EVAL_RESULT_DIR), a
    # sibling of the Magpie ``benchmark_*`` workspace. Searching from the
    # benchmark_* subdir (the pre-fix baseline root) cannot reach the sibling.
    _write_lm_eval_output(tmp_path)
    bench_ws = tmp_path / "benchmark_sglang_20260715_010101"
    bench_ws.mkdir(parents=True)
    out = parse_eval_results(bench_ws, framework="sglang")
    assert out.get("accuracy") is None


# --- Grid runner env wiring ---


def test_run_magpie_exports_eval_result_dir_mirrored_from_result_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
        )
    assert captured["env"].get("EVAL_RESULT_DIR") == captured["env"]["RESULT_DIR"]
    assert captured["env"]["EVAL_RESULT_DIR"] == str(tmp_path / "slot")


def test_run_magpie_eval_result_dir_follows_result_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=tmp_path / "config.yaml",
            output_dir=tmp_path / "slot",
            timeout_sec=5,
            cwd=str(tmp_path),
            result_dir="/tmp/redirect_leak",
        )
    assert captured["env"]["EVAL_RESULT_DIR"] == "/tmp/redirect_leak"
    assert captured["env"]["RESULT_DIR"] == "/tmp/redirect_leak"


# --- Baseline executor env wiring + accuracy parse ---


def _write_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
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


def _fake_workspace(slot: Path, *, tput: float = 1500.0) -> Path:
    ws = slot / "benchmark_sglang_20260715_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "sglang",
                "model": "/wekafs/models/Qwen-Qwen3-8B",
                "throughput": {
                    "request_throughput": tput / 256,
                    "output_throughput": tput,
                    "total_token_throughput": tput * 2,
                    "completed_requests": 64,
                    "duration_seconds": 25.0,
                },
                "latency": {
                    "ttft": {"mean_ms": 100.0, "p99_ms": 120.0},
                    "e2el": {"mean_ms": 2000.0, "p99_ms": 2300.0},
                },
            }
        )
    )
    return ws


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-baseline-eval-dir", params=params)
    return SimpleNamespace(task=task, extra={})


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


def test_baseline_exports_eval_result_dir_env(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        captured["env"] = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = asyncio.run(executor(ctx))

    assert result["status"] == "succeeded"
    assert captured["env"].get("EVAL_RESULT_DIR") == captured["env"]["RESULT_DIR"]
    assert captured["env"]["EVAL_RESULT_DIR"] == str(output_dir)


def test_baseline_parses_accuracy_from_eval_result_dir(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"
    # A dir OUTSIDE any search root, standing in for the pre-fix
    # ``/tmp/eval_out-*`` fallback so a missing $EVAL_RESULT_DIR loses the file.
    tmp_fallback = tmp_path / "tmp_eval_out_fallback"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        env = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        # Mimic InferenceX run_lm_eval: write to $EVAL_RESULT_DIR, else /tmp.
        eval_root = env.get("EVAL_RESULT_DIR") or str(tmp_fallback)
        _write_lm_eval_output(Path(eval_root))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = asyncio.run(executor(ctx))

    assert result["status"] == "succeeded"
    assert result.get("accuracy") == pytest.approx(0.83)
    assert result.get("accuracy_task") == "gsm8k"


def test_baseline_skips_accuracy_when_run_eval_disabled(tmp_path):
    """RUN_EVAL off -> no accuracy parse, even if the slot holds stale results.

    The eval-failure fallback reruns with ``RUN_EVAL=false`` reusing the same
    ``output_dir``; a prior attempt's ``results*.json`` may still sit in the slot
    (== ``$EVAL_RESULT_DIR``). Reading eval output must strictly follow running
    eval, so accuracy stays unset and cannot be promoted into baseline_accuracy.
    """
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        env = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        # A stale eval artifact already present in the reused slot: even though
        # THIS run has RUN_EVAL disabled (so lm-eval did not run), the file is
        # here from a prior attempt. It must be ignored.
        eval_root = env.get("EVAL_RESULT_DIR") or str(slot)
        _write_lm_eval_output(Path(eval_root))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {"output_dir": str(output_dir), "timeout_sec": 10, "disable_run_eval": True}
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = asyncio.run(executor(ctx))

    assert result["status"] == "succeeded"
    # Stale results*.json present in the slot, but RUN_EVAL was off this run:
    # accuracy must NOT be set (no stale promotion into baseline_accuracy).
    assert result.get("accuracy") is None