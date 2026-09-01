# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the ``$EVAL_RESULT_DIR`` wiring (P0 accuracy-gate fix).

InferenceX ``run_lm_eval`` (benchmark_lib.sh) reads ``$EVAL_RESULT_DIR`` for
lm-eval's ``--output_path``; unset, it falls back to ``/tmp/eval_out-*`` so the
``results*.json`` escape the task workspace and the accuracy gate sees no
baseline (``baseline_accuracy=0.0`` -> throughput-only KEEP). These tests pin
that:

* the baseline / grid subprocess env exports ``$EVAL_RESULT_DIR`` under, but
  separate from, ``$RESULT_DIR`` so lm-eval cleanup cannot delete Magpie traces;
  and
* the accuracy parse search root is aligned to that dir, where lm-eval
  (lm_eval 0.4.9.2) writes ``<root>/<model_sanitized>/results_<ts>.json``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    accuracy_passed,
    parse_eval_results,
)
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
    # lm-eval writes outside the Magpie ``benchmark_*`` workspace. Searching from
    # the benchmark_* subdir (the pre-fix baseline root) cannot reach it.
    _write_lm_eval_output(tmp_path)
    bench_ws = tmp_path / "benchmark_sglang_20260715_010101"
    bench_ws.mkdir(parents=True)
    out = parse_eval_results(bench_ws, framework="sglang")
    assert out.get("accuracy") is None


# --- Grid runner env wiring ---


def test_run_magpie_exports_eval_result_dir_under_result_dir(tmp_path, monkeypatch):
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
    assert captured["env"]["RESULT_DIR"] == str(tmp_path / "slot")
    assert captured["env"]["EVAL_RESULT_DIR"] == str(tmp_path / "slot" / "eval_output")


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
    assert captured["env"]["RESULT_DIR"] == "/tmp/redirect_leak"
    assert captured["env"]["EVAL_RESULT_DIR"] == "/tmp/redirect_leak/eval_output"


def test_run_magpie_keeps_magpie_traces_when_eval_output_is_cleaned(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "skip-kill")
    output_dir = tmp_path / "slot"
    trace_file = output_dir / "benchmark_sglang_20260716_010101" / "magpie_trace.json"

    def fake_run(cmd, *args, **kwargs):
        env = dict(kwargs.get("env") or {})
        trace_file.parent.mkdir(parents=True)
        trace_file.write_text("{}", encoding="utf-8")
        eval_dir = Path(env["EVAL_RESULT_DIR"])
        (eval_dir / "model__sanitized").mkdir(parents=True, exist_ok=True)
        (eval_dir / "model__sanitized" / "results.json").write_text("{}", encoding="utf-8")
        # Match benchmark_lib.sh cleanup after lm-eval output is processed.
        shutil.rmtree(eval_dir)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=fake_run,
    ):
        _run_magpie(
            magpie_python="/opt/venv/bin/python",
            config_path=tmp_path / "config.yaml",
            output_dir=output_dir,
            timeout_sec=5,
            cwd=str(tmp_path),
        )

    assert trace_file.exists()


# --- Baseline executor env wiring + accuracy parse ---


def _write_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
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
                "model": "/path/models/Qwen-Qwen3-8B",
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


class _StopRecorder:
    """Minimal SharedState stub capturing baseline stop requests."""

    def __init__(self) -> None:
        self.stop_reason = ""

    def set_stop_reason(self, value, **_kwargs):
        self.stop_reason = value
        return value


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
    # Single-round baseline so RESULT_DIR is the task ``output_dir`` itself; the
    # cold-start double-run guard otherwise runs warmup+measure in
    # ``output_dir/{warmup,measure}_round`` sub-slots, which this env-wiring
    # assertion is not about.
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "baseline_double_run": False})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = asyncio.run(executor(ctx))

    assert result["status"] == "succeeded"
    assert captured["env"]["RESULT_DIR"] == str(output_dir)
    assert captured["env"]["EVAL_RESULT_DIR"] == str(output_dir / "eval_output")


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


def test_baseline_anchors_relative_result_dir_before_accuracy_parse(tmp_path):
    """A relative result_dir must mean the same directory to Magpie and Hyperloom."""
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        env = dict(kwargs.get("env") or {})
        cwd = Path(kwargs["cwd"])
        captured["env"] = env
        _fake_workspace(slot)

        # The subprocess interprets relative RESULT_DIR from its cwd
        # (the per-task output_dir). Hyperloom must parse from the same absolute
        # location, not from the coordinator/repo cwd.
        result_root = Path(env["RESULT_DIR"])
        if not result_root.is_absolute():
            result_root = cwd / result_root
        _write_lm_eval_output(result_root / "eval_processed")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    shared_state = _StopRecorder()
    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=shared_state,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "baseline_double_run": False,
            "framework": "vllm",
            "result_dir": "runs/baseline/mp-backend",
        }
    )
    ctx.task.kind = "baseline"
    ctx.extra["shared_state"] = shared_state

    with (
        patch("hyperloom.orchestrator.actions.executors._ray_serving.maybe_serving_lease", return_value=None),
        patch(
            "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
            side_effect=fake_run,
        ),
    ):
        result = asyncio.run(executor(ctx))

    assert result["status"] == "succeeded"
    assert Path(captured["env"]["RESULT_DIR"]).is_absolute()
    assert result.get("output_dir") == str(output_dir)
    assert result.get("accuracy") == pytest.approx(0.83)
    assert result.get("accuracy_task") == "gsm8k"
    assert shared_state.stop_reason == ""


def test_baseline_parses_accuracy_after_eval_result_dir_cleanup(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        env = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        eval_root = Path(env["EVAL_RESULT_DIR"])
        raw_result = _write_lm_eval_output(eval_root)
        processed_dir = Path(env["RESULT_DIR"]) / "eval_processed" / raw_result.parent.name
        processed_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(raw_result.parent), str(processed_dir))
        shutil.rmtree(eval_root)
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
    assert "eval_processed" in (result.get("accuracy_source") or "")


def test_baseline_mn_warmup_eval_result_dir_is_discarded(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"
    calls: list[tuple[Path, dict]] = []

    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")

    async def fake_restart_server_for_round(*args, **kwargs):
        return None

    from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as mnl

    monkeypatch.setattr(mnl, "restart_server_for_round", fake_restart_server_for_round)

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        env = dict(kwargs.get("env") or {})
        calls.append((slot, env))
        if slot.name == "mn_warmup":
            _write_lm_eval_output(Path(env["EVAL_RESULT_DIR"]))
        else:
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
    assert [slot.name for slot, _env in calls] == ["mn_warmup", "ws"]
    assert calls[0][1]["RESULT_DIR"] == str(output_dir / "mn_warmup")
    assert calls[0][1]["EVAL_RESULT_DIR"] == str(output_dir / "mn_warmup" / "eval_output")
    assert calls[1][1]["EVAL_RESULT_DIR"] == str(output_dir / "eval_output")
    # The warmup keeps its eval output in its OWN slot (asserted above) so a
    # measured round never grades against it by accident. It is still a usable
    # accuracy source when it is the only one: accuracy is a property of the
    # model, not of a cold-vs-hot benchmark window, and the baseline double-run
    # now evaluates only in the warmup round.
    assert result.get("accuracy") == pytest.approx(0.83)


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
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "disable_run_eval": True})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = asyncio.run(executor(ctx))

    assert result["status"] == "succeeded"
    # Stale results*.json present in the slot, but RUN_EVAL was off this run:
    # accuracy must NOT be set (no stale promotion into baseline_accuracy).
    assert result.get("accuracy") is None


def _write_results_score(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score, "alias": "gsm8k"}}}),
        encoding="utf-8",
    )


# --- Finding 1: discarded warmup round must never be graded ---


def test_parse_eval_results_ignores_discarded_warmup_round(tmp_path):
    """integrate_patch grades from the grid slot (parent of the measured
    ``benchmark_*`` workspace). ``run_grid`` nests the discarded warmup eval under
    ``warmup_round/``, whose path sorts lexicographically AFTER the measured
    ``<model>/`` dir, so ``sorted(...)[-1]`` would wrongly pick the warmup score.
    The measured round must win.
    """
    slot = tmp_path / "variant_00_kv"
    # Measured round eval at the slot root: slot/<model>/results_<ts>.json.
    _write_results_score(slot / "Qwen__model" / "results_2026-07-15T10-00-00.000000.json", 0.90)
    # Discarded warmup round eval nested under warmup_round/ (worse score, and a
    # path that sorts last so the pre-fix sorted(...)[-1] would select it).
    _write_results_score(
        slot / "warmup_round" / "Qwen__model" / "results_2026-07-15T09-00-00.000000.json",
        0.50,
    )
    out = parse_eval_results(slot, framework="vllm")
    assert out.get("accuracy") == pytest.approx(0.90)
    assert "warmup_round" not in (out.get("source_file") or "")


def test_parse_eval_results_ignores_discarded_mn_warmup_round(tmp_path):
    slot = tmp_path / "baseline"
    _write_results_score(
        slot / "eval_processed" / "Qwen__model" / "results_2026-07-15T10-00-00.000000.json",
        0.90,
    )
    _write_results_score(
        slot / "mn_warmup" / "eval_output" / "Qwen__model" / "results_2026-07-15T09-00-00.000000.json",
        0.50,
    )
    out = parse_eval_results(slot, framework="vllm")
    assert out.get("accuracy") == pytest.approx(0.90)
    assert "mn_warmup" not in (out.get("source_file") or "")


def test_parse_eval_results_keeps_results_when_root_is_warmup_slot(tmp_path):
    """The warmup filter is workspace-relative, not absolute: a parse rooted AT a
    ``warmup_round`` slot (the baseline warmup round parses its own
    ``RESULT_DIR == .../warmup_round``) must still find its own results.
    """
    warm_slot = tmp_path / "warmup_round"
    _write_results_score(warm_slot / "Qwen__model" / "results_2026-07-15T10-00-00.000000.json", 0.77)
    out = parse_eval_results(warm_slot, framework="vllm")
    assert out.get("accuracy") == pytest.approx(0.77)


def test_warm_decision_gated_variant_grades_from_warmup_round(tmp_path):
    """Warm-decision explore runs the decision round with ``RUN_EVAL=false``, so a
    gated variant's only score sits under ``warmup_round/``. The gate must
    grade from it and PASS rather than REVERT as ``accuracy_unavailable``.
    """
    slot = tmp_path / "variant_00_kv"
    _write_results_score(
        slot / "warmup_round" / "Qwen__model" / "results_2026-07-15T09-00-00.000000.json",
        0.9462,
    )

    out = parse_eval_results(slot, framework="vllm")
    accuracy = out.get("accuracy")
    assert accuracy == pytest.approx(0.9462)
    assert accuracy_passed(0.9462, float(accuracy)) is True


# --- Finding 2: RUN_EVAL off in the base YAML (not extra_envs) is honored ---


def test_baseline_skips_accuracy_when_run_eval_off_in_base_yaml(tmp_path):
    """RUN_EVAL=false coming from the base YAML ``benchmark.envs`` (not
    ``extra_envs``) must be honored: baseline reads the effective RUN_EVAL from
    the materialized config, so a stale ``results*.json`` in the reused slot is
    not promoted into ``baseline_accuracy``.
    """
    base = tmp_path / "base.yaml"
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256, "RUN_EVAL": False},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with base.open("w") as f:
        yaml.safe_dump(cfg, f)
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        env = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        # Stale results in the slot from a prior attempt; RUN_EVAL is off in the
        # base YAML this run, so lm-eval did not run and this must be ignored.
        eval_root = env.get("EVAL_RESULT_DIR") or str(slot)
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
    assert result.get("accuracy") is None


# --- integrate_patch grade layer: warmup round must never gate the patch ---


def test_integrate_patch_grade_ignores_discarded_warmup_round(tmp_path):
    """``IntegratePatchExecutor._grade_accuracy`` grades from the grid slot (the
    parent of the measured ``benchmark_*`` workspace). ``run_grid`` writes the
    discarded warmup eval under ``warmup_round/``; if it were graded instead of
    the measured round, a good patch could be wrongly reverted. Grading must use
    the measured round's score.
    """
    from hyperloom.orchestrator.actions.executors.integrate_patch import (
        IntegratePatchExecutor,
    )

    slot = tmp_path / "variant_00_integrate-patch"
    # Measured round (slot root): high score that PASSES the gate vs baseline.
    _write_results_score(slot / "Qwen__model" / "results_2026-07-15T10-00-00.000000.json", 0.95)
    # Discarded warmup round (nested): low score that would FAIL the gate, and a
    # path that sorts last so a pre-fix sorted(...)[-1] would grade it.
    _write_results_score(
        slot / "warmup_round" / "Qwen__model" / "results_2026-07-15T09-00-00.000000.json",
        0.50,
    )
    # baseline 0.90: measured 0.95 is within tolerance (pass); warmup 0.50 is not.
    passed = IntegratePatchExecutor._grade_accuracy(str(slot), 0.90, framework="vllm")
    assert passed is True


# --- scriptable quality gate must survive a RUN_EVAL-off run ---------------


def _fake_scriptable_workspace(slot: Path, *, gate_passed: bool = True) -> Path:
    """A scriptable (xDiT) bench workspace: framework=xdit plus a fresh image
    ``quality_gate`` block embedded in ``benchmark_report.json``.

    ``RUN_EVAL`` gates only the serving lm-eval GSM8K run; the scriptable gate is
    computed by the bench script and written every run regardless, so it must
    still be read when ``RUN_EVAL`` is off.
    """
    ws = slot / "benchmark_xdit_20260715_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "xdit",
                "model": "/path/models/xdit-diffusion",
                "throughput": {
                    "request_throughput": 5.0,
                    "output_throughput": 5.0,
                    "total_token_throughput": 5.0,
                    "completed_requests": 8,
                    "duration_seconds": 25.0,
                },
                "latency": {
                    "ttft": {"mean_ms": 100.0, "p99_ms": 120.0},
                    "e2el": {"mean_ms": 2000.0, "p99_ms": 2300.0},
                },
                "quality_gate": {"passed": gate_passed},
            }
        )
    )
    return ws


def test_baseline_reads_scriptable_quality_gate_when_run_eval_disabled(tmp_path):
    """RUN_EVAL off must NOT drop a scriptable framework's image quality gate.

    ``RUN_EVAL`` governs only the serving lm-eval GSM8K run. Scriptable (xDiT)
    workloads carry no lm-eval; their sole correctness signal is the image
    ``quality_gate`` embedded in ``benchmark_report.json``, freshly written every
    run (``parse_quality_gate`` picks the newest by mtime -> no staleness). So
    when ``RUN_EVAL`` is off (here via ``disable_run_eval``), the accuracy parse
    must still resolve the quality gate rather than skip entirely and leave
    ``baseline_accuracy=0`` -> throughput-only KEEP.
    """
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_scriptable_workspace(slot, gate_passed=True)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10, "disable_run_eval": True})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = asyncio.run(executor(ctx))

    assert result["status"] == "succeeded"
    # RUN_EVAL off, but the scriptable gate passed -> accuracy=1.0 (not skipped).
    assert result.get("accuracy") == pytest.approx(1.0)
    assert result.get("accuracy_task") == "quality_gate"
