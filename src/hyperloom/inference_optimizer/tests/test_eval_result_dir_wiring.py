# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the ``$EVAL_RESULT_DIR`` wiring (P0 accuracy-gate fix)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.gain_math import accuracy_passed
from hyperloom.orchestrator.actions.executors._accuracy_gate import parse_eval_results


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
    # lm-eval writes one directory level below --output_path; the recursive ``**/results*.json`` glob must catch it
    # from the aligned root.
    _write_lm_eval_output(tmp_path)
    out = parse_eval_results(tmp_path, framework="sglang")
    assert out.get("accuracy") == pytest.approx(0.83)
    assert out.get("task") == "gsm8k"


def test_parse_eval_results_misses_when_root_is_benchmark_subdir(tmp_path):
    # lm-eval writes outside the ``benchmark_*`` workspace.
    _write_lm_eval_output(tmp_path)
    bench_ws = tmp_path / "benchmark_sglang_20260715_010101"
    bench_ws.mkdir(parents=True)
    out = parse_eval_results(bench_ws, framework="sglang")
    assert out.get("accuracy") is None


# --- Grid runner env wiring ---


# --- Baseline executor env wiring + accuracy parse ---


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-baseline-eval-dir", params=params)
    return SimpleNamespace(task=task, extra={})


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


def _write_results_score(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score, "alias": "gsm8k"}}}),
        encoding="utf-8",
    )


# --- Finding 1: discarded warmup round must never be graded ---


def test_parse_eval_results_ignores_discarded_warmup_round(tmp_path):
    """integrate_patch grades from the grid slot (parent of the measured ``benchmark_*`` workspace)."""
    slot = tmp_path / "variant_00_kv"
    # Measured round eval at the slot root: slot/<model>/results_<ts>.json.
    _write_results_score(slot / "Qwen__model" / "results_2026-07-15T10-00-00.000000.json", 0.90)
    # Discarded warmup round eval nested under warmup_round/ (worse score, and a path that sorts last so the pre-fix
    # sorted(...)[-1] would select it).
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
    """The warmup filter is workspace-relative, not absolute: a parse rooted AT a ``warmup_round`` slot (the baseline warmup round parses its own ``RESULT_DIR == .../warmup_round``) must still find its own results."""
    warm_slot = tmp_path / "warmup_round"
    _write_results_score(warm_slot / "Qwen__model" / "results_2026-07-15T10-00-00.000000.json", 0.77)
    out = parse_eval_results(warm_slot, framework="vllm")
    assert out.get("accuracy") == pytest.approx(0.77)


def test_warm_decision_gated_variant_grades_from_warmup_round(tmp_path):
    """Warm-decision explore runs the decision round with ``RUN_EVAL=false``, so a gated variant's only score sits under ``warmup_round/``."""
    slot = tmp_path / "variant_00_kv"
    _write_results_score(
        slot / "warmup_round" / "Qwen__model" / "results_2026-07-15T09-00-00.000000.json",
        0.9462,
    )

    out = parse_eval_results(slot, framework="vllm")
    accuracy = out.get("accuracy")
    assert accuracy == pytest.approx(0.9462)
    assert accuracy_passed(0.9462, float(accuracy)) is True


# --- integrate_patch grade layer: warmup round must never gate the patch ---


def test_integrate_patch_grade_ignores_discarded_warmup_round(tmp_path):
    """``grade_accuracy`` grades from the grid slot (the parent of the measured ``benchmark_*`` workspace)."""
    from hyperloom.orchestrator.actions.executors._accuracy_gate import AccuracyReference, grade_accuracy

    slot = tmp_path / "variant_00_integrate-patch"
    # Measured round (slot root): high score that PASSES the gate vs baseline.
    _write_results_score(slot / "Qwen__model" / "results_2026-07-15T10-00-00.000000.json", 0.95)
    # Discarded warmup round (nested): low score that would FAIL the gate, and a path that sorts last so a pre-fix
    # sorted(...)[-1] would grade it.
    _write_results_score(
        slot / "warmup_round" / "Qwen__model" / "results_2026-07-15T09-00-00.000000.json",
        0.50,
    )
    # baseline 0.90: measured 0.95 is within tolerance (pass); warmup 0.50 is not.
    passed = grade_accuracy(str(slot), AccuracyReference(0.90, ""), framework="vllm")
    assert passed is True


# --- scriptable quality gate must survive a RUN_EVAL-off run ---------------
