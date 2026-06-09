"""Tests for `driver.eval` — threshold resolution + acceptance decision."""

from __future__ import annotations

import pytest

from quantization_agent.driver.eval import (
    DEFAULT_ACCEPTABLE_GAP,
    EvalDecision,
    decide,
    resolve_threshold,
)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_threshold — priority chain (§3.1)
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_threshold_default_when_nothing_provided(tmp_path):
    t, src = resolve_threshold(tmp_path, acceptable_eval_gap=None)
    assert t == DEFAULT_ACCEPTABLE_GAP
    assert src == "default"


def test_resolve_threshold_arg_wins_over_file(tmp_path):
    (tmp_path / "eval_gap_threshold.txt").write_text("0.10", encoding="utf-8")
    t, src = resolve_threshold(tmp_path, acceptable_eval_gap=0.05)
    assert t == 0.05
    assert src == "arg"


def test_resolve_threshold_file_used_when_no_arg(tmp_path):
    (tmp_path / "eval_gap_threshold.txt").write_text("0.07\n", encoding="utf-8")
    t, src = resolve_threshold(tmp_path, acceptable_eval_gap=None)
    assert t == 0.07
    assert src == "file"


def test_resolve_threshold_malformed_file_falls_to_default(tmp_path):
    (tmp_path / "eval_gap_threshold.txt").write_text("not-a-float", encoding="utf-8")
    t, src = resolve_threshold(tmp_path, acceptable_eval_gap=None)
    assert t == DEFAULT_ACCEPTABLE_GAP
    assert src == "default"


def test_resolve_threshold_empty_file_falls_to_default(tmp_path):
    (tmp_path / "eval_gap_threshold.txt").write_text("   \n", encoding="utf-8")
    t, src = resolve_threshold(tmp_path, acceptable_eval_gap=None)
    assert t == DEFAULT_ACCEPTABLE_GAP
    assert src == "default"


def test_resolve_threshold_arg_zero_is_respected(tmp_path):
    # 0.0 is a valid (strict) threshold — arg path must not collapse to None.
    t, src = resolve_threshold(tmp_path, acceptable_eval_gap=0.0)
    assert t == 0.0
    assert src == "arg"


# ─────────────────────────────────────────────────────────────────────────────
# decide — gap acceptance
# ─────────────────────────────────────────────────────────────────────────────

def _report(gap: float) -> dict:
    return {
        "metric_name": "gsm8k",
        "dataset": "gsm8k",
        "backend": "vllm",
        "source_score": 0.50,
        "quantized_score": 0.50 * (1 - gap),
        "relative_gap": gap,
    }


def test_decide_within_threshold(tmp_path):
    d = decide(_report(0.02), workspace=tmp_path, acceptable_eval_gap=0.03)
    assert d.status == "within"
    assert d.relative_gap == 0.02
    assert d.threshold == 0.03
    assert d.threshold_source == "arg"


def test_decide_exactly_at_threshold_is_within(tmp_path):
    d = decide(_report(0.03), workspace=tmp_path, acceptable_eval_gap=0.03)
    assert d.status == "within"


def test_decide_exceeded(tmp_path):
    d = decide(_report(0.10), workspace=tmp_path, acceptable_eval_gap=0.03)
    assert d.status == "exceeded"
    assert d.relative_gap == 0.10


def test_decide_missing_when_report_is_none(tmp_path):
    d = decide(None, workspace=tmp_path, acceptable_eval_gap=None)
    assert d.status == "missing"
    assert d.relative_gap is None
    assert d.threshold == DEFAULT_ACCEPTABLE_GAP


def test_decide_missing_when_required_key_absent(tmp_path):
    d = decide({"metric_name": "gsm8k"}, workspace=tmp_path, acceptable_eval_gap=None)
    assert d.status == "missing"
    assert d.relative_gap is None


def test_decide_missing_when_gap_not_float(tmp_path):
    bad = _report(0.02)
    bad["relative_gap"] = "n/a"
    d = decide(bad, workspace=tmp_path, acceptable_eval_gap=None)
    assert d.status == "missing"


def test_decide_uses_file_threshold_when_no_arg(tmp_path):
    (tmp_path / "eval_gap_threshold.txt").write_text("0.05", encoding="utf-8")
    d = decide(_report(0.04), workspace=tmp_path, acceptable_eval_gap=None)
    assert d.status == "within"
    assert d.threshold == 0.05
    assert d.threshold_source == "file"


def test_decide_uses_default_threshold_when_nothing_set(tmp_path):
    d = decide(_report(0.029), workspace=tmp_path, acceptable_eval_gap=None)
    assert d.status == "within"
    assert d.threshold == DEFAULT_ACCEPTABLE_GAP
    assert d.threshold_source == "default"


def test_decide_negative_gap_is_within(tmp_path):
    # Quantized scored higher than source (rare but possible with noisy evals).
    d = decide(_report(-0.01), workspace=tmp_path, acceptable_eval_gap=0.03)
    assert d.status == "within"
    assert d.relative_gap == -0.01


def test_eval_decision_is_frozen():
    d = EvalDecision(status="within", relative_gap=0.0, threshold=0.03, threshold_source="default")
    with pytest.raises(Exception):
        d.status = "exceeded"  # type: ignore[misc]
