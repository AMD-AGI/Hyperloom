"""Tests for ``orchestrator.accuracy_gate`` — IMPL-CHECKLIST §3.47‒3.56."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import accuracy_gate as ag
from inference_optimizer.orchestrator.accuracy_gate import (
    ACCURACY_RISK_TABLE,
    AccuracyGateError,
    Verdict,
    compare_to_baseline,
    extract_score_from_summary,
    optional_kernel_micro_check,
    requires_gate,
    run_gsm8k,
)


# ---------------------------------------------------------------------------
# requires_gate / risk table
# ---------------------------------------------------------------------------
def test_requires_gate_returns_true_for_kernel_opt():
    assert requires_gate("kernel-opt") is True
    assert requires_gate("kernel_opt") is True   # underscore variant


def test_requires_gate_skips_setup_classify_baseline():
    for n in ("setup", "classify", "baseline", "profile", "sweep", "report"):
        assert requires_gate(n) is False


def test_requires_gate_returns_false_for_unknown_action():
    assert requires_gate("brand-new-action") is False


def test_risk_table_contains_known_actions():
    for n in ("backends", "kernel-opt", "integrate", "comm-optimization"):
        assert n in ACCURACY_RISK_TABLE


# ---------------------------------------------------------------------------
# compare_to_baseline
# ---------------------------------------------------------------------------
def test_compare_keep_when_within_threshold():
    assert compare_to_baseline(0.71, 0.705, threshold=0.01) == Verdict.KEEP


def test_compare_revert_when_drop_exceeds_threshold():
    assert compare_to_baseline(0.71, 0.69, threshold=0.01) == Verdict.REVERT


def test_compare_keep_when_new_higher():
    assert compare_to_baseline(0.71, 0.80) == Verdict.KEEP


def test_compare_fail_on_nan():
    assert compare_to_baseline(float("nan"), 0.5) == Verdict.FAIL
    assert compare_to_baseline(0.5, float("nan")) == Verdict.FAIL


def test_compare_fail_on_out_of_range():
    assert compare_to_baseline(-0.1, 0.5) == Verdict.FAIL
    assert compare_to_baseline(0.5, 1.5) == Verdict.FAIL


# ---------------------------------------------------------------------------
# extract_score_from_summary
# ---------------------------------------------------------------------------
def test_extract_flat_score(tmp_path: Path):
    p = tmp_path / "eval_summary_gsm8k.json"
    p.write_text(json.dumps({"score": 0.42}), encoding="utf-8")
    assert extract_score_from_summary(p) == 0.42


def test_extract_lm_eval_harness_acc(tmp_path: Path):
    p = tmp_path / "eval_summary_gsm8k.json"
    p.write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "acc,none": 0.638,
                        "acc_stderr,none": 0.005,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert extract_score_from_summary(p) == pytest.approx(0.638)


def test_extract_nested_dict(tmp_path: Path):
    p = tmp_path / "eval_summary_gsm8k.json"
    p.write_text(
        json.dumps({"meta": {"foo": "bar"}, "task": {"score": 0.5}}),
        encoding="utf-8",
    )
    assert extract_score_from_summary(p) == 0.5


def test_extract_missing_file_raises(tmp_path: Path):
    with pytest.raises(AccuracyGateError):
        extract_score_from_summary(tmp_path / "does_not_exist.json")


def test_extract_unparseable_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("not really json", encoding="utf-8")
    with pytest.raises(AccuracyGateError):
        extract_score_from_summary(p)


def test_extract_no_score_field_raises(tmp_path: Path):
    p = tmp_path / "eval_summary_gsm8k.json"
    p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(AccuracyGateError):
        extract_score_from_summary(p)


def test_extract_out_of_range_raises(tmp_path: Path):
    p = tmp_path / "eval_summary_gsm8k.json"
    p.write_text(json.dumps({"score": 1.5}), encoding="utf-8")
    with pytest.raises(AccuracyGateError):
        extract_score_from_summary(p)


# ---------------------------------------------------------------------------
# run_gsm8k (subprocess seam patched out)
# ---------------------------------------------------------------------------
def test_run_gsm8k_happy_path(monkeypatch, tmp_path: Path):
    summary = tmp_path / "eval_summary_gsm8k.json"

    def fake_run_eval(**kwargs):
        # Test patches the subprocess seam; write a fixture summary instead.
        results_dir = Path(kwargs["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "eval_summary_gsm8k.json").write_text(
            json.dumps({"score": 0.71}), encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(ag, "_run_eval", fake_run_eval)
    score = run_gsm8k(8000, "/models/llama-3-8b", tmp_path)
    assert score == pytest.approx(0.71)


def test_run_gsm8k_propagates_script_failure(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(ag, "_run_eval", lambda **kwargs: 7)
    with pytest.raises(AccuracyGateError):
        run_gsm8k(8000, "/models/x", tmp_path)


def test_run_gsm8k_passes_through_kwargs(monkeypatch, tmp_path: Path):
    captured: dict = {}

    def fake(**kw):
        captured.update(kw)
        # write summary so post-call extraction succeeds
        (Path(kw["results_dir"]) / f"eval_summary_{kw['eval_task']}.json").write_text(
            json.dumps({"score": 0.5})
        )
        return 0

    monkeypatch.setattr(ag, "_run_eval", fake)
    run_gsm8k(
        9000,
        "/m",
        tmp_path,
        eval_task="mmlu",
        num_fewshot=3,
    )
    assert captured["server_port"] == 9000
    assert captured["eval_task"] == "mmlu"
    assert captured["num_fewshot"] == 3


# ---------------------------------------------------------------------------
# optional_kernel_micro_check
# ---------------------------------------------------------------------------
def test_micro_check_python_lists_close():
    assert optional_kernel_micro_check([1.0, 2.0], [1.0001, 2.0001]) is True


def test_micro_check_python_lists_diverge():
    assert optional_kernel_micro_check([1.0, 2.0], [1.5, 2.0]) is False


def test_micro_check_nested_lists():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[1.0, 2.0], [3.0, 4.0]]
    assert optional_kernel_micro_check(a, b) is True


def test_micro_check_size_mismatch():
    assert optional_kernel_micro_check([1.0], [1.0, 2.0]) is False


def test_micro_check_nan_returns_false():
    assert optional_kernel_micro_check([1.0, float("nan")], [1.0, 1.0]) is False


def test_micro_check_numpy_arrays():
    np = pytest.importorskip("numpy")
    a = np.ones((4, 4)) * 1.5
    b = a + 1e-5
    assert optional_kernel_micro_check(a, b) is True
    c = a.copy()
    c[0, 0] = 9.9
    assert optional_kernel_micro_check(a, c) is False
