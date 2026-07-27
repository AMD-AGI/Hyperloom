"""Classifying an eval-rooted baseline failure, and who may redefine the gate.

When InferenceX's ``run_eval`` aborts the benchmark, the executor has to tell
that apart from an ordinary benchmark failure: the retry-with-``RUN_EVAL=false``
path is only correct for tasks that are *not* establishing the accuracy
reference. Getting this wrong either burns a second full benchmark for nothing
or silently redefines the quality reference from a throughput-only run.
"""

from __future__ import annotations

import hyperloom.orchestrator.actions.executors.baseline as baseline_mod


class _Executor(baseline_mod.BaselineExecutor):
    """Bare instance: these are pure classification helpers, no ctx needed."""

    def __init__(self):  # noqa: D107 - deliberately skips the real __init__
        pass


def test_error_text_carrying_a_run_eval_marker_is_eval_rooted():
    ex = _Executor()
    assert ex._is_eval_rooted_failure(
        {"error": "...\nERROR: run_eval failed with exit code 1\n"}
    ) is True


def test_the_rejected_flag_itself_counts_as_eval_rooted():
    """The flag message is the shape that killed real runs; it must classify."""
    ex = _Executor()
    assert ex._is_eval_rooted_failure(
        {"error": "Unknown parameter: --concurrent-requests"}
    ) is True


def test_marker_in_a_nonfatal_warning_still_classifies():
    ex = _Executor()
    assert ex._is_eval_rooted_failure(
        {"error": "", "nonfatal_warnings": ["run_eval failed with exit code 1"]}
    ) is True


def test_an_ordinary_benchmark_failure_is_not_eval_rooted():
    ex = _Executor()
    assert ex._is_eval_rooted_failure(
        {"error": "CUDA out of memory", "nonfatal_warnings": ["slow start"]}
    ) is False


def test_empty_result_is_not_eval_rooted_and_does_not_raise():
    ex = _Executor()
    assert ex._is_eval_rooted_failure({}) is False


def test_only_a_genuine_baseline_may_establish_the_quality_reference():
    assert baseline_mod._should_establish_quality_ref("baseline") is True
    # replay_warm_recipe reuses this executor but is a candidate: letting it
    # redefine the reference would mask its own deviation from the baseline.
    assert baseline_mod._should_establish_quality_ref("replay_warm_recipe") is False
    assert baseline_mod._should_establish_quality_ref("") is False
    assert baseline_mod._should_establish_quality_ref(None) is False
