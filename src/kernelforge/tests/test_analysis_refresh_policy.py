"""Tests for deterministic Analysis refresh admission."""

from __future__ import annotations

from kernelforge.loop.analysis_refresh_policy import decide_analysis_refresh


def _decide(**overrides):
    values = {
        "canonical_commit": "b" * 40,
        "evidence_commit": "a" * 40,
        "evidence_mean_case_speedup": 1.0,
        "evidence_status": "profiled",
        "current_mean_case_speedup": 1.0,
        "supervisor_due": False,
        "last_attempt_commit": "",
        "last_attempt_status": "",
        "last_attempt_iteration": -1,
        "current_iteration": 4,
    }
    values.update(overrides)
    return decide_analysis_refresh(**values)


def test_initial_analysis_is_required_without_evidence():
    decision = _decide(
        evidence_commit="",
        evidence_mean_case_speedup=None,
    )

    assert decision.refresh is True
    assert decision.reasons == ("INITIAL_ANALYSIS",)


def test_cumulative_gain_is_relative_to_last_evidence_score():
    below = _decide(
        evidence_mean_case_speedup=1.1,
        current_mean_case_speedup=1.1549,
    )
    reached = _decide(
        evidence_mean_case_speedup=1.1,
        current_mean_case_speedup=1.155,
    )

    assert below.refresh is False
    assert below.reasons == ("CUMULATIVE_GAIN_BELOW_THRESHOLD",)
    assert reached.refresh is True
    assert reached.reasons == ("CUMULATIVE_GAIN",)


def test_supervisor_refreshes_only_stale_evidence():
    stale = _decide(supervisor_due=True)
    current = _decide(
        canonical_commit="a" * 40,
        supervisor_due=True,
    )

    assert stale.refresh is True
    assert stale.reasons == ("SUPERVISOR_STALE_EVIDENCE",)
    assert current.refresh is False
    assert current.reasons == ("CURRENT_EVIDENCE",)


def test_threshold_and_supervisor_coalesce_into_one_refresh():
    decision = _decide(
        current_mean_case_speedup=1.05,
        supervisor_due=True,
    )

    assert decision.refresh is True
    assert decision.reasons == (
        "CUMULATIVE_GAIN",
        "SUPERVISOR_STALE_EVIDENCE",
    )


def test_failed_attempt_retries_only_in_later_planning_iteration():
    same_iteration = _decide(
        last_attempt_commit="b" * 40,
        last_attempt_status="failed",
        last_attempt_iteration=4,
        current_mean_case_speedup=1.2,
        supervisor_due=True,
    )
    next_iteration = _decide(
        last_attempt_commit="b" * 40,
        last_attempt_status="failed",
        last_attempt_iteration=4,
        current_iteration=5,
    )

    assert same_iteration.refresh is False
    assert same_iteration.reasons == ("ALREADY_ATTEMPTED_THIS_ITERATION",)
    assert next_iteration.refresh is True
    assert next_iteration.reasons == ("RETRY_FAILED_ANALYSIS",)


def test_exhausted_attempt_budget_blocks_same_commit():
    decision = _decide(
        last_attempt_commit="b" * 40,
        last_attempt_status="exhausted",
        last_attempt_iteration=3,
        current_iteration=5,
        supervisor_due=True,
    )

    assert decision.refresh is False
    assert decision.reasons == ("ANALYSIS_ATTEMPTS_EXHAUSTED",)


def test_partial_bundle_retries_only_in_later_iteration():
    same_iteration = _decide(
        canonical_commit="a" * 40,
        evidence_status="partial",
        last_attempt_iteration=4,
        current_iteration=4,
    )
    next_iteration = _decide(
        canonical_commit="a" * 40,
        evidence_status="partial",
        last_attempt_iteration=4,
        current_iteration=5,
    )

    assert same_iteration.refresh is False
    assert next_iteration.refresh is True
    assert next_iteration.reasons == ("PARTIAL_UPGRADE",)
