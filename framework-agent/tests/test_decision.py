"""Tests for framework_agent.decision (winner_decision + candidate_score)."""

from __future__ import annotations

import pytest

from dataclasses import replace

from framework_agent.decision import candidate_score, winner_decision
from framework_agent.models import ExploreRequest


def _req(min_ratio: float = 1.05, max_drop: float = 0.05) -> ExploreRequest:
    """Minimal ExploreRequest for gate / score testing."""
    return ExploreRequest.from_dict({
        "framework": "sglang",
        "repo_url": "https://github.com/x/y.git",
        "work_dir": "/tmp/x",
        "baseline": {"throughput": 100.0, "accuracy": 0.9, "completed": "1/1"},
        "thresholds": {"min_throughput_ratio": min_ratio, "max_accuracy_drop": max_drop},
    })


# winner_decision -----------------------------------------------------------


def test_winner_decision_pass() -> None:
    """All gates green -> winner."""
    win, reason = winner_decision(_req(), 200.0, 0.9, "1/1")
    assert win is True
    assert "gates passed" in reason


def test_winner_decision_missing_throughput() -> None:
    """Missing throughput fails the first gate."""
    win, reason = winner_decision(_req(), None, 0.9, "1/1")
    assert win is False
    assert "missing throughput" in reason


def test_winner_decision_throughput_below_ratio() -> None:
    """throughput / baseline < min_throughput_ratio -> rejected."""
    win, reason = winner_decision(_req(), 104.0, 0.9, "1/1")
    assert win is False
    assert "throughput ratio" in reason


def test_winner_decision_zero_baseline_throughput_rejects() -> None:
    """Baseline throughput == 0 cannot produce a ratio; defence-in-depth reject.

    ``Baseline.from_dict`` already enforces ``> 0`` so this branch is normally
    unreachable from JSON inputs; this guards against a future code path that
    builds Baseline programmatically (e.g. a unit test stub).
    """
    req = _req()
    bad_baseline = replace(req.baseline, throughput=0.0)
    bad_req = replace(req, baseline=bad_baseline)
    win, reason = winner_decision(bad_req, 100.0, 0.9, "1/1")
    assert win is False
    assert "baseline throughput is 0" in reason


def test_winner_decision_accuracy_drop_too_large() -> None:
    """Accuracy drop above the configured max -> rejected."""
    win, reason = winner_decision(_req(), 200.0, 0.5, "1/1")
    assert win is False
    assert "accuracy drop" in reason


def test_winner_decision_missing_accuracy_when_baseline_set() -> None:
    """Baseline accuracy present + candidate missing -> rejected."""
    win, reason = winner_decision(_req(), 200.0, None, "1/1")
    assert win is False
    assert "missing accuracy" in reason


def test_winner_decision_incomplete_benchmark() -> None:
    """completed='50/100' is a partial run and cannot be promoted."""
    win, reason = winner_decision(_req(), 200.0, 0.9, "50/100")
    assert win is False
    assert "incomplete" in reason


# candidate_score -----------------------------------------------------------


def test_candidate_score_returns_zero_on_missing_throughput() -> None:
    """Failed candidates score 0 so they sort to the tail."""
    assert candidate_score(_req(), None, 0.9) == 0.0
    assert candidate_score(_req(), 0.0, 0.9) == 0.0


def test_candidate_score_ratio_when_no_accuracy_drop() -> None:
    """Score equals throughput ratio when accuracy is intact."""
    score = candidate_score(_req(), 200.0, 0.9)
    assert score == pytest.approx(2.0)


def test_candidate_score_penalises_accuracy_drop() -> None:
    """Accuracy drop reduces score below ratio so a clean candidate ranks higher."""
    score_drop = candidate_score(_req(), 200.0, 0.85)
    score_clean = candidate_score(_req(), 200.0, 0.9)
    assert score_drop < score_clean


def test_candidate_score_orders_two_candidates_correctly() -> None:
    """Higher throughput at same accuracy must score higher."""
    high = candidate_score(_req(), 300.0, 0.9)
    low = candidate_score(_req(), 150.0, 0.9)
    assert high > low
