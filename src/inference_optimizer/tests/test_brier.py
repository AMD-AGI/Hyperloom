"""Tests for ``orchestrator.brier`` — IMPL-CHECKLIST §12.1‒12.4."""
from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.brier import (
    BrierEntry,
    BrierTracker,
    weight_for_score,
)


# ---------------------------------------------------------------------------
def test_brier_entry_perfect_prediction():
    e = BrierEntry.make("critic", 5.0, 5.0)
    assert e.score == 0.0


def test_brier_entry_max_score_is_one():
    e = BrierEntry.make("critic", 0.0, 200.0)
    assert e.score == 1.0


def test_brier_entry_normalises_to_unit_interval():
    e = BrierEntry.make("critic", 10.0, 60.0)
    assert e.score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
def test_score_for_unseen_agent_uses_prior():
    t = BrierTracker(prior=0.25, prior_k=5)
    assert t.score_for("critic") == 0.25


def test_score_for_after_one_record_smooths():
    t = BrierTracker(prior=0.25, prior_k=5)
    t.record("critic", predicted_gain_pct=5.0, actual_gain_pct=5.0)
    # observed=0.0, prior pull keeps us above the perfect score
    assert t.score_for("critic") == pytest.approx((0 + 0.25 * 5) / (1 + 5))


def test_score_for_converges_to_observed_with_many_samples():
    t = BrierTracker(prior=0.5, prior_k=3, window=200)
    for _ in range(100):
        t.record("critic", predicted_gain_pct=5.0, actual_gain_pct=5.0)
    assert t.score_for("critic") < 0.05


def test_record_keeps_window_bounded():
    t = BrierTracker(window=3)
    for i in range(5):
        t.record("critic", predicted_gain_pct=float(i), actual_gain_pct=0.0)
    h = t.history("critic")
    assert len(h) == 3


def test_weight_for_score_clamps_lower_bound():
    assert weight_for_score(1.0) == pytest.approx(0.5)
    assert weight_for_score(2.0) == pytest.approx(0.5)  # clamped via score min


def test_weight_for_score_perfect_returns_one():
    assert weight_for_score(0.0) == 1.0


def test_weight_for_agent_via_tracker():
    t = BrierTracker(prior=0.0, prior_k=1)
    t.record("a", predicted_gain_pct=0.0, actual_gain_pct=0.0)
    assert t.weight_for("a") == pytest.approx(1.0)


def test_per_agent_independence():
    t = BrierTracker(prior=0.5, prior_k=1)
    t.record("a", predicted_gain_pct=10.0, actual_gain_pct=10.0)  # perfect
    t.record("b", predicted_gain_pct=0.0, actual_gain_pct=100.0)  # max miss
    assert t.score_for("a") < t.score_for("b")


# ---------------------------------------------------------------------------
def test_snapshot_round_trip():
    t = BrierTracker(window=20, prior=0.3, prior_k=4)
    t.record("a", predicted_gain_pct=5.0, actual_gain_pct=5.0)
    t.record("a", predicted_gain_pct=20.0, actual_gain_pct=10.0)
    snap = t.snapshot()
    restored = BrierTracker.restore(snap)
    assert restored.score_for("a") == pytest.approx(t.score_for("a"))
    assert restored.window == 20
    assert restored.prior == 0.3
