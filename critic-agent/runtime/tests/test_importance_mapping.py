"""Tests for :mod:`runtime.importance_mapping`."""

from __future__ import annotations

from runtime.importance_mapping import (
    CRITIC_IMPORTANCE_CEILING,
    cap_importance,
    importance_for_kb_draft,
    importance_for_verdict,
)


def test_high_with_measurement_scores_above_default():
    score = importance_for_verdict(
        verdict="reject", confidence="high", has_measurement=True
    )
    assert score == 0.7


def test_high_without_measurement_drops_back_to_low():
    score = importance_for_verdict(
        verdict="reject", confidence="high", has_measurement=False
    )
    assert score == 0.4


def test_advise_clamped_low_regardless_of_confidence():
    assert importance_for_verdict(verdict="advise", confidence="high") == 0.4


def test_kb_draft_high_confidence_promoted():
    assert importance_for_kb_draft(confidence=0.85) == 0.6


def test_kb_draft_default_when_unknown():
    assert importance_for_kb_draft(confidence=None) == 0.5


def test_cap_importance_enforces_critic_ceiling():
    assert cap_importance(0.99) == CRITIC_IMPORTANCE_CEILING
    assert cap_importance(0.5) == 0.5
    assert cap_importance(-1.0) == 0.0
