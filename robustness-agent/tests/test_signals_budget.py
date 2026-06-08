# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for :mod:`robustness_agent.signals.budget`."""

from __future__ import annotations

import pytest

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals.budget import (
    BudgetConfig,
    evaluate_budget_signals,
)
from robustness_agent.signals.symptom import SymptomSeverity


def _ctx(
    *,
    elapsed_minutes: float = 0.0,
    remaining_minutes: float = 0.0,
    budget_minutes: float = 0.0,
    cumulative_gain_validated: float = 0.0,
    closing_phase: bool = False,
) -> ReactorContext:
    snap = SharedStateSnapshot(
        elapsed_minutes=elapsed_minutes,
        remaining_minutes=remaining_minutes,
        budget_minutes=budget_minutes,
        cumulative_gain_validated=cumulative_gain_validated,
        closing_phase=closing_phase,
    )
    return ReactorContext(shared_state=snap, now_unix=1700000000.0)


def test_no_budget_section_short_circuits():
    out = evaluate_budget_signals(_ctx())
    assert out == []


def test_below_warn_pct_emits_nothing():
    ctx = _ctx(
        elapsed_minutes=100.0,
        remaining_minutes=260.0,
        budget_minutes=360.0,
    )
    assert evaluate_budget_signals(ctx) == []


def test_warn_zone_no_validated_gain_emits_medium_alert():
    # 75% burnt, validated_gain=0 → medium (between 0.70 and 0.85).
    ctx = _ctx(
        elapsed_minutes=270.0,
        remaining_minutes=90.0,
        budget_minutes=360.0,
    )
    out = evaluate_budget_signals(ctx)
    assert len(out) == 1
    sym = out[0]
    assert sym.name == "budget_burn_no_gain"
    assert sym.severity is SymptomSeverity.MEDIUM
    assert sym.evidence["burn_pct"] == pytest.approx(0.75, rel=1e-3)


def test_imminent_zone_no_validated_gain_emits_deadline_imminent():
    # 90% burnt, validated_gain=0 → deadline_imminent (HIGH).
    ctx = _ctx(
        elapsed_minutes=324.0,
        remaining_minutes=36.0,
        budget_minutes=360.0,
    )
    out = evaluate_budget_signals(ctx)
    assert len(out) == 1
    sym = out[0]
    assert sym.name == "deadline_imminent"
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["burn_pct"] == pytest.approx(0.90, rel=1e-3)
    assert sym.evidence["elapsed_minutes"] == 324.0
    assert sym.evidence["budget_minutes"] == 360.0


def test_imminent_zone_with_validated_gain_is_suppressed():
    # 90% burnt but validated_gain > productive threshold → no symptom
    # (the session has shippable progress; let it finish naturally).
    ctx = _ctx(
        elapsed_minutes=324.0,
        remaining_minutes=36.0,
        budget_minutes=360.0,
        cumulative_gain_validated=10.5,
    )
    assert evaluate_budget_signals(ctx) == []


def test_closing_phase_suppresses_signal():
    # Already winding down; don't nag.
    ctx = _ctx(
        elapsed_minutes=324.0,
        remaining_minutes=36.0,
        budget_minutes=360.0,
        closing_phase=True,
    )
    assert evaluate_budget_signals(ctx) == []


def test_short_budget_below_minimum_is_silent():
    # 20 min budget < 30 min min_budget_minutes → never fires.
    ctx = _ctx(
        elapsed_minutes=18.0,
        remaining_minutes=2.0,
        budget_minutes=20.0,
    )
    assert evaluate_budget_signals(ctx) == []


def test_custom_thresholds_apply():
    cfg = BudgetConfig(
        warn_pct=0.5,
        imminent_pct=0.6,
        min_budget_minutes=10.0,
        productive_gain_pct=2.0,
        # Disable the absolute-time signals so we exercise the
        # percentage-axis custom thresholds in isolation.
        deadline_warning_minutes=0.0,
        deadline_hard_cutoff_minutes=0.0,
    )
    # 65% burnt, validated_gain=1.0% < productive_gain_pct → HIGH.
    ctx = _ctx(
        elapsed_minutes=39.0,
        remaining_minutes=21.0,
        budget_minutes=60.0,
        cumulative_gain_validated=1.0,
    )
    out = evaluate_budget_signals(ctx, config=cfg)
    assert len(out) == 1
    assert out[0].name == "deadline_imminent"


def test_dedup_key_is_session_wide():
    # Empty subject collapses to (name,) so cooldown applies globally.
    ctx = _ctx(
        elapsed_minutes=324.0,
        remaining_minutes=36.0,
        budget_minutes=360.0,
    )
    out = evaluate_budget_signals(ctx)
    names = [s.name for s in out]
    assert any(n == "deadline_imminent" for n in names)
    sym = next(s for s in out if s.name == "deadline_imminent")
    assert sym.dedup_key() == ("deadline_imminent",)


# ---------------------------------------------------------------------------
# H3 budget_strategy_drift — 50% burnt + 0 gain early warning
# ---------------------------------------------------------------------------

def test_strategy_drift_fires_at_half_burnt_with_zero_gain():
    """50% burnt, validated_gain=0 → MEDIUM early hint."""
    ctx = _ctx(
        elapsed_minutes=180.0,
        remaining_minutes=180.0,
        budget_minutes=360.0,
    )
    out = evaluate_budget_signals(ctx)
    sym = next((s for s in out if s.name == "budget_strategy_drift"), None)
    assert sym is not None
    assert sym.severity is SymptomSeverity.MEDIUM
    assert sym.evidence["burn_pct"] == pytest.approx(0.5, rel=1e-3)


def test_strategy_drift_silent_below_threshold():
    ctx = _ctx(
        elapsed_minutes=120.0,
        remaining_minutes=240.0,
        budget_minutes=360.0,  # 33% burnt
    )
    out = evaluate_budget_signals(ctx)
    assert all(s.name != "budget_strategy_drift" for s in out)


def test_strategy_drift_silent_when_validated_gain_present():
    ctx = _ctx(
        elapsed_minutes=180.0,
        remaining_minutes=180.0,
        budget_minutes=360.0,
        cumulative_gain_validated=5.0,
    )
    out = evaluate_budget_signals(ctx)
    assert all(s.name != "budget_strategy_drift" for s in out)


def test_strategy_drift_does_not_double_fire_with_warn():
    """Past warn_pct (0.7), warn_no_gain takes over and drift is suppressed."""
    ctx = _ctx(
        elapsed_minutes=270.0,
        remaining_minutes=90.0,
        budget_minutes=360.0,  # 75% burnt
    )
    out = evaluate_budget_signals(ctx)
    drift = [s for s in out if s.name == "budget_strategy_drift"]
    burn = [s for s in out if s.name == "budget_burn_no_gain"]
    assert drift == []
    assert len(burn) == 1


# ---------------------------------------------------------------------------
# H1 deadline_warning — absolute-time 30-min predictive warning
# ---------------------------------------------------------------------------

def test_deadline_warning_fires_at_30min_remaining_with_zero_gain_high():
    """Long session: 24h budget, 25 min remain, 0 validated gain → HIGH."""
    ctx = _ctx(
        elapsed_minutes=1415.0,
        remaining_minutes=25.0,
        budget_minutes=1440.0,  # 24h budget
    )
    out = evaluate_budget_signals(ctx)
    sym = next(s for s in out if s.name == "deadline_warning")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["remaining_minutes"] == 25.0


def test_deadline_warning_with_validated_gain_drops_to_medium():
    """Long session, 25 min remain, validated gain present → MEDIUM only."""
    ctx = _ctx(
        elapsed_minutes=1415.0,
        remaining_minutes=25.0,
        budget_minutes=1440.0,
        cumulative_gain_validated=8.5,
    )
    out = evaluate_budget_signals(ctx)
    sym = next(s for s in out if s.name == "deadline_warning")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_deadline_warning_silent_when_remaining_above_threshold():
    ctx = _ctx(
        elapsed_minutes=1380.0,
        remaining_minutes=60.0,
        budget_minutes=1440.0,
    )
    out = evaluate_budget_signals(ctx)
    assert all(s.name != "deadline_warning" for s in out)


def test_deadline_warning_independent_of_burn_pct():
    """Short 2h budget, 25 min remain = 79% burnt → between warn and
    imminent, but deadline_warning still fires on the time axis."""
    ctx = _ctx(
        elapsed_minutes=95.0,
        remaining_minutes=25.0,
        budget_minutes=120.0,
    )
    out = evaluate_budget_signals(ctx)
    names = [s.name for s in out]
    assert "deadline_warning" in names
    # The %-axis signal at this burn_pct is budget_burn_no_gain, not
    # deadline_imminent (which fires at 85%).
    assert "budget_burn_no_gain" in names


# ---------------------------------------------------------------------------
# H1 deadline_hard_cutoff — < 5 min emergency cut
# ---------------------------------------------------------------------------

def test_hard_cutoff_fires_at_5min_remaining_always_high():
    ctx = _ctx(
        elapsed_minutes=355.0,
        remaining_minutes=5.0,
        budget_minutes=360.0,
    )
    out = evaluate_budget_signals(ctx)
    sym = next(s for s in out if s.name == "deadline_hard_cutoff")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["remaining_minutes"] == 5.0


def test_hard_cutoff_ignores_validated_gain():
    """Even with healthy gain, < 5 min remain is an emergency."""
    ctx = _ctx(
        elapsed_minutes=355.0,
        remaining_minutes=4.0,
        budget_minutes=360.0,
        cumulative_gain_validated=15.0,
    )
    out = evaluate_budget_signals(ctx)
    sym = next(s for s in out if s.name == "deadline_hard_cutoff")
    assert sym.severity is SymptomSeverity.HIGH


def test_hard_cutoff_suppresses_deadline_warning():
    """Below the hard cutoff we don't also emit the soft warning."""
    ctx = _ctx(
        elapsed_minutes=357.0,
        remaining_minutes=3.0,
        budget_minutes=360.0,
    )
    out = evaluate_budget_signals(ctx)
    names = [s.name for s in out]
    assert "deadline_hard_cutoff" in names
    assert "deadline_warning" not in names


def test_hard_cutoff_silent_when_remaining_above_threshold():
    ctx = _ctx(
        elapsed_minutes=350.0,
        remaining_minutes=10.0,
        budget_minutes=360.0,
    )
    out = evaluate_budget_signals(ctx)
    assert all(s.name != "deadline_hard_cutoff" for s in out)


# ---------------------------------------------------------------------------
# Axis coexistence — absolute and percentage signals can overlap
# ---------------------------------------------------------------------------

def test_absolute_and_percentage_can_both_fire():
    """24h budget, 25 min remain, 95% burnt, no validated gain.

    Both ``deadline_warning`` (time-axis) and ``deadline_imminent``
    (percentage-axis) fire because they cover complementary failure
    modes. The Classifier's ``_dedup`` collapses identical names, not
    different ones.
    """
    ctx = _ctx(
        elapsed_minutes=1415.0,
        remaining_minutes=25.0,
        budget_minutes=1440.0,
    )
    out = evaluate_budget_signals(ctx)
    names = {s.name for s in out}
    assert "deadline_warning" in names
    assert "deadline_imminent" in names


def test_short_budget_below_min_silences_absolute_signals_too():
    """20 min budget < 30 min min_budget_minutes → all signals silent."""
    ctx = _ctx(
        elapsed_minutes=18.0,
        remaining_minutes=2.0,
        budget_minutes=20.0,
    )
    out = evaluate_budget_signals(ctx)
    assert out == []


def test_closing_phase_suppresses_all_new_signals():
    ctx = _ctx(
        elapsed_minutes=355.0,
        remaining_minutes=5.0,
        budget_minutes=360.0,
        closing_phase=True,
    )
    out = evaluate_budget_signals(ctx)
    assert out == []


def test_custom_thresholds_apply_to_new_signals():
    cfg = BudgetConfig(
        deadline_warning_minutes=60.0,
        deadline_hard_cutoff_minutes=15.0,
        strategy_drift_pct=0.3,
        warn_pct=0.7,
        imminent_pct=0.85,
        min_budget_minutes=10.0,
        productive_gain_pct=0.5,
    )
    ctx = _ctx(
        elapsed_minutes=30.0,
        remaining_minutes=30.0,
        budget_minutes=60.0,  # 50% burnt, 30 min remain
    )
    out = evaluate_budget_signals(ctx, config=cfg)
    names = [s.name for s in out]
    # strategy_drift_pct = 0.3, so 50% burnt triggers drift.
    assert "budget_strategy_drift" in names
    # deadline_warning_minutes = 60, so 30 min remain trips warning.
    assert "deadline_warning" in names
