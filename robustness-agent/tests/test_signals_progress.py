"""Unit tests for ``gain_plateau`` and ``no_levers_found`` signals (B2 / B3)."""

from __future__ import annotations

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import SymptomSeverity
from robustness_agent.signals.progress import ProgressConfig, ProgressDetector
from robustness_agent.sources.base import SourceData


def _ctx(
    *,
    tick: int = 0,
    cumulative_gain_validated: float = 0.0,
    elapsed_minutes: float = 0.0,
    optimization_stack_size: int = 0,
    closing_phase: bool = False,
    stop_reason: str = "",
) -> ReactorContext:
    snap = SharedStateSnapshot(
        session_id="sess-1",
        tick=tick,
        cumulative_gain_validated=cumulative_gain_validated,
        elapsed_minutes=elapsed_minutes,
        optimization_stack_size=optimization_stack_size,
        closing_phase=closing_phase,
        stop_reason=stop_reason,
    )
    return ReactorContext(tick_index=tick, shared_state=snap, now_unix=1.0)


# ---------------------------------------------------------------------------
# gain_plateau
# ---------------------------------------------------------------------------

def test_no_history_silent():
    det = ProgressDetector()
    out = det.evaluate(_ctx(tick=0), SourceData())
    assert out == []


def test_short_history_silent_until_window_full():
    det = ProgressDetector(ProgressConfig(gain_window_ticks=3))
    det.evaluate(_ctx(tick=0, cumulative_gain_validated=5.0), SourceData())
    out = det.evaluate(_ctx(tick=1, cumulative_gain_validated=5.0), SourceData())
    assert all(s.name != "gain_plateau" for s in out)


def test_plateau_with_productive_gain_fires_medium():
    """Productive gain ≥ threshold → flat = "exhausted but shippable" → medium."""
    det = ProgressDetector(ProgressConfig(
        gain_window_ticks=3, gain_epsilon_pct=0.5, productive_gain_pct=0.5,
    ))
    det.evaluate(_ctx(tick=0, cumulative_gain_validated=10.0), SourceData())
    det.evaluate(_ctx(tick=1, cumulative_gain_validated=10.1), SourceData())
    out = det.evaluate(_ctx(tick=2, cumulative_gain_validated=10.2), SourceData())
    sym = next(s for s in out if s.name == "gain_plateau")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_plateau_with_zero_gain_fires_high():
    det = ProgressDetector(ProgressConfig(
        gain_window_ticks=3, gain_epsilon_pct=0.5, productive_gain_pct=0.5,
    ))
    det.evaluate(_ctx(tick=0, cumulative_gain_validated=0.0), SourceData())
    det.evaluate(_ctx(tick=1, cumulative_gain_validated=0.1), SourceData())
    out = det.evaluate(_ctx(tick=2, cumulative_gain_validated=0.0), SourceData())
    sym = next(s for s in out if s.name == "gain_plateau")
    assert sym.severity is SymptomSeverity.HIGH


def test_plateau_resets_on_movement():
    det = ProgressDetector(ProgressConfig(
        gain_window_ticks=3, gain_epsilon_pct=0.5,
    ))
    det.evaluate(_ctx(tick=0, cumulative_gain_validated=5.0), SourceData())
    det.evaluate(_ctx(tick=1, cumulative_gain_validated=5.0), SourceData())
    out = det.evaluate(_ctx(tick=2, cumulative_gain_validated=7.0), SourceData())
    # Delta=2 > epsilon=0.5 → no plateau.
    assert all(s.name != "gain_plateau" for s in out)


def test_closing_phase_short_circuits():
    det = ProgressDetector(ProgressConfig(gain_window_ticks=2))
    det.evaluate(_ctx(tick=0, closing_phase=True,
                      cumulative_gain_validated=0.0), SourceData())
    out = det.evaluate(
        _ctx(tick=1, closing_phase=True, cumulative_gain_validated=0.0),
        SourceData(),
    )
    assert out == []


def test_stop_reason_short_circuits():
    det = ProgressDetector(ProgressConfig(gain_window_ticks=2))
    out = det.evaluate(
        _ctx(tick=0, stop_reason="time_exhausted",
             cumulative_gain_validated=0.0),
        SourceData(),
    )
    assert out == []


# ---------------------------------------------------------------------------
# no_levers_found
# ---------------------------------------------------------------------------

def test_no_levers_silent_before_min_elapsed():
    det = ProgressDetector(ProgressConfig(
        no_levers_min_minutes=45.0, no_levers_min_ticks=8,
    ))
    out = det.evaluate(
        _ctx(tick=20, elapsed_minutes=20.0, optimization_stack_size=0),
        SourceData(),
    )
    assert all(s.name != "no_levers_found" for s in out)


def test_no_levers_silent_before_min_ticks():
    det = ProgressDetector(ProgressConfig(
        no_levers_min_minutes=45.0, no_levers_min_ticks=8,
    ))
    out = det.evaluate(
        _ctx(tick=5, elapsed_minutes=60.0, optimization_stack_size=0),
        SourceData(),
    )
    assert all(s.name != "no_levers_found" for s in out)


def test_no_levers_fires_high_when_quotas_met():
    det = ProgressDetector(ProgressConfig(
        no_levers_min_minutes=45.0, no_levers_min_ticks=8,
    ))
    out = det.evaluate(
        _ctx(tick=20, elapsed_minutes=70.0, optimization_stack_size=0),
        SourceData(),
    )
    sym = next(s for s in out if s.name == "no_levers_found")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["optimization_stack_size"] == 0


def test_no_levers_silent_when_stack_not_empty():
    det = ProgressDetector()
    out = det.evaluate(
        _ctx(tick=20, elapsed_minutes=70.0, optimization_stack_size=2),
        SourceData(),
    )
    assert all(s.name != "no_levers_found" for s in out)


def test_no_levers_silent_when_validated_gain_present():
    det = ProgressDetector(ProgressConfig(productive_gain_pct=0.5))
    out = det.evaluate(
        _ctx(
            tick=20,
            elapsed_minutes=70.0,
            optimization_stack_size=0,
            cumulative_gain_validated=5.0,
        ),
        SourceData(),
    )
    assert all(s.name != "no_levers_found" for s in out)
