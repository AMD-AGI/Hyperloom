# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
    kernel_opt_attempts_count: int = 0,
    has_keep_pending_integrate: bool = False,
    explore_started: bool = False,
) -> ReactorContext:
    snap = SharedStateSnapshot(
        session_id="sess-1",
        tick=tick,
        cumulative_gain_validated=cumulative_gain_validated,
        elapsed_minutes=elapsed_minutes,
        optimization_stack_size=optimization_stack_size,
        closing_phase=closing_phase,
        stop_reason=stop_reason,
        kernel_opt_attempts_count=kernel_opt_attempts_count,
        has_keep_pending_integrate=has_keep_pending_integrate,
        explore_started=explore_started,
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
    """Productive gain ≥ threshold → flat = "exhausted but shippable" → medium (stack must be > 0)."""
    det = ProgressDetector(ProgressConfig(
        gain_window_ticks=3, gain_epsilon_pct=0.5, productive_gain_pct=0.5,
    ))
    det.evaluate(_ctx(tick=0, cumulative_gain_validated=10.0,
                      optimization_stack_size=2), SourceData())
    det.evaluate(_ctx(tick=1, cumulative_gain_validated=10.1,
                      optimization_stack_size=2), SourceData())
    out = det.evaluate(_ctx(tick=2, cumulative_gain_validated=10.2,
                            optimization_stack_size=2), SourceData())
    sym = next(s for s in out if s.name == "gain_plateau")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_plateau_with_zero_gain_also_fires_medium():
    """Loosen P3_19: ``gain_plateau`` is now uniformly MEDIUM advisory (previously HIGH on still-zero validated gain)."""
    det = ProgressDetector(ProgressConfig(
        gain_window_ticks=3, gain_epsilon_pct=0.5, productive_gain_pct=0.5,
    ))
    det.evaluate(_ctx(tick=0, cumulative_gain_validated=0.0,
                      optimization_stack_size=1), SourceData())
    det.evaluate(_ctx(tick=1, cumulative_gain_validated=0.1,
                      optimization_stack_size=1), SourceData())
    out = det.evaluate(_ctx(tick=2, cumulative_gain_validated=0.0,
                            optimization_stack_size=1), SourceData())
    sym = next(s for s in out if s.name == "gain_plateau")
    assert sym.severity is SymptomSeverity.MEDIUM
    assert "skip_to_close" in (sym.suggestion or "") or "report" in (sym.suggestion or "")


def test_plateau_suppressed_when_stack_empty():
    """Cold-start guard: empty stack must not fire ``gain_plateau`` (``no_levers_found`` owns that case); else two HIGH escalations bias toward early report. Repro: primus-claw-20260522020448-z6rg6 tick=6."""
    det = ProgressDetector(ProgressConfig(
        gain_window_ticks=3, gain_epsilon_pct=0.5, productive_gain_pct=0.5,
    ))
    det.evaluate(_ctx(tick=0, cumulative_gain_validated=0.0,
                      optimization_stack_size=0), SourceData())
    det.evaluate(_ctx(tick=1, cumulative_gain_validated=0.0,
                      optimization_stack_size=0), SourceData())
    out = det.evaluate(_ctx(tick=2, cumulative_gain_validated=0.0,
                            optimization_stack_size=0), SourceData())
    assert all(s.name != "gain_plateau" for s in out)


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


def test_no_levers_fires_medium_when_quotas_met():
    """Loosen P3_19: ``no_levers_found`` is MEDIUM advisory once floors met with empty stack; ``explore_started=True`` distinguishes genuine no-lever from cold-start delay."""
    det = ProgressDetector(ProgressConfig(
        no_levers_min_minutes=45.0, no_levers_min_ticks=8,
    ))
    out = det.evaluate(
        _ctx(tick=20, elapsed_minutes=70.0, optimization_stack_size=0,
             explore_started=True),
        SourceData(),
    )
    sym = next(s for s in out if s.name == "no_levers_found")
    assert sym.severity is SymptomSeverity.MEDIUM
    assert sym.evidence["optimization_stack_size"] == 0


def test_no_levers_silent_when_stack_not_empty():
    det = ProgressDetector()
    out = det.evaluate(
        _ctx(tick=20, elapsed_minutes=70.0, optimization_stack_size=2,
             explore_started=True),
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
            explore_started=True,
        ),
        SourceData(),
    )
    assert all(s.name != "no_levers_found" for s in out)


# ---------------------------------------------------------------------------
# PR-B Fix 2: in-flight kernel-opt short-circuits no_levers_found
# ---------------------------------------------------------------------------
def test_no_levers_silent_when_kernel_opt_attempts_in_progress():
    """In-flight kernel_opt (``kernel_opt_attempts_count > 0``) with empty stack must short-circuit so Orch doesn't race to report. Repro: Qwen3-30B-A3B-Base 20260522T093903Z."""
    det = ProgressDetector(ProgressConfig(
        no_levers_min_minutes=45.0, no_levers_min_ticks=8,
    ))
    out = det.evaluate(
        _ctx(
            tick=20, elapsed_minutes=70.0, optimization_stack_size=0,
            kernel_opt_attempts_count=3,  # batch of 3 kernels in flight
            explore_started=True,
        ),
        SourceData(),
    )
    assert all(s.name != "no_levers_found" for s in out), \
        "kernel_opt_attempts_count>0 must silence no_levers_found"


def test_no_levers_silent_when_keep_pending_integrate():
    """A waiting multi-KEEP queue means integrate is about to fire -- not a "no lever found" condition."""
    det = ProgressDetector(ProgressConfig(
        no_levers_min_minutes=45.0, no_levers_min_ticks=8,
    ))
    out = det.evaluate(
        _ctx(
            tick=20, elapsed_minutes=70.0, optimization_stack_size=0,
            kernel_opt_attempts_count=2,
            has_keep_pending_integrate=True,
            explore_started=True,
        ),
        SourceData(),
    )
    assert all(s.name != "no_levers_found" for s in out)


def test_no_levers_silent_before_explore_started():
    """Cold-start guard (PR #239 followup 97318ee): before explore starts, stack=0 + gain=0 are by-construction, so ``no_levers_found`` must stay silent. Repro: primus-claw-20260522034541-xkk9f turn=7."""
    det = ProgressDetector(ProgressConfig(
        no_levers_min_minutes=45.0, no_levers_min_ticks=8,
    ))
    out = det.evaluate(
        _ctx(tick=20, elapsed_minutes=70.0, optimization_stack_size=0,
             explore_started=False),
        SourceData(),
    )
    assert all(s.name != "no_levers_found" for s in out)


def test_no_levers_evidence_includes_in_flight_fields():
    """When no_levers fires, the evidence dict carries the in-flight bookkeeping for post-hoc inspection."""
    det = ProgressDetector(ProgressConfig(
        no_levers_min_minutes=45.0, no_levers_min_ticks=8,
    ))
    out = det.evaluate(
        _ctx(
            tick=20, elapsed_minutes=70.0, optimization_stack_size=0,
            kernel_opt_attempts_count=0,
            has_keep_pending_integrate=False,
            explore_started=True,
        ),
        SourceData(),
    )
    sym = next(s for s in out if s.name == "no_levers_found")
    assert sym.evidence["kernel_opt_attempts_count"] == 0
    assert sym.evidence["has_keep_pending_integrate"] is False
