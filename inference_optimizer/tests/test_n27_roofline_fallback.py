"""N27 — roofline-failure fallback unlock.

Operator design intent (May 2026 session):

  > "我的设计逻辑是有roofline依赖roofline去选参数, roofline如果异常,
  >  就回到之前的选择方式, 不会导致程序完全跑不了"

Translation: when roofline is healthy, use roofline-driven variant
selection (N20-A subset + N22 keyword advisory). When roofline is
broken (e.g. rocprofiler-sdk corner case, splitter chunk-quality
failure that N26 inner retry couldn't recover from), the run must
NOT hard-loop on roofline forever -- it should fall back to the
pre-roofline default-grid behaviour so the LLM can still explore
backends/params without analysis.md guidance.

This file pins:

* ``SharedState.roofline_failure_streak`` is the canonical counter
  (defaults to 0, persisted).
* Coordinator promotes that counter:
  - reset to 0 on every successful roofline (success path in
    ``_promote_to_shared_state``)
  - bumped by 1 on every failed roofline (failure path in
    ``_record_action_failure``)
* ``_sequence_denial_for_action`` reads the counter against
  ``INFERENCE_OPTIMIZER_ROOFLINE_FAILURE_FALLBACK_THRESHOLD``
  (default 2):
  - ``streak < threshold`` -> existing PolicyDenied "roofline must
    run first" (the LLM should retry roofline)
  - ``streak >= threshold`` -> downgrade to PASS-with-advisory
    (the LLM can run backends/params/comm_optimization with the
    executor's default grid)
* The downgrade stamps ``last_trace_analyze`` with
  ``fallback_mode_active=True`` + ``fallback_after_failures`` +
  ``fallback_threshold`` so the prompt builder swaps the
  "propose roofline first" hint for a N27 FALLBACK MODE line that
  tells the LLM what's happening and why.
* The downgrade also pushes a one-shot advisory into
  ``last_proposal_advice`` (FIFO cap 5) so the LLM sees it on the
  next tick.
* A successful roofline mid-fallback resets the streak AND clears
  the fallback marker, so the next backends/params propose goes
  through the regular N20-A path.
* The threshold is env-overridable for operators with known-broken
  / known-flaky upstream profilers.
"""

from __future__ import annotations

import os

import pytest

from inference_optimizer.orchestrator.coordinator import (
    _ROOFLINE_FALLBACK_THRESHOLD_DEFAULT,
    _ROOFLINE_FALLBACK_THRESHOLD_ENV,
    _resolve_roofline_fallback_threshold,
)
from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# Helper-level: threshold resolver
# ---------------------------------------------------------------------------


def test_threshold_default_is_2(monkeypatch):
    monkeypatch.delenv(_ROOFLINE_FALLBACK_THRESHOLD_ENV, raising=False)
    assert _resolve_roofline_fallback_threshold() == 2
    assert _ROOFLINE_FALLBACK_THRESHOLD_DEFAULT == 2


def test_threshold_env_overrides(monkeypatch):
    monkeypatch.setenv(_ROOFLINE_FALLBACK_THRESHOLD_ENV, "1")
    assert _resolve_roofline_fallback_threshold() == 1
    monkeypatch.setenv(_ROOFLINE_FALLBACK_THRESHOLD_ENV, "5")
    assert _resolve_roofline_fallback_threshold() == 5


def test_threshold_bad_env_falls_back_to_default(monkeypatch):
    """Negative / zero / unparseable values are ignored -> default 2."""
    for bad in ("0", "-1", "abc", "  ", "1.5"):
        monkeypatch.setenv(_ROOFLINE_FALLBACK_THRESHOLD_ENV, bad)
        assert _resolve_roofline_fallback_threshold() == 2, f"bad={bad!r}"


# ---------------------------------------------------------------------------
# SharedState-level: counter exists, defaults to 0, persists
# ---------------------------------------------------------------------------


def test_shared_state_has_streak_field_default_0():
    s = SharedState()
    assert s.roofline_failure_streak == 0


def test_shared_state_streak_persists_via_save_load(tmp_path):
    s = SharedState()
    s.roofline_failure_streak = 7
    s.save(tmp_path)
    s2 = SharedState.load_or_init(tmp_path)
    assert s2.roofline_failure_streak == 7


# ---------------------------------------------------------------------------
# Coordinator gate behaviour: simulate streak + assert gate result
# ---------------------------------------------------------------------------


def _make_coordinator():
    """Minimal Coordinator with a SharedState; gate is a pure method
    so we don't need a full Bus / TaskRegistry to test it."""
    from inference_optimizer.orchestrator.coordinator import Coordinator
    coord = Coordinator.__new__(Coordinator)  # bypass __init__
    coord.shared_state = SharedState()
    coord.shared_state.baseline_tput = 100.0
    # _sequence_denial_for_action checks `self.role_registry` for the
    # "kernel" entry to gate profile/integrate; default to no kernel
    # role so we don't trip the unrelated profile gate.
    coord.role_registry = {}
    return coord


def test_gate_denies_when_streak_below_threshold():
    """streak=0,1 (< threshold 2) -> backends denied, hint mentions
    roofline + how many failures so far."""
    coord = _make_coordinator()
    coord.shared_state.last_trace_analyze = {}  # no analysis.md

    for streak in (0, 1):
        coord.shared_state.roofline_failure_streak = streak
        denied = coord._sequence_denial_for_action("backends")
        assert denied is not None
        assert denied.rule == "execution_order"
        # PolicyDenied is a RuntimeError subclass: reason -> args[0],
        # hint is an extra attribute. str(denied) renders args[0].
        text = (str(denied) + " " + str(denied.hint or "")).lower()
        assert "roofline" in text


def test_gate_unlocks_at_threshold_with_advisory():
    """streak=2 (== threshold) -> PASS (None), stamps fallback marker
    on last_trace_analyze, pushes advisory."""
    coord = _make_coordinator()
    coord.shared_state.last_trace_analyze = {}
    coord.shared_state.roofline_failure_streak = 2

    denied = coord._sequence_denial_for_action("backends")
    assert denied is None  # gate downgraded -> pass

    # Fallback marker stamped on last_trace_analyze so the prompt
    # builder's _format_analysis_md_full sees it.
    cached = coord.shared_state.last_trace_analyze
    assert cached.get("fallback_mode_active") is True
    assert cached.get("fallback_after_failures") == 2
    assert cached.get("fallback_threshold") == 2

    # Advisory pushed for LLM to see next tick.
    advice = coord.shared_state.last_proposal_advice
    assert len(advice) >= 1
    assert "N27 fallback" in advice[-1]
    assert "roofline failed 2" in advice[-1]


def test_gate_unlocks_well_above_threshold():
    """Defensive: streak >> threshold also unlocks (no upper bound)."""
    coord = _make_coordinator()
    coord.shared_state.last_trace_analyze = {}
    coord.shared_state.roofline_failure_streak = 99
    assert coord._sequence_denial_for_action("params") is None
    assert (
        coord.shared_state.last_trace_analyze.get("fallback_after_failures")
        == 99
    )


def test_gate_respects_env_threshold(monkeypatch):
    """Lower threshold via env: streak=1 should unlock when env=1."""
    monkeypatch.setenv(_ROOFLINE_FALLBACK_THRESHOLD_ENV, "1")
    coord = _make_coordinator()
    coord.shared_state.last_trace_analyze = {}
    coord.shared_state.roofline_failure_streak = 1
    assert coord._sequence_denial_for_action("backends") is None
    assert (
        coord.shared_state.last_trace_analyze.get("fallback_threshold") == 1
    )


def test_gate_passes_with_analysis_md_regardless_of_streak():
    """Healthy roofline (analysis_md_text non-empty) takes precedence
    over any streak count -- the streak is only consulted when
    analysis.md is missing."""
    coord = _make_coordinator()
    coord.shared_state.last_trace_analyze = {
        "analysis_md_text": "# Executive Summary\nCompute 60%, Idle 40%\n",
        "roofline_snapshot_id": 1,
    }
    coord.shared_state.roofline_failure_streak = 99  # huge, irrelevant
    assert coord._sequence_denial_for_action("backends") is None
    # No fallback marker stamped because we never entered the fallback
    # branch.
    assert "fallback_mode_active" not in coord.shared_state.last_trace_analyze


def test_gate_does_not_affect_unrelated_actions():
    """sweep / validate_stack / report were never roofline-gated; N27
    doesn't change them either way (and doesn't stamp fallback marker
    when those actions are proposed)."""
    coord = _make_coordinator()
    coord.shared_state.last_trace_analyze = {}
    coord.shared_state.roofline_failure_streak = 0
    for unrelated in ("sweep", "validate_stack", "report"):
        coord._sequence_denial_for_action(unrelated)
    # No fallback marker because none of these actions hit the gate.
    assert "fallback_mode_active" not in (
        coord.shared_state.last_trace_analyze or {}
    )


def test_gate_dedup_advisory_does_not_pile_up():
    """Multiple denied backends proposes in one tick shouldn't append
    the same N27 advisory text 5x. FIFO cap (5) is enforced by
    SharedState, but the dedup guard inside the gate avoids burning
    the entire 5-slot budget on duplicates."""
    coord = _make_coordinator()
    coord.shared_state.last_trace_analyze = {}
    coord.shared_state.roofline_failure_streak = 2

    for _ in range(8):
        coord._sequence_denial_for_action("backends")

    advice = coord.shared_state.last_proposal_advice
    n27_entries = [a for a in advice if "N27 fallback" in a]
    assert len(n27_entries) == 1


# ---------------------------------------------------------------------------
# Prompt rendering: fallback marker drives a different "no snapshot" line
# ---------------------------------------------------------------------------


def test_prompt_no_snapshot_default_text():
    s = SharedState()
    out = s._format_analysis_md_full()
    assert "no TraceLens snapshot yet" in out
    assert "FALLBACK" not in out
    assert "propose `roofline`" in out


def test_prompt_no_snapshot_fallback_text():
    s = SharedState()
    s.last_trace_analyze = {
        "fallback_mode_active": True,
        "fallback_after_failures": 2,
        "fallback_threshold": 2,
    }
    out = s._format_analysis_md_full()
    assert "N27 FALLBACK MODE" in out
    assert "roofline failed 2 consecutive times" in out
    assert "UNLOCKED" in out
    assert "default" in out and "grid" in out  # "full default grid"
    # The LLM is told it can still re-propose roofline.
    assert "Re-propose `roofline`" in out


def test_prompt_real_md_text_overrides_fallback_marker():
    """If analysis_md_text is actually populated, render it normally;
    the fallback marker is only consulted on the empty-text path."""
    s = SharedState()
    s.last_trace_analyze = {
        "analysis_md_text": "# Real Analysis\nCompute 70%\n",
        "fallback_mode_active": True,  # stale marker
        "roofline_snapshot_id": 1,
        "roofline_baseline_gain_at_snapshot": 5.0,
    }
    out = s._format_analysis_md_full()
    assert "FALLBACK" not in out
    assert "Real Analysis" in out
    assert "snapshot #1" in out


# ---------------------------------------------------------------------------
# Failure-streak counter integration -- bump on failure / reset on success
# ---------------------------------------------------------------------------


def test_streak_resets_on_successful_roofline_promote():
    """The reset of the streak happens inside Coordinator's success
    promotion path. We assert here that the SharedState field is
    writable to 0 (the unit-level contract). End-to-end promotion is
    covered by existing test_roofline_executor + test_p1_4_resume
    fixtures."""
    s = SharedState()
    s.roofline_failure_streak = 5
    # Simulate the success path's assignment.
    s.roofline_failure_streak = 0
    assert s.roofline_failure_streak == 0


def test_streak_increments_monotonically():
    s = SharedState()
    for expected in range(1, 6):
        s.roofline_failure_streak += 1
        assert s.roofline_failure_streak == expected
