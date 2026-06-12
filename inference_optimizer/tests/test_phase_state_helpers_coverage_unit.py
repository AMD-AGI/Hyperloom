# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for phase_state pure helpers: escalate hints, budget normalization,
time/budget remaining math, post-prelude target, and history-row builder."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator import phase_state as ps


# -- escalate hints --------------------------------------------------------
def test_is_pause_specialist_hint() -> None:
    prefix = ps.ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX
    assert ps.is_pause_specialist_hint(prefix + "serving_specialist") is True
    assert ps.is_pause_specialist_hint(prefix) is False  # no domain suffix
    assert ps.is_pause_specialist_hint("something_else") is False


def test_is_valid_escalate_hint() -> None:
    prefix = ps.ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX
    assert ps.is_valid_escalate_hint(prefix + "kernel_switch_specialist") is True
    assert ps.is_valid_escalate_hint("not-a-real-hint") is False
    # at least one vocab member should validate
    some_vocab = next(iter(ps.ESCALATE_HINT_VOCAB))
    assert ps.is_valid_escalate_hint(some_vocab) is True


# -- budget normalization + bump ------------------------------------------
def test_normalize_budget_pct_defaults_and_filters() -> None:
    assert ps.normalize_budget_pct(None) == dict(ps.DEFAULT_PHASE_BUDGET_PCT)
    out = ps.normalize_budget_pct({
        "EXPLORE": 0.4,
        "BOGUS_PHASE": 0.5,      # unknown phase dropped
        "KERNEL": "bad",          # non-numeric dropped
        "SWEEP": 2.0,             # out of (0,1] dropped
    })
    assert out["EXPLORE"] == 0.4
    assert "BOGUS_PHASE" not in out


def test_apply_escalate_budget_bump() -> None:
    # unknown phase -> returned unchanged (copy)
    base = {"EXPLORE": 0.3}
    assert ps.apply_escalate_budget_bump(base, phase="nope") == base
    # valid phase -> bumped and capped
    out = ps.apply_escalate_budget_bump(
        {"EXPLORE": 0.3}, phase="explore", delta=0.1, cap=0.8,
    )
    assert out["EXPLORE"] == pytest.approx(0.4)
    capped = ps.apply_escalate_budget_bump(
        {"EXPLORE": 0.75}, phase="explore", delta=0.5, cap=0.8,
    )
    assert capped["EXPLORE"] == 0.8


# -- time/budget helpers ---------------------------------------------------
def test_now_unix_injected() -> None:
    state = SimpleNamespace(_now_unix=lambda: 1234.0)
    assert ps._now_unix(state) == 1234.0


def test_phase_started_unix_bad_value() -> None:
    assert ps._phase_started_unix(SimpleNamespace(phase_started_unix="bad")) == 0.0
    assert ps._phase_started_unix(SimpleNamespace(phase_started_unix=10.0)) == 10.0


def test_pending_escalate_hint() -> None:
    valid = next(iter(ps.ESCALATE_HINT_VOCAB))
    assert ps._pending_escalate_hint(SimpleNamespace(pending_escalate_hint=valid)) == valid
    assert ps._pending_escalate_hint(SimpleNamespace(pending_escalate_hint="garbage")) == ""
    assert ps._pending_escalate_hint(SimpleNamespace(pending_escalate_hint="")) == ""


def test_max_minutes_coercion() -> None:
    assert ps._max_minutes(SimpleNamespace(max_minutes=30)) == 30.0
    assert ps._max_minutes(SimpleNamespace(max_minutes="bad")) == 0.0
    assert ps._max_minutes(SimpleNamespace(max_minutes=0)) == 0.0


def test_phase_elapsed_seconds() -> None:
    # not started -> 0
    assert ps.phase_elapsed_seconds(SimpleNamespace(phase_started_unix=0.0)) == 0.0
    # started -> now - started, clamped non-negative
    state = SimpleNamespace(phase_started_unix=100.0)
    assert ps.phase_elapsed_seconds(state, now_unix=160.0) == 60.0
    assert ps.phase_elapsed_seconds(state, now_unix=50.0) == 0.0


def test_phase_budget_remaining_seconds() -> None:
    # unlimited -> None
    assert ps.phase_budget_remaining_seconds(SimpleNamespace(max_minutes=0)) is None
    # phase not present in the budget map -> pct 0 -> None
    state = SimpleNamespace(
        max_minutes=60, phase="UNKNOWN_PHASE", phase_started_unix=0.0,
        phase_budget_pct={"EXPLORE": 0.5},
    )
    assert ps.phase_budget_remaining_seconds(state) is None
    # normal: 60min * 0.5 = 1800s budget, minus elapsed
    state2 = SimpleNamespace(
        max_minutes=60, phase="EXPLORE", phase_started_unix=1000.0,
        phase_budget_pct={"EXPLORE": 0.5},
    )
    rem = ps.phase_budget_remaining_seconds(state2, now_unix=1300.0)
    assert rem == pytest.approx(1800.0 - 300.0)


def test_session_remaining_seconds() -> None:
    assert ps.session_remaining_seconds(SimpleNamespace(max_minutes=0)) is None
    # no start_ts -> None
    assert ps.session_remaining_seconds(
        SimpleNamespace(max_minutes=60, start_ts=""),
    ) is None
    # bad ts -> None
    assert ps.session_remaining_seconds(
        SimpleNamespace(max_minutes=60, start_ts="not-a-date"),
    ) is None
    # valid recent start -> positive remaining
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    rem = ps.session_remaining_seconds(
        SimpleNamespace(max_minutes=60, start_ts=now_iso),
    )
    assert rem is not None and 0.0 < rem <= 3600.0


# -- post-prelude target + history row ------------------------------------
def test_post_prelude_target() -> None:
    assert ps._post_prelude_target(explore_enabled=True, kernel_enabled=True) == ps.PHASE_EXPLORE
    assert ps._post_prelude_target(explore_enabled=False, kernel_enabled=True) == ps.PHASE_KERNEL
    assert ps._post_prelude_target(explore_enabled=False, kernel_enabled=False) == ps.PHASE_SWEEP


def test_make_history_row() -> None:
    row = ps.make_history_row(
        from_phase="explore", to_phase="kernel", reason="  plateau  ",
        evidence={"k": 1}, ts="2026-06-09T00:00:00Z", ts_unix=12.0,
    )
    assert row["from_phase"] == "EXPLORE"
    assert row["to_phase"] == "KERNEL"
    assert row["reason"] == "plateau"
    assert row["evidence"] == {"k": 1}
    assert row["ts_unix"] == 12.0
