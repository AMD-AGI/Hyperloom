# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the KERNEL idle-spin wind-down (kernel_idle_no_progress).

When the phase stops moving, the kernel_agent cannot emit the
``escalate_strategy_change`` hint (PolicyGate denies it for that role), so
KERNEL would otherwise spin until the wall-clock cap. ``exit_normal_kernel``
therefore winds down to SWEEP once the idle streak has lasted both
``KERNEL_IDLE_MAX_TICKS`` ticks and ``KERNEL_IDLE_MIN_SECONDS`` of wall clock.

The streak is measured by the phase machine against an observable progress
fingerprint; ``kernel_work_pending`` no longer takes part. It reports whether
the ledger still lists anything unresolved, which stays true forever once an
attempt can no longer be advanced — a production session sat in KERNEL for
10.4h with all 8 GPUs idle because that predicate kept the counter at zero. The
protection against ending KERNEL mid-build now lives where the evidence is: the
phase machine freezes the streak while a kernel-lane task is in flight (see
``test_kernel_idle_streak.py``).
"""

from __future__ import annotations

from types import SimpleNamespace

from hyperloom.orchestrator.phases import machine_state as ms


def _state(idle_ticks, *, idle_seconds=ms.KERNEL_IDLE_MIN_SECONDS):
    # ``now_unix`` is pinned at 10_000.0 by the callers below, so backdating the
    # streak start by ``idle_seconds`` gives an exact idle window.
    return SimpleNamespace(
        kernel_idle_ticks=idle_ticks,
        kernel_idle_since_unix=10_000.0 - idle_seconds,
        rejected_kernel_ids=[],
        phase="KERNEL_AGENT",
    )


def _patch(monkeypatch, *, work_pending, hint=""):
    monkeypatch.setattr(ms, "kernel_work_pending", lambda s: work_pending)
    monkeypatch.setattr(ms, "_pending_escalate_hint", lambda s: hint)
    # Keep the hard budget/cap exits inert so only the idle logic decides.
    monkeypatch.setattr(ms, "phase_budget_remaining_seconds", lambda s, **k: 9_999.0)
    monkeypatch.setattr(ms, "phase_cap_exceeded", lambda s, **k: False)


def test_idle_at_threshold_winds_down_to_sweep(monkeypatch):
    _patch(monkeypatch, work_pending=False)
    state = _state(ms.KERNEL_IDLE_MAX_TICKS)
    result = ms.exit_normal_kernel(state, now_unix=10_000.0)
    assert result is not None
    reason, evidence = result
    assert reason == "kernel_no_more_leverage"
    assert evidence["evidence"] == "kernel_idle_no_progress"
    assert evidence["idle_ticks"] == ms.KERNEL_IDLE_MAX_TICKS
    assert evidence["idle_max_ticks"] == ms.KERNEL_IDLE_MAX_TICKS
    assert evidence["idle_seconds"] == ms.KERNEL_IDLE_MIN_SECONDS
    assert evidence["idle_min_seconds"] == ms.KERNEL_IDLE_MIN_SECONDS


def test_idle_below_tick_threshold_does_not_exit(monkeypatch):
    _patch(monkeypatch, work_pending=False)
    state = _state(ms.KERNEL_IDLE_MAX_TICKS - 1)
    # Below the tick threshold and budget healthy -> KERNEL keeps running.
    assert ms.exit_normal_kernel(state, now_unix=10_000.0) is None


def test_idle_below_wall_clock_floor_does_not_exit(monkeypatch):
    # Ticks are cheap (a few seconds each, and the phase machine is scanned more
    # than once per tick), so the tick threshold alone must not wind the phase
    # down: a healthy gap between a kernel result landing and the next dispatch
    # would otherwise look like a stall.
    _patch(monkeypatch, work_pending=False)
    state = _state(
        ms.KERNEL_IDLE_MAX_TICKS * 100,
        idle_seconds=ms.KERNEL_IDLE_MIN_SECONDS - 1.0,
    )
    assert ms.exit_normal_kernel(state, now_unix=10_000.0) is None


def test_unstamped_streak_start_does_not_exit(monkeypatch):
    # A hand-built or partially-initialised state has no measured idle window;
    # the guard must refuse to act on a tick count it did not observe itself.
    _patch(monkeypatch, work_pending=False)
    state = _state(ms.KERNEL_IDLE_MAX_TICKS)
    state.kernel_idle_since_unix = 0.0
    assert ms.exit_normal_kernel(state, now_unix=10_000.0) is None


def test_work_pending_no_longer_blocks_idle_exit(monkeypatch):
    # Regression for the production stall: ``kernel_work_pending`` answered True
    # for 1130 consecutive idle ticks because three attempts on the ledger could
    # never be advanced. The ledger's opinion must not be able to veto a
    # wind-down that the observed streak has already justified.
    _patch(monkeypatch, work_pending=True)
    state = _state(ms.KERNEL_IDLE_MAX_TICKS + 5)
    result = ms.exit_normal_kernel(state, now_unix=10_000.0)
    assert result is not None
    reason, evidence = result
    assert reason == "kernel_no_more_leverage"
    assert evidence["evidence"] == "kernel_idle_no_progress"


def test_skip_to_sweep_hint_still_exits_when_no_work(monkeypatch):
    # The explicit escalate-hint path (when available) still yields the
    # non-terminal leverage exit.
    _patch(monkeypatch, work_pending=False, hint=ms.ESCALATE_HINT_SKIP_TO_SWEEP)
    state = _state(0)
    result = ms.exit_normal_kernel(state, now_unix=10_000.0)
    assert result is not None
    reason, evidence = result
    assert reason == "kernel_no_more_leverage"
    assert evidence["hint"] == ms.ESCALATE_HINT_SKIP_TO_SWEEP


def test_default_idle_max_ticks_is_three(monkeypatch):
    # Lock in the documented default when the env override is absent.
    monkeypatch.delenv("INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS", raising=False)
    assert ms._kernel_idle_max_ticks() == 3


def test_idle_max_ticks_env_override(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS", "5")
    assert ms._kernel_idle_max_ticks() == 5
    # Non-positive / garbage falls back to 3.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS", "0")
    assert ms._kernel_idle_max_ticks() == 3
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS", "xx")
    assert ms._kernel_idle_max_ticks() == 3


def test_default_idle_min_seconds_is_ten_minutes(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS", raising=False)
    assert ms._kernel_idle_min_seconds() == 600.0


def test_idle_min_seconds_env_override(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS", "90")
    assert ms._kernel_idle_min_seconds() == 90.0
    # Non-positive / garbage falls back to the 600s default.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS", "0")
    assert ms._kernel_idle_min_seconds() == 600.0
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS", "xx")
    assert ms._kernel_idle_min_seconds() == 600.0
