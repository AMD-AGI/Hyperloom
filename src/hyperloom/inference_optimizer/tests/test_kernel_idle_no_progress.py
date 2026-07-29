# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the KERNEL idle-spin wind-down (kernel_idle_no_progress).

When candidates are exhausted, the kernel_agent cannot emit the
``escalate_strategy_change`` hint (PolicyGate denies it for that role), so
KERNEL would otherwise spin until the wall-clock cap. ``exit_normal_kernel``
therefore winds down to SWEEP after ``KERNEL_IDLE_MAX_TICKS`` consecutive
no-work ticks. The critical safety property is that this must NOT fire while
work is still pending (e.g. a forge attempt in progress) or before the tick
threshold is reached — otherwise KERNEL exits prematurely.
"""
from __future__ import annotations

from types import SimpleNamespace

from hyperloom.orchestrator.phases import machine_state as ms


def _state(idle_ticks):
    return SimpleNamespace(
        kernel_idle_ticks=idle_ticks,
        rejected_kernel_ids=[],
        phase="KERNEL_AGENT",
    )


def _patch(monkeypatch, *, work_pending, hint=""):
    monkeypatch.setattr(ms, "kernel_work_pending", lambda s: work_pending)
    monkeypatch.setattr(ms, "_pending_escalate_hint", lambda s: hint)
    # Keep the hard budget/cap exits inert so only the idle logic decides.
    monkeypatch.setattr(
        ms, "phase_budget_remaining_seconds", lambda s, **k: 9_999.0
    )
    monkeypatch.setattr(ms, "phase_cap_exceeded", lambda s, **k: False)


def test_idle_at_threshold_winds_down_to_sweep(monkeypatch):
    _patch(monkeypatch, work_pending=False)
    state = _state(ms.KERNEL_IDLE_MAX_TICKS)
    result = ms.exit_normal_kernel(state)
    assert result is not None
    reason, evidence = result
    assert reason == "kernel_no_more_leverage"
    assert evidence["evidence"] == "kernel_idle_no_progress"
    assert evidence["idle_ticks"] == ms.KERNEL_IDLE_MAX_TICKS
    assert evidence["idle_max_ticks"] == ms.KERNEL_IDLE_MAX_TICKS


def test_idle_below_threshold_does_not_exit(monkeypatch):
    _patch(monkeypatch, work_pending=False)
    state = _state(ms.KERNEL_IDLE_MAX_TICKS - 1)
    # Below the threshold and budget healthy -> KERNEL keeps running.
    assert ms.exit_normal_kernel(state) is None


def test_work_pending_blocks_idle_exit(monkeypatch):
    # Even with idle_ticks way over the cap, pending work must veto the wind-down
    # (guards against exiting while a forge attempt is still in flight).
    _patch(monkeypatch, work_pending=True)
    state = _state(ms.KERNEL_IDLE_MAX_TICKS + 5)
    assert ms.exit_normal_kernel(state) is None


def test_skip_to_sweep_hint_still_exits_when_no_work(monkeypatch):
    # The explicit escalate-hint path (when available) still yields the
    # non-terminal leverage exit.
    _patch(monkeypatch, work_pending=False, hint=ms.ESCALATE_HINT_SKIP_TO_SWEEP)
    state = _state(0)
    result = ms.exit_normal_kernel(state)
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
