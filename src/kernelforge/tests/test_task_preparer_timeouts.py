"""Regression guards for the forge preflight/prepare timeout knobs.

Two things are locked in here:

1. ``_deadline_timeout`` clamps a per-subprocess timeout to the shared absolute
   wall-clock deadline (never below 1s, never above the phase default).
2. The raised default timeouts stay raised. They were bumped well above the old
   120-300s values because a cold CK JIT compile on gfx950 runs for many minutes
   and the low defaults made every first-run preflight time out. A future edit
   that accidentally lowers them back would silently reintroduce that failure,
   so assert the committed defaults when the env override is absent.
"""

from __future__ import annotations

import os

from kernelforge.loop import task_preparer


# ---------------------------------------------------------------------------
# _deadline_timeout clamp
# ---------------------------------------------------------------------------


def test_deadline_zero_returns_default():
    assert task_preparer._deadline_timeout(0, 1800) == 1800
    assert task_preparer._deadline_timeout(-5, 42) == 42


def test_deadline_far_future_capped_at_default(monkeypatch):
    monkeypatch.setattr(task_preparer.time, "time", lambda: 1000.0)
    # Deadline is 10000s away but default is the ceiling.
    assert task_preparer._deadline_timeout(11000.0, 900) == 900


def test_deadline_near_clamps_below_default(monkeypatch):
    monkeypatch.setattr(task_preparer.time, "time", lambda: 1000.0)
    # Only 120s left before the shared deadline -> clamp under the 900 default.
    assert task_preparer._deadline_timeout(1120.0, 900) == 120.0


def test_deadline_past_floors_at_one_second(monkeypatch):
    monkeypatch.setattr(task_preparer.time, "time", lambda: 1000.0)
    # Deadline already blown -> never returns <= 0, floors at 1.0.
    assert task_preparer._deadline_timeout(500.0, 900) == 1.0


# ---------------------------------------------------------------------------
# Raised defaults stay raised (only when not env-overridden)
# ---------------------------------------------------------------------------


def test_prepare_defaults_stay_raised():
    if "FORGE_PREPARE_MAX_WALL" not in os.environ:
        assert task_preparer.PREPARE_MAX_WALL_SEC >= 3000
    if "FORGE_PREPARE_ATTEMPT_CAP" not in os.environ:
        assert task_preparer.PER_ATTEMPT_CAP_SEC >= 900


def test_preflight_defaults_stay_raised():
    if "FORGE_PREFLIGHT_CORRECTNESS_TIMEOUT" not in os.environ:
        assert task_preparer.PREFLIGHT_CORRECTNESS_TIMEOUT_S >= 1800
    if "FORGE_PREFLIGHT_BENCH_TIMEOUT" not in os.environ:
        assert task_preparer.PREFLIGHT_BENCH_TIMEOUT_S >= 1800
    if "FORGE_PREFLIGHT_GRAPH_TIMEOUT" not in os.environ:
        assert task_preparer.PREFLIGHT_GRAPH_TIMEOUT_S >= 900
    if "FORGE_PREFLIGHT_PROFILE_TIMEOUT" not in os.environ:
        assert task_preparer.PREFLIGHT_PROFILE_TIMEOUT_S >= 900
