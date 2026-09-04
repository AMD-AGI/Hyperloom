# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A signal-killed subprocess must not pin its kernel as busy forever."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.common import in_flight_liveness as ifl


def _alive(_: int) -> bool:
    return True


def _dead(_: int) -> bool:
    return False


def test_a_running_marker_with_a_live_pid_is_in_flight() -> None:
    verdict = ifl.evaluate_marker(state="running", pid=4242, mtime=100.0, now=100.0, pid_alive=_alive)
    assert verdict.in_flight is True
    assert verdict.stale_reason == ifl.STALE_NONE


def test_a_dead_pid_frees_the_kernel() -> None:
    """The case this exists for: killed by a signal, marker never cleared."""
    verdict = ifl.evaluate_marker(state="running", pid=4242, mtime=100.0, now=100.0, pid_alive=_dead)
    assert verdict.in_flight is False
    assert verdict.stale_reason == ifl.STALE_PID_GONE


def test_a_silent_marker_frees_the_kernel() -> None:
    """Second signal, for a writer that recorded no pid or a recycled one."""
    verdict = ifl.evaluate_marker(
        state="running",
        pid=None,
        mtime=0.0,
        now=ifl.DEFAULT_MAX_SILENCE_SEC + 1.0,
        pid_alive=_alive,
    )
    assert verdict.in_flight is False
    assert verdict.stale_reason == ifl.STALE_SILENT


def test_silence_exactly_at_the_limit_is_still_in_flight() -> None:
    verdict = ifl.evaluate_marker(
        state="running", pid=None, mtime=0.0, now=ifl.DEFAULT_MAX_SILENCE_SEC, pid_alive=_alive
    )
    assert verdict.in_flight is True


@pytest.mark.parametrize("state", ["succeeded", "failed", "", None, "RUNNING_LATER"])
def test_only_a_running_marker_can_be_in_flight(state: Any) -> None:
    assert ifl.evaluate_marker(state=state, pid=4242, pid_alive=_alive).in_flight is False


def test_a_completed_marker_carries_no_stale_reason() -> None:
    """Finishing normally is not staleness; only a running marker can be stale."""
    assert ifl.evaluate_marker(state="succeeded", pid=4242, pid_alive=_dead).stale_reason == ifl.STALE_NONE


@pytest.mark.parametrize("state", ["running", " Running ", "RUNNING"])
def test_running_is_matched_case_insensitively(state: str) -> None:
    assert ifl.evaluate_marker(state=state, pid=4242, mtime=1.0, now=1.0, pid_alive=_alive).in_flight is True


@pytest.mark.parametrize("pid", [None, 0, -1, "abc", True, 1.5])
def test_an_unusable_pid_falls_back_to_the_silence_check(pid: Any) -> None:
    """Cannot check by pid, so freshness is the only remaining signal."""
    fresh = ifl.evaluate_marker(state="running", pid=pid, mtime=10.0, now=10.0, pid_alive=_dead)
    assert fresh.in_flight is True
    stale = ifl.evaluate_marker(
        state="running", pid=pid, mtime=0.0, now=ifl.DEFAULT_MAX_SILENCE_SEC + 1.0, pid_alive=_dead
    )
    assert stale.stale_reason == ifl.STALE_SILENT


def test_a_valid_float_pid_is_accepted() -> None:
    assert ifl.evaluate_marker(state="running", pid=4242.0, mtime=1.0, now=1.0, pid_alive=_dead).in_flight is False


def test_no_mtime_skips_the_silence_check() -> None:
    assert ifl.evaluate_marker(state="running", pid=4242, mtime=None, pid_alive=_alive).in_flight is True


def test_a_disabled_silence_limit_skips_the_check() -> None:
    verdict = ifl.evaluate_marker(state="running", pid=None, mtime=0.0, now=1e9, max_silence_sec=0.0, pid_alive=_alive)
    assert verdict.in_flight is True


def test_a_dead_pid_wins_over_a_fresh_marker() -> None:
    """Freshness cannot resurrect a process that is provably gone."""
    verdict = ifl.evaluate_marker(state="running", pid=4242, mtime=100.0, now=100.0, pid_alive=_dead)
    assert verdict.stale_reason == ifl.STALE_PID_GONE


def test_the_default_probe_reports_a_missing_process() -> None:
    """Exercise the real signal-0 path once, with a pid that cannot exist."""
    assert ifl._pid_is_alive(2**31 - 1) is False


def test_the_default_probe_reports_this_process_alive() -> None:
    import os

    assert ifl._pid_is_alive(os.getpid()) is True
