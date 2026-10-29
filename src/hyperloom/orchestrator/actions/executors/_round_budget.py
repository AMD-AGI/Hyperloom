# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a benchmark round may still spend, and when a phase may no longer start one.

Admission is the single owner of whether a round boots: the predicates here are what
``BenchmarkRunExecutor`` prices a round against, and a phase terminal that names the same
line (:func:`...phases.prelude_exits._prelude_budget_line`) only records a decision these
already made.
"""

from __future__ import annotations

from typing import Any

from ...phases import machine_state as _phase_state

RETRY_PAST_THE_RESERVE = "retry_past_the_optimization_reserve"


def _round_headroom_sec(state: Any) -> tuple[float | None, dict[str, Any]]:
    """Seconds this round's budget may still spend, and the numbers behind it."""
    if state is None:
        return None, {"reason": "no_session_state"}
    usable_sec = _phase_state.session_usable_seconds(state)
    if usable_sec is None:
        return None, {"reason": "unbounded_session_budget"}
    return usable_sec, {"bound": "session_usable", "affordable_sec": round(usable_sec, 1)}


def _a_use_must_follow_the_round(state: Any) -> bool:
    """Whether this round is only worth running if something can be measured after it."""
    phase = str(getattr(state, "phase", "") or "").strip().upper()
    return phase == _phase_state.PHASE_PRELUDE


def _a_retry_past_the_reserve_line(state: Any) -> tuple[bool, dict[str, Any]]:
    """Whether this is a PRELUDE baseline retry the phase can no longer pay for.

    The first baseline is what the optimization reserve is held for, so it is priced on the
    session clock like any other round. A retry after a failed baseline is extra work PRELUDE
    buys out of its own allowance (:func:`machine_state.prelude_affordable_seconds`), and once
    that allowance is gone the phase may not start one. :func:`prelude_exits._prelude_budget_line`
    reads the same allowance to close an idle PRELUDE, so the terminal only records the refusal
    admission already makes.
    """
    if state is None:
        return False, {}
    if str(getattr(state, "phase", "") or "").strip().upper() != _phase_state.PHASE_PRELUDE:
        return False, {}
    streak = int(getattr(state, "baseline_failure_streak", 0) or 0)
    if streak < 1:
        return False, {}
    affordable_sec, evidence = _phase_state.prelude_affordable_seconds(state)
    if affordable_sec is None or affordable_sec > 0.0:
        return False, {}
    return True, {"reason": RETRY_PAST_THE_RESERVE, "baseline_failure_streak": streak, **evidence}
