# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How work the run itself stopped is told apart from work that failed."""

from __future__ import annotations

import time
from typing import NamedTuple

from hyperloom.common.deadline import Deadline

__all__ = [
    "ORCHESTRATOR_CANCELLED_CLASS",
    "PHASE_TIME_EXHAUSTED_CLASS",
    "SESSION_TIME_EXHAUSTED_CLASS",
    "STOPPED_BY_THE_RUN",
    "StoppedByTheRun",
    "stopped_by_the_run_class",
    "variant_budget_stop",
]

# Labels work that never ran, or did not finish, because the session wall-clock budget was spent.
SESSION_TIME_EXHAUSTED_CLASS = "session_time_exhausted"

# Labels work that never started because the current phase's wall-clock budget could not fit it.
PHASE_TIME_EXHAUSTED_CLASS = "phase_time_exhausted"

# Labels work the orchestrator stopped from outside -- a shutdown, or a budget the dispatcher found spent.
ORCHESTRATOR_CANCELLED_CLASS = "orchestrator_cancelled"


class StoppedByTheRun(NamedTuple):
    """How work that the run stopped from outside is recorded."""

    error_class: str
    interrupted: str
    never_started: str
    ends_the_batch: bool


# The causes, keyed by the class the ledgers carry.
STOPPED_BY_THE_RUN: dict[str, StoppedByTheRun] = {
    SESSION_TIME_EXHAUSTED_CLASS: StoppedByTheRun(
        error_class=SESSION_TIME_EXHAUSTED_CLASS,
        interrupted="session wall-clock budget exhausted while this round was running",
        never_started="session wall-clock budget exhausted before this round ran",
        ends_the_batch=False,
    ),
    PHASE_TIME_EXHAUSTED_CLASS: StoppedByTheRun(
        error_class=PHASE_TIME_EXHAUSTED_CLASS,
        interrupted="phase wall-clock budget exhausted while this round was running",
        never_started="phase wall-clock budget exhausted before this round ran",
        ends_the_batch=False,
    ),
    ORCHESTRATOR_CANCELLED_CLASS: StoppedByTheRun(
        error_class=ORCHESTRATOR_CANCELLED_CLASS,
        interrupted="the orchestrator cancelled this action while this round was running",
        never_started="the orchestrator cancelled this action before this round ran",
        ends_the_batch=True,
    ),
}


def stopped_by_the_run_class(error_class: str | None) -> StoppedByTheRun | None:
    """Return how to record work the run itself stopped, if it did."""
    if not error_class:
        return None
    return STOPPED_BY_THE_RUN.get(str(error_class))


def variant_budget_stop(
    session_deadline_sec: float | None,
    phase_deadline: Deadline | None,
    needed_sec: float,
) -> StoppedByTheRun | None:
    """Which wall-clock budget, if any, cannot fit the next variant of a grid.

    Args:
        session_deadline_sec: Monotonic instant the session's grid work must end by, or ``None`` when unbounded.
        phase_deadline: Instant the current phase's budget runs out, or ``None`` when no phase budget bounds the work.
        needed_sec: What the next variant is expected to take.

    Returns:
        How to record the variants that will not start, or ``None`` when the next one fits.
    """
    if session_deadline_sec is not None and session_deadline_sec - time.monotonic() < needed_sec:
        return STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS]
    if phase_deadline is not None and phase_deadline.remaining() < needed_sec:
        return STOPPED_BY_THE_RUN[PHASE_TIME_EXHAUSTED_CLASS]
    return None
