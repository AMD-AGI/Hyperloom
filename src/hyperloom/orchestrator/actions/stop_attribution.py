# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How work the run itself stopped is told apart from work that failed.

Two causes stop a unit of work without saying anything about what it was
measuring: the session's wall-clock budget running out, and the orchestrator
cancelling the action (a shutdown, or a budget the dispatcher found spent).
Neither is evidence about the model under test, so nothing that ends this way
may be graded as a failed measurement.

The distinction has to survive into every ledger, not just the one nearest the
subprocess. A grid records the variant, ``explore`` records the fingerprint the
KB reads, and the baseline arm records a failure streak that stops the session
-- three places that each decide what a round *meant*, and each of which would
otherwise file "the run ran out of time" as a verdict about the model. So the
notion lives here, in a leaf every one of them can import, rather than as three
local judgements that can drift apart.

This module is the error-class side, which is what the ledgers carry. The
returncode side of the same distinction is deliberately not here, and is itself
in two pieces: the two sentinel codes are allocated in
:mod:`..executors._subprocess_kill`, beside every other code that space hands
out so that a collision is visible in one file, and ``stopped_by_the_run``, the
function that reads a code back, sits with the session-budget helpers in
``..executors._grid_runner`` that both benching arms already share. Neither
piece can move here: this leaf is imported by the executors package, so naming
a returncode would close an import cycle. A reader following a sentinel
returncode therefore starts there and arrives at the classes below.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = [
    "ORCHESTRATOR_CANCELLED_CLASS",
    "SESSION_BUDGET_BELOW_ONE_ROUND_CLASS",
    "SESSION_TIME_EXHAUSTED_CLASS",
    "STOPPED_BY_THE_RUN",
    "StoppedByTheRun",
    "stopped_by_the_run_class",
]

# Labels work that never ran, or did not finish, because the session wall-clock
# budget was spent. Distinct from any measurement failure: work that was not run
# is not evidence about what it would have measured.
SESSION_TIME_EXHAUSTED_CLASS = "session_time_exhausted"

# Labels work the orchestrator stopped from outside -- a shutdown, or a budget
# the dispatcher found spent. Kept apart from the class above for the same
# reason their returncodes are: a resume faces the spent budget again and does
# not face the shutdown.
ORCHESTRATOR_CANCELLED_CLASS = "orchestrator_cancelled"

# Labels work refused, or cut short, because what is left of the budget cannot
# pay for a whole round of it. A benchmark round is two passes -- a discarded
# warmup and the measured pass it makes comparable -- so a budget that fits one
# fits none: the pass it could pay for is the one whose number nothing may use.
# Distinct from the class above because the deadline has not passed. Nothing was
# measured either way, but this one is also known to be permanent: the clock
# only shrinks, so the round that does not fit now never will.
SESSION_BUDGET_BELOW_ONE_ROUND_CLASS = "session_budget_below_one_round"


class StoppedByTheRun(NamedTuple):
    """How work that the run stopped from outside is recorded.

    Attributes:
        error_class: The ledger class for the cause.
        interrupted: What to report when the work was already running.
        never_started: What to report when it never began.
        ends_the_batch: Whether the caller should stop launching further work on
            its own, rather than leave that to a budget check it may not have.
    """

    error_class: str
    interrupted: str
    never_started: str
    ends_the_batch: bool


# The two causes, keyed by the class the ledgers carry. The budget leaves the
# rest of a batch to the fit check its caller runs, which sees a deadline in the
# past and skips them all under the same label; a cancel has no such check, and
# nothing new should start under one.
STOPPED_BY_THE_RUN: dict[str, StoppedByTheRun] = {
    SESSION_TIME_EXHAUSTED_CLASS: StoppedByTheRun(
        error_class=SESSION_TIME_EXHAUSTED_CLASS,
        interrupted="session wall-clock budget exhausted while this round was running",
        never_started="session wall-clock budget exhausted before this round ran",
        ends_the_batch=False,
    ),
    ORCHESTRATOR_CANCELLED_CLASS: StoppedByTheRun(
        error_class=ORCHESTRATOR_CANCELLED_CLASS,
        interrupted="the orchestrator cancelled this action while this round was running",
        never_started="the orchestrator cancelled this action before this round ran",
        ends_the_batch=True,
    ),
    SESSION_BUDGET_BELOW_ONE_ROUND_CLASS: StoppedByTheRun(
        error_class=SESSION_BUDGET_BELOW_ONE_ROUND_CLASS,
        interrupted=(
            "the warmup pass was stopped at its share of the remaining budget, "
            "which is therefore too small to fit both passes of this round"
        ),
        never_started="the remaining budget cannot fit both passes of this round",
        ends_the_batch=True,
    ),
}


def stopped_by_the_run_class(error_class: str | None) -> StoppedByTheRun | None:
    """Return how to record work the run itself stopped, if it did.

    Args:
        error_class: The ``error_class`` a result carries; anything that is not
            one of the two causes (including ``None`` and ``""``) reads as work
            that has something to say about what it measured.

    Returns:
        StoppedByTheRun | None: How to record it, or ``None`` when the class
            names a failure of the thing under test rather than of the run.
    """
    if not error_class:
        return None
    return STOPPED_BY_THE_RUN.get(str(error_class))
