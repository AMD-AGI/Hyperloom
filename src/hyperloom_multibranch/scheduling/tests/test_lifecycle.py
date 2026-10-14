# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's exit code: 0 only for a session that ran its course, chose a commit and left nothing behind."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.toolbox import QueueState
from hyperloom_multibranch.scheduling.close import CloseReport
from hyperloom_multibranch.scheduling.final import FinalCandidate, FinalChoice
from hyperloom_multibranch.scheduling.lifecycle import SessionOutcome
from hyperloom_multibranch.scheduling.processes import ProcessRecord
from hyperloom_multibranch.scheduling.stops import SessionStopReason, StopReport

CHOSEN = FinalChoice(
    FinalCandidate("integration", INTEGRATION_BRANCH, "i1", 1041.0), (), "the higher value", "output_throughput", 64
)
CLEAN = StopReport((), QueueState())
LEFTOVER = ProcessRecord(4242, 4242, 0.0, "b01", ("sleep", "600"), "scope")


def _outcome(reason: SessionStopReason, final: FinalChoice | None, stop: StopReport, error: str = "") -> SessionOutcome:
    close = CloseReport(reason, final, None, stop, error)
    return SessionOutcome(reason, final, close, error)


@pytest.mark.parametrize("reason", [SessionStopReason.TIME_EXHAUSTED, SessionStopReason.GLOBAL_CONVERGED])
def test_a_session_that_ran_its_course_and_stopped_clean_exits_zero(reason: SessionStopReason) -> None:
    assert _outcome(reason, CHOSEN, CLEAN).exit_code == 0


@pytest.mark.parametrize(
    "outcome",
    [
        _outcome(SessionStopReason.TIME_EXHAUSTED, CHOSEN, StopReport((LEFTOVER,), QueueState())),
        _outcome(SessionStopReason.TIME_EXHAUSTED, CHOSEN, StopReport((), QueueState(running_by_branch={"b01": 1}))),
        _outcome(SessionStopReason.TIME_EXHAUSTED, CHOSEN, StopReport((), None)),
        _outcome(SessionStopReason.TIME_EXHAUSTED, None, CLEAN, "the session deadline passed before FINAL"),
        _outcome(SessionStopReason.TIME_EXHAUSTED, None, CLEAN),
        _outcome(SessionStopReason.TIME_EXHAUSTED, replace(CHOSEN, chosen=None), CLEAN),
        _outcome(SessionStopReason.SIGNAL, None, CLEAN),
        _outcome(SessionStopReason.PRELUDE_FAILED, None, CLEAN, "no baseline"),
        SessionOutcome(SessionStopReason.TIME_EXHAUSTED, CHOSEN, None, ""),
    ],
    ids=[
        "process-left",
        "test-running",
        "queue-unknown",
        "deadline-before-final",
        "final-never-ran",
        "nothing-chosen",
        "signal",
        "prelude-failed",
        "close-never-ran",
    ],
)
def test_every_other_ending_exits_one(outcome: SessionOutcome) -> None:
    assert outcome.exit_code == 1
