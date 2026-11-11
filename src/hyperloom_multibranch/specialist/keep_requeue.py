# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the import of an autonomous branch's keep that measured nothing goes once more (:mod:`autonomous_session`).

A test whose every attempt failed for an infrastructure reason measured nothing of the keep: cards a process outside the
test held, or a platform short of sandboxes, for the few minutes the toolbox's retries take. The keep goes once more,
:data:`REQUEUE_SECONDS` after its import ended, unless the reason is one a later attempt meets again; a keep goes once
more only once, and after :data:`REQUEUE_FAILURES` requeued imports in a row measured nothing again none does, so a node
or a platform that stays broken does not loop.
"""

from __future__ import annotations

from enum import Enum

from hyperloom.contracts.measurement import FailureClass, TestResult, Validity
from hyperloom.measure.failures import SANDBOX_ACQUIRE_TIMEOUT
from hyperloom_multibranch.toolbox.attempts import UNRETRIED_INFRA_REASONS

#: How long after the run saw the import of a keep end measuring nothing it goes once more: the toolbox's own retries of
#: the test are spent within minutes, and what held the cards or the sandboxes is most often gone within a few more.
REQUEUE_SECONDS = 5 * 60.0
#: How many requeued imports in a row may measure nothing again before the branch requeues no more: a node or a
#: platform that stays broken fails every requeue alike, and each spends a test's cards.
REQUEUE_FAILURES = 2
#: The infrastructure failures a requeue minutes later meets again: those a fresh sandbox repeats
#: (``hyperloom_multibranch.toolbox.attempts``), but for a wait for a sandbox, which a later queue may not meet.
_REPEATED_INFRA = UNRETRIED_INFRA_REASONS - {SANDBOX_ACQUIRE_TIMEOUT}
#: How the toolbox's reason starts for a test it ended as an infrastructure failure for a program error.
_PROGRAM_ERRORS = ("measurement refused", "executor error")


class Ruling(Enum):
    """What becomes of a keep whose import ended: ``SETTLED``, nothing more; ``MEASURED``, nothing more, and a requeued
    import measured something, so the count of failed requeues starts over; ``SPENT``, its requeued import measured
    nothing again; ``CAPPED``, it measured nothing after the last :data:`REQUEUE_FAILURES` requeues did too;
    ``REQUEUE``, it goes once more."""

    SETTLED = "settled"
    MEASURED = "measured"
    SPENT = "spent"
    CAPPED = "capped"
    REQUEUE = "requeue"


def requeueable(result: TestResult) -> bool:
    """Whether ``result`` measured nothing for an infrastructure reason a requeue minutes later may not meet: not a
    program error, nor one every sandbox repeats, which the toolbox ends a test with as is."""

    if result.validity is not Validity.MEASUREMENT_FAILED or result.failure_class is not FailureClass.INFRA:
        return False
    reason = result.failure_reason or ""
    return not reason.startswith(_PROGRAM_ERRORS) and reason.partition(":")[0].strip() not in _REPEATED_INFRA


def rule(result: TestResult | None, *, again: bool, failures: int) -> Ruling:
    """The ruling on a keep whose import ended with ``result`` (``None``: withdrawn or unknown); ``again`` when the
    import was the keep's requeue, ``failures`` the requeued imports in a row that measured nothing again."""

    if result is None or not requeueable(result):
        measured = again and result is not None and result.validity is not Validity.MEASUREMENT_FAILED
        return Ruling.MEASURED if measured else Ruling.SETTLED
    if again:
        return Ruling.SPENT
    return Ruling.CAPPED if failures >= REQUEUE_FAILURES else Ruling.REQUEUE


def said(ruling: Ruling, failures: int) -> str:
    """What the branch log says of a keep whose import measured nothing, by the ``ruling`` on it."""

    if ruling is Ruling.REQUEUE:
        return "it goes once more"
    if ruling is Ruling.SPENT:
        return "it went once more already"
    return f"the last {failures} requeued imports measured nothing again"
