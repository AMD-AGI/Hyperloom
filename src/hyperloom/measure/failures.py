# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Split a failed test into infrastructure, candidate and unclear failures.

One function decides for the local backend, the hosted backend and the runner, so the same log lands in the same class
wherever it is read. The classes drive different follow-ups in the tool layer:

* ``infra``: nothing about the candidate was learned; retry in a fresh sandbox, write no negative result;
* ``candidate``: the log carries a known failure signature; record a negative result;
* ``unclear``: non-zero with no known signature, or a timeout; rerun once in a fresh sandbox.

A refused instruction sheet or a rejected result bundle is none of these. It is a program error (or a tampered result),
so it gets no failure class at all: calling it ``candidate`` would write a false negative, and calling it ``unclear`` or
``infra`` would retry something that cannot succeed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from hyperloom.common.failure_signature import classify_failure
from hyperloom.contracts.measurement import FailureClass
from hyperloom.measure.returncodes import (
    HOSTED_INFRA_FAILURE_RC,
    HOSTED_RESULT_REJECTED_RC,
    HOSTED_RUNNER_UNAVAILABLE_RC,
    HOSTED_SPEC_REJECTED_RC,
)

#: Only the tail of a log is read, so a runner and a local backend holding the same log in different amounts decide
#: alike. Failure signatures sit at the end of a log, where the process died.
LOG_TAIL_CHARS = 128 * 1024
#: A line longer than this keeps its first and last halves. The signature rules backtrack quadratically on a long run
#: of text without whitespace (a progress bar redrawn with carriage returns is one such line), so one unbounded line
#: could stall a runner or the tool layer for minutes.
LOG_LINE_CHARS = 2000

#: ``error_class`` values a runner, the hosted backend or the platform reports that are not about the candidate.
INFRA_ERROR_CLASSES: frozenset[str] = frozenset(
    {
        "runner_start_failed",
        "runner_start_timeout",
        "runner_lost",
        "preflight_mismatch",
        "node_fault",
        "no_spec",
        "sandbox_create_failed",
        "capacity_unavailable",
        "image_pull_failed",
        "result_read_failed",
        "archive_failed",
    }
)
#: ``error_class`` values that mean the framework or the result is wrong, not the measurement or the candidate.
PROGRAM_ERROR_CLASSES: frozenset[str] = frozenset({"spec_rejected", "result_rejected"})

#: The server could not bind its port. The launcher picks a free port and releases it before the server binds, so
#: another process can take it in between; the candidate has no say in that, and a fresh sandbox does not repeat it.
PORT_IN_USE_REASON = "port_in_use"
_PORT_IN_USE = re.compile(r"address already in use|\[Errno 98\]", re.IGNORECASE)

# Sentinel return codes of the local flow (``orchestrator/actions/executors/_subprocess_kill.py``). The names stay
# literal here because ``measure`` must not import another package's private module; ``test_failures`` pins every
# value to its definition.
_SERVER_STALLED_RC = -911
_AGENTX_PREFLIGHT_RC = -912
_EVAL_PROBE_UNPATCHABLE_RC = -914
_SESSION_TIME_EXHAUSTED_RC = -915
_ORCHESTRATOR_CANCELLED_RC = -917

_INFRA_RETURNCODES = {
    HOSTED_RUNNER_UNAVAILABLE_RC: "runner_unavailable",
    HOSTED_INFRA_FAILURE_RC: "infra_failure",
    # The AgentX client is missing or not the pinned build: an environment failure.
    _AGENTX_PREFLIGHT_RC: "agentx_preflight",
}
_PROGRAM_ERROR_RETURNCODES = {
    HOSTED_SPEC_REJECTED_RC: "spec_rejected",
    HOSTED_RESULT_REJECTED_RC: "result_rejected",
    # The eval probe cannot be installed although its target is present: a framework defect, deterministic.
    _EVAL_PROBE_UNPATCHABLE_RC: "eval_probe_unpatchable",
}
_STOPPED_RETURNCODES = {
    _SESSION_TIME_EXHAUSTED_RC: "session_time_exhausted",
    _ORCHESTRATOR_CANCELLED_RC: "orchestrator_cancelled",
}


class Disposition(str, Enum):
    """What kind of end a test had, which decides whether a failure class applies at all."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PROGRAM_ERROR = "program_error"
    STOPPED = "stopped"


@dataclass(frozen=True)
class FailureClassification:
    """The verdict on one test end; ``failure_class`` is set exactly when ``disposition`` is ``FAILED``."""

    disposition: Disposition
    failure_class: FailureClass | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.disposition is Disposition.FAILED) != (self.failure_class is not None):
            raise ValueError("a failure class is set exactly when the test failed")
        if self.disposition is not Disposition.SUCCEEDED and not self.failure_reason:
            raise ValueError("every end that is not a success names its reason")
        if self.disposition is Disposition.SUCCEEDED and self.failure_reason is not None:
            raise ValueError("a success has no failure reason")


def classify_test_failure(
    *,
    returncode: int,
    timed_out: bool = False,
    error_class: str | None = None,
    logs: Iterable[str] = (),
) -> FailureClassification:
    """Classify how one test ended.

    ``error_class`` is what the runner's ``status.json`` or the hosted backend reports; ``logs`` are the texts a known
    failure signature may be found in (stdout and stderr tails, ``server.log`` tail).
    """
    if error_class is not None:
        return _from_error_class(error_class)
    if returncode in _PROGRAM_ERROR_RETURNCODES:
        return FailureClassification(Disposition.PROGRAM_ERROR, failure_reason=_PROGRAM_ERROR_RETURNCODES[returncode])
    if returncode in _STOPPED_RETURNCODES:
        return FailureClassification(Disposition.STOPPED, failure_reason=_STOPPED_RETURNCODES[returncode])
    if returncode in _INFRA_RETURNCODES:
        return _infra(_INFRA_RETURNCODES[returncode])
    if timed_out:
        return _unclear("timeout")
    if returncode == 0:
        return FailureClassification(Disposition.SUCCEEDED)
    # The silence watchdog reaped a ready server that stopped writing; a crash it logged before going quiet names it.
    return _from_logs(logs, unmatched="server_stalled" if returncode == _SERVER_STALLED_RC else "no_known_signature")


def _from_error_class(error_class: str) -> FailureClassification:
    if error_class in INFRA_ERROR_CLASSES:
        return _infra(error_class)
    if error_class in PROGRAM_ERROR_CLASSES:
        return FailureClassification(Disposition.PROGRAM_ERROR, failure_reason=error_class)
    # A class nobody handles is a framework defect. Calling it a candidate failure would write a false negative and
    # calling it infra would retry it forever, so it stops the line like any other program error.
    return FailureClassification(Disposition.PROGRAM_ERROR, failure_reason=f"unknown_error_class:{error_class}")


def _from_logs(logs: Iterable[str], *, unmatched: str) -> FailureClassification:
    text = "\n".join(_bounded(part) for part in logs if part)
    if _PORT_IN_USE.search(text):
        return _infra(PORT_IN_USE_REASON)
    signature = classify_failure(text)
    if signature.is_actionable:
        return FailureClassification(Disposition.FAILED, FailureClass.CANDIDATE, signature.kind)
    return _unclear(unmatched)


def _bounded(log: str) -> str:
    """The tail of ``log`` with every over-long line shortened, so classification time is bounded."""
    half = LOG_LINE_CHARS // 2
    lines = (
        line if len(line) <= LOG_LINE_CHARS else f"{line[:half]} ... {line[-half:]}"
        for line in log[-LOG_TAIL_CHARS:].split("\n")
    )
    return "\n".join(lines)


def _infra(reason: str) -> FailureClassification:
    return FailureClassification(Disposition.FAILED, FailureClass.INFRA, reason)


def _unclear(reason: str) -> FailureClassification:
    return FailureClassification(Disposition.FAILED, FailureClass.UNCLEAR, reason)
