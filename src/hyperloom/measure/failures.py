# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Split a failed test into infrastructure, candidate and unclear failures.

One function decides for the local backend, the hosted backend and the runner, so the same log lands in the same class
wherever it is read. The classes drive different follow-ups in the tool layer:

* ``infra``: nothing about the candidate was learned; retry in a fresh sandbox, write no negative result;
* ``candidate``: the log carries a known failure signature, the server refused its own command line, or it did not
  become ready within its boot limit; record a negative result;
* ``unclear``: non-zero with no known signature, a stalled server, or a timeout; rerun once in a fresh sandbox. The reason still names
  what the log says (a refused evaluation request, a Python exception) when it says anything.

A refused instruction sheet or a rejected result bundle is none of these. It is a program error (or a tampered result),
so it gets no failure class at all: calling it ``candidate`` would write a false negative, and calling it ``unclear`` or
``infra`` would retry something that cannot succeed.
"""

from __future__ import annotations

import errno
import re
from dataclasses import dataclass
from enum import Enum

from inference_suite.contracts import FailedStep

from hyperloom.common.failure_signature import (
    EVAL_RUNTIME_FAILURE,
    HIP_KERNEL_MISSING,
    HIP_RUNTIME_ERROR,
    IMPORT_ERROR,
    SERVE_FLAG,
    classify_failure,
    framework_unimportable_line,
)
from hyperloom.contracts.measurement import FailureClass
from hyperloom.measure import log_causes
from hyperloom.measure.boot_watch import HUNG_BEFORE_READY
from hyperloom.measure.returncodes import (
    HOSTED_INFRA_FAILURE_RC,
    HOSTED_RESULT_REJECTED_RC,
    HOSTED_RUNNER_UNAVAILABLE_RC,
    HOSTED_SPEC_REJECTED_RC,
    SERVER_BOOT_TIMEOUT_RC,
)

#: Only the tail of a log is read, so a runner and a local backend holding the same log in different amounts decide
#: alike. Failure signatures sit at the end of a log, where the process died.
LOG_TAIL_CHARS = 128 * 1024
#: A line longer than this keeps its first and last halves. The signature rules backtrack quadratically on a long run
#: of text without whitespace (a progress bar redrawn with carriage returns is one such line), so one unbounded line
#: could stall a runner or the tool layer for minutes.
LOG_LINE_CHARS = 2000

#: A process outside the test held one of the test's cards, so nothing the test showed is the candidate's: retried on
#: other cards like any infrastructure failure (``hyperloom.measure.runner.card_watch``).
FOREIGN_PROCESS_ON_CARD = "foreign_process_on_card"

#: The runner's view of the machine refused it the kernel's list for its cards (a container shown only some GPUs), so the
#: card guard cannot look, and the guard enforces and refuses such a result (``hyperloom.measure.runner.card_watch``). A
#: fresh sandbox of the same platform is refused the same way, so it is not retried: it stops the line like any other
#: program error until the platform shows the runner its cards or the guard is set to record it.
CARDS_UNOBSERVABLE = "cards_unobservable"

#: A write failed because the disk or the user's quota was full (``ENOSPC``, ``EDQUOT``), in the runner or in the server
#: (a Triton cache under the sandbox's work directory, say). A candidate's patch cannot fill a shared volume, and the next
#: sandbox may find room, so nothing about the candidate was learned: retried like any infrastructure failure. The reason
#: is ``disk_full: <the log's line>``; a runner that failed this way reports it as its ``error_class``.
DISK_FULL = "disk_full"
_DISK_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})

#: The runner process exited without writing ``status.json`` (it died importing, say): the host stops waiting at once
#: instead of polling until the test's timeout. Retried like any infrastructure failure.
RUNNER_EXITED = "runner_exited"

#: The server's scheduler watchdog fired while a rank was still building an aiter JIT module (a tuned table named
#: kernel instances the image's prebuilt module lacks, so aiter rebuilt it at the first prefill). Nothing about the
#: candidate's speed was learned, so it is an infrastructure failure, but a fresh sandbox rebuilds the same module: the
#: attempts retry it once only (``hyperloom_multibranch.toolbox.attempts``). The reason is ``jit_build_timeout: <the
#: watchdog's line>``.
JIT_BUILD_TIMEOUT = "jit_build_timeout"
#: A ``git`` fetch the benchmark client runs for its own dependencies timed out; the reason is ``<this>: <the line>``.
GIT_FETCH_TIMEOUT = "git_fetch_timeout"
#: The Hugging Face Hub refused a download of the benchmark client (its trace dataset, say) for the rate limit of the
#: address the sandbox reaches it from, after the client's own retries; the reason is ``<this>: <the line>``. No candidate
#: causes it and the node has no part in it (``hyperloom_multibranch.toolbox.attempts`` does not report it against one).
HUB_RATE_LIMITED = "hub_rate_limited"
#: A Hugging Face cache (the ``datasets`` lock of the evaluation's dataset, say) refused the test with ``[Errno 13]
#: Permission denied``; the reason is ``<this>: <the path refused>``. The cache is a volume every sandbox of the platform
#: mounts alike, so no candidate causes it and a fresh sandbox meets it again: ``hyperloom_multibranch.toolbox.attempts``
#: ends the test with it at once and does not hold it against the node.
HF_CACHE_PERMISSION = "hf_cache_permission"
#: The test's own timeout passed while its attempt waited for a slot of the sandbox pool or for room in the grant's
#: quota, before any sandbox was made (``hyperloom.measure.sandboxes.AcquireTimedOut``); the reason is ``<this>: <what it
#: waited for and what held it>``. Nothing ran, so no node and no candidate is to blame, and a retry would wait on the same
#: sandboxes: ``hyperloom_multibranch.toolbox.attempts`` ends the test with it.
SANDBOX_ACQUIRE_TIMEOUT = "sandbox_acquire_timeout"

#: The AgentX checkpoint stopped the measured round for trailing its reference by more than the allowed drop
#: (``measure.checkpoint``): a ruling on the candidate, never retried. The ruling itself is in the round's slot.
AGENTX_EARLY_STOP = "agentx_early_stop"

#: ``error_class`` values a runner, the hosted backend or the platform reports that are not about the candidate.
INFRA_ERROR_CLASSES: frozenset[str] = frozenset(
    {
        "runner_start_failed",
        "runner_start_timeout",
        SANDBOX_ACQUIRE_TIMEOUT,
        "runner_lost",
        "preflight_mismatch",
        FOREIGN_PROCESS_ON_CARD,
        DISK_FULL,
        "node_fault",
        "no_spec",
        "sandbox_create_failed",
        "capacity_unavailable",
        "image_pull_failed",
        "result_read_failed",
        "archive_failed",
        # The test went silent before its server was ready (``boot_watch``): a hang a new sandbox does not repeat.
        HUNG_BEFORE_READY,
        RUNNER_EXITED,
    }
)
#: ``error_class`` values that mean the framework or the result is wrong, not the measurement or the candidate.
PROGRAM_ERROR_CLASSES: frozenset[str] = frozenset({"spec_rejected", "result_rejected", CARDS_UNOBSERVABLE})

#: The test ran past its own timeout: unclear, since a hung server and a test longer than its timeout end alike.
#: ``hyperloom_multibranch.toolbox.attempts`` does not rerun a sweep that ends so.
TEST_TIMEOUT_REASON = "timeout"
#: The server could not bind its port. The launcher picks a free port and releases it before the server binds, so
#: another process can take it in between; the candidate has no say in that, and a fresh sandbox does not repeat it.
PORT_IN_USE_REASON = "port_in_use"
#: The server did not report ready within the boot limit (``hyperloom.measure.boot_watch``).
SERVER_BOOT_TIMEOUT_REASON = "server_boot_timeout"
#: A requested GPU was not visible, or another process held it, when the suite checked before boot.
HOST_GPUS_UNAVAILABLE_REASON = "host_gpus_unavailable"
#: The server went quiet while serving (``FailedStep.SERVER_STALLED``): a hang or a slow candidate end alike.
SERVER_STALLED_REASON = "server_stalled"
#: The serving interpreter could not report its framework (``FailedStep.FRAMEWORK_VERSION_UNKNOWN``) in a test that
#: changed nothing of the framework: the image or the sandbox's environment cannot serve it, whatever the candidate.
FRAMEWORK_UNIMPORTABLE_REASON = "framework_unimportable"
#: The steps inference_suite names a round failed at that are not about the candidate, or that name the cause better
#: than any line of a log that holds only the symptom. Every other step leaves the verdict to the logs.
_FAILED_STEP_VERDICTS: dict[FailedStep, tuple[FailureClass, str]] = {
    # The suite parses the argv before the server starts, so no server log holds the refusal.
    FailedStep.FLAG_UNKNOWN: (FailureClass.CANDIDATE, SERVE_FLAG),
    FailedStep.PORT_IN_USE: (FailureClass.INFRA, PORT_IN_USE_REASON),
    FailedStep.GPUS_UNAVAILABLE: (FailureClass.INFRA, HOST_GPUS_UNAVAILABLE_REASON),
    FailedStep.SERVER_SILENT_BEFORE_READY: (FailureClass.INFRA, HUNG_BEFORE_READY),
    FailedStep.DISK_FULL: (FailureClass.INFRA, DISK_FULL),
    FailedStep.TEARDOWN_INCOMPLETE: (FailureClass.INFRA, FailedStep.TEARDOWN_INCOMPLETE.value),
    FailedStep.SERVER_STALLED: (FailureClass.UNCLEAR, SERVER_STALLED_REASON),
}
#: The steps whose reason quotes the report's ``failure_detail``, as ``<reason>: <detail>``.
_DETAILED_STEPS = frozenset({FailedStep.FLAG_UNKNOWN})
#: The two causes an unclear failure names when its log carries them; the reason is ``<prefix>: <the log's line>``.
#: A failure the log only calls a crashed evaluation names the refused request too, when the log carries one.
LM_EVAL_BAD_REQUEST_REASON = "lm_eval_http_400"
PYTHON_EXCEPTION_REASON = "python_exception"
NO_KNOWN_SIGNATURE_REASON = "no_known_signature"
#: The kinds of the shared signature table whose reason also quotes the log line that states them, with the reader of it.
_QUOTED_KINDS = {
    SERVE_FLAG: log_causes.argparse_error,
    IMPORT_ERROR: log_causes.import_error,
    HIP_KERNEL_MISSING: log_causes.missing_hip_kernel,
    HIP_RUNTIME_ERROR: log_causes.hip_error,
}
#: The causes that decide a test on their own once the warmup or the accuracy round states one: the server crashed or
#: refused to serve, or its evaluation failed. Measuring what is left would only boot a server that cannot serve; any
#: other end (a port taken, an unnamed log, a stalled server) keeps the rounds after it, since a later round's logs can
#: still name the cause. The names are the part of a reason before its ``: <the log's line>``.
_DECISIVE_REASONS = frozenset(
    {
        HIP_RUNTIME_ERROR,
        IMPORT_ERROR,
        SERVE_FLAG,
        EVAL_RUNTIME_FAILURE,
        LM_EVAL_BAD_REQUEST_REASON,
        PYTHON_EXCEPTION_REASON,
        SERVER_BOOT_TIMEOUT_REASON,
    }
)
_PORT_IN_USE = re.compile(r"address already in use|\[Errno 98\]", re.IGNORECASE)

# Sentinel return codes of the local flow (``orchestrator/actions/executors/_subprocess_kill.py``). The names stay
# literal here because ``measure`` must not import another package's private module; ``test_failures`` pins every
# value to its definition.
_SESSION_TIME_EXHAUSTED_RC = -915
_ORCHESTRATOR_CANCELLED_RC = -917
_AGENTX_EARLY_STOP_RC = -919

_INFRA_RETURNCODES = {
    HOSTED_RUNNER_UNAVAILABLE_RC: "runner_unavailable",
    HOSTED_INFRA_FAILURE_RC: "infra_failure",
}
_PROGRAM_ERROR_RETURNCODES = {
    HOSTED_SPEC_REJECTED_RC: "spec_rejected",
    HOSTED_RESULT_REJECTED_RC: "result_rejected",
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


@dataclass(frozen=True)
class StepFailure:
    """The step a failed round's ``report.json`` names, and its ``failure_detail`` (``""`` when it gave none)."""

    step: FailedStep
    detail: str = ""


@dataclass(frozen=True)
class FailureEvidence:
    """What a failed test left to read, consulted only when neither its error class nor its return code decides it.

    ``logs`` are the texts a known failure signature may be found in (stdout and stderr tails, the tails of
    ``server.log`` and of the benchmark's stderr); ``client_logs`` are the benchmark client's own output, read only for
    the infrastructure cause it states (a Hub download refused), never for a candidate's signature, since a client that
    failed on its own says nothing of the server; ``detail`` is what the caller observed of a boot timeout, which the
    return code alone does not say; ``step_failure`` is the step the failed round's ``report.json`` names, with its
    detail, which inference_suite records in place of the return code it exits with; ``unrun`` is ``<error_class>:
    <message>`` of a failed round refused before it ran, which left no log: the only account of it there is;
    ``framework_changed`` is whether the test put a rebuilt framework on its server's import path
    (:attr:`~hyperloom.measure.socket.TestCall.framework_changed`).
    """

    logs: tuple[str, ...] = ()
    client_logs: tuple[str, ...] = ()
    detail: str = ""
    step_failure: StepFailure | None = None
    unrun: str = ""
    framework_changed: bool = False


def classify_test_failure(
    *,
    returncode: int,
    timed_out: bool = False,
    error_class: str | None = None,
    evidence: FailureEvidence = FailureEvidence(),
) -> FailureClassification:
    """Classify how one test ended; ``error_class`` is what the runner's ``status.json`` or the hosted backend
    reports."""
    if error_class is not None:
        return _from_error_class(error_class)
    if returncode in _PROGRAM_ERROR_RETURNCODES:
        return FailureClassification(Disposition.PROGRAM_ERROR, failure_reason=_PROGRAM_ERROR_RETURNCODES[returncode])
    if returncode in _STOPPED_RETURNCODES:
        return FailureClassification(Disposition.STOPPED, failure_reason=_STOPPED_RETURNCODES[returncode])
    if returncode in _INFRA_RETURNCODES:
        return _infra(_INFRA_RETURNCODES[returncode])
    if returncode == _AGENTX_EARLY_STOP_RC:
        return _candidate(AGENTX_EARLY_STOP)
    if timed_out:
        return _unclear(TEST_TIMEOUT_REASON)
    if returncode == 0:
        return FailureClassification(Disposition.SUCCEEDED)
    parts = [_bounded(part) for part in evidence.logs if part]
    text = "\n".join(parts)
    refused = log_causes.hub_rate_limited("\n".join([text, *(_bounded(part) for part in evidence.client_logs if part)]))
    if refused:
        # Before the signatures: the client that was refused may then fail to load a tokenizer, which reads as one.
        return _infra(f"{HUB_RATE_LIMITED}: {refused}")
    reported = evidence.step_failure
    step = reported.step if reported else None
    by_step = _from_step(reported, evidence.framework_changed) if reported else None
    if by_step is not None:
        return by_step
    named = _named_by_logs(text, parts)
    if named is not None:
        return named
    if returncode == SERVER_BOOT_TIMEOUT_RC or step is FailedStep.SERVER_BOOT_TIMEOUT:
        detail = evidence.detail
        return _candidate(f"{SERVER_BOOT_TIMEOUT_REASON}: {detail}" if detail else SERVER_BOOT_TIMEOUT_REASON)
    return _from_unnamed_logs(text, evidence.unrun)


def decides_test(verdict: FailureClassification) -> bool:
    """Whether ``verdict`` names a cause that settles the test, so the rounds that are left need not run."""
    reason = verdict.failure_reason or ""
    return verdict.disposition is Disposition.FAILED and reason.split(":", 1)[0] in _DECISIVE_REASONS


def is_disk_full(exc: BaseException) -> bool:
    """Whether ``exc`` is a write the file system refused for want of room (:data:`DISK_FULL`)."""
    return isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS


def _from_error_class(error_class: str) -> FailureClassification:
    if error_class in INFRA_ERROR_CLASSES:
        return _infra(error_class)
    if error_class in PROGRAM_ERROR_CLASSES:
        return FailureClassification(Disposition.PROGRAM_ERROR, failure_reason=error_class)
    if error_class == AGENTX_EARLY_STOP:
        return _candidate(AGENTX_EARLY_STOP)
    # A class nobody handles is a framework defect. Calling it a candidate failure would write a false negative and
    # calling it infra would retry it forever, so it stops the line like any other program error.
    return FailureClassification(Disposition.PROGRAM_ERROR, failure_reason=f"unknown_error_class:{error_class}")


def _from_step(reported: StepFailure, framework_changed: bool) -> FailureClassification | None:
    """The verdict the step a round failed at gives by itself; ``None`` leaves it to the logs."""
    if reported.step is FailedStep.FRAMEWORK_VERSION_UNKNOWN:
        return _framework_unimportable(reported.detail.strip(), framework_changed)
    if reported.step not in _FAILED_STEP_VERDICTS:
        return None
    failure_class, reason = _FAILED_STEP_VERDICTS[reported.step]
    if reported.step in _DETAILED_STEPS and reported.detail.strip():
        reason = f"{reason}: {log_causes.one_line(reported.detail)}"
    return FailureClassification(Disposition.FAILED, failure_class, reason)


def _framework_unimportable(detail: str, changed: bool) -> FailureClassification | None:
    """The probe found no framework or its import raised: the rebuilt framework broke the import when the test laid one,
    else the image's own install cannot be imported. Any other probe failure says neither and is left to the logs."""
    unimportable = framework_unimportable_line(detail)
    if unimportable is None:
        return None
    line = log_causes.one_line(unimportable)
    if changed:
        return _candidate(f"{IMPORT_ERROR}: {line}")
    return _infra(f"{FRAMEWORK_UNIMPORTABLE_REASON}: {line}")


def _named_by_logs(text: str, parts: list[str]) -> FailureClassification | None:
    """The verdict a log gives by itself: a port taken, a known signature, a refused command line; else ``None``.

    ``text`` is the logs ``parts`` joined.
    """
    if _PORT_IN_USE.search(text):
        return _infra(PORT_IN_USE_REASON)
    full = log_causes.disk_full(text)
    if full:
        # Before the signatures: a full disk breaks whatever was writing, and the traceback it leaves can match one.
        return _infra(f"{DISK_FULL}: {full}")
    refused_cache = _hf_cache_refusal(parts)
    if refused_cache:
        # Before the signatures: the evaluation that could not take the lock dies on it with a Python exception.
        return _infra(f"{HF_CACHE_PERMISSION}: {refused_cache}")
    stalled_fetch = log_causes.git_fetch_timeout(text)
    if stalled_fetch:
        return _infra(f"{GIT_FETCH_TIMEOUT}: {stalled_fetch}")
    hang = log_causes.jit_build_timeout(text)
    if hang:
        return _infra(f"{JIT_BUILD_TIMEOUT}: {hang}")
    signature = classify_failure(text)
    if signature.kind == EVAL_RUNTIME_FAILURE:
        refusal = log_causes.bad_request(text)
        if refusal:
            return _candidate(f"{LM_EVAL_BAD_REQUEST_REASON}: {refusal}")
    if signature.is_actionable:
        # The signature names the kind; the line of the log that states it names the flag or the module.
        reader = _QUOTED_KINDS.get(signature.kind)
        cause = reader(text) if reader else None
        return _candidate(f"{signature.kind}: {cause}" if cause else signature.kind)
    flag_error = log_causes.argparse_error(text)
    if flag_error:
        return _candidate(f"{SERVE_FLAG}: {flag_error}")
    return None


def _hf_cache_refusal(parts: list[str]) -> str | None:
    """The Hugging Face cache path whose refusal ended the test, else ``None``.

    The refusal must be the error a log ended on (:func:`log_causes.terminal_error`): ``huggingface_hub`` logs a refusal
    of a read-only cache and carries on, and a refusal survived is no cause. When another log ended on a failure a
    candidate is known to cause (a failed import, say), that failure is the cause and the refusal is not.
    """
    endings = [ending for ending in map(log_causes.terminal_error, parts) if ending]
    refused = [path for path in (log_causes.hf_cache_permission(e) for e in endings) if path]
    if not refused or any(classify_failure(e).is_actionable for e in endings):
        return None
    return refused[-1]


def _from_unnamed_logs(text: str, unrun: str) -> FailureClassification:
    """An unclear failure, named by what the runner recorded of a round that never ran, else by the line of the log that
    says what happened when there is one."""
    if unrun:
        return _unclear(" ".join(unrun.split())[:LOG_LINE_CHARS])
    for reason, cause in (
        (LM_EVAL_BAD_REQUEST_REASON, log_causes.bad_request(text)),
        (PYTHON_EXCEPTION_REASON, log_causes.python_exception(text)),
    ):
        if cause:
            return _unclear(f"{reason}: {cause}")
    return _unclear(NO_KNOWN_SIGNATURE_REASON)


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


def _candidate(reason: str) -> FailureClassification:
    return FailureClassification(Disposition.FAILED, FailureClass.CANDIDATE, reason)


def _unclear(reason: str) -> FailureClassification:
    return FailureClassification(Disposition.FAILED, FailureClass.UNCLEAR, reason)
