# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attempts and reruns of one test, the code the fake and the real executor share.

An :class:`Executor` runs ONE attempt in a fresh sandbox and returns a ``TestResult`` describing it. This module turns
attempts into the request's single result:

* ``infra`` failure: another sandbox, away from every node an earlier attempt failed on for a reason held against it,
  up to ``max_infra_retries`` retries, then ``measurement_failed``; the node is reported as an infrastructure failure.
  A retry after a failure that is not the node's (:data:`WAITED_INFRA_REASONS`: cards a process outside the test held,
  a platform short of sandboxes, a failure never held against its node) waits first, longer each time
  (:class:`InfraBackoff`), so a minute in which the cards or the platform are busy does not spend every attempt; the
  wait never runs past the time its branch can still have a test measured (``latest_start``). Of those, a failure that
  clearly says nothing about the node (:data:`UNAVOIDED_INFRA_REASONS`) does not put its node in the avoid set, so the
  retry may land on the same node again; a runner that did not start in time, could not start or was lost still does,
  since a slow, faulty or dead node is one way to get there. Cards a process outside the test held do not report their
  node either, and their retry waits on the same cards, unless the two attempts just before it were both refused that
  way on one node (:func:`_held_foreign_node`): a holder that stays, which the next attempt avoids. A retry after any
  other infrastructure failure goes at once, since the avoid set and the quarantine already move it to another node.
  An infrastructure
  failure every sandbox repeats (:data:`UNRETRIED_INFRA_REASONS`) ends as ``measurement_failed`` at once, and its node
  is not reported; one it likely repeats (:data:`ONCE_RETRIED_INFRA_REASONS`: an aiter build past the watchdog, a
  sandbox that failed before it ran, a merge smoke that could not run, a profile's server that died in its window) is
  retried once, and its node is not reported either. One that says nothing about
  the node (:data:`NODE_BLAMELESS_INFRA_REASONS`: a runner that could not start, a runner lost, a Hub download refused
  for its rate limit, a result file whose download came back short) is retried like any other, and its node is not
  reported. A profile whose server was ready before it failed (:func:`_served`) retries every infrastructure failure
  once only.
* ``candidate`` failure (a known failure signature): ``candidate_failed`` at once.
* ``unclear`` failure: one rerun away from the first node. A second failure is ``candidate_failed``; a success is the
  result, and the first attempt is re-labelled an infrastructure failure of its node and reported as a failure that
  did not reproduce. A profile whose server was ready before it failed, and a sweep that ran out of its time
  (:data:`TEST_TIMEOUT_REASON`), are not rerun: ``measurement_failed`` at once (:func:`_rerun_refused`).
* an executor that raises :class:`MeasurementRefused` (a rejected spec, a result that fails verification): no retry,
  ``measurement_failed``.

A node that was to be avoided but carried the attempt anyway (one node, or Claw could not avoid it) marks the
attempt ``same_node``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import NamedTuple, Protocol

from hyperloom.measure.failures import (
    CAPACITY_UNAVAILABLE,
    FRAMEWORK_VERSION_MISMATCH,
    HF_CACHE_PERMISSION,
    FOREIGN_PROCESS_ON_CARD,
    HUB_RATE_LIMITED,
    JIT_BUILD_TIMEOUT,
    PROFILER_DIED,
    PROFILER_HUNG,
    PROFILER_SNAPSHOT_FAILED,
    RESULT_DOWNLOAD_INCOMPLETE,
    RUNNER_START_TIMEOUT,
    SANDBOX_ACQUIRE_TIMEOUT,
    TEST_TIMEOUT_REASON,
)
from hyperloom.measure.round_marks import SERVED_PHASES
from hyperloom.measure.runner.import_outcome import INERT_PATCH
from hyperloom.contracts.measurement import (
    Attempt,
    FailureClass,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.merge_smoke import SMOKE_NOT_RUN

log = logging.getLogger(__name__)

DEFAULT_MAX_INFRA_RETRIES = 3
#: What the reason of a test whose executor raised :class:`MeasurementRefused` starts with, before the refusal.
MEASUREMENT_REFUSED = "measurement refused"
#: What the reason of a test whose infrastructure retries ran out says after the count of its attempts.
RETRIES_EXHAUSTED = "attempts failed for infrastructure reasons"
#: How long the attempts wait before each infrastructure retry that waits (:data:`WAITED_INFRA_REASONS`), the first, the
#: second and every later one. Back to back, the four attempts of a test spent about forty seconds, so cards a process
#: outside the test held for a minute ended it ``measurement_failed``; these spread them over seven minutes. The wait
#: holds no card or sandbox, but the test stays running in the queue: it keeps its dispatch slot and its branch's K.
INFRA_BACKOFF_SECONDS: tuple[float, ...] = (60.0, 120.0, 240.0)
#: Infrastructure failures a fresh sandbox repeats, by the name a reason starts with (before its ``: <detail>``). They are
#: about how the test was built, not about a node: a patch whose files the server did not import from the farm imports
#: them from the same place in every sandbox, so a retry only spends GPU time. An attempt that waited the test's whole
#: timeout for a sandbox slot or for room in the quota (``SANDBOX_ACQUIRE_TIMEOUT``) never started, and a retry would
#: wait on the same sandboxes: the test ends with what held it instead. A Hugging Face cache that refused the test
#: (``HF_CACHE_PERMISSION``) is a volume every sandbox mounts alike, so every sandbox is refused the same way. A profile
#: whose server died in the profiler's memory-snapshot dump (``PROFILER_SNAPSHOT_FAILED``) dies there in every sandbox of
#: the image, and says nothing of the candidate. A profile whose server went silent with its profile window open
#: (``PROFILER_HUNG``) hangs again at the same point on the same image; a retry only holds the cards for another silence
#: limit, or meets them still held by the hung server's ranks. The failed profile tells the session its backend cannot
#: profile, so it asks for none again. A sandbox whose framework is another version than the one the candidate's files
#: were compiled against (``FRAMEWORK_VERSION_MISMATCH``) runs the same image as every other sandbox of the test.
UNRETRIED_INFRA_REASONS: frozenset[str] = frozenset(
    {
        INERT_PATCH,
        SANDBOX_ACQUIRE_TIMEOUT,
        HF_CACHE_PERMISSION,
        PROFILER_SNAPSHOT_FAILED,
        PROFILER_HUNG,
        FRAMEWORK_VERSION_MISMATCH,
    }
)
#: The hosted backend's ``error_class`` when the platform made the sandbox and it failed before it ran
#: (``hyperloom.measure.hosted.SANDBOX_CREATE_FAILED``).
SANDBOX_CREATE_FAILED = "sandbox_create_failed"
#: Infrastructure failures a fresh sandbox likely repeats but may not, retried once only and not held against the node:
#: an aiter module rebuilt past the server's watchdog builds again in the next sandbox, unless the shared compiled-code
#: cache took the first build. A sandbox that failed before it ran most often failed on its template or on what the
#: platform mounts into it, which the next sandbox meets the same way; one more sandbox (seconds, a new key) still
#: separates that from a platform hiccup, where giving up at once would lose the test.
#: A merge smoke that could not run (:data:`~.merge_smoke.SMOKE_NOT_RUN`: no interpreter, a child past its timeout) ran
#: on this host, not on a node; one more try separates a host busy for a moment from one that cannot run it.
#: A profile whose server died with its profile window open (``PROFILER_DIED``) most often dies again when the tracer
#: starts on the same image, after the same boot and warmup; the retry, on a node the avoid set has not seen fail,
#: separates that from a fault of the first node, which would otherwise end the test and, through the failed profile,
#: switch profiling off for the session. A profile whose retry dies the same way ends ``measurement_failed``, and the
#: session asks for no profile again.
ONCE_RETRIED_INFRA_REASONS: frozenset[str] = frozenset(
    {JIT_BUILD_TIMEOUT, SANDBOX_CREATE_FAILED, SMOKE_NOT_RUN, PROFILER_DIED}
)
#: The runner's ``error_class`` when it could not start (``hyperloom.measure.runner``).
RUNNER_START_FAILED = "runner_start_failed"
#: The hosted backend's ``error_class`` when it lost the sandbox (``hyperloom.measure.hosted.RUNNER_LOST``).
RUNNER_LOST = "runner_lost"
#: Infrastructure failures retried like any other but never held against the node they ran on. A runner that could not
#: start failed on what its sandbox was given (the template, an entry of the runner cache that is not what its manifest
#: recorded, a signing key mounted read-only), which any node gives it alike. A lost runner is most often the platform
#: taking its sandbox back when the node's allocation ended, which the toolbox cannot tell from a node that died; a node
#: that is sick shows it in the failures that stay held against it (GPU and node faults), and the next attempt of the
#: request still avoids the node it was lost on (:data:`UNAVOIDED_INFRA_REASONS` leaves it out). A sandbox that
#: failed before it ran failed on its template or what the platform mounted, and was never placed on a node it could blame.
#: A download the Hugging Face Hub refused for its rate limit is refused for the egress address every node shares, and a
#: Hugging Face cache that refused the test is the shared volume's permissions, not the node's. A framework of another
#: version than the candidate's files were compiled against is the image's, which every node runs alike.
#: A result file whose download came back short after the re-copies (:data:`RESULT_DOWNLOAD_INCOMPLETE`) is the
#: platform's file API, not the node.
NODE_BLAMELESS_INFRA_REASONS: frozenset[str] = frozenset(
    {
        FRAMEWORK_VERSION_MISMATCH,
        RUNNER_START_FAILED,
        RUNNER_LOST,
        SANDBOX_CREATE_FAILED,
        HUB_RATE_LIMITED,
        SANDBOX_ACQUIRE_TIMEOUT,
        HF_CACHE_PERMISSION,
        RESULT_DOWNLOAD_INCOMPLETE,
    }
)
#: Infrastructure failures that never quarantine their node: the blameless ones, and cards a process outside the test
#: held (:data:`FOREIGN_PROCESS_ON_CARD`), which the retry waits out, since a result measured beside such a process is
#: refused on any node.
_UNQUARANTINED_INFRA_REASONS: frozenset[str] = NODE_BLAMELESS_INFRA_REASONS | frozenset({FOREIGN_PROCESS_ON_CARD})
#: Infrastructure failures whose node is not reported.
_UNREPORTED_INFRA_REASONS = ONCE_RETRIED_INFRA_REASONS | _UNQUARANTINED_INFRA_REASONS
#: Infrastructure failures a retry waits out first (:class:`InfraBackoff`): the ones that are not the node's. Cards a
#: process outside the test held (:data:`FOREIGN_PROCESS_ON_CARD`) or a platform with no sandbox to give
#: (:data:`RUNNER_START_TIMEOUT`, :data:`CAPACITY_UNAVAILABLE`) may be free a minute later, and a failure never held
#: against its node (:data:`NODE_BLAMELESS_INFRA_REASONS`) is not escaped by moving elsewhere. Any other infrastructure
#: failure is held against its node, and the avoid set and the quarantine already send the retry to another one, so it
#: goes at once: a wait would only keep the test's dispatch slot and its branch's K idle.
WAITED_INFRA_REASONS: frozenset[str] = NODE_BLAMELESS_INFRA_REASONS | frozenset(
    {FOREIGN_PROCESS_ON_CARD, RUNNER_START_TIMEOUT, CAPACITY_UNAVAILABLE}
)
#: Infrastructure failures that do not put their node in the next attempt's avoid set (:func:`_facts_nodes`), since
#: they clearly say nothing about the node: cards a process outside the test held (:data:`FOREIGN_PROCESS_ON_CARD`),
#: whose retry waits and may run on the same node unless two attempts in a row were refused that way on it
#: (:func:`_held_foreign_node`), and the failures never held against their node that are the image's, the platform's or
#: a shared service's (:data:`NODE_BLAMELESS_INFRA_REASONS` less the two below). A runner that could not start
#: (:data:`RUNNER_START_FAILED`) or was lost (:data:`RUNNER_LOST`) does not quarantine its node but still avoids it for
#: the next attempt, since a lost runner cannot be told from a node that died; a runner that did not start in time
#: (:data:`RUNNER_START_TIMEOUT`) and a platform with no capacity to give (:data:`CAPACITY_UNAVAILABLE`) avoid it too.
UNAVOIDED_INFRA_REASONS: frozenset[str] = (
    NODE_BLAMELESS_INFRA_REASONS - frozenset({RUNNER_START_FAILED, RUNNER_LOST})
) | frozenset({FOREIGN_PROCESS_ON_CARD})
#: The kinds whose result holds no measured round: a profile names its traces, a repro its verdict.
UNMEASURED_KINDS = (TestKind.PROFILE, TestKind.REPRO)


def carries_request_classes(workload: WorkloadKind, request: TestRequest, validity: Validity) -> bool:
    """Whether a result counts, for each measured round, the requests its client marked failed by class.

    Only a succeeded test of an AgentX campaign does, and only a kind that has measured rounds.
    """
    return workload is WorkloadKind.AGENTX and validity is Validity.SUCCEEDED and request.kind not in UNMEASURED_KINDS


class MeasurementRefused(Exception):
    """The measurement itself is broken (spec rejected, result failed verification): a program error, never retried."""


@dataclass(frozen=True)
class InfraBackoff:
    """How long the attempts of a test wait before an infrastructure retry that waits (:data:`WAITED_INFRA_REASONS`):
    ``seconds[n - 1]`` before the ``n``-th such retry, the last of them before any later one, none when ``seconds`` is
    empty. The test keeps its dispatch slot and its branch's K while it waits. ``latest_start`` gives the time by which a retry
    of a request must start for its branch to still have it measured (``None``, or a ``latest_start`` of ``None``: no
    such time), and a wait never runs past it: a retry that cannot wait goes at once, as before. ``clock`` and ``sleep``
    are the real ones by default."""

    seconds: tuple[float, ...] = INFRA_BACKOFF_SECONDS
    latest_start: Callable[[TestRequest], float | None] | None = None
    clock: Callable[[], float] = field(default=time.time)
    sleep: Callable[[float], Awaitable[None]] = field(default=asyncio.sleep)

    def before(self, retry: int, request: TestRequest) -> float:
        """The seconds to wait before the ``retry``-th waiting infrastructure retry of ``request`` (counted from 1)."""
        if not self.seconds or retry < 1:
            return 0.0
        wait = self.seconds[min(retry, len(self.seconds)) - 1]
        latest = None if self.latest_start is None else self.latest_start(request)
        if latest is not None:
            wait = min(wait, latest - self.clock())
        return max(0.0, wait)


@dataclass(frozen=True)
class TestJob:
    """One attempt for an executor: the accepted request, what acceptance compiled, and the nodes to stay away from."""

    __test__ = False

    request: TestRequest
    attempt: int
    avoid_nodes: tuple[str, ...]
    priority: int
    environment: object = None


class Executor(Protocol):
    """Runs one attempt. Raising :class:`MeasurementRefused` ends the test; ``CancelledError`` is a cancel."""

    async def run(self, job: TestJob) -> TestResult:
        """Run one attempt of ``job`` and return its result."""

    async def reverify(self, result: TestResult) -> None:
        """Check a recorded ``result`` against the evidence of the attempt that made it; raises
        :class:`~hyperloom_multibranch.contracts.gate.ResultUnverified` when it does not match or cannot be checked."""

    def withdraw(self, request_id: str) -> None:
        """End the running attempt of ``request_id`` and release what it holds (its sandbox and cards); returns at
        once, and does nothing for a request that is not running.

        Cancelling the task that awaits :meth:`run` does not do this: the measurement runs in a worker thread no cancel
        reaches, so a withdrawn test would keep its cards to the end. The toolbox calls it for a test it withdraws while
        it runs, never when it only closes (a restart re-attaches to what is still measuring).
        """


NodeFailure = Callable[[str, str, bool], Awaitable[None]]
"""Hears ``(node, reason, reproduced)``: ``reproduced`` is False for an unclear failure that did not recur on its rerun
and True for an infrastructure failure the executor classified."""


def _reason_name(reason: str | None) -> str:
    """The name a failure's ``reason`` starts with, before its ``: <detail>``."""
    return (reason or "").partition(":")[0].strip()


def ending_cause(reason: str | None) -> str:
    """The name of the failure that ended a test, from the ``reason`` of its result: a refused measurement's own reason
    under its :data:`MEASUREMENT_REFUSED` prefix, and empty when the infrastructure retries ran out
    (:data:`RETRIES_EXHAUSTED`), whose attempts may each have failed another way."""
    name, _, detail = (reason or "").partition(":")
    if name.strip() == MEASUREMENT_REFUSED:
        return _reason_name(detail)
    return "" if name.endswith(RETRIES_EXHAUSTED) else name.strip()


def node_unquarantined(reason: str | None) -> bool:
    """Whether a failure's ``reason`` names one that never quarantines its node (:data:`_UNQUARANTINED_INFRA_REASONS`):
    one never held against it, or cards a process outside the test held."""
    return _reason_name(reason) in _UNQUARANTINED_INFRA_REASONS


def _retry_refused(request: TestRequest, result: TestResult, attempts: list[Attempt]) -> bool:
    """Whether an infrastructure failure ``result`` names one that a retry in a fresh sandbox repeats, or one retried
    once only that ``attempts`` (this one included) already met before. A profile whose server was ready before it
    failed (:func:`_served`) retries every reason once only: its failures then mostly come from the profiler on the
    image, and a retry that met one again met it on a second node, where it would only hold the cards for a third."""
    name = _reason_name(result.failure_reason)
    if name in UNRETRIED_INFRA_REASONS:
        return True
    once = name in ONCE_RETRIED_INFRA_REASONS or (request.kind is TestKind.PROFILE and _served(result))
    return once and sum(_reason_name(a.failure_reason) == name for a in attempts) > 1


def _rerun_refused(request: TestRequest, result: TestResult) -> bool:
    """Whether an unclear failure ``result`` ends ``measurement_failed`` without its rerun, nothing held against the
    candidate or the node.

    A profile whose server was ready before it failed (:func:`_served`) nearly always fails its rerun for the same cause
    (a server that dies or stalls under the tracer, a client that times out), after another boot and window on its
    cards; a profile is evidence for the session, not a verdict on its candidate. One that failed while its server
    booted keeps its rerun: a boot also fails for a passing fault of its node, and a profile that ended on one would
    switch profiling off for the session. A sweep that ran out of its time gets the same time for the same concs on its
    rerun, which a sweep at a low conc outruns again; FINAL, the sweep's one requester, ranks a candidate whose sweep
    gave nothing on its reproduction (``hyperloom_multibranch.scheduling.final``), so a rerun only holds FINAL for one
    more timeout."""
    if request.kind is TestKind.PROFILE:
        return _served(result)
    return request.kind is TestKind.SWEEP and _reason_name(result.failure_reason) == TEST_TIMEOUT_REASON


def _served(result: TestResult) -> bool:
    """Whether an attempt's server was ready before the attempt failed: its phases hold one a round begins only at its
    server's ready stamp (:data:`~hyperloom.measure.round_marks.SERVED_PHASES`). An attempt that stamped no phases (no
    result landed, or a backend that stamps none) counts as not ready."""
    return any(phase.name in SERVED_PHASES for phase in result.phases)


def _attempt_of(result: TestResult) -> Attempt:
    return Attempt(
        sandbox=result.sandbox,
        failure_class=result.failure_class,
        failure_reason=result.failure_reason,
        started_at=result.started_at,
        finished_at=result.finished_at,
    )


def _mark_same_node(result: TestResult, avoided: set[str]) -> TestResult:
    sandbox = result.sandbox
    if sandbox is None or sandbox.node_name not in avoided or sandbox.same_node:
        return result
    return replace(result, sandbox=replace(sandbox, same_node=True))


def _facts_nodes(attempts: list[Attempt]) -> list[str]:
    """The nodes the next attempt avoids: each earlier attempt's, less the ones whose infrastructure failure said nothing
    about the node (:data:`UNAVOIDED_INFRA_REASONS`), whose retry may run on the same node; and the node a process
    outside the test held cards on for the two attempts just before it (:func:`_held_foreign_node`)."""
    nodes = [attempt.node for attempt in attempts if attempt.node and not _unavoided(attempt)]
    held = _held_foreign_node(attempts)
    return nodes if held is None or held in nodes else [*nodes, held]


def _held_foreign_node(attempts: list[Attempt]) -> str | None:
    """The node the last two of ``attempts`` were both refused on for cards a process outside the test held
    (:data:`FOREIGN_PROCESS_ON_CARD`), or ``None``. A holder that outlasts a wait likely stays, so the next attempt
    avoids that node; that one attempt only, since an attempt after it no longer has both refusals just before it. The
    node is still neither reported nor quarantined."""
    if len(attempts) < 2:
        return None
    previous, last = attempts[-2], attempts[-1]
    if last.node and previous.node == last.node and _foreign(previous) and _foreign(last):
        return last.node
    return None


def _foreign(attempt: Attempt) -> bool:
    return (
        attempt.failure_class is FailureClass.INFRA and _reason_name(attempt.failure_reason) == FOREIGN_PROCESS_ON_CARD
    )


def _unavoided(attempt: Attempt) -> bool:
    return (
        attempt.failure_class is FailureClass.INFRA and _reason_name(attempt.failure_reason) in UNAVOIDED_INFRA_REASONS
    )


def _waited(attempt: Attempt) -> bool:
    return attempt.failure_class is FailureClass.INFRA and _reason_name(attempt.failure_reason) in WAITED_INFRA_REASONS


class _Ending(NamedTuple):
    """How a request ends: its validity and, when it failed, the failure class and the reason."""

    validity: Validity
    failure: FailureClass | None
    reason: str | None


class AttemptRunner:
    """Runs a request's attempts with ``executor``; ``avoid`` supplies the quarantine list, ``on_node_failure`` hears of
    a node that failed, ``backoff`` paces the infrastructure retries."""

    def __init__(
        self,
        executor: Executor,
        *,
        avoid: Callable[[], frozenset[str]],
        on_node_failure: NodeFailure,
        max_infra_retries: int = DEFAULT_MAX_INFRA_RETRIES,
        backoff: InfraBackoff = InfraBackoff(),
    ) -> None:
        self._executor = executor
        self._avoid = avoid
        self._on_node_failure = on_node_failure
        self._max_infra_retries = max_infra_retries
        self._backoff = backoff

    async def run(self, job: TestJob) -> TestResult:
        attempts: list[Attempt] = []
        infra_retries, reran = 0, False
        last: TestResult | None = None
        while True:
            avoided = set(job.request.avoid_nodes) | self._avoid() | set(_facts_nodes(attempts))
            attempt_job = replace(job, attempt=len(attempts) + 1, avoid_nodes=tuple(sorted(avoided)))
            if attempts:
                log.info(
                    "test %s: attempt %d starts after %s (%s)",
                    job.request.request_id,
                    attempt_job.attempt,
                    attempts[-1].failure_class.value if attempts[-1].failure_class else "a failure",
                    attempts[-1].failure_reason or "no reason given",
                )
            try:
                result = await self._executor.run(attempt_job)
            except MeasurementRefused as error:
                ending = _Ending(Validity.MEASUREMENT_FAILED, FailureClass.INFRA, f"{MEASUREMENT_REFUSED}: {error}")
                return self._final(job, last, attempts, ending)
            result = _mark_same_node(result, avoided)
            attempts.append(_attempt_of(result))
            last = result
            failure = result.failure_class
            if result.validity is Validity.SUCCEEDED:
                return await self._succeeded(job, result, attempts)
            if failure is FailureClass.CANDIDATE:
                return self._final(
                    job, result, attempts, _Ending(Validity.CANDIDATE_FAILED, failure, result.failure_reason)
                )
            if failure is FailureClass.UNCLEAR:
                if reran or _rerun_refused(job.request, result):
                    validity = Validity.CANDIDATE_FAILED if reran else Validity.MEASUREMENT_FAILED
                    return self._final(job, result, attempts, _Ending(validity, failure, result.failure_reason))
                reran = True
                continue
            if _retry_refused(job.request, result, attempts):
                return self._final(
                    job,
                    result,
                    attempts,
                    _Ending(Validity.MEASUREMENT_FAILED, FailureClass.INFRA, result.failure_reason),
                )
            if _reason_name(result.failure_reason) not in _UNREPORTED_INFRA_REASONS:
                await self._report(attempts[-1], result.failure_reason, reproduced=True)
            if infra_retries >= self._max_infra_retries:
                reason = f"{infra_retries + 1} {RETRIES_EXHAUSTED}: {result.failure_reason}"
                return self._final(
                    job, result, attempts, _Ending(Validity.MEASUREMENT_FAILED, FailureClass.INFRA, reason)
                )
            infra_retries += 1
            await self._back_off(job.request, attempts)

    async def _back_off(self, request: TestRequest, attempts: list[Attempt]) -> None:
        """Wait before the next infrastructure retry when the last attempt failed for a reason that is not its node's
        (:data:`WAITED_INFRA_REASONS`), the ``n``-th such wait of the request for the ``n``-th such failure; after any
        other failure the retry goes at once, to a node the avoid set has not yet seen fail."""
        waited = [a for a in attempts if _waited(a)]
        if not waited or waited[-1] is not attempts[-1]:
            return
        retry = len(waited)
        wait = self._backoff.before(retry, request)
        if wait > 0:
            log.info("test %s: infrastructure retry %d waits %.0f s first", request.request_id, retry, wait)
            await self._backoff.sleep(wait)

    async def _succeeded(self, job: TestJob, result: TestResult, attempts: list[Attempt]) -> TestResult:
        first = attempts[0]
        if len(attempts) > 1 and first.failure_class is FailureClass.UNCLEAR:
            reason = f"unclear failure did not reproduce on a rerun: {first.failure_reason or 'no reason given'}"
            attempts[0] = replace(first, failure_class=FailureClass.INFRA, failure_reason=reason)
            await self._report(attempts[0], reason, reproduced=False)
        return self._final(job, result, attempts, _Ending(result.validity, None, None))

    async def _report(self, attempt: Attempt, reason: str | None, *, reproduced: bool) -> None:
        if attempt.node:
            await self._on_node_failure(attempt.node, reason or "infrastructure failure", reproduced)

    @staticmethod
    def _final(job: TestJob, result: TestResult | None, attempts: list[Attempt], ending: _Ending) -> TestResult:
        base = result or TestResult(request_id=job.request.request_id or "", validity=ending.validity)
        return replace(
            base,
            validity=ending.validity,
            failure_class=ending.failure,
            failure_reason=ending.reason,
            attempts=tuple(attempts),
        )
