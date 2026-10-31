# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attempts and reruns of one test (design 4.5), the code the fake and the real executor share.

An :class:`Executor` runs ONE attempt in a fresh sandbox and returns a ``TestResult`` describing it. This module turns
attempts into the request's single result:

* ``infra`` failure: another sandbox, away from every node tried, up to ``max_infra_retries`` retries, then
  ``measurement_failed``; the node is reported as an infrastructure failure. An infrastructure failure every sandbox
  repeats (:data:`UNRETRIED_INFRA_REASONS`) ends as ``measurement_failed`` at once, and its node is not reported; one it
  likely repeats (:data:`ONCE_RETRIED_INFRA_REASONS`: an aiter build past the watchdog, a sandbox that failed before it
  ran) is retried once, and its node is not reported either. One that says nothing about the node
  (:data:`NODE_BLAMELESS_INFRA_REASONS`: a runner that could not start, a runner lost, a Hub download refused for its
  rate limit) is retried like any other, and its node is not reported.
* ``candidate`` failure (a known failure signature): ``candidate_failed`` at once.
* ``unclear`` failure: one rerun away from the first node. A second failure is ``candidate_failed``; a success is the
  result, and the first attempt is re-labelled an infrastructure failure of its node and reported as a failure that
  did not reproduce.
* an executor that raises :class:`MeasurementRefused` (a rejected spec, a result that fails verification): no retry,
  ``measurement_failed``.

A node that was to be avoided but carried the attempt anyway (one node, or Claw could not avoid it) marks the
attempt ``same_node``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol

from hyperloom.measure.failures import HUB_RATE_LIMITED, JIT_BUILD_TIMEOUT, SANDBOX_ACQUIRE_TIMEOUT
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

log = logging.getLogger(__name__)

DEFAULT_MAX_INFRA_RETRIES = 3
#: Infrastructure failures a fresh sandbox repeats, by the name a reason starts with (before its ``: <detail>``). They are
#: about how the test was built, not about a node: a patch whose files the server did not import from the farm imports
#: them from the same place in every sandbox, so a retry only spends GPU time. An attempt that waited the test's whole
#: timeout for a sandbox slot or for room in the quota (``SANDBOX_ACQUIRE_TIMEOUT``) never started, and a retry would
#: wait on the same sandboxes: the test ends with what held it instead.
UNRETRIED_INFRA_REASONS: frozenset[str] = frozenset({INERT_PATCH, SANDBOX_ACQUIRE_TIMEOUT})
#: The hosted backend's ``error_class`` when the platform made the sandbox and it failed before it ran
#: (``hyperloom.measure.hosted.SANDBOX_CREATE_FAILED``).
SANDBOX_CREATE_FAILED = "sandbox_create_failed"
#: Infrastructure failures a fresh sandbox likely repeats but may not, retried once only and not held against the node:
#: an aiter module rebuilt past the server's watchdog builds again in the next sandbox, unless the shared compiled-code
#: cache took the first build. A sandbox that failed before it ran most often failed on its template or on what the
#: platform mounts into it, which the next sandbox meets the same way; one more sandbox (seconds, a new key) still
#: separates that from a platform hiccup, where giving up at once would lose the test.
ONCE_RETRIED_INFRA_REASONS: frozenset[str] = frozenset({JIT_BUILD_TIMEOUT, SANDBOX_CREATE_FAILED})
#: The runner's ``error_class`` when it could not start (``hyperloom.measure.runner``).
RUNNER_START_FAILED = "runner_start_failed"
#: The hosted backend's ``error_class`` when it lost the sandbox (``hyperloom.measure.hosted.RUNNER_LOST``).
RUNNER_LOST = "runner_lost"
#: Infrastructure failures retried like any other but never held against the node they ran on. A runner that could not
#: start failed on what its sandbox was given (the template, an entry of the runner cache that is not what its manifest
#: recorded, a signing key mounted read-only), which any node gives it alike. A lost runner is most often the platform
#: taking its sandbox back when the node's allocation ended, which the toolbox cannot tell from a node that died; a node
#: that is sick shows it in the failures that stay held against it (the card guard, GPU and node faults). A sandbox that
#: failed before it ran failed on its template or what the platform mounted, and was never placed on a node it could blame.
#: A download the Hugging Face Hub refused for its rate limit is refused for the egress address every node shares.
NODE_BLAMELESS_INFRA_REASONS: frozenset[str] = frozenset(
    {RUNNER_START_FAILED, RUNNER_LOST, SANDBOX_CREATE_FAILED, HUB_RATE_LIMITED, SANDBOX_ACQUIRE_TIMEOUT}
)
#: Infrastructure failures whose node is not reported.
_UNREPORTED_INFRA_REASONS = ONCE_RETRIED_INFRA_REASONS | NODE_BLAMELESS_INFRA_REASONS
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
class TestJob:
    """One attempt for an executor: the accepted request, what acceptance compiled, and the nodes to stay away from."""

    __test__ = False

    request: TestRequest
    attempt: int
    avoid_nodes: tuple[str, ...]
    priority: int
    sandbox_priority: int
    environment: object = None


class Executor(Protocol):
    """Runs one attempt. Raising :class:`MeasurementRefused` ends the test; ``CancelledError`` is a cancel."""

    async def run(self, job: TestJob) -> TestResult: ...

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


def node_blameless(reason: str | None) -> bool:
    """Whether a failure's ``reason`` names one never held against its node (:data:`NODE_BLAMELESS_INFRA_REASONS`)."""
    return _reason_name(reason) in NODE_BLAMELESS_INFRA_REASONS


def _retry_refused(reason: str | None, attempts: list[Attempt]) -> bool:
    """Whether an infrastructure failure's ``reason`` names one that a retry in a fresh sandbox repeats, or one retried
    once only that ``attempts`` (this one included) already met before."""
    name = _reason_name(reason)
    if name in UNRETRIED_INFRA_REASONS:
        return True
    return name in ONCE_RETRIED_INFRA_REASONS and sum(_reason_name(a.failure_reason) == name for a in attempts) > 1


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
    return [attempt.node for attempt in attempts if attempt.node]


class _Ending(NamedTuple):
    """How a request ends: its validity and, when it failed, the failure class and the reason."""

    validity: Validity
    failure: FailureClass | None
    reason: str | None


class AttemptRunner:
    """Runs a request's attempts with ``executor``; ``avoid`` supplies the quarantine list, ``on_node_failure`` hears of
    a node that failed."""

    def __init__(
        self,
        executor: Executor,
        *,
        avoid: Callable[[], frozenset[str]],
        on_node_failure: NodeFailure,
        max_infra_retries: int = DEFAULT_MAX_INFRA_RETRIES,
    ) -> None:
        self._executor = executor
        self._avoid = avoid
        self._on_node_failure = on_node_failure
        self._max_infra_retries = max_infra_retries

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
                ending = _Ending(Validity.MEASUREMENT_FAILED, FailureClass.INFRA, f"measurement refused: {error}")
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
                if reran:
                    return self._final(
                        job, result, attempts, _Ending(Validity.CANDIDATE_FAILED, failure, result.failure_reason)
                    )
                reran = True
                continue
            if _retry_refused(result.failure_reason, attempts):
                return self._final(
                    job,
                    result,
                    attempts,
                    _Ending(Validity.MEASUREMENT_FAILED, FailureClass.INFRA, result.failure_reason),
                )
            if _reason_name(result.failure_reason) not in _UNREPORTED_INFRA_REASONS:
                await self._report(attempts[-1], result.failure_reason, reproduced=True)
            if infra_retries >= self._max_infra_retries:
                reason = f"{infra_retries + 1} attempts failed for infrastructure reasons: {result.failure_reason}"
                return self._final(
                    job, result, attempts, _Ending(Validity.MEASUREMENT_FAILED, FailureClass.INFRA, reason)
                )
            infra_retries += 1

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
