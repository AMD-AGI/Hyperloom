# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A toolbox that measures nothing: the same interface as the real one, with configurable, seeded results.

:class:`FakeToolbox` implements :class:`~hyperloom_multibranch.contracts.toolbox.Toolbox`. It accepts a request
(rejection and backpressure as the real one raises them), queues it by priority under the per-branch K and the global
limit, runs attempts with the retry rules of design section 4.5 and writes the request, the result, a withdrawal or a
quarantine to the notebook it is given. The values come from :mod:`hyperloom_multibranch.toolbox.fake_model`.

The queue (:class:`RunQueue`) and the retry rules (:meth:`FakeToolbox._attempt_loop`) are the parts the real toolbox
replaces; both are reached only through ``FakeToolbox``'s own methods, so a replacement keeps the interface above.

Attempts and reruns, as the real toolbox does them: an infrastructure failure retries on a fresh sandbox, away from the
nodes already tried, up to ``max_infra_retries`` times, then the result is ``measurement_failed``; an unclear failure
reruns once, a failing rerun is ``candidate_failed`` and a passing one is the result, the first attempt counting as
that node's infrastructure failure; a candidate failure is final; a failure of the measurement itself is final and
never the candidate's. All attempts share the ``request_id`` and the submitter gets one result.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from hyperloom.contracts.measurement import (
    Attempt,
    Purpose,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom_multibranch.contracts.notebook.records import QuarantineEvent, Withdrawal
from hyperloom_multibranch.contracts.notebook.vocab import QuarantineAction
from hyperloom_multibranch.contracts.toolbox import (
    Backpressure,
    CancelOutcome,
    QueueState,
    RequestRejected,
    UnknownTestRequest,
)
from hyperloom_multibranch.toolbox.fake_model import (
    AttemptOutcome,
    Fate,
    FakeMeasurer,
    FakeToolboxConfig,
)

#: Used when a request leaves ``priority`` empty; ``baseline``, ``health`` and ``final`` have none in the design (the
#: scheduler gives them), so the fake takes a middle one, the integration one for the two the run waits on.
DEFAULT_PRIORITY = {
    Purpose.CONFIRM: 0,
    Purpose.REPRO: 1,
    Purpose.INTEGRATION: 2,
    Purpose.PROFILE: 3,
    Purpose.EXPLORE: 4,
    Purpose.ACCURACY: 4,
    Purpose.BASELINE: 2,
    Purpose.FINAL: 2,
    Purpose.HEALTH: 4,
}
#: Purposes that do not take up one of their branch's K slots.
OUTSIDE_K = frozenset({Purpose.CONFIRM, Purpose.REPRO})
CANCELLED_REASON = "cancelled by the submitter"


class ToolboxNotebook(Protocol):
    """What the fake writes to: the notebook's toolbox-side writers (``TestRecordSink`` and ``NodeHealth``)."""

    async def write_request(self, request: TestRequest) -> object: ...

    async def write_result(self, result: TestResult) -> object: ...

    async def write_withdrawal(self, withdrawal: Withdrawal) -> object: ...

    async def record_quarantine(self, event: QuarantineEvent) -> None: ...


@dataclass(frozen=True)
class QueuedTest:
    request_id: str
    request: TestRequest
    priority: int
    queued_at: float


class RunQueue:
    """Priority queue, first in first out within a priority; knows nothing of K or parallelism."""

    def __init__(self) -> None:
        self._entries: dict[int, deque[QueuedTest]] = {}

    def push(self, entry: QueuedTest) -> None:
        self._entries.setdefault(entry.priority, deque()).append(entry)

    def remove(self, request_id: str) -> bool:
        for entries in self._entries.values():
            for entry in entries:
                if entry.request_id == request_id:
                    entries.remove(entry)
                    return True
        return False

    def pop(self, eligible: Callable[[QueuedTest], bool]) -> QueuedTest | None:
        """The most urgent, oldest entry ``eligible`` accepts; the others keep their place."""

        for priority in sorted(self._entries):
            for entry in self._entries[priority]:
                if eligible(entry):
                    self._entries[priority].remove(entry)
                    return entry
        return None

    def entries(self) -> tuple[QueuedTest, ...]:
        return tuple(entry for priority in sorted(self._entries) for entry in self._entries[priority])


@dataclass
class _Running:
    entry: QueuedTest
    task: asyncio.Task[None] | None = None
    withdrawn: bool = False


@dataclass
class _Run:
    """The attempts of one request so far."""

    attempts: list[Attempt] = field(default_factory=list)
    tried: set[str] = field(default_factory=set)
    infra_failures: int = 0
    unclear_node: str | None = None
    last: AttemptOutcome | None = None


class FakeToolbox:
    """The toolbox without GPUs. ``autorun`` starts accepted tests at once; without it :meth:`drain` runs them."""

    def __init__(
        self,
        config: FakeToolboxConfig | None = None,
        *,
        notebook: ToolboxNotebook | None = None,
        autorun: bool = True,
        parent_exists: Callable[[str], bool] | None = None,
        queue: RunQueue | None = None,
    ) -> None:
        self.config = config or FakeToolboxConfig()
        self.requests: dict[str, TestRequest] = {}
        self.results: dict[str, TestResult] = {}
        self.withdrawn: dict[str, str] = {}
        self.quarantined: set[str] = set()
        self.now = self.config.start_time
        self._notebook = notebook
        self._autorun = autorun
        self._parent_exists = parent_exists
        self._queue = queue or RunQueue()
        self._measurer = FakeMeasurer(self.config)
        self._running: dict[str, _Running] = {}
        self._k: dict[str, int] = {}
        self._scripts: list[tuple[Callable[[TestRequest], bool], list[Fate]]] = []
        self._rejections: list[Callable[[TestRequest], RequestRejected | None]] = []
        self._submitted = 0

    # --- the interface ---------------------------------------------------------------------------------------------

    async def submit_test(self, request: TestRequest) -> str:
        self._check_acceptable(request)
        queued = sum(1 for entry in self._queue.entries() if entry.request.branch_id == request.branch_id)
        if queued >= self.config.branch_queue_limit:
            raise Backpressure(request.branch_id, queued, self.config.branch_queue_limit)
        request_id = str(uuid.uuid5(self.config.campaign_id, f"{request.branch_id}/{self._submitted}"))
        self._submitted += 1
        accepted = request.accepted(request_id)
        priority = request.priority if request.priority is not None else DEFAULT_PRIORITY[request.purpose]
        if self._notebook is not None:
            await self._notebook.write_request(accepted)
        self.requests[request_id] = accepted
        self._queue.push(QueuedTest(request_id, accepted, priority, self.now))
        if self._autorun:
            self._pump()
        return request_id

    async def cancel(self, request_id: str) -> CancelOutcome:
        if request_id not in self.requests:
            raise UnknownTestRequest(request_id)
        if request_id in self.results:
            return CancelOutcome.FINISHED
        if request_id in self.withdrawn:
            return CancelOutcome.WITHDRAWN
        running = self._running.get(request_id)
        if running is None:
            self._queue.remove(request_id)
            await self._withdraw(request_id)
        else:
            running.withdrawn = True
            self._running.pop(request_id, None)  # a task cancelled before its first step never reaches its cleanup
            if running.task is not None:
                running.task.cancel()
            await self._withdraw(request_id)
            if self._autorun:
                self._pump()
        return CancelOutcome.WITHDRAWN

    async def queue_state(self) -> QueueState:
        queued = self._queue.entries()
        by_priority = dict.fromkeys(range(5), 0)
        by_branch: dict[str, int] = {}
        for entry in queued:
            by_priority[entry.priority] += 1
            by_branch[entry.request.branch_id] = by_branch.get(entry.request.branch_id, 0) + 1
        running: dict[str, int] = {}
        for item in self._running.values():
            branch = item.entry.request.branch_id
            running[branch] = running.get(branch, 0) + 1
        return QueueState(by_priority, running, by_branch, self.config.attempt_seconds if self.results else None)

    # --- controls for tests ----------------------------------------------------------------------------------------

    def set_k(self, branch_id: str, k: int) -> None:
        if k < 1:
            raise ValueError("a branch needs room for at least one test")
        self._k[branch_id] = k

    def script(self, match: Callable[[TestRequest], bool], fates: Sequence[Fate]) -> None:
        """Force the fate of the attempts of the requests ``match`` accepts: ``fates[n]`` for attempt ``n``, then random."""

        self._scripts.append((match, list(fates)))

    def reject_if(self, check: Callable[[TestRequest], RequestRejected | None]) -> None:
        """Refuse requests at acceptance: ``check`` returns the rejection to raise, or ``None`` to accept."""

        self._rejections.append(check)

    async def drain(self) -> None:
        """Run everything queued and wait for every running test; returns when the toolbox is idle."""

        while self._queue.entries() or self._running:
            self._pump()
            tasks = [item.task for item in self._running.values() if item.task is not None]
            if not tasks:
                raise RuntimeError("queued tests are not eligible and nothing is running")
            await asyncio.wait(tasks)

    # --- acceptance ------------------------------------------------------------------------------------------------

    def _check_acceptable(self, request: TestRequest) -> None:
        if self._parent_exists is not None and not self._parent_exists(request.parent_commit):
            raise RequestRejected(f"parent commit {request.parent_commit!r} is not in the repository", "unknown_parent")
        if request.retest_of is not None and request.retest_of not in self.requests:
            raise RequestRejected(f"retest_of {request.retest_of!r} names no accepted request", "unknown_retest")
        for check in self._rejections:
            rejection = check(request)
            if rejection is not None:
                raise rejection

    # --- running ---------------------------------------------------------------------------------------------------

    def _eligible(self, entry: QueuedTest) -> bool:
        if len(self._running) >= self.config.max_parallel_tests:
            return False
        if entry.request.purpose in OUTSIDE_K:
            return True
        branch = entry.request.branch_id
        in_k = sum(
            1
            for item in self._running.values()
            if item.entry.request.branch_id == branch and item.entry.request.purpose not in OUTSIDE_K
        )
        return in_k < self._k.get(branch, self.config.default_k)

    def _pump(self) -> None:
        while (entry := self._queue.pop(self._eligible)) is not None:
            item = _Running(entry)
            self._running[entry.request_id] = item
            item.task = asyncio.ensure_future(self._run(item))

    async def _run(self, item: _Running) -> None:
        request_id = item.entry.request_id
        try:
            if self.config.run_delay_s:
                await asyncio.sleep(self.config.run_delay_s)
            result = await self._attempt_loop(item.entry)
            if not item.withdrawn:
                self.results[request_id] = result
                if self._notebook is not None:
                    await self._notebook.write_result(result)
        except asyncio.CancelledError:
            pass
        finally:
            self._running.pop(request_id, None)
            if self._autorun:
                self._pump()

    async def _withdraw(self, request_id: str) -> None:
        self.withdrawn[request_id] = CANCELLED_REASON
        if self._notebook is not None:
            await self._notebook.write_withdrawal(Withdrawal(request_id, CANCELLED_REASON))

    def _forced(self, request: TestRequest, attempt: int) -> Fate | None:
        for match, fates in self._scripts:
            if match(request):
                return fates[attempt] if attempt < len(fates) else None
        return None

    async def _attempt_loop(self, entry: QueuedTest) -> TestResult:
        request, run = entry.request, _Run()
        run.tried.update(request.avoid_nodes)
        reran = False
        while True:
            outcome = self._measurer.run(
                request,
                entry.request_id,
                len(run.attempts),
                avoid=frozenset(run.tried),
                unavailable=frozenset(self.quarantined),
                forced=self._forced(request, len(run.attempts)),
            )
            await self._record_attempt(entry, run, outcome)
            if outcome.fate is Fate.OK:
                await self._quarantine_unclear(entry, run)
                return self._result(entry, run, Validity.SUCCEEDED)
            if outcome.fate is Fate.CANDIDATE or (outcome.fate is Fate.UNCLEAR and reran):
                return self._result(entry, run, Validity.CANDIDATE_FAILED)
            if outcome.fate is Fate.MEASUREMENT:
                return self._result(entry, run, Validity.MEASUREMENT_FAILED)
            if outcome.fate is Fate.UNCLEAR:
                reran, run.unclear_node = True, outcome.sandbox.node_name if outcome.sandbox else None
            elif run.infra_failures > self.config.max_infra_retries:
                return self._result(entry, run, Validity.MEASUREMENT_FAILED)

    async def _record_attempt(self, entry: QueuedTest, run: _Run, outcome: AttemptOutcome) -> None:
        started, self.now = self.now, self.now + self.config.attempt_seconds
        run.attempts.append(Attempt(outcome.sandbox, outcome.failure_class, outcome.failure_reason, started, self.now))
        run.last = outcome
        if outcome.sandbox is not None and outcome.sandbox.node_name is not None:
            run.tried.add(outcome.sandbox.node_name)
        if outcome.fate is Fate.INFRA:
            run.infra_failures += 1
            if outcome.sandbox is not None:
                await self._quarantine(outcome.sandbox.node_name, "infra failure", entry.request_id)

    async def _quarantine_unclear(self, entry: QueuedTest, run: _Run) -> None:
        if run.unclear_node is not None:
            await self._quarantine(run.unclear_node, "unclear failure that a rerun passed", entry.request_id)

    async def _quarantine(self, node: str | None, reason: str, request_id: str) -> None:
        if node is None or not self.config.quarantine_nodes or node in self.quarantined:
            return
        self.quarantined.add(node)
        if self._notebook is not None:
            await self._notebook.record_quarantine(
                QuarantineEvent(node, QuarantineAction.QUARANTINE, reason, request_ids=(request_id,))
            )

    def _result(self, entry: QueuedTest, run: _Run, validity: Validity) -> TestResult:
        last, request, request_id = run.last, entry.request, entry.request_id
        assert last is not None
        measured = validity is Validity.SUCCEEDED
        digest = hashlib.sha256(f"{request_id}/{validity.value}".encode()).hexdigest()
        gpu_seconds = sum(
            (a.finished_at - a.started_at) * a.sandbox.gpu_count for a in run.attempts if a.sandbox is not None
        )
        return TestResult(
            request_id=request_id,
            validity=validity,
            failure_class=None if measured else last.failure_class,
            failure_reason=None if measured else last.failure_reason,
            rounds=last.rounds,
            accuracy=last.accuracy,
            sandbox=last.sandbox,
            attempts=tuple(run.attempts),
            manifest_sha256=digest if measured else None,
            signing_key_fingerprint="fake-signing-key" if measured else None,
            spec_fingerprint=digest[:16] if measured else None,
            surface_sha256=digest if measured else None,
            output_dir=f"tests/{request_id}",
            queued_at=entry.queued_at,
            started_at=run.attempts[0].started_at,
            finished_at=run.attempts[-1].finished_at,
            gpu_seconds=gpu_seconds,
            profile_location=f"tests/{request_id}/profile" if measured and request.kind is TestKind.PROFILE else "",
        )


__all__ = ["DEFAULT_PRIORITY", "FakeToolbox", "QueuedTest", "RunQueue", "ToolboxNotebook"]
