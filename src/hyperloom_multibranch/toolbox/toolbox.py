# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real toolbox: acceptance, queue, attempts, quarantine and results into the notebook (design 4.5, 4.17).

``submit_test`` accepts synchronously: when it returns the request is in the submit log, in the notebook (its request
row, with the claim handed over) and in the queue. Execution is behind the :class:`~.attempts.Executor` seam, so the
fake executor and the hosted one run the same queue and retry code. The outcome of a test is logged before it is
written to the notebook and settled after, so a crash loses neither a request nor a result and renumbers nothing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import replace

from hyperloom.contracts.measurement import FailureClass, TestRequest, TestResult, Validity
from hyperloom_multibranch.contracts.notebook.interfaces import (
    ClaimLost,
    ClaimService,
    ConflictingWrite,
    RecordSearch,
    UnknownRequest,
)
from hyperloom_multibranch.contracts.notebook.records import ClaimKey, Withdrawal
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import (
    CancelOutcome,
    NodeHealth,
    QueueState,
    RequestRejected,
    TestRecordSink,
    UnknownTestRequest,
)
from hyperloom_multibranch.toolbox.acceptance import Acceptance, MeasurementPlanner
from hyperloom_multibranch.toolbox.attempts import DEFAULT_MAX_INFRA_RETRIES, AttemptRunner, Executor, TestJob
from hyperloom_multibranch.toolbox.quarantine import Quarantine
from hyperloom_multibranch.toolbox.queue import QueuedTest, QueuePolicy, WorkQueue, sandbox_priority
from hyperloom_multibranch.toolbox.submission_log import LogEntry, SubmissionLog
from hyperloom_multibranch.toolbox.tools import JobStatus, ToolJobs, ToolRequest

logger = logging.getLogger(__name__)

CLAIM_TTL_SECONDS = 600.0
WRITE_RETRIES = 3


class RealToolbox:
    """Implements :class:`~hyperloom_multibranch.contracts.toolbox.Toolbox`; call :meth:`start` before use."""

    def __init__(
        self,
        *,
        log: SubmissionLog,
        repo: ExperimentRepo,
        sink: TestRecordSink,
        health: NodeHealth,
        executor: Executor,
        planner: MeasurementPlanner,
        active_image_digest: str | None,
        policy: QueuePolicy | None = None,
        claims: ClaimService | None = None,
        records: RecordSearch | None = None,
        tools: ToolJobs | None = None,
        max_infra_retries: int = DEFAULT_MAX_INFRA_RETRIES,
        claim_ttl_seconds: float = CLAIM_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._log, self._sink, self._claims, self._records = log, sink, claims, records
        self._queue = WorkQueue(policy or QueuePolicy())
        self._quarantine = Quarantine(health)
        self._acceptance = Acceptance(
            repo, planner, active_image_digest=active_image_digest, is_accepted=self._is_accepted
        )
        self._runner = AttemptRunner(
            executor,
            avoid=lambda: self._quarantine.nodes,
            on_node_failure=self._node_failed,
            max_infra_retries=max_infra_retries,
        )
        self._ttl, self._clock = claim_ttl_seconds, clock
        self._tools = tools
        self._closed = False
        self._tool_jobs: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._renewer: asyncio.Task[None] | None = None

    # --- lifecycle ---------------------------------------------------------------------------------------------

    async def start(self) -> None:
        """Read the quarantine list, recover what the log says is unfinished, and start renewing claims."""

        await self._quarantine.load()
        await self.recover()
        self._renewer = asyncio.create_task(self._renew_loop())

    async def close(self) -> None:
        """Stop without withdrawing anything: running tests are picked up from the log by the next start."""

        self._closed = True
        tasks = [*self._tasks.values(), *([self._renewer] if self._renewer else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._renewer = None

    async def drain(self) -> None:
        """Wait until nothing is queued or running (for tests and orderly shutdown)."""

        while self._tasks or any(item.state.value == "queued" for item in self._queue.items()):
            if self._tasks:
                await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)
            else:
                await asyncio.sleep(0)
                self._pump()

    # --- the Toolbox interface -----------------------------------------------------------------------------------

    async def submit_test(self, request: TestRequest) -> str:
        async with self._lock:
            prepared = await self._acceptance.prepare(request)
            self._queue.check_backpressure(request)
            priority = self._queue.policy.priority_of(request)
            sequence = self._log.next_sequence(request.branch_id)
            accepted = request.accepted(self._log.request_id(request.branch_id, sequence))
            entry = LogEntry(
                accepted.request_id or "", request.branch_id, sequence, accepted.to_document(), priority, self._clock()
            )
            self._log.accept(entry)
            await self._publish(entry, accepted)
            self._enqueue(entry, accepted, prepared.environment)
        self._pump()
        return entry.request_id

    async def cancel(self, request_id: str) -> CancelOutcome:
        entry = self._log.get(request_id)
        if entry is None or entry.aborted:
            raise UnknownTestRequest(request_id)
        if entry.outcome is not None:
            return CancelOutcome.WITHDRAWN if "withdrawal" in entry.outcome else CancelOutcome.FINISHED
        await self._withdraw(entry, "cancelled by the submitter")
        return CancelOutcome.WITHDRAWN

    async def queue_state(self) -> QueueState:
        return self._queue.state()

    # --- beyond the interface ------------------------------------------------------------------------------------

    async def set_policy(self, policy: QueuePolicy) -> None:
        """The scheduler's quota and priority policy; takes effect for the next test that starts."""

        self._queue.set_policy(policy)
        self._queue.reprioritise()
        self._pump()

    async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
        """Withdraw every queued and running test of a stopped branch; returns how many."""

        pending = [e for e in self._log.entries() if e.branch == branch_id and e.waiting]
        for entry in pending:
            await self._withdraw(entry, reason)
        cancelled = sum(self._tools.cancel(job) for job in self._tool_jobs.pop(branch_id, [])) if self._tools else 0
        return len(pending) + cancelled

    def submit_tool(self, request: ToolRequest) -> str:
        """Start an asynchronous tool (profile, roofline, build, ...) for a branch; returns its job id at once.

        Tool jobs are not tests: they take no slot of K and write no test record. Their result reaches the branch's
        inbox through the notebook, and a stopped branch's running jobs are cancelled with :meth:`cancel_branch`.
        """

        if self._tools is None:
            raise RequestRejected("this toolbox serves no asynchronous tools")
        job_id = self._tools.submit(request)
        self._tool_jobs.setdefault(request.branch, []).append(job_id)
        return job_id

    def tool_status(self, job_id: str) -> JobStatus:
        if self._tools is None:
            raise UnknownTestRequest(job_id)
        return self._tools.status(job_id)

    def record_cold_start(self, seconds: float) -> None:
        self._queue.cold_start_seconds = seconds

    @property
    def quarantined(self) -> frozenset[str]:
        return self._quarantine.nodes

    async def recover(self) -> None:
        """Put the log's unfinished work back: write measured outcomes, re-queue what never ran, in log order."""

        for entry in self._log.entries():
            if entry.aborted or entry.settled:
                continue
            if entry.outcome is not None:
                await self._write_outcome(entry)
            elif await self._already_final(entry):
                self._log.settle(entry.request_id)
            else:
                await self._recover_waiting(entry)
        self._pump()

    async def tick(self) -> None:
        """One round of housekeeping: renew the claims of live tests and retry outcomes the notebook refused."""

        await self.renew_claims()
        for entry in self._log.entries():
            if entry.unwritten:
                await self._write_outcome(entry)

    async def renew_claims(self) -> None:
        if self._claims is None:
            return
        for item in self._queue.items():
            if item.request.intent_fingerprint is None:
                continue
            try:
                await self._claims.renew(self._key(item.request), ttl_seconds=self._ttl)
            except Exception:
                logger.warning("could not renew the claim of %s", item.request_id, exc_info=True)

    # --- accepting and recovering --------------------------------------------------------------------------------

    def _is_accepted(self, request_id: str) -> bool:
        entry = self._log.get(request_id)
        return entry is not None and not entry.aborted

    @staticmethod
    def _key(request: TestRequest) -> ClaimKey:
        return ClaimKey(request.parent_commit, request.intent_fingerprint or "")

    async def _publish(self, entry: LogEntry, accepted: TestRequest) -> None:
        """Write the request row and hand the claim to the test; undo the acceptance if either is refused."""

        try:
            await self._sink.write_request(replace(accepted, priority=entry.priority))
            if self._claims is not None and accepted.intent_fingerprint is not None:
                await self._claims.attach_request(self._key(accepted), entry.request_id, ttl_seconds=self._ttl)
        except ClaimLost as error:
            await self._sink.write_withdrawal(Withdrawal(entry.request_id, f"claim lost: {error}"))
            self._log.abort(entry.request_id)
            raise RequestRejected(f"the claim of this request is not held: {error}") from error
        except BaseException:
            self._log.abort(entry.request_id)
            raise

    def _enqueue(self, entry: LogEntry, request: TestRequest, environment: object) -> None:
        self._queue.push(QueuedTest(request, entry.priority, 0, environment, entry.queued_at))

    async def _already_final(self, entry: LogEntry) -> bool:
        if self._records is None:
            return False
        try:
            record = await self._records.record(entry.request_id)
        except UnknownRequest:  # logged but the notebook never heard of it: the crash came between the two writes
            return False
        return record.result is not None or record.withdrawn_at is not None

    async def _recover_waiting(self, entry: LogEntry) -> None:
        request = TestRequest.from_document(entry.request)
        try:
            prepared = await self._acceptance.prepare(request, replay=True)
            await self._publish(entry, request)
        except RequestRejected as error:
            await self._record_and_write(entry, {"withdrawal": f"not recoverable: {error}"})
            return
        self._enqueue(entry, request, prepared.environment)

    # --- running -------------------------------------------------------------------------------------------------

    def _pump(self) -> None:
        if self._closed:
            return
        while (item := self._queue.take()) is not None:
            self._tasks[item.request_id] = asyncio.create_task(self._run(item))

    async def _run(self, item: QueuedTest) -> None:
        try:
            result = await self._execute(item)
            entry = self._log.get(item.request_id)
            if entry is not None:
                await self._record_and_write(entry, self._outcome_of(result))
        finally:
            self._queue.remove(item.request_id)
            self._tasks.pop(item.request_id, None)
            self._pump()

    async def _execute(self, item: QueuedTest) -> TestResult:
        job = TestJob(item.request, 1, (), item.priority, sandbox_priority(item.priority), item.environment)
        try:
            result = await self._runner.run(job)
        except Exception as error:
            logger.exception("test %s failed in the executor", item.request_id)
            return TestResult(
                request_id=item.request_id,
                validity=Validity.MEASUREMENT_FAILED,
                failure_class=FailureClass.INFRA,
                failure_reason=f"executor error: {error!r}",
                queued_at=item.queued_at,
            )
        return replace(result, request_id=item.request_id, queued_at=item.queued_at)

    @staticmethod
    def _outcome_of(result: TestResult) -> dict[str, object]:
        if result.validity is Validity.MEASUREMENT_FAILED:
            return {"withdrawal": f"measurement_failed: {result.failure_reason}", "result": result.to_document()}
        return {"result": result.to_document()}

    async def _withdraw(self, entry: LogEntry, reason: str) -> None:
        self._queue.remove(entry.request_id)
        task = self._tasks.pop(entry.request_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._record_and_write(entry, {"withdrawal": reason})
        self._pump()

    async def _record_and_write(self, entry: LogEntry, outcome: dict[str, object]) -> None:
        self._log.record_outcome(entry.request_id, outcome)
        await self._write_outcome(entry)

    async def _write_outcome(self, entry: LogEntry) -> None:
        """Write the logged outcome to the notebook and settle it; an outcome the notebook refuses stays unsettled and
        is retried by :meth:`tick` and the next start."""

        outcome = entry.outcome or {}
        for attempt in range(WRITE_RETRIES):
            try:
                if "withdrawal" in outcome:
                    await self._sink.write_withdrawal(Withdrawal(entry.request_id, str(outcome["withdrawal"])))
                else:
                    await self._sink.write_result(TestResult.from_document(outcome["result"]))
            except ConflictingWrite:
                break  # the notebook already holds another final state for this request; there is nothing to write
            except Exception:
                logger.warning(
                    "the notebook refused the outcome of %s (try %d)", entry.request_id, attempt + 1, exc_info=True
                )
                await asyncio.sleep(0)
                continue
            break
        else:
            return
        self._log.settle(entry.request_id)

    # --- quarantine ----------------------------------------------------------------------------------------------

    async def _node_failed(self, node: str, reason: str) -> None:
        await self._quarantine.add(node, reason)

    async def _renew_loop(self) -> None:
        while True:
            await asyncio.sleep(self._ttl / 3)
            await self.tick()


__all__ = ["RealToolbox"]
