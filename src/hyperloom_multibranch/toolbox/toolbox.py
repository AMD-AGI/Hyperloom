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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from hyperloom.contracts.measurement import AttachmentRef, FailureClass, TestKind, TestRequest, TestResult, Validity
from hyperloom.contracts.recipe import KernelArtifact, KernelKind
from hyperloom_multibranch.contracts.notebook.interfaces import (
    ClaimLost,
    ClaimService,
    ConflictingWrite,
    RecordSearch,
    UnknownRequest,
)
from hyperloom_multibranch.contracts.notebook.records import ClaimKey, Withdrawal
from hyperloom_multibranch.contracts.toolbox import (
    BranchLease,
    CancelOutcome,
    MeasurementCapacity,
    NodeHealth,
    QueueState,
    RequestRejected,
    TestProgress,
    TestRecordSink,
    TestWithdrawn,
    UnknownTestRequest,
)
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.toolbox.analysis import (
    ANALYSIS_TOOLS,
    CODE_NO_EXECUTOR,
    CODE_TOOL_UNAVAILABLE,
    ROOT_PROFILE_JOB_PREFIX,
    AnalysisAsk,
    analysis_request,
    root_profile_request,
)
from hyperloom_multibranch.toolbox.acceptance import Acceptance, AcceptanceSetup, check_kind
from hyperloom_multibranch.toolbox.attachments import store_attachment
from hyperloom_multibranch.toolbox.attempts import DEFAULT_MAX_INFRA_RETRIES, AttemptRunner, Executor, TestJob
from hyperloom_multibranch.toolbox.kernel_artifacts import KernelVariables, mint_kernel_artifact
from hyperloom_multibranch.toolbox.quarantine import Quarantine
from hyperloom_multibranch.toolbox.queue import QueuedTest, QueuePolicy, State, WorkQueue, sandbox_priority
from hyperloom_multibranch.toolbox.submission_log import LogEntry, SubmissionLog
from hyperloom_multibranch.toolbox.tools import TRACELENS_TOOL, JobStatus, ToolJobs, ToolRequest

logger = logging.getLogger(__name__)

CLAIM_TTL_SECONDS = 600.0
WRITE_RETRIES = 3


@dataclass(frozen=True)
class ToolboxStores:
    """Where the toolbox keeps what it must not lose: the submit log and the notebook's ports.

    ``sink`` takes the requests, results and withdrawals, ``health`` holds the quarantine list, ``claims`` hands the
    claim of a request over, ``records`` is searched when a restart recovers unfinished tests (both may be ``None``).
    """

    log: SubmissionLog
    sink: TestRecordSink
    health: NodeHealth
    claims: ClaimService | None = None
    records: RecordSearch | None = None


@dataclass(frozen=True)
class ToolboxTuning:
    """The queue policy, the infrastructure retries, the claim lifetime and the clock; the defaults are the real ones."""

    policy: QueuePolicy | None = None
    max_infra_retries: int = DEFAULT_MAX_INFRA_RETRIES
    claim_ttl_seconds: float = CLAIM_TTL_SECONDS
    clock: Callable[[], float] = time.time


class RealToolbox:
    """Implements :class:`~hyperloom_multibranch.contracts.toolbox.Toolbox`; call :meth:`start` before use.

    ``trace_root`` is the directory a PROFILE record's relative ``profile_location`` is read from (the session
    directory of a builder that lands test output there); ``None`` passes the location on as recorded.
    """

    trace_root: Path | None = None
    #: The session's blob store (``put``/``get``) that ``put_attachment`` writes to; ``None``: attachments unsupported.
    blobs: Any = None

    def __init__(
        self,
        stores: ToolboxStores,
        executor: Executor,
        acceptance: AcceptanceSetup,
        *,
        tools: ToolJobs | None = None,
        tuning: ToolboxTuning = ToolboxTuning(),
    ) -> None:
        self._log, self._sink, self._claims, self._records = stores.log, stores.sink, stores.claims, stores.records
        self._executor = executor
        self._queue = WorkQueue(tuning.policy or QueuePolicy())
        self._quarantine = Quarantine(stores.health)
        self._acceptance = Acceptance(acceptance, is_accepted=self._is_accepted, live_profile=self._live_profile)
        self._runner = AttemptRunner(
            executor,
            avoid=lambda: self._quarantine.nodes,
            on_node_failure=self._node_failed,
            max_infra_retries=tuning.max_infra_retries,
        )
        self._ttl, self._clock = tuning.claim_ttl_seconds, tuning.clock
        self._tools, self._repo, self._kinds = tools, acceptance.repo, acceptance.supported_kinds
        self._closed = False
        self._tool_jobs: dict[str, list[str]] = {}
        self._kernel_variables = KernelVariables()
        self._lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._outcome_events: dict[str, asyncio.Event] = {}
        self._renewer: asyncio.Task[None] | None = None

    # --- lifecycle ---------------------------------------------------------------------------------------------

    async def start(self) -> None:
        """Read the quarantine list, recover what the log says is unfinished, and start renewing claims."""

        await self._quarantine.load()
        await self.recover()
        self._renewer = asyncio.create_task(self._renew_loop())

    async def close(self) -> None:
        """Stop without withdrawing anything: running tests are picked up from the log by the next start.

        Tool jobs are not recoverable, so they are cancelled, and their child processes ended, before this returns.
        """

        self._closed = True
        tasks = [*self._tasks.values(), *([self._renewer] if self._renewer else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._renewer = None
        if self._tools is not None:
            await self._tools.close()

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
        check_kind(request, self._kinds)
        async with self._lock:
            request = self._acceptance.with_required_accuracy(request)
            prepared = await self._acceptance.prepare(request)
            self._queue.check_branch_listed(request)
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

    async def progress(self, request_id: str) -> TestProgress | None:
        item = self._queue.get(request_id)
        if item is None:
            return None
        if item.state is State.QUEUED:
            return TestProgress("queued", queued_ahead=self._queue.queued_ahead(request_id))
        running = None if item.started_at is None else max(0.0, self._clock() - item.started_at)
        ago, line = await self._last_output(request_id) or (None, None)
        return TestProgress("running", running_seconds=running, last_output_seconds_ago=ago, last_line=line)

    async def measurement_capacity(self) -> MeasurementCapacity:
        """The cards this toolbox can measure on; a toolbox with no card pool of its own reports it as unknown."""

        return MeasurementCapacity.unknown()

    async def lease_branch_sandbox(self, branch_id: str, cards: int) -> BranchLease:
        """Hold ``cards`` of the toolbox's cards and a working directory of their own for ``branch_id``'s agent.

        The cards leave the measurement pool until :meth:`release_branch_sandbox` or :meth:`close`; see
        :class:`~hyperloom_multibranch.contracts.toolbox.BranchLease`. A toolbox whose backend has no cards of its own
        to lease refuses with ``unsupported_kind``; a backend that leases raises ``lease_exists`` and
        ``lease_leaves_no_measurement`` as :class:`RequestRejected`.
        """

        raise RequestRejected("this toolbox's backend has no cards to lease", "unsupported_kind")

    async def release_branch_sandbox(self, lease_id: str) -> None:
        """Return the cards of a lease; idempotent, and safe to call from several tasks at once."""

    async def put_attachment(self, attachment_id: str, kind: Literal["file", "dir"], data: bytes) -> AttachmentRef:
        """Keep ``data`` in the session's blob store for a repro request to carry; see the interface."""

        if self.blobs is None:
            raise RequestRejected("this toolbox has no blob store for attachments", "attachments_unsupported")
        return store_attachment(self.blobs, attachment_id, kind, data)

    async def put_kernel_artifact(
        self, kind: KernelKind, target: str, env: str, data: bytes, rows_only: bool = False
    ) -> KernelArtifact:
        """Keep a tuned table in the session's blob store, the one the experiment repository reads; see the interface."""

        if self.blobs is None:
            raise RequestRejected("this toolbox has no blob store for kernel artifacts", "attachments_unsupported")
        artifact = mint_kernel_artifact(kind, target, env, data, rows_only)
        self._kernel_variables.bind(artifact)
        self.blobs.put(data)
        return artifact

    # --- beyond the interface ------------------------------------------------------------------------------------

    async def check_test(self, request: TestRequest) -> None:
        """Run acceptance on ``request`` as :meth:`submit_test` does, and accept nothing.

        Raises :class:`RequestRejected` for the check that refuses it; nothing is queued, logged or written to the
        notebook either way. The queue's own checks (branch listed, backpressure) are left out: they say when a test
        may start, not whether it may run.
        """

        check_kind(request, self._kinds)
        async with self._lock:
            await self._acceptance.prepare(self._acceptance.with_required_accuracy(request))

    async def set_policy(self, policy: QueuePolicy) -> None:
        """The scheduler's quota and priority policy; takes effect for the next test that starts."""

        self._queue.set_policy(policy)
        self._queue.reprioritise()
        self._pump()

    async def outcome(self, request_id: str) -> TestResult:
        """Wait for the final outcome of an accepted test and return its result.

        Raises :class:`TestWithdrawn` when the test was withdrawn instead, :class:`UnknownTestRequest` for an id the
        toolbox never accepted. A ``measurement_failed`` result is returned like any other: it is a result.
        """

        entry = self._log.get(request_id)
        if entry is None or entry.aborted:
            raise UnknownTestRequest(request_id)
        while entry.outcome is None:
            await self._outcome_events.setdefault(request_id, asyncio.Event()).wait()
        if "result" in entry.outcome:
            return TestResult.from_document(entry.outcome["result"])
        raise TestWithdrawn(request_id, str(entry.outcome["withdrawal"]))

    async def reverify(self, request_id: str) -> None:
        """Check the notebook's result of ``request_id`` against the evidence of the attempt that made it (L26).

        Fails closed: raises :class:`ResultUnverified` when there is no record, no result or no manifest hash to check
        it by; otherwise what the executor raises.
        """

        if self._records is None:
            raise ResultUnverified(request_id, "the toolbox reads no notebook records")
        try:
            record = await self._records.record(request_id)
        except UnknownRequest:
            raise ResultUnverified(request_id, "the notebook holds no record of this request") from None
        result = record.result
        if result is None:
            raise ResultUnverified(request_id, "the record holds no result")
        if not result.manifest_sha256:
            raise ResultUnverified(request_id, "the result names no manifest hash")
        await self._executor.reverify(result)

    async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
        """Withdraw every queued and running test of a stopped branch; returns how many."""

        tests = await self._withdraw_tests(lambda entry: entry.branch == branch_id, reason)
        return len(tests) + self._cancel_tool_jobs(branch_id)

    async def withdraw_branch(self, branch_id: str, reason: str = "branch stopped") -> tuple[str, ...]:
        """Like :meth:`cancel_branch`, returning the request ids of the tests it withdrew."""

        tests = await self._withdraw_tests(lambda entry: entry.branch == branch_id, reason)
        self._cancel_tool_jobs(branch_id)
        return tests

    async def withdraw_all(self, reason: str = "session stopped") -> tuple[str, ...]:
        """Withdraw every queued and running test of every branch; returns their request ids."""

        tests = await self._withdraw_tests(lambda entry: True, reason)
        for branch in list(self._tool_jobs):
            self._cancel_tool_jobs(branch)
        return tests

    async def _withdraw_tests(self, wanted: Callable[[LogEntry], bool], reason: str) -> tuple[str, ...]:
        pending = [e for e in self._log.entries() if e.waiting and wanted(e)]
        for entry in pending:
            await self._withdraw(entry, reason)
        return tuple(entry.request_id for entry in pending)

    def _cancel_tool_jobs(self, branch_id: str) -> int:
        if self._tools is None:
            return 0
        return sum(self._tools.cancel(job) for job in self._tool_jobs.pop(branch_id, []))

    def submit_tool(self, request: ToolRequest, *, job_id: str | None = None) -> str:
        """Start an asynchronous tool (profile, roofline, build, ...) for a branch; returns its job id at once.

        Tool jobs are not tests: they take no slot of K and write no test record. Their result reaches the branch's
        inbox through the notebook, and a stopped branch's running jobs are cancelled with :meth:`cancel_branch`.
        """

        if self._tools is None:
            raise RequestRejected("this toolbox serves no asynchronous tools", CODE_NO_EXECUTOR)
        job_id = self._tools.submit(request, job_id=job_id)
        self._tool_jobs.setdefault(request.branch, []).append(job_id)
        return job_id

    @property
    def analysis_tools(self) -> tuple[str, ...]:
        """The analysis tools this toolbox serves: those of :data:`ANALYSIS_TOOLS` its tool registry holds.

        The one answer to "which analysis tools does this toolbox serve": the MCP surface advertises exactly these and
        :meth:`submit_analysis` refuses the others, both from the registry the toolbox was built with.
        """
        held = self._tools.tool_names if self._tools is not None else ()
        return tuple(tool for tool in ANALYSIS_TOOLS if tool in held)

    def _analysis_jobs(self, tool: str) -> ToolJobs:
        if self._tools is None:
            raise RequestRejected("this toolbox serves no asynchronous tools", CODE_NO_EXECUTOR)
        if tool in ANALYSIS_TOOLS and tool not in self.analysis_tools:
            raise RequestRejected(f"{tool!r} is not run by this toolbox's backend", CODE_TOOL_UNAVAILABLE)
        return self._tools

    async def submit_analysis(
        self, branch: str, tool: str, params: Mapping[str, Any], profile_request_id: str = ""
    ) -> str:
        """Start ``tracelens``, ``roofline`` or ``kernel_tune`` (those of :attr:`analysis_tools`) for ``branch``; returns the job id at once.

        The commit of ``tracelens`` and ``roofline`` is the head of the branch in the experiment repository, read here at
        submission; it is never taken from ``params``. ``profile_request_id`` is kept only for a PROFILE test of that commit,
        whichever branch ran it. ``kernel_tune`` reads the landed output of the test ``params`` names (the notebook's record
        of it) and leases a card of the toolbox for the job, which ``lease_leaves_no_measurement`` refuses when the card
        cannot be spared.
        """

        self._analysis_jobs(tool)
        request = await analysis_request(
            self._repo, self._records, AnalysisAsk(branch, tool, params, profile_request_id), self.trace_root
        )
        return self.submit_tool(request)

    async def analyze_root_profile(self, commit_id: str, trace_dir: Path, point_config: Path, *, source_id: str) -> str:
        """Start the deterministic TraceLens report of the PRELUDE child's trace as job ``prelude:<source_id>``; return its id.

        For the scheduler, not a specialist: ``trace_dir`` and ``point_config`` (whose ``benchmark.envs`` give the
        splitter its hints) must lie inside the session directory, and ``trace_dir`` must hold a ``*.trace.json*``.
        The job takes no GPU lease and no slot of any branch; its pointer reaches the integration branch's inbox with the
        op shares on ``commit_id``, or with none when the report is degraded or failed. A second call with the same
        ``source_id`` returns the job already started.
        """

        jobs = self._analysis_jobs(TRACELENS_TOOL)
        job_id = f"{ROOT_PROFILE_JOB_PREFIX}{source_id}"
        if jobs.has_job(job_id):
            return job_id
        return self.submit_tool(
            root_profile_request(self.trace_root, commit_id, trace_dir, point_config), job_id=job_id
        )

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
        await self._housekeeping()
        for entry in self._log.entries():
            if entry.unwritten:
                await self._write_outcome(entry)

    async def _housekeeping(self) -> None:
        """What a backend reports on its own once a test has settled and on every tick; nothing here by default."""

    async def _last_output(self, request_id: str) -> tuple[float, str] | None:
        """``(seconds since, last line)`` of the newest log a running test wrote, where a backend can read its live
        files; ``None`` here by default."""

        return None

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

    def _live_profile(self, request: TestRequest) -> str | None:
        """The id of the earliest PROFILE of ``request``'s commit and recipe that the submit log holds as queued,
        running or measured successfully; ``None`` when there is none.

        Read from the log, never from a set of its own, so a restarted toolbox answers as the one before it did. A
        profile that failed, was withdrawn or was aborted covers nothing, and so does one the log knows only as
        settled in the notebook with no outcome of its own. :meth:`submit_test` calls this under the lock that also
        logs the accepted request, so two submits of one profile cannot both pass.
        """

        recipe = request.delta.fingerprint()
        for entry in self._log.entries():
            if entry.aborted or entry.request.get("kind") != TestKind.PROFILE.value:
                continue
            if entry.request.get("parent_commit") != request.parent_commit:
                continue
            if not self._holds_profile(entry) or TestRequest.from_document(entry.request).delta.fingerprint() != recipe:
                continue
            return entry.request_id
        return None

    @staticmethod
    def _holds_profile(entry: LogEntry) -> bool:
        if entry.outcome is None:
            return not entry.settled
        result = entry.outcome.get("result")
        return isinstance(result, Mapping) and result.get("validity") == Validity.SUCCEEDED.value

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

    def _parallel_limit(self) -> int | None:
        """The most tests the toolbox's own cards run at once; ``None`` for a backend that has none to count."""

        return None

    def _pump(self) -> None:
        if self._closed:
            return
        while (item := self._queue.take(self._parallel_limit())) is not None:
            item.started_at = self._clock()
            self._tasks[item.request_id] = asyncio.create_task(self._run(item))

    async def _run(self, item: QueuedTest) -> None:
        try:
            result = await self._execute(item)
            if result.cold_start_seconds is not None:
                self.record_cold_start(result.cold_start_seconds)
            entry = self._log.get(item.request_id)
            if entry is not None:
                await self._record_and_write(entry, {"result": result.to_document()})
            await self._housekeeping()
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

    async def _withdraw(self, entry: LogEntry, reason: str) -> None:
        self._queue.remove(entry.request_id)
        task = self._tasks.pop(entry.request_id, None)
        if task is not None:
            self._executor.withdraw(entry.request_id)  # the task's cancel cannot reach the thread that measures
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._record_and_write(entry, {"withdrawal": reason})
        self._pump()

    async def _record_and_write(self, entry: LogEntry, outcome: dict[str, object]) -> None:
        self._log.record_outcome(entry.request_id, outcome)
        waiter = self._outcome_events.pop(entry.request_id, None)
        if waiter is not None:
            waiter.set()
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

    @property
    def single_node(self) -> bool:
        """Whether every test of this toolbox runs on one node, which a quarantine would only mark ``same_node``.

        A toolbox that does not know its nodes says ``False`` and quarantines as the rule has it.
        """

        return False

    @property
    def node_failures(self) -> dict[str, int]:
        """The failures reported against each node this session, quarantined or not."""

        return dict(self._quarantine.failures)

    async def _node_failed(self, node: str, reason: str, reproduced: bool) -> None:
        failures = self._quarantine.count_failure(node)
        if self.single_node:
            logger.info(
                "node %s not quarantined: single node (%d failures counted; latest: %s)", node, failures, reason
            )
            return
        if reproduced:
            await self._quarantine.add(node, reason)
        else:
            await self._quarantine.add_suspect(node, reason)

    async def _renew_loop(self) -> None:
        while True:
            await asyncio.sleep(self._ttl / 3)
            await self.tick()


__all__ = ["AcceptanceSetup", "RealToolbox", "ToolboxStores", "ToolboxTuning"]
