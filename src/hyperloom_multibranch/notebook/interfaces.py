# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notebook as the other multi-branch modules call it: one protocol per interface of design section 5.1.

The modules are written in parallel against these protocols. The local notebook implements all of them and is the
fake the end-to-end skeleton runs on; the notebook server's client will be the second implementation. Every method is
a coroutine, so callers keep their code when the notebook moves behind a server.

A write checks the role of the caller's :class:`~hyperloom_multibranch.notebook.records.Writer` and raises
:class:`WriterNotAllowed` when that role does not own the state; reads are open to every role.
"""

from __future__ import annotations

from typing import Protocol

from hyperloom_multibranch.notebook.peers import (
    BranchLogEntry,
    BranchLogType,
    Intent,
    Judgement,
    TestRequest,
    TestResult,
)
from hyperloom_multibranch.notebook.records import (
    BaselineRecord,
    BranchLogRecord,
    BranchProgress,
    Claim,
    ClaimKey,
    CommittedCommit,
    DedupAnswer,
    InboxPage,
    LineageEvent,
    LineageView,
    MergeAttempt,
    QuarantineEvent,
    RecordedValue,
    RegressionCase,
    SearchHit,
    StartupBrief,
    TestRecord,
    ToolResultPointer,
)
from hyperloom_multibranch.notebook.vocab import Conclusion


class NotebookError(Exception):
    """Base of every error the notebook raises."""


class WriterNotAllowed(NotebookError, PermissionError):
    """The caller's role, or its branch, does not own the state it tried to write."""


class _DedupRefusal(NotebookError):
    def __init__(self, answer: DedupAnswer) -> None:
        super().__init__(f"{answer.status.value}: {answer.key.intent_fingerprint} on {answer.key.parent_commit}")
        self.answer = answer


class ClaimHeld(_DedupRefusal):
    """Someone holds the key, or the same action is in progress on the state; ``answer`` says who."""


class AlreadyTested(_DedupRefusal):
    """The action was measured on the state, or the intent was and no reason to repeat it was given."""


class ClaimLost(NotebookError):
    """The claim is no longer the caller's to renew, attach to or release."""


class UnknownRequest(NotebookError, LookupError):
    """No test request has this ``request_id``."""


class ConflictingWrite(NotebookError):
    """A request or result already written under this ``request_id`` says something else."""


class NotRecorded(NotebookError, LookupError):
    """The notebook holds no such commit, baseline or record."""


class TestRecordSink(Protocol):
    """The toolbox: one row per test, written when it accepts the test and when the final result is in."""

    async def write_request(self, request: TestRequest) -> TestRecord: ...

    async def write_result(self, result: TestResult) -> TestRecord:
        """Store the final outcome; the claim the request measures is completed by it."""

    async def notify_tool_result(self, pointer: ToolResultPointer) -> None:
        """Push a pointer to an asynchronous tool's result to the branch that asked for it."""


class NodeHealth(Protocol):
    """The toolbox's quarantine list, stored in the notebook."""

    async def record_quarantine(self, event: QuarantineEvent) -> None: ...

    async def quarantined(self) -> tuple[QuarantineEvent, ...]:
        """Every node and GPU set taken out of service and not returned since."""


class JudgementSink(Protocol):
    """The rule gate for specialist branches and the integrator for the integration branch."""

    async def write_judgement(
        self, request_id: str, judgement: Judgement, conclusion: Conclusion, commit_id: str | None = None
    ) -> TestRecord:
        """Append a judgement; the latest one of a request is in force, earlier ones stay in the history."""

    async def record_merge_attempt(self, attempt: MergeAttempt) -> None: ...


class GateRecords(Protocol):
    """What the rule gate stores besides judgements."""

    async def write_baseline(self, record: BaselineRecord) -> None: ...

    async def add_regression_case(self, case: RegressionCase) -> None: ...


class LineageSink(Protocol):
    """Branch starts and stops (the scheduler) and splits (the rule gate)."""

    async def record_lineage(self, event: LineageEvent) -> None: ...


class ClaimService(Protocol):
    """Same-state deduplication: the specialist runtime claims, the toolbox carries the claim through its test."""

    async def lookup(self, key: ClaimKey, action_fingerprint: str) -> DedupAnswer: ...

    async def acquire(self, intent: Intent, action_fingerprint: str, *, ttl_seconds: float, reason: str = "") -> Claim:
        """Claim ``intent`` for the caller's branch; ``reason`` overrides a suspected duplicate.

        Raises :class:`ClaimHeld` or :class:`AlreadyTested` with the answer that refused it.
        """

    async def renew(self, key: ClaimKey, *, ttl_seconds: float) -> Claim: ...

    async def attach_request(self, key: ClaimKey, request_id: str, *, ttl_seconds: float) -> Claim:
        """Hand the claim to the toolbox's test; the toolbox renews it from then on."""

    async def release(self, key: ClaimKey) -> Claim: ...

    async def release_branch(self, branch: str) -> tuple[Claim, ...]:
        """Release every claim a stopped branch holds."""


class BranchLogStore(Protocol):
    """The specialist runtime's long-term memory; only a branch's own runtime appends to it."""

    async def append_branch_log(self, entry: BranchLogEntry) -> BranchLogRecord: ...

    async def branch_log(
        self, branch: str, *, kinds: frozenset[BranchLogType] = frozenset(), after: str = ""
    ) -> tuple[BranchLogRecord, ...]:
        """The branch's entries, oldest first, of ``kinds`` (all when empty), after the entry ``after``."""

    async def current_plan(self, branch: str) -> BranchLogRecord | None: ...


class Inbox(Protocol):
    """Pointers pushed to one branch; the tool layer turns polling into pushes."""

    async def poll(self, branch: str, cursor: str = "") -> InboxPage: ...


class ProgressQuery(Protocol):
    """What the scheduler and the dashboard read."""

    async def progress(self) -> tuple[BranchProgress, ...]: ...

    async def lineage(self) -> LineageView: ...


class CommitQuery(Protocol):
    """What the integrator and the rule gate read about commits."""

    async def committed(self) -> tuple[CommittedCommit, ...]:
        """Specialist commits, foundation fixes first, then by their own gain."""

    async def recorded_value(self, commit_id: str) -> RecordedValue: ...

    async def baseline(self) -> BaselineRecord: ...

    async def regression_cases_for(self, commit_id: str) -> tuple[RegressionCase, ...]:
        """The reproduction cases of every fix ``commit_id`` builds on."""


class StartupBriefProvider(Protocol):
    async def startup_brief(self, start_commit: str) -> StartupBrief: ...


class RecordSearch(Protocol):
    """Record lookups a specialist makes through the tool layer."""

    async def search(self, query: str, *, limit: int = 20) -> tuple[SearchHit, ...]:
        """Entries matching ``query``: ``,`` separates alternatives, ``&`` joins terms that must all appear."""

    async def record(self, request_id: str) -> TestRecord: ...

    async def profiles(self, commit_id: str) -> tuple[TestRecord, ...]:
        """The profile tests of ``commit_id`` that produced an artifact."""


class Notebook(
    TestRecordSink,
    NodeHealth,
    JudgementSink,
    GateRecords,
    LineageSink,
    ClaimService,
    BranchLogStore,
    Inbox,
    ProgressQuery,
    CommitQuery,
    StartupBriefProvider,
    RecordSearch,
    Protocol,
):
    """Every interface at once: what one notebook handle offers."""


__all__ = [
    "AlreadyTested",
    "BranchLogStore",
    "ClaimHeld",
    "ClaimLost",
    "ClaimService",
    "CommitQuery",
    "ConflictingWrite",
    "GateRecords",
    "Inbox",
    "JudgementSink",
    "LineageSink",
    "NodeHealth",
    "NotRecorded",
    "Notebook",
    "NotebookError",
    "ProgressQuery",
    "RecordSearch",
    "StartupBriefProvider",
    "TestRecordSink",
    "UnknownRequest",
    "WriterNotAllowed",
]
