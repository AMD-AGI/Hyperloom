# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notebook as the other multi-branch modules call it: one protocol per interface of design section 5.1.

The modules are written in parallel against these protocols. The local notebook implements all of them and is the
fake the end-to-end skeleton runs on; the notebook server's client will be the second implementation. Every method is
a coroutine, so callers keep their code when the notebook moves behind a server.

A write checks the role of the caller's :class:`~hyperloom_multibranch.contracts.notebook.records.Writer` and raises
:class:`WriterNotAllowed` when that role does not own the state; reads are open to every role.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from hyperloom_multibranch.contracts.notebook.peers import (
    BranchLogEntry,
    BranchLogType,
    Intent,
    Judgement,
    TestRequest,
    TestResult,
)
from hyperloom_multibranch.contracts.notebook.records import (
    BaselineRecord,
    BranchLogRecord,
    BranchProgress,
    Claim,
    ClaimKey,
    CommittedCommit,
    DedupAnswer,
    InboxPage,
    Learning,
    LearningRecord,
    LineageEvent,
    LineageView,
    LlmUsage,
    LlmUsageRecord,
    ManifestFingerprint,
    MergeAttempt,
    NotebookSnapshot,
    OpShares,
    QuarantineEvent,
    RecordedValue,
    RegressionCase,
    SearchHit,
    SessionProfile,
    StartupBrief,
    TestRecord,
    ToolResultPointer,
    Withdrawal,
)
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion


class NotebookError(Exception):
    """Base of every error the notebook raises."""


class WriterNotAllowed(NotebookError, PermissionError):
    """The caller's role, or its branch, does not own the state it tried to write."""


class _DedupRefusal(NotebookError):
    def __init__(self, answer: DedupAnswer) -> None:
        super().__init__(f"{answer.status.value}: {answer.key.intent_fingerprint} on {answer.key.parent_commit}")
        self.answer = answer


class ClaimHeld(_DedupRefusal):
    """Someone holds the key, or the same action is in progress on the state, or another branch is testing the intent
    on its own state; ``answer`` says who."""


class AlreadyTested(_DedupRefusal):
    """The action was measured on the state, or the intent was and no reason to repeat it was given."""


class CeilingTooLow(NotebookError):
    """The intent's Amdahl ceiling is below the rule gate's bar: no speedup of its target alone could be kept.

    ``profiled`` is the op shares the notebook holds on the intent's parent commit, if any, to pick a larger target.
    """

    def __init__(self, intent: Intent, ceiling_pct: float, bar_pct: float, profiled: OpShares | None) -> None:
        super().__init__(
            f"a target worth {intent.target_share_pct:.2f}% of GPU time on {intent.parent_commit} can gain at most "
            f"{ceiling_pct:.2f}%, below the bar of {bar_pct:.2f}%"
        )
        self.intent = intent
        self.ceiling_pct = ceiling_pct
        self.bar_pct = bar_pct
        self.profiled = profiled


class ClaimLost(NotebookError):
    """The claim is no longer the caller's to renew, attach to or release."""


class UnknownRequest(NotebookError, LookupError):
    """No test request has this ``request_id``."""


class ConflictingWrite(NotebookError):
    """What the notebook already holds says something else: the request, result or withdrawal of this ``request_id``,
    the recipe digest of this commit, or the start or stop of this branch."""


class NotRecorded(NotebookError, LookupError):
    """The notebook holds no such commit, baseline or record."""


class TestRecordSink(Protocol):
    """The toolbox: one row per test, written when it accepts the test and when the final result is in, or when it
    takes the test back unmeasured."""

    async def write_request(self, request: TestRequest) -> TestRecord: ...

    async def write_result(self, result: TestResult) -> TestRecord:
        """Store the final outcome; the claim the request measures is completed by it. A withdrawn test takes none."""

    async def write_withdrawal(self, withdrawal: Withdrawal) -> TestRecord:
        """Settle an unmeasured test and release the claim it carried; a measured test is not withdrawn."""

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
        self,
        request_id: str,
        judgement: Judgement,
        conclusion: Conclusion,
        commit_id: str | None = None,
        recipe_digest: str | None = None,
    ) -> TestRecord:
        """Append a judgement; the latest one of a request is in force, earlier ones stay in the history.

        A judgement that makes ``commit_id`` may name ``recipe_digest``, the ``Recipe.digest()`` of the recipe the
        experiment repository stores for it; a commit keeps the digest it was first recorded with.
        """

    async def record_merge_attempt(self, attempt: MergeAttempt) -> None: ...


class GateRecords(Protocol):
    """What the rule gate stores besides judgements."""

    async def write_baseline(self, record: BaselineRecord) -> None: ...

    async def add_regression_case(self, case: RegressionCase) -> None: ...


class LineageSink(Protocol):
    """Branch starts and stops (the scheduler) and splits (the rule gate)."""

    async def record_lineage(self, event: LineageEvent) -> None:
        """Start a branch once (``ROOT``, ``OPEN`` or ``SPLIT``) and stop it once; the same event again is a no-op.

        A split starts its branch, so the branch takes no ``OPEN`` after it. Another start of a started branch, another
        stop of a stopped one or a second root raises :class:`ConflictingWrite`; stopping a branch that never started
        raises :class:`NotRecorded`.
        """


class ClaimService(Protocol):
    """Same-state deduplication: the specialist runtime claims, the toolbox carries the claim through its test."""

    async def lookup(self, key: ClaimKey, action_fingerprint: str) -> DedupAnswer: ...

    async def acquire(self, intent: Intent, action_fingerprint: str, *, ttl_seconds: float, reason: str = "") -> Claim:
        """Claim ``intent`` for the caller's branch; ``reason`` overrides a suspected duplicate.

        Raises :class:`ClaimHeld` or :class:`AlreadyTested` with the answer that refused it, and :class:`CeilingTooLow`
        when the intent gives its target's share and the ceiling it puts on the gain is below the smallest
        ``threshold_pct`` of the baseline. Before a baseline there is no bar, and no intent is refused for its ceiling.
        """

    async def renew(self, key: ClaimKey, *, ttl_seconds: float) -> Claim: ...

    async def attach_request(self, key: ClaimKey, request_id: str, *, ttl_seconds: float) -> Claim:
        """Hand the claim to the toolbox's test; the toolbox renews it from then on.

        ``ClaimLost`` refuses an explore whose change another branch asked first to explore on the same commit, while
        that test is in flight or once it is measured; the toolbox then withdraws the request, which releases the claim.
        """

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


class LearningStore(Protocol):
    """What the specialists learnt: a branch's own runtime appends its learnings, and every branch reads them all.

    An append raises ``ValueError`` for a learning too short, too long or too generic to help a later decision, and
    :class:`NotRecorded` when it cites a test or supersedes a learning the notebook does not hold.
    """

    async def append_learning(self, learning: Learning) -> LearningRecord: ...

    async def learnings(self, *, branch: str = "") -> tuple[LearningRecord, ...]:
        """The learnings of ``branch``, every branch's when empty, oldest first, the superseded ones included."""


class AgentRecords(Protocol):
    """What an autonomous branch's runtime records of the third-party agent that drives it."""

    async def record_agent_manifest(self, branch: str, document: Mapping[str, object]) -> None:
        """Keep the manifest ``document`` of the agent driving ``branch``, once its preflight passed; only the branch's
        own runtime records it, and the notebook drops every credential from it first."""


class UsageSink(Protocol):
    """What the LLMs cost: a branch's specialist runtime records each launch's spend, the scheduler its own."""

    async def record_llm_usage(self, usage: LlmUsage) -> LlmUsageRecord:
        """Keep one launch's or call's cost; a specialist runtime records only its own branch's, the scheduler only its
        own, with no branch."""


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


class SessionRecords(Protocol):
    """What the session is and what it ran with; the scheduler writes both, the profile once at PRELUDE."""

    async def write_session_profile(self, profile: SessionProfile) -> None: ...

    async def session_profile(self) -> SessionProfile:
        """The latest profile; raises :class:`NotRecorded` before PRELUDE wrote one."""

    async def write_manifest_fingerprint(self, fingerprint: ManifestFingerprint) -> None: ...

    async def manifest_fingerprints(self) -> tuple[ManifestFingerprint, ...]: ...


class SnapshotQuery(Protocol):
    """Everything at once, for the notebook owner's own consumers: the RSI, the dashboard and the exports."""

    async def snapshot(self) -> NotebookSnapshot: ...


class RecordSearch(Protocol):
    """Record lookups a specialist makes through the tool layer."""

    async def search(self, query: str, *, limit: int = 20) -> tuple[SearchHit, ...]:
        """Entries matching ``query``: ``,`` separates alternatives, ``&`` joins terms that must all appear."""

    async def record(self, request_id: str) -> TestRecord: ...

    async def profiles(self, commit_id: str) -> tuple[TestRecord, ...]:
        """The profile tests of ``commit_id`` that produced an artifact."""

    async def op_shares(self, commit_id: str) -> OpShares | None:
        """The latest TraceLens op shares a tool reported on ``commit_id``; ``None`` when none was analysed there."""


class Notebook(
    TestRecordSink,
    NodeHealth,
    JudgementSink,
    GateRecords,
    LineageSink,
    ClaimService,
    BranchLogStore,
    LearningStore,
    AgentRecords,
    UsageSink,
    Inbox,
    ProgressQuery,
    CommitQuery,
    StartupBriefProvider,
    RecordSearch,
    SessionRecords,
    SnapshotQuery,
    Protocol,
):
    """Every interface at once: what one notebook handle offers."""


__all__ = [
    "AgentRecords",
    "AlreadyTested",
    "BranchLogStore",
    "CeilingTooLow",
    "ClaimHeld",
    "ClaimLost",
    "ClaimService",
    "CommitQuery",
    "ConflictingWrite",
    "GateRecords",
    "Inbox",
    "JudgementSink",
    "LearningStore",
    "LineageSink",
    "NodeHealth",
    "NotRecorded",
    "Notebook",
    "NotebookError",
    "ProgressQuery",
    "RecordSearch",
    "SessionRecords",
    "SnapshotQuery",
    "StartupBriefProvider",
    "TestRecordSink",
    "UnknownRequest",
    "UsageSink",
    "WriterNotAllowed",
]
