# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One participant's handle on a session's notebook directory, implementing every notebook protocol.

The directory holds ``segments/`` (one append-only file per handle) and ``claims/`` (claim generations), and works the
same on a local disk and on a shared NFS folder. A handle writes as one :class:`Writer`, and each write checks that the
writer's role owns the state it writes: test requests, results and withdrawals the toolbox, specialist-branch
judgements the rule gate, integration-branch judgements and merge attempts the integrator, branch starts and stops the
scheduler, claims, branch logs and learnings the specialist runtime of that branch.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from hyperloom.contracts.recipe import DIGEST_PREFIX
from hyperloom_multibranch.contracts.notebook.interfaces import (
    AlreadyTested,
    BranchLogStore,
    CeilingTooLow,
    ClaimHeld,
    ClaimLost,
    ClaimService,
    CommitQuery,
    ConflictingWrite,
    GateRecords,
    Inbox,
    JudgementSink,
    LearningStore,
    LineageSink,
    NodeHealth,
    NotebookError,
    NotRecorded,
    ProgressQuery,
    RecordSearch,
    SessionRecords,
    SnapshotQuery,
    StartupBriefProvider,
    TestRecordSink,
    UsageSink,
    WriterNotAllowed,
)
from hyperloom_multibranch.contracts.notebook.peers import (
    BranchLogEntry,
    BranchLogType,
    Intent,
    Judgement,
    TestRequest,
    TestResult,
    Validity,
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
    Writer,
    amdahl_ceiling_pct,
)
from hyperloom_multibranch.contracts.notebook.vocab import (
    COMMITTING,
    INTEGRATION_BRANCH,
    INTEGRATION_CONCLUSIONS,
    LINEAGE_WRITERS,
    ClaimStatus,
    Conclusion,
    DedupStatus,
    LineageKind,
    Role,
    UsageSource,
)
from hyperloom_multibranch.notebook import codec
from hyperloom_multibranch.notebook.local import feeds, views
from hyperloom_multibranch.notebook.local.feeds import BriefTriggers
from hyperloom_multibranch.notebook.local.claims import LeaseStore, LeaseTaken, effective, is_live
from hyperloom_multibranch.notebook.local.segments import (
    Entry,
    EntryKind,
    SegmentReader,
    SegmentWriter,
    clip_summary,
    make_writer_id,
    require_token,
)
from hyperloom_multibranch.notebook.local.state import ClaimDecision, JudgementWrite, NotebookState

SEGMENTS_DIR = "segments"
CLAIMS_DIR = "claims"
#: The fields each lineage event must name.
_LINEAGE_FIELDS: dict[LineageKind, tuple[str, ...]] = {
    LineageKind.ROOT: ("commit_id",),
    LineageKind.OPEN: ("from_commit",),
    LineageKind.SPLIT: ("commit_id", "from_commit", "from_branch"),
    LineageKind.STOP: (),
}
_RECIPE_DIGEST = re.compile(re.escape(DIGEST_PREFIX) + "[0-9a-f]{64}")
LEARNING_MIN_CHARS = 20
LEARNING_MAX_CHARS = 600
#: Learnings too generic for a later decision to use.
GENERIC_LEARNINGS = frozenset(
    {"improve performance", "improve throughput", "optimize", "optimization", "test", "try", "it works", "no gain"}
)


def _check_learning(learning: Learning, state: NotebookState) -> Learning:
    """``learning`` with its text trimmed; raises ``ValueError`` or :class:`NotRecorded` for what the store refuses."""

    content = " ".join(learning.content.split())
    if len(content) < LEARNING_MIN_CHARS or content.lower().rstrip(".") in GENERIC_LEARNINGS:
        raise ValueError(f"a learning says what was learnt, in at least {LEARNING_MIN_CHARS} characters: {content!r}")
    if len(content) > LEARNING_MAX_CHARS:
        raise ValueError(f"a learning is at most {LEARNING_MAX_CHARS} characters, got {len(content)}")
    unknown = [request_id for request_id in learning.request_ids if request_id not in state.requests]
    if unknown:
        raise NotRecorded(f"no test request {', '.join(unknown)}")
    own = {record.entry_id for record in state.learnings if record.learning.branch == learning.branch}
    if learning.supersedes is not None and learning.supersedes not in own:
        raise NotRecorded(f"branch {learning.branch} has no learning {learning.supersedes}")
    return replace(learning, content=content, target=learning.target.strip(), mechanism=learning.mechanism.strip())


def _refusal(answer: DedupAnswer, reason: str) -> NotebookError | None:
    if answer.status is DedupStatus.IN_PROGRESS:
        return ClaimHeld(answer)
    if answer.status is DedupStatus.TESTED or (answer.status is DedupStatus.SUSPECTED and not reason.strip()):
        return AlreadyTested(answer)
    return None


def _bar_pct(state: NotebookState) -> float | None:
    """The smallest gain the rule gate keeps, from the latest baseline; ``None`` before there is one."""

    return min(state.baselines[-1].threshold_pct.values(), default=None) if state.baselines else None


def _ceiling_pct(intent: Intent) -> float | None:
    return amdahl_ceiling_pct(intent.target_share_pct) if intent.target_share_pct is not None else None


def _ceiling_refusal(state: NotebookState, intent: Intent, bar_pct: float | None) -> CeilingTooLow | None:
    ceiling = _ceiling_pct(intent)
    if ceiling is None or bar_pct is None or ceiling >= bar_pct:
        return None
    return CeilingTooLow(intent, ceiling, bar_pct, state.op_shares.get(intent.parent_commit))


def _released(claim: Claim, now: float) -> Claim:
    if claim.status is not ClaimStatus.HELD:
        raise ClaimLost(f"the claim on {claim.key.parent_commit} is {claim.status.value}")
    return replace(claim, status=ClaimStatus.RELEASED, heartbeat_at=now)


def _check_lineage(event: LineageEvent) -> None:
    missing = [name for name in _LINEAGE_FIELDS[event.kind] if not getattr(event, name)]
    if missing:
        raise ValueError(f"a {event.kind.value} event needs {', '.join(missing)}")
    if event.kind is not LineageKind.STOP and (event.kind is LineageKind.ROOT) != (event.branch == INTEGRATION_BRANCH):
        raise ValueError(f"the root commit, and only it, starts the {INTEGRATION_BRANCH} branch")
    if event.recipe_digest is not None and event.kind is not LineageKind.ROOT:
        raise ValueError("only the root event names its commit's recipe digest; judgements name the others'")


def _new_lineage(state: NotebookState, event: LineageEvent) -> bool:
    """Whether ``event`` adds to the lineage; raises when its branch already started, already stopped or never began."""

    recorded = [item for item in state.lineage if item.branch == event.branch]
    if event in recorded:
        return False
    started = [item for item in recorded if item.kind is not LineageKind.STOP]
    if event.kind is not LineageKind.STOP:
        if started:
            raise ConflictingWrite(f"branch {event.branch} already started with a {started[0].kind.value} event")
    elif not started:
        raise NotRecorded(f"branch {event.branch} never started")
    elif any(item.kind is LineageKind.STOP for item in recorded):
        raise ConflictingWrite(f"branch {event.branch} is already stopped")
    return True


def _check_recipe_digest(state: NotebookState, commit_id: str | None, digest: str | None) -> None:
    """Refuse a malformed digest, one without a commit, or another digest than ``commit_id`` was recorded with."""

    if digest is None:
        return
    if commit_id is None:
        raise ValueError("a recipe digest names the recipe of the commit a write makes")
    if not _RECIPE_DIGEST.fullmatch(digest):
        raise ValueError(f"a recipe digest is {DIGEST_PREFIX} and 64 hex digits, got {digest!r}")
    known = views.lineage_view(state).commit(commit_id)
    if known is not None and known.recipe_digest not in (None, digest):
        raise ConflictingWrite(f"commit {commit_id} was recorded with recipe {known.recipe_digest}")


class LocalNotebook(
    TestRecordSink,
    NodeHealth,
    JudgementSink,
    GateRecords,
    LineageSink,
    ClaimService,
    BranchLogStore,
    LearningStore,
    UsageSink,
    Inbox,
    ProgressQuery,
    CommitQuery,
    StartupBriefProvider,
    RecordSearch,
    SessionRecords,
    SnapshotQuery,
):
    """A handle on the notebook in ``root``, writing as ``writer``; ``clock`` gives Unix seconds.

    It is a :class:`~hyperloom_multibranch.contracts.notebook.interfaces.Notebook`: it implements each interface it lists.
    With ``briefs``, the inbox it polls also holds ``brief`` items (:class:`~feeds.BriefTriggers`); the handles the
    specialists poll through want them, the others need nothing.
    """

    def __init__(
        self,
        root: Path,
        writer: Writer,
        *,
        clock: Callable[[], float] = time.time,
        briefs: BriefTriggers | None = None,
    ) -> None:
        require_token(writer.name, "writer name")
        if writer.branch:
            require_token(writer.branch, "branch")
        elif writer.role is Role.SPECIALIST_RUNTIME:
            raise ValueError("a specialist runtime writes for its branch; give its writer one")
        self.root = Path(root)
        self.writer = writer
        self._clock = clock
        self._briefs = briefs
        self._segment = SegmentWriter(self.root / SEGMENTS_DIR, make_writer_id(writer.role, writer.name))
        self._reader = SegmentReader(self.root / SEGMENTS_DIR)
        self._leases = LeaseStore(self.root / CLAIMS_DIR)
        self._lock = threading.Lock()
        self._folded: NotebookState | None = None

    # --- plumbing -------------------------------------------------------------------------------------------------

    def _require(self, *roles: Role) -> None:
        if self.writer.role not in roles:
            owners = ", ".join(role.value for role in roles)
            raise WriterNotAllowed(f"{self.writer.role.value} may not write this; it belongs to {owners}")

    def _own_branch(self, branch: str) -> None:
        if branch != self.writer.branch:
            raise WriterNotAllowed(f"{self.writer.name} writes for {self.writer.branch or 'no branch'}, not {branch}")

    def _state(self) -> NotebookState:
        entries = self._reader.entries()
        folded = self._folded
        if folded is None or len(folded.entries) != len(entries):
            folded = self._folded = NotebookState(entries)
        return folded

    def _append(self, kind: EntryKind, payload: object, *, branch: str, parent: str, summary: str) -> Entry:
        detail = codec.encode(payload)
        writer = self._segment.writer

        def build(seq: int) -> Entry:
            return Entry(
                entry_id=f"{writer}:{seq}",
                seq=seq,
                at=self._clock(),
                writer=writer,
                role=self.writer.role,
                actor=self.writer.name,
                kind=kind,
                branch=branch,
                parent=parent,
                summary=clip_summary(summary),
                detail=detail,
            )

        with self._lock:
            return self._segment.append(build)

    def _check_holds(self, claim: Claim) -> None:
        """Whether the caller may renew or release ``claim``: its branch's runtime, or the toolbox carrying its test."""

        role = self.writer.role
        ours = (
            role is Role.SCHEDULER
            or (role is Role.SPECIALIST_RUNTIME and claim.holder == self.writer.branch)
            or (role is Role.TOOLBOX and claim.request_id is not None)
        )
        if claim.status is not ClaimStatus.HELD or not ours:
            raise ClaimLost(f"the claim on {claim.key.parent_commit} is not {self.writer.name}'s to change")

    # --- test records ---------------------------------------------------------------------------------------------

    async def write_request(self, request: TestRequest) -> TestRecord:
        self._require(Role.TOOLBOX)
        if request.request_id is None:
            raise ValueError("the notebook stores accepted requests: the toolbox sets request_id when it accepts one")
        require_token(request.request_id, "request_id")
        existing = self._state().requests.get(request.request_id)
        if existing is None:
            summary = f"{request.kind.value} {request.request_id} on {request.branch} at {request.parent_commit}"
            self._append(
                EntryKind.REQUEST, request, branch=request.branch, parent=request.parent_commit, summary=summary
            )
        elif existing != request:
            raise ConflictingWrite(f"request {request.request_id} was written with other content")
        return self._state().record(request.request_id)

    async def write_result(self, result: TestResult) -> TestRecord:
        self._require(Role.TOOLBOX)
        state = self._state()
        request = state.record(result.request_id).request
        if result.request_id in state.withdrawals:
            raise ConflictingWrite(f"request {result.request_id} was withdrawn; it takes no result")
        existing = state.results.get(result.request_id)
        if existing is None:
            failure = f": {result.failure.value} {result.failure_reason or ''}" if result.failure is not None else ""
            summary = f"{result.validity.value} {result.request_id} on {request.branch}{failure}"
            self._append(EntryKind.RESULT, result, branch=request.branch, parent=request.parent_commit, summary=summary)
            failed = result.validity is Validity.MEASUREMENT_FAILED
            self._settle_claim(request, ClaimStatus.RELEASED if failed else ClaimStatus.COMPLETED)
        elif existing != result:
            raise ConflictingWrite(f"the result of {result.request_id} was written with other content")
        return self._state().record(result.request_id)

    async def write_withdrawal(self, withdrawal: Withdrawal) -> TestRecord:
        self._require(Role.TOOLBOX)
        state = self._state()
        request = state.record(withdrawal.request_id).request
        if withdrawal.request_id in state.results:
            raise ConflictingWrite(f"request {withdrawal.request_id} was measured; a measured test is not withdrawn")
        existing = state.withdrawals.get(withdrawal.request_id)
        if existing is None:
            summary = f"withdrawn {withdrawal.request_id} on {request.branch}: {withdrawal.reason}"
            self._append(
                EntryKind.WITHDRAWAL, withdrawal, branch=request.branch, parent=request.parent_commit, summary=summary
            )
            self._settle_claim(request, ClaimStatus.RELEASED)
        elif existing != withdrawal:
            raise ConflictingWrite(f"request {withdrawal.request_id} was withdrawn for another reason")
        return self._state().record(withdrawal.request_id)

    def _settle_claim(self, request: TestRequest, status: ClaimStatus) -> None:
        """End the claim ``request`` measures: completed by its result, released when nothing was measured."""

        if request.intent_fingerprint is None:
            return
        now = self._clock()

        def settle(claim: Claim) -> Claim:
            ours = claim.holder == request.branch and claim.request_id in (None, request.request_id)
            if claim.status is not ClaimStatus.HELD or not ours:
                raise ClaimLost(f"the claim on {claim.key.parent_commit} is not the claim of {request.request_id}")
            return replace(claim, status=status, request_id=request.request_id, heartbeat_at=now)

        with suppress(ClaimLost):
            self._leases.change(ClaimKey(request.parent_commit, request.intent_fingerprint), settle)

    async def notify_tool_result(self, pointer: ToolResultPointer) -> None:
        self._require(Role.TOOLBOX)
        summary = f"tool {pointer.job_id}: {pointer.summary}"
        self._append(EntryKind.TOOL_RESULT, pointer, branch=pointer.branch, parent="", summary=summary)

    # --- node health ----------------------------------------------------------------------------------------------

    async def record_quarantine(self, event: QuarantineEvent) -> None:
        self._require(Role.TOOLBOX)
        gpus = f" GPUs {','.join(map(str, event.gpu_ids))}" if event.gpu_ids else ""
        summary = f"{event.action.value} {event.node}{gpus}: {event.reason}"
        self._append(EntryKind.QUARANTINE, event, branch="", parent="", summary=summary)

    async def quarantined(self) -> tuple[QuarantineEvent, ...]:
        return views.quarantined(self._state())

    # --- judgements and gate records -------------------------------------------------------------------------------

    async def write_judgement(
        self,
        request_id: str,
        judgement: Judgement,
        conclusion: Conclusion,
        commit_id: str | None = None,
        recipe_digest: str | None = None,
    ) -> TestRecord:
        state = self._state()
        request = state.record(request_id).request
        integration = request.branch == INTEGRATION_BRANCH
        self._require(Role.INTEGRATOR if integration else Role.GATE)
        if (conclusion in INTEGRATION_CONCLUSIONS) != integration:
            raise WriterNotAllowed(f"{conclusion.value} is no conclusion for a test on {request.branch}")
        if request_id not in state.results:
            raise NotRecorded(f"request {request_id} has no result to judge")
        if state.results[request_id].validity is Validity.MEASUREMENT_FAILED:
            raise NotRecorded(f"the measurement of {request_id} failed; there is nothing to judge")
        if (conclusion in COMMITTING or conclusion is Conclusion.MERGED) != (commit_id is not None):
            raise ValueError(f"a {conclusion.value} judgement names a commit exactly when it makes one")
        _check_recipe_digest(state, commit_id, recipe_digest)
        summary = f"{conclusion.value}: {judgement.verdict.value} {judgement.gain_pct:+.2f}% on {request_id}"
        write = JudgementWrite(request_id, judgement, conclusion, commit_id, recipe_digest)
        self._append(EntryKind.JUDGEMENT, write, branch=request.branch, parent=request.parent_commit, summary=summary)
        return self._state().record(request_id)

    async def record_merge_attempt(self, attempt: MergeAttempt) -> None:
        self._require(Role.INTEGRATOR)
        if attempt.request_id is not None:
            if self._state().record(attempt.request_id).request.branch != INTEGRATION_BRANCH:
                raise ValueError(f"a merge is tested on the {INTEGRATION_BRANCH} branch; {attempt.request_id} is not")
        elif not attempt.skipped_reason:
            raise ValueError("a merge attempt without a test says why it was skipped")
        tested = f"tested by {attempt.request_id}" if attempt.request_id else f"skipped: {attempt.skipped_reason}"
        summary = f"merge {', '.join(attempt.source_commits)} onto {attempt.integration_head}, {tested}"
        self._append(
            EntryKind.MERGE_ATTEMPT,
            attempt,
            branch=INTEGRATION_BRANCH,
            parent=attempt.integration_head,
            summary=summary,
        )

    async def write_baseline(self, record: BaselineRecord) -> None:
        self._require(Role.GATE)
        values = ", ".join(f"{metric} {value:g}" for metric, value in sorted(record.values.items()))
        summary = f"baseline of {record.root_commit}: {values}"
        self._append(EntryKind.BASELINE, record, branch=INTEGRATION_BRANCH, parent=record.root_commit, summary=summary)

    async def add_regression_case(self, case: RegressionCase) -> None:
        self._require(Role.GATE)
        self._state().record(case.request_id)
        summary = f"regression case {case.case_ref} of fix {case.fix_commit}"
        self._append(EntryKind.REGRESSION, case, branch="", parent=case.fix_commit, summary=summary)

    async def record_lineage(self, event: LineageEvent) -> None:
        self._require(LINEAGE_WRITERS[event.kind])
        _check_lineage(event)
        state = self._state()
        _check_recipe_digest(state, event.commit_id, event.recipe_digest)
        if not _new_lineage(state, event):
            return
        start = event.commit_id or event.from_commit
        summary = f"{event.kind.value} {event.branch} at {start or 'its head'}: {event.reason}"
        self._append(EntryKind.LINEAGE, event, branch=event.branch, parent=event.from_commit, summary=summary)

    # --- claims ---------------------------------------------------------------------------------------------------

    async def lookup(self, key: ClaimKey, action_fingerprint: str) -> DedupAnswer:
        return self._answer(self._state(), key, action_fingerprint, self._clock())

    def _answer(self, state: NotebookState, key: ClaimKey, action_fingerprint: str, now: float) -> DedupAnswer:
        measured = [
            record for record in state.records() if record.measured and record.request.intent_fingerprint is not None
        ]
        here = [record for record in measured if record.request.parent_commit == key.parent_commit]
        exact = tuple(record for record in here if record.request.action_fingerprint == action_fingerprint)
        same_intent = tuple(record for record in here if record.request.intent_fingerprint == key.intent_fingerprint)
        elsewhere = tuple(
            record
            for record in measured
            if record.request.intent_fingerprint == key.intent_fingerprint
            and record.request.parent_commit != key.parent_commit
        )
        live = sorted(
            (
                claim
                for claim in self._leases.all_current()
                if is_live(claim, now)
                and claim.key.parent_commit == key.parent_commit
                and (claim.key == key or claim.action_fingerprint == action_fingerprint)
            ),
            key=lambda claim: claim.key != key,
        )
        if exact:
            return DedupAnswer(key, action_fingerprint, DedupStatus.TESTED, records=exact, elsewhere=elsewhere)
        if live:
            return DedupAnswer(key, action_fingerprint, DedupStatus.IN_PROGRESS, claim=live[0], elsewhere=elsewhere)
        status = DedupStatus.SUSPECTED if same_intent else DedupStatus.FREE
        return DedupAnswer(key, action_fingerprint, status, records=same_intent, elsewhere=elsewhere)

    async def acquire(self, intent: Intent, action_fingerprint: str, *, ttl_seconds: float, reason: str = "") -> Claim:
        self._require(Role.SPECIALIST_RUNTIME)
        self._own_branch(intent.branch)
        now = self._clock()
        key = ClaimKey(intent.parent_commit, intent.fingerprint)
        state = self._state()
        answer = self._answer(state, key, action_fingerprint, now)
        bar = _bar_pct(state)
        too_low = _ceiling_refusal(state, intent, bar)
        refusal = _refusal(answer, reason) or too_low
        claim: Claim | None = None
        if refusal is None:
            wanted = Claim(
                key,
                intent.branch,
                intent,
                action_fingerprint,
                ttl_seconds,
                now,
                now,
                now + ttl_seconds,
                ClaimStatus.HELD,
            )
            try:
                claim = self._leases.acquire(wanted, now)
            except LeaseTaken as taken:
                answer = replace(answer, status=DedupStatus.IN_PROGRESS, claim=taken.claim)
                refusal = ClaimHeld(answer)
        matched = tuple(record.request_id for record in answer.records)
        decision = ClaimDecision(
            intent, action_fingerprint, answer.status, claim is not None, reason, matched, _ceiling_pct(intent), bar
        )
        found = answer.status.value
        if too_low is not None and refusal is too_low:
            found = f"ceiling {too_low.ceiling_pct:.2f}% below the bar of {too_low.bar_pct:.2f}%"
        outcome = "claimed" if claim is not None else f"refused ({found})"
        summary = f"{outcome} {intent.target} / {intent.mechanism} on {intent.parent_commit}"
        self._append(EntryKind.CLAIM, decision, branch=intent.branch, parent=intent.parent_commit, summary=summary)
        if claim is None:
            raise refusal or ClaimHeld(answer)
        return claim

    async def renew(self, key: ClaimKey, *, ttl_seconds: float) -> Claim:
        self._require(Role.SPECIALIST_RUNTIME, Role.TOOLBOX)
        now = self._clock()

        def renewed(claim: Claim) -> Claim:
            self._check_holds(claim)
            return replace(claim, heartbeat_at=now, expires_at=now + ttl_seconds)

        return effective(self._leases.change(key, renewed), now)

    async def attach_request(self, key: ClaimKey, request_id: str, *, ttl_seconds: float) -> Claim:
        self._require(Role.TOOLBOX)
        request = self._state().record(request_id).request
        if request.intent_fingerprint is None:
            raise ConflictingWrite(f"request {request_id} names no intent, so it carries no claim")
        if ClaimKey(request.parent_commit, request.intent_fingerprint) != key:
            raise ConflictingWrite(f"request {request_id} measures another claim")
        now = self._clock()

        def attached(claim: Claim) -> Claim:
            ours = claim.holder == request.branch and claim.request_id in (None, request_id)
            if claim.status is not ClaimStatus.HELD or not ours:
                raise ClaimLost(f"the claim on {key.parent_commit} is not the one {request_id} measures")
            return replace(claim, request_id=request_id, heartbeat_at=now, expires_at=now + ttl_seconds)

        return effective(self._leases.change(key, attached), now)

    async def release(self, key: ClaimKey) -> Claim:
        self._require(Role.SPECIALIST_RUNTIME, Role.TOOLBOX, Role.SCHEDULER)
        now = self._clock()

        def released(claim: Claim) -> Claim:
            self._check_holds(claim)
            return _released(claim, now)

        return self._leases.change(key, released)

    async def release_branch(self, branch: str) -> tuple[Claim, ...]:
        self._require(Role.SCHEDULER)
        now = self._clock()
        released = []
        for claim in self._leases.all_current():
            if claim.holder == branch and claim.status is ClaimStatus.HELD:
                with suppress(ClaimLost):
                    released.append(self._leases.change(claim.key, lambda current: _released(current, now)))
        return tuple(released)

    # --- branch logs and inbox ------------------------------------------------------------------------------------

    async def append_branch_log(self, entry: BranchLogEntry) -> BranchLogRecord:
        self._require(Role.SPECIALIST_RUNTIME)
        self._own_branch(entry.branch)
        known = {record.entry_id for record in self._state().branch_logs.get(entry.branch, [])}
        if entry.supersedes is not None and entry.supersedes not in known:
            raise NotRecorded(f"branch {entry.branch} has no log entry {entry.supersedes}")
        summary = f"{entry.kind.value}: {entry.content}"
        written = self._append(EntryKind.BRANCH_LOG, entry, branch=entry.branch, parent="", summary=summary)
        return BranchLogRecord(written.entry_id, written.at, entry)

    async def branch_log(
        self, branch: str, *, kinds: frozenset[BranchLogType] = frozenset(), after: str = ""
    ) -> tuple[BranchLogRecord, ...]:
        records = self._state().branch_logs.get(branch, [])
        ids = [record.entry_id for record in records]
        if after and after not in ids:
            raise NotRecorded(f"branch {branch} has no log entry {after}")
        start = ids.index(after) + 1 if after else 0
        return tuple(record for record in records[start:] if not kinds or record.entry.kind in kinds)

    async def current_plan(self, branch: str) -> BranchLogRecord | None:
        records = self._state().branch_logs.get(branch, [])
        superseded = {record.entry.supersedes for record in records if record.entry.supersedes}
        plans = [r for r in records if r.entry.kind is BranchLogType.PLAN and r.entry_id not in superseded]
        return plans[-1] if plans else None

    async def append_learning(self, learning: Learning) -> LearningRecord:
        self._require(Role.SPECIALIST_RUNTIME)
        self._own_branch(learning.branch)
        checked = _check_learning(learning, self._state())
        summary = f"learning/{checked.kind.value}: {checked.content}"
        written = self._append(EntryKind.LEARNING, checked, branch=checked.branch, parent="", summary=summary)
        return LearningRecord(written.entry_id, written.at, checked)

    async def learnings(self, *, branch: str = "") -> tuple[LearningRecord, ...]:
        return tuple(record for record in self._state().learnings if not branch or record.learning.branch == branch)

    async def poll(self, branch: str, cursor: str = "") -> InboxPage:
        if self._briefs is None:
            return feeds.poll(self._state(), branch, cursor)
        now = self._clock()
        held = [claim for claim in self._leases.all_current() if claim.holder == branch and is_live(claim, now)]
        return feeds.poll(self._state(), branch, cursor, self._briefs, held)

    async def record_llm_usage(self, usage: LlmUsage) -> LlmUsageRecord:
        self._require(Role.SPECIALIST_RUNTIME, Role.SCHEDULER)
        if self.writer.role is Role.SPECIALIST_RUNTIME:
            self._own_branch(usage.branch)
            if usage.source is not UsageSource.SPECIALIST:
                raise WriterNotAllowed(f"a specialist runtime records its specialist's spend, not the {usage.source}'s")
        elif usage.branch or usage.source is UsageSource.SPECIALIST:
            raise WriterNotAllowed("the scheduler records its own judge's and resolver's spend, with no branch")
        summary = (
            f"llm usage/{usage.source.value}: ${usage.cost_usd:.4f}, {usage.input_tokens}+{usage.output_tokens} tokens"
        )
        written = self._append(EntryKind.LLM_USAGE, usage, branch=usage.branch, parent="", summary=summary)
        return LlmUsageRecord(written.entry_id, written.at, usage)

    # --- queries --------------------------------------------------------------------------------------------------

    async def progress(self) -> tuple[BranchProgress, ...]:
        state = self._state()
        return views.progress(state, views.lineage_view(state))

    async def lineage(self) -> LineageView:
        return views.lineage_view(self._state())

    async def committed(self) -> tuple[CommittedCommit, ...]:
        state = self._state()
        return views.committed(state, views.lineage_view(state))

    async def recorded_value(self, commit_id: str) -> RecordedValue:
        state = self._state()
        return views.recorded_value(state, views.lineage_view(state), commit_id)

    async def baseline(self) -> BaselineRecord:
        baselines = self._state().baselines
        if not baselines:
            raise NotRecorded("the rule gate has not written a baseline yet")
        return baselines[-1]

    async def regression_cases_for(self, commit_id: str) -> tuple[RegressionCase, ...]:
        state = self._state()
        return views.regression_cases_for(state, views.lineage_view(state), commit_id)

    async def startup_brief(self, start_commit: str) -> StartupBrief:
        state = self._state()
        return views.startup_brief(state, views.lineage_view(state), start_commit)

    async def search(self, query: str, *, limit: int = 20) -> tuple[SearchHit, ...]:
        return feeds.search(self._state(), query, limit)

    async def record(self, request_id: str) -> TestRecord:
        return self._state().record(request_id)

    async def profiles(self, commit_id: str) -> tuple[TestRecord, ...]:
        return feeds.profiles(self._state(), commit_id)

    async def op_shares(self, commit_id: str) -> OpShares | None:
        return self._state().op_shares.get(commit_id)

    # --- session records and snapshots ----------------------------------------------------------------------------

    async def write_session_profile(self, profile: SessionProfile) -> None:
        self._require(Role.SCHEDULER)
        require_token(profile.session_id, "session_id")
        summary = f"session {profile.session_id}: {profile.model} on {profile.hardware}, TP {profile.tp}"
        self._append(EntryKind.SESSION_PROFILE, profile, branch="", parent="", summary=summary)

    async def session_profile(self) -> SessionProfile:
        profiles = self._state().profiles
        if not profiles:
            raise NotRecorded("PRELUDE has not written the session profile yet")
        return profiles[-1]

    async def write_manifest_fingerprint(self, fingerprint: ManifestFingerprint) -> None:
        self._require(Role.SCHEDULER)
        parts = (len(fingerprint.sockets), len(fingerprint.prompt_versions), len(fingerprint.policy_versions))
        summary = f"manifest of {fingerprint.activity}: {parts[0]} sockets, {parts[1]} prompts, {parts[2]} policies"
        self._append(EntryKind.MANIFEST, fingerprint, branch="", parent="", summary=summary)

    async def manifest_fingerprints(self) -> tuple[ManifestFingerprint, ...]:
        return tuple(self._state().fingerprints)

    async def snapshot(self) -> NotebookSnapshot:
        now = self._clock()
        claims = tuple(effective(claim, now) for claim in self._leases.all_current())
        return views.snapshot(self._state(), claims, now)


__all__ = [
    "CLAIMS_DIR",
    "GENERIC_LEARNINGS",
    "LEARNING_MAX_CHARS",
    "LEARNING_MIN_CHARS",
    "SEGMENTS_DIR",
    "LocalNotebook",
]
