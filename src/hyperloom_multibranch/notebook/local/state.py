# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notebook's entries folded into indexed state, oldest first; rebuilt whenever an entry is added."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

from hyperloom_multibranch.contracts.notebook.interfaces import UnknownRequest
from hyperloom_multibranch.contracts.notebook.peers import (
    BranchLogEntry,
    Intent,
    Judgement,
    TestRequest,
    TestResult,
)
from hyperloom_multibranch.contracts.notebook.records import (
    AgentManifestRecord,
    BaselineRecord,
    BranchLogRecord,
    BuildNotice,
    Learning,
    LearningRecord,
    LeaseEnd,
    LeaseStart,
    LineageEvent,
    LlmUsage,
    LlmUsageRecord,
    ManifestFingerprint,
    MergeAttempt,
    OpShares,
    QuarantineEvent,
    RegressionCase,
    SessionProfile,
    TestRecord,
    ToolResultPointer,
    Withdrawal,
)
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, DedupStatus, LineageKind, Role
from hyperloom_multibranch.notebook import codec
from hyperloom_multibranch.notebook.local.segments import Entry, EntryKind

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgementWrite:
    """A judgement as stored: the rule gate's or the integrator's, on one request."""

    request_id: str
    judgement: Judgement
    conclusion: Conclusion
    commit_id: str | None = None
    recipe_digest: str | None = None


@dataclass(frozen=True)
class ClaimDecision:
    """What one claim attempt found and whether it got the claim: what Q6 counts wrong refusals from.

    ``ceiling_pct`` is the Amdahl ceiling of an intent that gave its target's share, and ``bar_pct`` the bar it was held
    to, ``None`` before a baseline; a refusal for the ceiling has ``ceiling_pct`` below ``bar_pct``. A refusal while
    another branch tested the intent, or the same change on the same state, names those branches in ``holders`` and
    their tests in flight in ``matched``: it is what tells the branch how those tests ended. The same change refused at
    submit, after the claim was taken, is recorded the same way.
    """

    intent: Intent
    action_fingerprint: str
    found: DedupStatus
    acquired: bool
    reason: str = ""
    matched: tuple[str, ...] = ()
    ceiling_pct: float | None = None
    bar_pct: float | None = None
    holders: tuple[str, ...] = ()


#: The type each kind of entry stores in its detail.
PAYLOADS: dict[EntryKind, type] = {
    EntryKind.REQUEST: TestRequest,
    EntryKind.RESULT: TestResult,
    EntryKind.WITHDRAWAL: Withdrawal,
    EntryKind.JUDGEMENT: JudgementWrite,
    EntryKind.MERGE_ATTEMPT: MergeAttempt,
    EntryKind.LINEAGE: LineageEvent,
    EntryKind.BASELINE: BaselineRecord,
    EntryKind.REGRESSION: RegressionCase,
    EntryKind.QUARANTINE: QuarantineEvent,
    EntryKind.BRANCH_LOG: BranchLogEntry,
    EntryKind.LEARNING: Learning,
    EntryKind.TOOL_RESULT: ToolResultPointer,
    EntryKind.CLAIM: ClaimDecision,
    EntryKind.SESSION_PROFILE: SessionProfile,
    EntryKind.MANIFEST: ManifestFingerprint,
    EntryKind.LLM_USAGE: LlmUsage,
    EntryKind.AGENT_MANIFEST: AgentManifestRecord,
    EntryKind.BUILD: BuildNotice,
    EntryKind.LEASE_START: LeaseStart,
    EntryKind.LEASE_END: LeaseEnd,
}


class NotebookState:
    """Every readable entry, decoded and indexed by what the notebook's queries look things up by."""

    def __init__(self, entries: list[Entry]) -> None:
        self.entries = entries
        self.decoded: list[tuple[Entry, object]] = []
        self.requests: dict[str, TestRequest] = {}
        self.results: dict[str, TestResult] = {}
        self.withdrawals: dict[str, Withdrawal] = {}
        self.judgements: dict[str, list[JudgementWrite]] = {}
        self.commit_writes: dict[str, JudgementWrite] = {}
        self.merge_attempts: list[MergeAttempt] = []
        self.merge_attempted_at: list[float] = []
        self.lineage: list[LineageEvent] = []
        self.baselines: list[BaselineRecord] = []
        self.regressions: list[RegressionCase] = []
        self.quarantine: list[QuarantineEvent] = []
        self.lease_starts: dict[str, LeaseStart] = {}
        self.lease_ends: list[LeaseEnd] = []
        self.toolbox_seen_at: float | None = None
        self.branch_logs: dict[str, list[BranchLogRecord]] = {}
        self.learnings: list[LearningRecord] = []
        self.llm_usage: list[LlmUsageRecord] = []
        self.agent_manifests: dict[str, AgentManifestRecord] = {}
        self.intents: dict[str, Intent] = {}
        self.op_shares: dict[str, OpShares] = {}
        self.profiles: list[SessionProfile] = []
        self.fingerprints: list[ManifestFingerprint] = []
        self.requested_at: dict[str, float] = {}
        self.measured_at: dict[str, float] = {}
        self.withdrawn_at: dict[str, float] = {}
        self.judged_at: dict[str, float] = {}
        self._replay(entries)

    def _replay(self, entries: list[Entry]) -> None:
        """Apply ``entries`` in order; an entry this build cannot decode is skipped with a warning."""

        handlers = self._handlers()
        for entry in entries:
            try:
                value = codec.decode(PAYLOADS[entry.kind], entry.detail)
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("notebook: skipping undecodable %s entry %s: %s", entry.kind.value, entry.entry_id, exc)
                continue
            handlers[entry.kind](entry, value)
            self.decoded.append((entry, value))
            if entry.role is Role.TOOLBOX:
                self.toolbox_seen_at = max(entry.at, self.toolbox_seen_at or entry.at)

    def _handlers(self) -> dict[EntryKind, Callable[[Entry, object], None]]:
        """How each kind of entry changes the state."""

        return {
            EntryKind.REQUEST: self._request,
            EntryKind.RESULT: self._result,
            EntryKind.WITHDRAWAL: self._withdrawal,
            EntryKind.JUDGEMENT: self._judgement,
            EntryKind.SESSION_PROFILE: lambda _entry, value: self.profiles.append(value),
            EntryKind.MANIFEST: lambda _entry, value: self.fingerprints.append(value),
            EntryKind.MERGE_ATTEMPT: self._merge_attempt,
            EntryKind.LINEAGE: lambda _entry, value: self.lineage.append(value),
            EntryKind.BASELINE: lambda _entry, value: self.baselines.append(value),
            EntryKind.REGRESSION: lambda _entry, value: self.regressions.append(value),
            EntryKind.QUARANTINE: lambda _entry, value: self.quarantine.append(value),
            EntryKind.BRANCH_LOG: self._branch_log,
            EntryKind.LEARNING: self._learning,
            EntryKind.TOOL_RESULT: self._tool_result,
            EntryKind.CLAIM: self._claim,
            EntryKind.LLM_USAGE: lambda entry, value: self.llm_usage.append(
                LlmUsageRecord(entry.entry_id, entry.at, value)
            ),
            EntryKind.AGENT_MANIFEST: self._agent_manifest,
            # A build notice indexes nothing: the inbox reads it from ``decoded``.
            EntryKind.BUILD: lambda _entry, _notice: None,
            EntryKind.LEASE_START: lambda _entry, value: self.lease_starts.setdefault(value.lease_id, value),
            EntryKind.LEASE_END: lambda _entry, value: self.lease_ends.append(value),
        }

    def _request(self, entry: Entry, request: TestRequest) -> None:
        self.requests[request.request_id] = request
        self.requested_at.setdefault(request.request_id, entry.at)

    def _merge_attempt(self, entry: Entry, attempt: MergeAttempt) -> None:
        self.merge_attempts.append(attempt)
        self.merge_attempted_at.append(entry.at)

    def _result(self, entry: Entry, result: TestResult) -> None:
        self.results[result.request_id] = result
        self.measured_at.setdefault(result.request_id, entry.at)

    def _withdrawal(self, entry: Entry, withdrawal: Withdrawal) -> None:
        self.withdrawals[withdrawal.request_id] = withdrawal
        self.withdrawn_at.setdefault(withdrawal.request_id, entry.at)

    def _judgement(self, entry: Entry, write: JudgementWrite) -> None:
        self.judgements.setdefault(write.request_id, []).append(write)
        self.judged_at[write.request_id] = entry.at
        if write.commit_id is not None:
            self.commit_writes.setdefault(write.commit_id, write)

    def _branch_log(self, entry: Entry, log_entry: BranchLogEntry) -> None:
        self.branch_logs.setdefault(log_entry.branch, []).append(BranchLogRecord(entry.entry_id, entry.at, log_entry))

    def _learning(self, entry: Entry, learning: Learning) -> None:
        self.learnings.append(LearningRecord(entry.entry_id, entry.at, learning))

    def _agent_manifest(self, entry: Entry, record: AgentManifestRecord) -> None:
        self.agent_manifests[record.branch] = replace(record, at=entry.at)

    def _claim(self, _entry: Entry, decision: ClaimDecision) -> None:
        self.intents.setdefault(decision.intent.fingerprint, decision.intent)

    def _tool_result(self, _entry: Entry, pointer: ToolResultPointer) -> None:
        if pointer.op_shares:
            shares = OpShares(pointer.commit_id, pointer.job_id, pointer.profile_request_id, pointer.op_shares)
            # Re-inserted, so the shares of the commit profiled last come last.
            self.op_shares.pop(pointer.commit_id, None)
            self.op_shares[pointer.commit_id] = shares

    def latest_judgement(self, request_id: str) -> JudgementWrite | None:
        history = self.judgements.get(request_id)
        return history[-1] if history else None

    def record(self, request_id: str) -> TestRecord:
        """The test record of ``request_id``; raises :class:`UnknownRequest` when no request has it."""

        request = self.requests.get(request_id)
        if request is None:
            raise UnknownRequest(f"no test request {request_id}")
        write = self.latest_judgement(request_id)
        withdrawal = self.withdrawals.get(request_id)
        return TestRecord(
            request=request,
            result=self.results.get(request_id),
            judgement=write.judgement if write is not None else None,
            conclusion=write.conclusion if write is not None else None,
            commit_id=write.commit_id if write is not None else None,
            requested_at=self.requested_at[request_id],
            measured_at=self.measured_at.get(request_id),
            judged_at=self.judged_at.get(request_id),
            withdrawn_at=self.withdrawn_at.get(request_id),
            withdrawal_reason=withdrawal.reason if withdrawal is not None else "",
        )

    def records(self) -> list[TestRecord]:
        """Every test record, in the order the requests were written."""

        return [self.record(request_id) for request_id in self.requests]

    def recipe_digests(self) -> dict[str, str]:
        """Each commit's recipe digest where its writer named one: the root's, and that of every commit a judgement
        made, a merge's included."""

        found = {
            event.commit_id: event.recipe_digest
            for event in self.lineage
            if event.kind is LineageKind.ROOT and event.commit_id and event.recipe_digest
        }
        found.update(
            (commit, write.recipe_digest) for commit, write in self.commit_writes.items() if write.recipe_digest
        )
        return found

    def shares_by_commit(self) -> dict[str, OpShares]:
        """Every commit's op shares: the latest a tool reported on it, else the latest reported on a commit of the
        same recipe, which runs the same ops, as an integration commit that took a branch's head as it was does; a
        commit whose recipe no profiled commit has gets none."""

        digests = self.recipe_digests()
        by_recipe = {digests[commit]: shares for commit, shares in self.op_shares.items() if commit in digests}
        shared = {commit: by_recipe[digest] for commit, digest in digests.items() if digest in by_recipe}
        return {**shared, **self.op_shares}

    def shares_of(self, commit_id: str) -> OpShares | None:
        """The op shares of ``commit_id``, as :meth:`shares_by_commit` gives them."""

        own = self.op_shares.get(commit_id)
        return own if own is not None else self.shares_by_commit().get(commit_id)

    def split_branches(self) -> dict[str, str]:
        """The branch each split's fix commit opened."""

        return {event.commit_id: event.branch for event in self.lineage if event.kind is LineageKind.SPLIT}

    def merge_sources(self, request_id: str) -> tuple[str, ...]:
        """The source commits of the merge attempt ``request_id`` tested."""

        attempt = next((item for item in self.merge_attempts if item.request_id == request_id), None)
        return attempt.source_commits if attempt is not None else ()


__all__ = ["PAYLOADS", "ClaimDecision", "JudgementWrite", "NotebookState"]
