# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the notebook's queries derive from its state: the branch tree, progress, commits, values and briefs.

Each fact is written once and every view is derived from it: a specialist commit comes from the judgement that
carries its commit id, a merge from the integrator's merge judgement and its merge attempt, a branch's start, split
and stop from lineage events.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook.interfaces import NotRecorded
from hyperloom_multibranch.contracts.notebook.peers import TestKind
from hyperloom_multibranch.contracts.notebook.records import (
    BranchNode,
    BranchProgress,
    Claim,
    CommitNode,
    CommittedCommit,
    LineageEvent,
    LineageView,
    MergeAttempt,
    MergeState,
    NotebookSnapshot,
    QuarantineEvent,
    RecordedValue,
    RegressionCase,
    StartupBrief,
    TestRecord,
    TriedDirection,
    priced_usd,
)
from hyperloom_multibranch.contracts.notebook.vocab import (
    INTEGRATION_BRANCH,
    CommitOrigin,
    Conclusion,
    LineageKind,
    MergeOutcome,
    QuarantineAction,
)
from hyperloom_multibranch.notebook.changes import describe_change
from hyperloom_multibranch.notebook.local.state import JudgementWrite, NotebookState

_SPECIALIST_ORIGINS = frozenset({CommitOrigin.COMMIT, CommitOrigin.SPLIT})


@dataclass
class _BranchDraft:
    branch: str
    origin: LineageKind
    start_commit: str
    opened_at: float
    from_branch: str = ""
    head: str = ""
    stop_reason: str | None = None
    stopped_at: float | None = None

    def node(self) -> BranchNode:
        return BranchNode(
            branch=self.branch,
            origin=self.origin,
            start_commit=self.start_commit,
            head_commit=self.head or self.start_commit,
            from_branch=self.from_branch,
            stop_reason=self.stop_reason,
            opened_at=self.opened_at,
            stopped_at=self.stopped_at,
        )


def lineage_view(state: NotebookState) -> LineageView:
    """Every commit and branch, in the order the notebook learnt of them."""

    splits = state.split_branches()
    branches: dict[str, _BranchDraft] = {}
    commits: dict[str, CommitNode] = {}
    for entry, value in state.decoded:
        if isinstance(value, LineageEvent):
            _apply_event(branches, commits, value, entry.at)
            continue
        node = _commit_node(state, value, splits, entry.at) if isinstance(value, JudgementWrite) else None
        if node is not None and node.commit_id not in commits:
            commits[node.commit_id] = node
            implicit = _BranchDraft(node.branch, LineageKind.OPEN, node.parent_commit, opened_at=entry.at)
            branches.setdefault(node.branch, implicit).head = node.commit_id
    return LineageView(tuple(commits.values()), tuple(draft.node() for draft in branches.values()))


def _apply_event(
    branches: dict[str, _BranchDraft], commits: dict[str, CommitNode], event: LineageEvent, at: float
) -> None:
    draft = branches.get(event.branch)
    if event.kind is LineageKind.ROOT:
        root = CommitNode(
            event.commit_id, "", event.branch, CommitOrigin.ROOT, at=at, recipe_digest=event.recipe_digest
        )
        commits.setdefault(event.commit_id, root)
        branches[event.branch] = _BranchDraft(event.branch, LineageKind.ROOT, event.commit_id, at, head=event.commit_id)
    elif event.kind is LineageKind.STOP:
        if draft is not None:
            draft.stop_reason, draft.stopped_at = event.reason, at
    elif draft is None:
        branches[event.branch] = _BranchDraft(event.branch, event.kind, event.from_commit, at, event.from_branch)
    else:
        draft.origin, draft.start_commit, draft.from_branch = event.kind, event.from_commit, event.from_branch
        draft.opened_at = min(draft.opened_at, at)


def _commit_node(
    state: NotebookState, write: JudgementWrite, splits: Mapping[str, str], at: float
) -> CommitNode | None:
    """The commit a judgement made; a judgement names a commit exactly when its conclusion makes one."""

    request = state.requests.get(write.request_id)
    if write.commit_id is None or request is None:
        return None
    if write.conclusion is Conclusion.MERGED:
        sources = state.merge_sources(write.request_id)
        branch, origin = INTEGRATION_BRANCH, CommitOrigin.MERGE
    elif write.conclusion is Conclusion.SPLIT_COMMITTED:
        sources = ()
        branch, origin = splits.get(write.commit_id, request.branch), CommitOrigin.SPLIT
    else:
        sources = ()
        branch, origin = request.branch, CommitOrigin.COMMIT
    return CommitNode(
        write.commit_id,
        request.parent_commit,
        branch,
        origin,
        write.request_id,
        sources,
        at,
        recipe_digest=write.recipe_digest,
    )


def merge_state(state: NotebookState, attempt: MergeAttempt) -> MergeState:
    """Where a merge attempt stands, from the judgement of its merge test."""

    if attempt.request_id is None:
        return MergeState(MergeOutcome.CONFLICT_SKIPPED, attempt.integration_head)
    write = state.latest_judgement(attempt.request_id)
    if write is None:
        return MergeState(MergeOutcome.PENDING, attempt.integration_head, attempt.request_id)
    if write.conclusion is Conclusion.MERGED:
        return MergeState(MergeOutcome.MERGED, attempt.integration_head, attempt.request_id, write.commit_id)
    return MergeState(MergeOutcome.NOT_STACKABLE, attempt.integration_head, attempt.request_id)


def committed(state: NotebookState, view: LineageView) -> tuple[CommittedCommit, ...]:
    """Specialist commits, foundation fixes first, then by their own gain over their parent."""

    merges = [node for node in view.commits if node.origin is CommitOrigin.MERGE]
    found = []
    for node in view.commits:
        if node.origin not in _SPECIALIST_ORIGINS:
            continue
        judgement = state.commit_writes[node.commit_id].judgement
        attempts = [attempt for attempt in state.merge_attempts if node.commit_id in attempt.source_commits]
        found.append(
            CommittedCommit(
                commit_id=node.commit_id,
                branch=node.branch,
                parent_commit=node.parent_commit,
                request_id=str(node.request_id),
                gain_pct=judgement.gain_pct,
                fixes_correctness=judgement.fixes_correctness,
                foundation_fix=judgement.foundation_fix,
                merged_into=next((merge.commit_id for merge in merges if node.commit_id in merge.merged_from), None),
                last_merge=merge_state(state, attempts[-1]) if attempts else None,
            )
        )
    return tuple(sorted(found, key=lambda commit: (not commit.foundation_fix, -commit.gain_pct)))


def recorded_value(state: NotebookState, view: LineageView, commit_id: str) -> RecordedValue:
    """The root's baseline, or the value the judgement that made the commit recorded."""

    node = view.commit(commit_id)
    if node is None:
        raise NotRecorded(f"no commit {commit_id}")
    if node.origin is CommitOrigin.ROOT:
        baseline = next((record for record in reversed(state.baselines) if record.root_commit == commit_id), None)
        if baseline is None:
            raise NotRecorded(f"no baseline for the root commit {commit_id}")
        return RecordedValue(commit_id, baseline.values, baseline.request_ids)
    judgement = state.commit_writes[commit_id].judgement
    return RecordedValue(commit_id, judgement.recorded_value, judgement.request_ids)


def regression_cases_for(state: NotebookState, view: LineageView, commit_id: str) -> tuple[RegressionCase, ...]:
    if view.commit(commit_id) is None:
        raise NotRecorded(f"no commit {commit_id}")
    ancestry = set(view.ancestry(commit_id))
    return tuple(case for case in state.regressions if case.fix_commit in ancestry)


def judged_since_commit(records: Sequence[TestRecord], view: LineageView, branch: str, until: float = math.inf) -> int:
    """The tests of ``branch`` judged by ``until`` and after its last commit by then, or since it opened before any.

    A test counts once it was measured and judged; a withdrawn or failed measurement, or a result that did not
    re-verify, never does.
    """

    commits = [
        node.at
        for node in view.commits
        if node.branch == branch and node.origin is not CommitOrigin.ROOT and node.at <= until
    ]
    since = max(commits, default=-math.inf)
    return sum(
        1
        for record in records
        if record.request.branch == branch
        and record.judged
        and record.judged_at is not None
        and since < record.judged_at <= until
    )


def progress(state: NotebookState, view: LineageView) -> tuple[BranchProgress, ...]:
    records = state.records()
    found = []
    for branch in view.branches:
        requests = [
            request
            for request in state.requests.values()
            if request.branch == branch.branch and request.request_id not in state.withdrawals
        ]
        results = [state.results[request.request_id] for request in requests if request.request_id in state.results]
        commits = [
            node for node in view.commits if node.branch == branch.branch and node.origin is not CommitOrigin.ROOT
        ]
        found.append(
            BranchProgress(
                branch=branch.branch,
                running=branch.stop_reason is None,
                start_commit=branch.start_commit,
                head_commit=branch.head_commit,
                tests=len(requests),
                commits=len(commits),
                gpu_seconds=sum(result.gpu_seconds for result in results),
                head_value=_value_or_empty(state, view, branch.head_commit),
                judged_since_commit=judged_since_commit(records, view, branch.branch),
                llm_usd=priced_usd(item.usage for item in state.llm_usage if item.usage.branch == branch.branch),
            )
        )
    return tuple(found)


def _value_or_empty(state: NotebookState, view: LineageView, commit_id: str) -> Mapping[str, float]:
    try:
        return recorded_value(state, view, commit_id).values
    except NotRecorded:
        return {}


def startup_brief(state: NotebookState, view: LineageView, start_commit: str) -> StartupBrief:
    """What specialists tried on the start commit and on every state it builds on, merged branches included."""

    if view.commit(start_commit) is None:
        raise NotRecorded(f"no commit {start_commit}")
    ancestry = set(view.ancestry(start_commit))
    tried = tuple(
        _tried(state, record)
        for record in state.records()
        if record.request.kind is TestKind.TEST
        and record.request.branch != INTEGRATION_BRANCH
        and record.request.parent_commit in ancestry
        and (record.in_flight or record.measured)
    )
    return StartupBrief(start_commit=start_commit, tried=tried, op_shares=state.op_shares.get(start_commit))


def _tried(state: NotebookState, record: TestRecord) -> TriedDirection:
    intent = state.intents.get(record.request.intent_fingerprint)
    judgement = record.judgement
    return TriedDirection(
        commit_id=record.request.parent_commit,
        request_id=record.request_id,
        target=intent.target if intent is not None else "",
        mechanism=intent.mechanism if intent is not None else "",
        layer=intent.layer if intent is not None else None,
        verdict=judgement.verdict if judgement is not None else None,
        conclusion=record.conclusion,
        gain_pct=judgement.gain_pct if judgement is not None else None,
        change=describe_change(record.request.delta),
    )


def quarantined(state: NotebookState) -> tuple[QuarantineEvent, ...]:
    """Every node or GPU set out of service; lifting a whole node returns its GPU sets too."""

    current: dict[tuple[str, tuple[int, ...]], QuarantineEvent] = {}
    for event in state.quarantine:
        if event.action is QuarantineAction.QUARANTINE:
            current[(event.node, event.gpu_ids)] = event
            continue
        for slot in [
            slot for slot in current if slot[0] == event.node and (not event.gpu_ids or slot[1] == event.gpu_ids)
        ]:
            del current[slot]
    return tuple(current.values())


def snapshot(state: NotebookState, claims: tuple[Claim, ...], taken_at: float) -> NotebookSnapshot:
    """Every view at once; ``claims`` are the claims in force as their holders see them at ``taken_at``."""

    view = lineage_view(state)
    return NotebookSnapshot(
        taken_at=taken_at,
        profile=state.profiles[-1] if state.profiles else None,
        fingerprints=tuple(state.fingerprints),
        records=tuple(state.records()),
        lineage=view,
        progress=progress(state, view),
        committed=committed(state, view),
        claims=claims,
        intents=dict(state.intents),
        baselines=tuple(state.baselines),
        regressions=tuple(state.regressions),
        quarantined=quarantined(state),
        branch_logs={branch: tuple(records) for branch, records in state.branch_logs.items()},
        learnings=tuple(state.learnings),
        op_shares=dict(state.op_shares),
        llm_usage=tuple(state.llm_usage),
    )


__all__ = [
    "committed",
    "judged_since_commit",
    "lineage_view",
    "merge_state",
    "progress",
    "quarantined",
    "recorded_value",
    "regression_cases_for",
    "snapshot",
    "startup_brief",
]
