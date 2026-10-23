# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the notebook's queries derive from its state: the branch tree, progress, commits, values and briefs.

Each fact is written once and every view is derived from it: a specialist commit comes from the judgement that
carries its commit id, a merge from the integrator's merge judgement and its merge attempt, a branch's start, split
and stop from lineage events.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook.interfaces import NotRecorded
from hyperloom_multibranch.contracts.notebook.peers import Purpose, TestKind, TestResult, Validity
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
    PendingCandidate,
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
    BranchDriver,
    CommitOrigin,
    Conclusion,
    LineageKind,
    MergeOutcome,
    QuarantineAction,
)
from hyperloom_multibranch.notebook.changes import describe_change
from hyperloom_multibranch.notebook.local.state import JudgementWrite, NotebookState

_SPECIALIST_ORIGINS = frozenset({CommitOrigin.COMMIT, CommitOrigin.SPLIT})
_MEASURE_PHASE = "measure"


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
    repairs: str = ""
    driver: BranchDriver = BranchDriver.SPECIALIST

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
            repairs=self.repairs,
            driver=self.driver,
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
        branches[event.branch] = _BranchDraft(
            event.branch,
            event.kind,
            event.from_commit,
            at,
            event.from_branch,
            repairs=event.repairs,
            driver=event.driver,
        )
    else:
        draft.origin, draft.start_commit, draft.from_branch = event.kind, event.from_commit, event.from_branch
        draft.repairs, draft.driver = event.repairs or draft.repairs, event.driver
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
        return MergeState(attempt.skipped_outcome, attempt.integration_head)
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


def measured_value(result: TestResult, metric: str, concurrency: int | None) -> float | None:
    """``metric`` as the gate reads it off ``result``: the mean over its measure-phase rounds at ``concurrency``, or over
    all of them when none ran there; ``None`` when no measure-phase round reports it."""

    rounds = [item for item in result.rounds if item.phase == _MEASURE_PHASE and metric in item.metrics]
    chosen = [item for item in rounds if item.concurrency == concurrency] or rounds
    return statistics.fmean(item.metrics[metric] for item in chosen) if chosen else None


def _candidate(
    record: TestRecord, confirm: str, head: Mapping[str, float], bars: Mapping[str, float], concurrency: int | None
) -> PendingCandidate | None:
    if record.result is None:
        return None
    for metric, bar in sorted(bars.items()):
        value, anchor = measured_value(record.result, metric, concurrency), head.get(metric)
        if value is None or not anchor:
            continue
        gain = (value / anchor - 1.0) * 100.0
        if gain >= bar:
            return PendingCandidate(record.request_id, confirm, record.request.parent_commit, metric, value, gain)
    return None


def _pending(
    state: NotebookState, records: Sequence[TestRecord], branch: BranchNode, head: Mapping[str, float]
) -> tuple[PendingCandidate, ...]:
    """``branch``'s measured explores on its head, not judged yet, whose confirm is in flight and whose gain over the head
    reaches the gate's bar."""

    if not state.baselines:
        return ()
    bars = state.baselines[-1].threshold_pct
    concurrency = state.profiles[-1].conc if state.profiles else None
    confirms = {record.request.retest_of: record.request_id for record in records if record.request.retest_of}
    waiting = {of: confirm for of, confirm in confirms.items() if state.record(confirm).in_flight}
    found = (
        _candidate(record, waiting[record.request_id], head, bars, concurrency)
        for record in records
        if record.request_id in waiting
        and record.request.branch == branch.branch
        and record.request.purpose is Purpose.EXPLORE
        and record.request.parent_commit == branch.head_commit
        and record.judgement is None
        and record.result is not None
        and record.result.validity is Validity.SUCCEEDED
    )
    return tuple(candidate for candidate in found if candidate is not None)


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
        head = _value_or_empty(state, view, branch.head_commit)
        found.append(
            BranchProgress(
                branch=branch.branch,
                running=branch.stop_reason is None,
                start_commit=branch.start_commit,
                head_commit=branch.head_commit,
                tests=len(requests),
                commits=len(commits),
                gpu_seconds=sum(result.gpu_seconds for result in results),
                head_value=head,
                judged_since_commit=judged_since_commit(records, view, branch.branch),
                llm_usd=priced_usd(item.usage for item in state.llm_usage if item.usage.branch == branch.branch),
                pending=_pending(state, records, branch, head),
            )
        )
    return tuple(found)


def _value_or_empty(state: NotebookState, view: LineageView, commit_id: str) -> Mapping[str, float]:
    try:
        return recorded_value(state, view, commit_id).values
    except NotRecorded:
        return {}


def startup_brief(state: NotebookState, view: LineageView, start_commit: str) -> StartupBrief:
    """What specialists tried on the start commit and on every state it builds on, merged branches included.

    The start commit's tests come first, then those of the commits the fewest parent or merge steps from it, the
    latest first on each step, so a brief that shows only the first of them shows the nearest.
    """

    if view.commit(start_commit) is None:
        request = state.requests.get(start_commit)
        if request is not None:
            raise NotRecorded(
                f"no commit {start_commit}: that is a test's request_id, and start_commit names a commit, such as "
                f"{request.parent_commit}, the one that test was made on"
            )
        raise NotRecorded(f"no commit {start_commit}")
    steps = _steps_from(view, start_commit)
    found = [
        record
        for record in state.records()
        if record.request.kind is TestKind.TEST
        and record.request.branch != INTEGRATION_BRANCH
        and record.request.parent_commit in steps
        and (record.in_flight or record.measured)
    ]
    nearest = sorted(range(len(found)), key=lambda index: (steps[found[index].request.parent_commit], -index))
    tried = tuple(_tried(state, found[index]) for index in nearest)
    return StartupBrief(start_commit=start_commit, tried=tried, op_shares=state.op_shares.get(start_commit))


def _steps_from(view: LineageView, commit: str) -> dict[str, int]:
    """``commit`` and every commit it builds on, through parents and the sources merges brought in, by the fewest
    steps from it."""

    by_id = {node.commit_id: node for node in view.commits}
    steps = {commit: 0}
    pending = deque([commit])
    while pending:
        current = pending.popleft()
        node = by_id.get(current)
        for source in (node.parent_commit, *node.merged_from) if node is not None else ():
            if source in by_id and source not in steps:
                steps[source] = steps[current] + 1
                pending.append(source)
    return steps


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
        resolved_patches=tuple(patch for attempt in state.merge_attempts for patch in attempt.resolved_patches),
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
