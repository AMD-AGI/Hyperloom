# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Frontier scoring, a prototype: how much one more unit of budget is expected to gain on each specialist branch.

:class:`FrontierScoring` is a ``ProgressQuery`` over the notebook's: the same rows, each running specialist branch's
``frontier`` filled with ``FrontierScore(expected_gain_pct, components, basis, scorer)``. A score has two layers.
:data:`FRONTIER_V1`, the formula below, gives the core :class:`Estimate`. The envelope, :func:`enveloped`, then holds
it to the same guarantees: a branch scored on its own short record keeps ``new_branch_floor_pct`` until
:data:`SETTLED_TESTS` of its explore tests succeeded and were judged, as the judge's grace counts them, so a flat
start does not stop it before it has tried a few directions; a repair branch that young scores at least the gain of
the blocked candidate it repairs; a candidate waiting for its confirm raises the score to its own gain; the headroom
and the Amdahl ceiling cap it; and it is never negative.

``frontier-v1``'s components, and the envelope's:

* ``trend_pct``: the cumulative gain the branch reached over the last window, in points over the baseline; the window
  spans at least :data:`WINDOW_TESTS` of the session's longest root tests (:func:`paced`), and reaches back to the
  branch's :data:`WINDOW_TESTS`-th last ended explore test when that is earlier (:func:`trend_start`), so a long
  profile or a wait for a slot does not empty it;
* ``tests`` and ``gpu_hours``: what the branch has spent, its tests counted once measured;
* ``observing``: 1 while a kernel-layer direction the branch tested lately may still be paying off;
* ``headroom_pct``: how far the head's recorded value is from the roofline bound PRELUDE recorded for the metric;
* ``amdahl_ceiling_pct``: the most that speeding up the head's largest op could gain, when the head was profiled;
* ``repairing_pct``: the gain of the blocked candidate a young repair branch repairs, in points over the baseline;
* ``pending_pct``: the gain of the branch's best candidate waiting for its confirm, in points over the baseline.

``frontier-v1`` expects the trend, and a branch under observation keeps at least ``observation_floor_pct``.
``basis`` names which part of either layer the score rests on, and ``scorer`` the scorer.
The formula is a prototype: what the score means is the scheduler's call (gongzheng's), and it stays here until then.
:func:`score_at` scores a branch as an archived tree has it, so the live scoring and a replay score alike.

Beside every score of a running branch the scoring writes its stall evidence (:func:`stall_evidence`), the facts the
judge's stop rule turns on: ``stalled_tests``, its tests judged since its last commit that did not fail to merge,
against the bar ``stall_bar`` (:class:`StopRule`, the stall odds of :mod:`rsi.history` by default);
``merged_commits`` and ``merged_gain_pct``, what of it reached the integration branch; ``near_bar_tests`` and
``stranded_lead_pct``, the leads it holds; and ``stop_candidate``, 1 when it has stalled past the bar with none of
those, no candidate waiting for its confirm and its grace over. A ``probation_share`` of the candidates is drawn,
once a stall, to run ``probation_tests`` more first, with ``probation`` 1 meanwhile: what such a branch does next is
how the stop rule learns whether stopping it was right.

Given a ``log`` path, the scoring appends one :data:`FRONTIER_LOG_FORMAT` line per round it scored anything: when, on
the session clock too, the policy's parameters, the scorer, and each scored branch's spend, head value and score.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from hyperloom_multibranch.contracts.notebook import NotRecorded
from hyperloom_multibranch.contracts.notebook.interfaces import ProgressQuery, SnapshotQuery
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import (
    BranchProgress,
    FrontierScore,
    LineageEvent,
    LineageView,
    NotebookSnapshot,
    OpShares,
    TestRecord,
    amdahl_ceiling_pct,
)
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, CommitOrigin, FrontierBasis, LineageKind
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.local.files import encode_json
from hyperloom_multibranch.notebook.local.unfinished import near_misses
from hyperloom_multibranch.notebook.rsi.archive import (
    SearchTree,
    TreeBranch,
    longest_root_test_seconds,
    session_start,
    tree_from_snapshot,
)
from hyperloom_multibranch.notebook.rsi.history import HORIZON_TESTS, stall_odds
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions
from hyperloom_multibranch.notebook.rsi.stranded import stranded_leads

log = logging.getLogger(__name__)

FRONTIER_LOG_FORMAT = "multibranch.frontier-log.v1"
HOUR = 3600.0
WINDOW_SECONDS = HOUR / 2
OBSERVATION_SECONDS = HOUR
OBSERVATION_FLOOR_PCT = 0.5
NEW_BRANCH_FLOOR_PCT = OBSERVATION_FLOOR_PCT
#: The session length the default window and observation suit; a shorter session scales both down in proportion.
REFERENCE_BUDGET_SECONDS = 4 * HOUR
#: How many of the longest root tests a window spans at least, so that it holds an explore and its confirm.
WINDOW_TESTS = 4
#: How many judged tests end a branch's new-branch floor; the judge's grace reads it too, so the grace never ends
#: before the floor does.
SETTLED_TESTS = 5
FRONTIER_V1_LABEL = "frontier-v1"
REPAIRING_COMPONENT = "repairing_pct"
PENDING_COMPONENT = "pending_pct"
HEADROOM_COMPONENT = "headroom_pct"
AMDAHL_COMPONENT = "amdahl_ceiling_pct"
STALLED_COMPONENT = "stalled_tests"
STALL_BAR_COMPONENT = "stall_bar"
MERGED_COMPONENT = "merged_commits"
MERGED_GAIN_COMPONENT = "merged_gain_pct"
NEAR_BAR_COMPONENT = "near_bar_tests"
STRANDED_COMPONENT = "stranded_lead_pct"
CANDIDATE_COMPONENT = "stop_candidate"
PROBATION_COMPONENT = "probation"
#: The stall evidence's components, written beside every score of a running branch.
EVIDENCE_COMPONENTS = frozenset(
    {
        STALLED_COMPONENT,
        STALL_BAR_COMPONENT,
        MERGED_COMPONENT,
        MERGED_GAIN_COMPONENT,
        NEAR_BAR_COMPONENT,
        STRANDED_COMPONENT,
        CANDIDATE_COMPONENT,
        PROBATION_COMPONENT,
    }
)
#: The share of stop candidates drawn for probation, once a stall.
PROBATION_SHARE = 0.25
#: How many judged tests past the bar a branch drawn for probation runs before it is a stop candidate: the horizon
#: the stall odds read a commit within, so what it does then measures them.
PROBATION_TESTS = HORIZON_TESTS


@dataclass(frozen=True)
class FrontierPolicy:
    """The scoring's parameters; ``metric`` is the recorded-value metric gains are read off, and ``settled_tests``
    how many explore tests that succeeded and were judged end a new branch's floor, as they end the judge's grace."""

    metric: str
    window_seconds: float = WINDOW_SECONDS
    observation_seconds: float = OBSERVATION_SECONDS
    observation_floor_pct: float = OBSERVATION_FLOOR_PCT
    new_branch_floor_pct: float = NEW_BRANCH_FLOOR_PCT
    settled_tests: int = SETTLED_TESTS

    @classmethod
    def for_budget(cls, metric: str, budget_seconds: float) -> FrontierPolicy:
        """The policy of a session that may run ``budget_seconds``: the defaults, shrunk with a shorter session.

        A two-hour session looks back 15 minutes for its trend and observes a kernel direction for 30.
        """

        if budget_seconds <= 0:
            raise ValueError(f"a session budget must be positive, got {budget_seconds}")
        scale = min(1.0, budget_seconds / REFERENCE_BUDGET_SECONDS)
        return cls(metric, window_seconds=WINDOW_SECONDS * scale, observation_seconds=OBSERVATION_SECONDS * scale)


def paced(policy: FrontierPolicy, longest_test_seconds: float | None) -> FrontierPolicy:
    """``policy`` with a window of at least :data:`WINDOW_TESTS` of the longest root tests, when that is longer."""

    if not longest_test_seconds or WINDOW_TESTS * longest_test_seconds <= policy.window_seconds:
        return policy
    return replace(policy, window_seconds=WINDOW_TESTS * longest_test_seconds)


def _observing(snapshot: NotebookSnapshot, branch: str, since: float) -> bool:
    kernel = {fingerprint for fingerprint, intent in snapshot.intents.items() if intent.layer is IntentLayer.KERNEL}
    tested = any(
        record.request.branch == branch and record.request.intent_fingerprint in kernel and record.requested_at >= since
        for record in snapshot.records
    )
    held = any(
        claim.status is ClaimStatus.HELD and claim.intent.branch == branch and claim.key.intent_fingerprint in kernel
        for claim in snapshot.claims
    )
    return tested or held


def headroom_pct(ceiling: float | None, head: float | None) -> float | None:
    """How far ``head`` is below the roofline ``ceiling``, in percent of the head; ``None`` without either."""

    return max(0.0, (ceiling / head - 1.0) * 100.0) if ceiling and head else None


def largest_ceiling_pct(shares: OpShares | None) -> float | None:
    """The Amdahl ceiling of the largest op in ``shares``; ``None`` for a commit that was not profiled."""

    largest = shares.largest(1) if shares is not None else ()
    return amdahl_ceiling_pct(largest[0][1]) if largest else None


@dataclass(frozen=True)
class ScoringContext:
    """What a branch is scored against: its session's ``task``, the archived ``history``, the ``policy``, and ``tree``,
    the session's own search tree as it stood then, for a scorer that sets a branch beside its siblings; ``None`` where
    the caller built none."""

    task: TaskConditions
    history: tuple[SearchTree, ...]
    policy: FrontierPolicy
    tree: SearchTree | None = None


@dataclass(frozen=True)
class Readings:
    """What the scoring reads about a branch besides its tree: whether it is ``observing`` and its two bounds.

    ``observing`` is whether a kernel-layer direction the branch tested within the observation period may still be
    paying off, ``headroom_pct`` how far its head is below the roofline bound (:func:`headroom_pct`),
    ``amdahl_pct`` the ceiling of the head's largest op (:func:`largest_ceiling_pct`), ``pending_pct`` the gain of
    its best candidate waiting for its confirm, and ``repairing_pct`` the gain of the blocked candidate a repair branch
    was opened to repair, both in points over the baseline.
    """

    observing: bool = False
    headroom_pct: float | None = None
    amdahl_pct: float | None = None
    pending_pct: float | None = None
    repairing_pct: float | None = None


@dataclass(frozen=True)
class Estimate:
    """``frontier-v1``'s estimate of a branch, before the envelope: the gain expected of one more unit of budget, in
    points over the baseline, the ``components`` it read, and the ``basis`` it rests on, its trend or its
    observation."""

    expected_gain_pct: float
    components: Mapping[str, float] = field(default_factory=dict)
    basis: FrontierBasis = FrontierBasis.TREND


def is_young(branch: TreeBranch, t_s: float, policy: FrontierPolicy) -> bool:
    """Too new to have a trend: fewer than ``policy.settled_tests`` explore tests succeeded and were judged, as the
    judge's grace counts them, so the floor and the grace end at the same test, and a branch whose tests failed keeps
    its floor. A tree archived before that was kept counts its ended explore tests, then its judged tests; one
    archived before judgements were kept, too, is young while open less than a window and without a commit."""

    for counted in (branch.settled_by, branch.ended_by, branch.judged_by):
        settled = counted(t_s)
        if settled is not None:
            return settled < policy.settled_tests
    return t_s - branch.opened_s < policy.window_seconds and not branch.commits


def trend_start(branch: TreeBranch, t_s: float, window_seconds: float) -> float:
    """Where the trend window starts: ``window_seconds`` back, or earlier, at the branch's :data:`WINDOW_TESTS`-th
    last explore test that ended by ``t_s``, so a long profile or a wait for a slot cannot empty the window; a tree
    archived before ended tests were kept, or a branch with fewer, keeps the wall-clock window."""

    start = t_s - window_seconds
    ended = sorted(when for when in branch.ended_times or () if when <= t_s)
    return min(start, ended[-WINDOW_TESTS]) if len(ended) >= WINDOW_TESTS else start


@dataclass(frozen=True)
class FrontierV1:
    """Today's formula: the branch's trend, and at least ``observation_floor_pct`` while a kernel-layer direction it
    tested lately may still pay off."""

    @property
    def label(self) -> str:
        return FRONTIER_V1_LABEL

    def estimate(self, branch: TreeBranch, t_s: float, context: ScoringContext, readings: Readings) -> Estimate:
        policy = context.policy
        trend = branch.gain_at(t_s) - branch.gain_at(trend_start(branch, t_s, policy.window_seconds))
        tests = len(branch.spent_times)
        components = {
            "trend_pct": trend,
            "tests": float(tests),
            "gpu_hours": branch.gpu_seconds / HOUR,
            "observing": float(readings.observing),
        }
        expected, basis = trend, FrontierBasis.TREND
        if readings.observing and expected < policy.observation_floor_pct:
            expected, basis = policy.observation_floor_pct, FrontierBasis.OBSERVING
        return Estimate(expected, components, basis)


#: The built-in formula, :class:`FrontierV1`.
FRONTIER_V1 = FrontierV1()


def enveloped(
    estimate: Estimate, branch: TreeBranch, t_s: float, context: ScoringContext, readings: Readings
) -> FrontierScore:
    """``estimate`` held to the envelope. ``basis`` names the envelope's part where it changed the score: the
    new-branch floor of a young branch scored on its own trend, the blocked gain a young repair branch repairs, the
    gain of a candidate waiting for its confirm, or the headroom or the Amdahl ceiling that capped it."""

    policy = context.policy
    expected, basis = estimate.expected_gain_pct, estimate.basis
    components = dict(estimate.components)
    young = is_young(branch, t_s, policy)
    if basis is FrontierBasis.TREND and young and expected < policy.new_branch_floor_pct:
        expected, basis = policy.new_branch_floor_pct, FrontierBasis.NEW
    if readings.repairing_pct is not None and young:
        components[REPAIRING_COMPONENT] = readings.repairing_pct
        if readings.repairing_pct > expected:
            expected, basis = readings.repairing_pct, FrontierBasis.REPAIR
    if readings.pending_pct is not None:
        components[PENDING_COMPONENT] = readings.pending_pct
        if readings.pending_pct > expected:
            expected, basis = readings.pending_pct, FrontierBasis.PENDING
    for key, bound, capped in (
        (HEADROOM_COMPONENT, readings.headroom_pct, FrontierBasis.HEADROOM),
        (AMDAHL_COMPONENT, readings.amdahl_pct, FrontierBasis.AMDAHL),
    ):
        if bound is not None:
            components[key] = bound
            if bound < expected:
                expected, basis = bound, capped
    return FrontierScore(expected_gain_pct=max(0.0, expected), components=components, basis=basis)


def score_at(branch: TreeBranch, t_s: float, context: ScoringContext, readings: Readings = Readings()) -> FrontierScore:
    """The frontier score of ``branch`` as it stood at ``t_s`` on its session's clock: ``frontier-v1``'s estimate held
    to the envelope. The scoring reads the ``readings`` off the notebook; a replay reads them, and the branch as of
    ``t_s``, off an archived tree."""

    estimate = FRONTIER_V1.estimate(branch, t_s, context, readings)
    return replace(enveloped(estimate, branch, t_s, context, readings), scorer=FRONTIER_V1_LABEL)


@dataclass(frozen=True)
class StopRule:
    """When a running branch is a stop candidate: ``stall_tests`` judged tests past its last commit that did not fail
    to merge, ``None`` for the bar the stall odds set each round (:func:`~.history.stall_odds`, the session's own
    branches counted in); a ``probation_share`` of the candidates, drawn once a stall, runs ``probation_tests`` more
    before it is one."""

    stall_tests: int | None = None
    probation_share: float = PROBATION_SHARE
    probation_tests: int = PROBATION_TESTS


@dataclass(frozen=True)
class StallBar:
    """The stop rule as one round applies it: the ``rule``, and the ``bar`` it stands at this round."""

    rule: StopRule
    bar: int


def _last_mergeable_commit(snapshot: NotebookSnapshot, branch: str) -> tuple[str, float]:
    """The branch's last commit that did not fail to merge, and when it was made; ``("", -inf)`` before any."""

    failed = {
        item.commit_id for item in snapshot.committed if item.first_failed_at is not None and item.merged_into is None
    }
    made = [
        node
        for node in snapshot.lineage.commits
        if node.branch == branch and node.origin is not CommitOrigin.ROOT and node.commit_id not in failed
    ]
    last = max(made, key=lambda node: node.at, default=None)
    return (last.commit_id, last.at) if last is not None else ("", -math.inf)


def _judged_since(snapshot: NotebookSnapshot, branch: str, since: float) -> list[TestRecord]:
    judged = [
        record
        for record in snapshot.records
        if record.request.branch == branch
        and record.judged
        and record.judged_at is not None
        and record.judged_at > since
    ]
    return sorted(judged, key=lambda record: record.judged_at or 0.0)


def drawn_for_probation(snapshot: NotebookSnapshot, branch: str, anchor: str, share: float) -> bool:
    """Whether the stall of ``branch`` since its commit ``anchor`` (``""`` before any) is drawn for probation, with
    chance ``share``: the same draw every round of the stall."""

    session = snapshot.profile.session_id if snapshot.profile is not None else ""
    digest = hashlib.sha256(f"{session}:{branch}:{anchor}".encode()).hexdigest()
    return int(digest, 16) < share * 16 ** len(digest)


def stall_evidence(
    snapshot: NotebookSnapshot, row: BranchProgress, metric: str, stall: StallBar, young: bool
) -> dict[str, float]:
    """The stall evidence of the running branch ``row``, ``young`` while its grace lasts.

    The stall counts the tests judged since the branch's last commit that did not fail to merge, or since it opened.
    A branch holds a near-bar lead when one of its last ``stall.bar`` judged tests is a near miss
    (:func:`~hyperloom_multibranch.notebook.local.unfinished.near_misses`), and a stranded one when its head leads
    integration over the bar on commits that did not merge (:func:`~.stranded.stranded_leads`). A branch with either,
    a candidate waiting for its confirm, or its grace, is no stop candidate however long it stalled.
    """

    anchor, since = _last_mergeable_commit(snapshot, row.branch)
    judged = _judged_since(snapshot, row.branch, since)
    recent = judged[-stall.bar :] if stall.bar > 0 else []
    near = near_misses(recent)
    merged = [item for item in snapshot.committed if item.branch == row.branch and item.merged_into is not None]
    stranded = next((lead.lead_pct for lead in stranded_leads(snapshot, metric) if lead.branch == row.branch), None)
    evidence = {
        STALLED_COMPONENT: float(len(judged)),
        STALL_BAR_COMPONENT: float(stall.bar),
        MERGED_COMPONENT: float(len(merged)),
        MERGED_GAIN_COMPONENT: sum(item.gain_pct for item in merged),
        NEAR_BAR_COMPONENT: float(near),
    }
    if stranded is not None:
        evidence[STRANDED_COMPONENT] = stranded
    held = young or bool(row.pending) or near > 0 or stranded is not None
    stalled = len(judged) >= stall.bar
    probation = (
        not held
        and stalled
        and len(judged) < stall.bar + stall.rule.probation_tests
        and drawn_for_probation(snapshot, row.branch, anchor, stall.rule.probation_share)
    )
    evidence[CANDIDATE_COMPONENT] = float(stalled and not held and not probation)
    if probation:
        evidence[PROBATION_COMPONENT] = 1.0
    return evidence


def probation_event(
    snapshot: NotebookSnapshot, row: BranchProgress, rule: StopRule | None = None
) -> LineageEvent | None:
    """The ``PROBATION`` event the scheduler records while ``row``'s stall evidence holds it on probation under
    ``rule``, :class:`StopRule`'s defaults without one; ``None`` otherwise. It is the same event every round of the
    stall, and the notebook records one a stall."""

    if row.frontier is None or not row.frontier.components.get(PROBATION_COMPONENT):
        return None
    anchor, _ = _last_mergeable_commit(snapshot, row.branch)
    return LineageEvent(
        LineageKind.PROBATION,
        row.branch,
        commit_id=anchor,
        reason=f"stalled past the stop bar since {anchor or 'it opened'}, drawn for probation",
        tests=(rule or StopRule()).probation_tests,
    )


def score_branch(
    snapshot: NotebookSnapshot, row: BranchProgress, context: ScoringContext, stall: StallBar | None = None
) -> FrontierScore:
    """The frontier score of one running specialist branch of ``context.tree``, the session's own search tree; with
    ``stall``, its stall evidence beside it."""

    if context.tree is None:
        raise ValueError("a running branch is scored on its session's search tree")
    policy = context.policy
    branch = next(item for item in context.tree.branches if item.name == row.branch)
    ceiling = snapshot.profile.roofline_ceiling.get(policy.metric) if snapshot.profile is not None else None
    readings = Readings(
        observing=_observing(snapshot, row.branch, snapshot.taken_at - policy.observation_seconds),
        headroom_pct=headroom_pct(ceiling, row.head_value.get(policy.metric)),
        amdahl_pct=largest_ceiling_pct(snapshot.op_shares.get(row.head_commit)),
        pending_pct=_pending_pct(snapshot, row, policy.metric),
        repairing_pct=_repairing_pct(snapshot, row, policy.metric),
    )
    t_s = snapshot.taken_at - session_start(snapshot)
    score = score_at(branch, t_s, context, readings)
    if stall is None:
        return score
    evidence = stall_evidence(snapshot, row, policy.metric, stall, is_young(branch, t_s, policy))
    return replace(score, components={**score.components, **evidence})


def _over_head(snapshot: NotebookSnapshot, row: BranchProgress, metric: str, values: Sequence[float]) -> float | None:
    """The best of ``values`` over ``row``'s head, in points over the baseline, the score's unit."""

    anchor = snapshot.baseline.values.get(metric) if snapshot.baseline is not None else None
    head = row.head_value.get(metric)
    if not anchor or not head:
        return None
    return max(((value - head) / anchor * 100.0 for value in values), default=None)


def _pending_pct(snapshot: NotebookSnapshot, row: BranchProgress, metric: str) -> float | None:
    """The best of ``row``'s candidates waiting for their confirm."""

    return _over_head(snapshot, row, metric, [item.value for item in row.pending if item.metric == metric])


def _repairing_pct(snapshot: NotebookSnapshot, row: BranchProgress, metric: str) -> float | None:
    """The blocked candidate ``row``'s branch was opened to repair, read as a pending candidate is; ``None`` for a
    branch that repairs nothing."""

    node = next((item for item in snapshot.lineage.branches if item.branch == row.branch), None)
    if node is None or not node.repairs:
        return None
    record = next((item for item in snapshot.records if item.request.request_id == node.repairs), None)
    value = record.judgement.recorded_value.get(metric) if record is not None and record.judgement else None
    return _over_head(snapshot, row, metric, [value] if value is not None else [])


def score_progress(
    snapshot: NotebookSnapshot, history: tuple[SearchTree, ...], policy: FrontierPolicy, rule: StopRule | None = None
) -> tuple[BranchProgress, ...]:
    """The notebook's progress, each running branch the judge steers scored, with its stall evidence under ``rule``
    (:class:`StopRule`'s defaults without one); unscored without a profile or a baseline."""

    try:
        tree = tree_from_snapshot(snapshot, policy.metric)
    except (NotRecorded, ValueError):
        return snapshot.progress
    context = ScoringContext(tree.task, history, paced(policy, tree.longest_root_test_seconds), tree)
    stop = rule or StopRule()
    bar = stop.stall_tests if stop.stall_tests is not None else stall_odds(history, tree.task, session=tree).stall_tests
    stall = StallBar(stop, bar)
    steered = {branch.name for branch in tree.branches if branch.steered}
    return tuple(
        replace(row, frontier=score_branch(snapshot, row, context, stall))
        if row.running and row.branch in steered
        else row
        for row in snapshot.progress
    )


def frontier_log_line(
    snapshot: NotebookSnapshot, rows: Sequence[BranchProgress], policy: FrontierPolicy
) -> dict[str, object] | None:
    """One round of scores as a log line; ``None`` when the round scored no branch."""

    scored = [row for row in rows if row.frontier is not None]
    if not scored:
        return None
    return {
        "format": FRONTIER_LOG_FORMAT,
        "at": snapshot.taken_at,
        "t_s": snapshot.taken_at - session_start(snapshot),
        "metric": policy.metric,
        "scorer": FRONTIER_V1_LABEL,
        "window_seconds": policy.window_seconds,
        "observation_seconds": policy.observation_seconds,
        "new_branch_floor_pct": policy.new_branch_floor_pct,
        "branches": [
            {
                "branch": row.branch,
                "tests": row.tests,
                "commits": row.commits,
                "gpu_seconds": row.gpu_seconds,
                "head_value": dict(row.head_value),
                "frontier": encode(row.frontier),
            }
            for row in scored
        ],
    }


@dataclass(frozen=True)
class FrontierScoring(ProgressQuery):
    """The notebook's progress with frontier scores, against the archived ``history``; ``log`` keeps every round.
    ``stop_rule`` gives the stop rule each round's stall evidence reads: the playbook's the session plays, which a
    fallback changes."""

    notebook: SnapshotQuery
    policy: FrontierPolicy
    history: tuple[SearchTree, ...] = ()
    log: Path | None = None
    stop_rule: Callable[[], StopRule] = StopRule

    async def progress(self) -> tuple[BranchProgress, ...]:
        snapshot = await self.notebook.snapshot()
        rows = score_progress(snapshot, self.history, self.policy, self.stop_rule())
        if self.log is not None:
            policy = paced(self.policy, longest_root_test_seconds(snapshot))
            line = frontier_log_line(snapshot, rows, policy)
            if line is not None:
                self._append(self.log, encode_json(line))
        return rows

    @staticmethod
    def _append(path: Path, data: bytes) -> None:
        # Losing a round of the log costs the record one round; the scheduler's decision must not wait on it.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(data)
        except OSError as exc:
            log.warning("could not append the frontier scores to %s: %s", path, exc)

    async def lineage(self) -> LineageView:
        return (await self.notebook.snapshot()).lineage


__all__ = [
    "AMDAHL_COMPONENT",
    "CANDIDATE_COMPONENT",
    "EVIDENCE_COMPONENTS",
    "FRONTIER_LOG_FORMAT",
    "FRONTIER_V1",
    "FRONTIER_V1_LABEL",
    "HEADROOM_COMPONENT",
    "MERGED_COMPONENT",
    "MERGED_GAIN_COMPONENT",
    "NEAR_BAR_COMPONENT",
    "NEW_BRANCH_FLOOR_PCT",
    "OBSERVATION_FLOOR_PCT",
    "OBSERVATION_SECONDS",
    "PENDING_COMPONENT",
    "PROBATION_COMPONENT",
    "PROBATION_SHARE",
    "PROBATION_TESTS",
    "REFERENCE_BUDGET_SECONDS",
    "REPAIRING_COMPONENT",
    "SETTLED_TESTS",
    "STALLED_COMPONENT",
    "STALL_BAR_COMPONENT",
    "STRANDED_COMPONENT",
    "WINDOW_SECONDS",
    "WINDOW_TESTS",
    "Estimate",
    "FrontierPolicy",
    "FrontierScoring",
    "FrontierV1",
    "Readings",
    "ScoringContext",
    "StallBar",
    "StopRule",
    "drawn_for_probation",
    "enveloped",
    "frontier_log_line",
    "headroom_pct",
    "is_young",
    "largest_ceiling_pct",
    "paced",
    "probation_event",
    "score_at",
    "score_branch",
    "score_progress",
    "stall_evidence",
    "trend_start",
]
