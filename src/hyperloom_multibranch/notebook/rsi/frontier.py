# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Frontier scoring, a prototype: how much one more unit of budget is expected to gain on each specialist branch.

:class:`FrontierScoring` is a ``ProgressQuery`` over the notebook's: the same rows, each running specialist branch's
``frontier`` filled with ``FrontierScore(expected_gain_pct, components, basis)``. The components:

* ``trend_pct``: the cumulative gain the branch reached over the last window, in points over the baseline; the window
  spans at least :data:`WINDOW_TESTS` of the session's longest root tests (:func:`paced`);
* ``tests`` and ``gpu_hours``: what the branch has spent, its tests counted once measured;
* ``headroom_pct``: how far the head's recorded value is from the roofline bound PRELUDE recorded for the metric;
* ``prior_remaining_pct``: what similar archived branches still gained after spending as many tests, weighted by task
  similarity, with ``prior_effective_sessions``, the sessions behind it each counted by its similarity; a session less
  than half similar does not count, and the prior is left out while they add up to less than 1.5;
* ``observing``: 1 while a kernel-layer direction the branch tested lately may still be paying off;
* ``amdahl_ceiling_pct``: the most that speeding up the head's largest op could gain, when the head was profiled;
* ``pending_pct``: the gain of the branch's best candidate waiting for its confirm, in points over the baseline.

The expected gain is the prior where there is one, else the trend; a branch under observation keeps at least
``observation_floor_pct``, and a branch scored by its trend keeps ``new_branch_floor_pct`` until :data:`SETTLED_TESTS` of
its tests are judged, so a flat trend does not stop it before it has tried a few directions; a candidate waiting for its
confirm raises it to its own gain; the headroom and the Amdahl ceiling cap it, and it is never negative.
``basis`` names which of them the score rests on. The formula is a prototype: what the score means is the scheduler's
call (gongzheng's), and it stays here until then. :func:`score_at` is the formula alone, on a branch as an archived
tree has it, so the live scoring and the calibration score with the same one.

Given a ``log`` path, the scoring appends one :data:`FRONTIER_LOG_FORMAT` line per round it scored anything: when, on
the session clock too, the parameters the calibration varies, and each scored branch's spend, head value and score.
Checking the scores against the gains that followed is how they get calibrated.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, NotRecorded
from hyperloom_multibranch.contracts.notebook.interfaces import ProgressQuery, SnapshotQuery
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import (
    BranchProgress,
    FrontierScore,
    LineageView,
    NotebookSnapshot,
    OpShares,
    amdahl_ceiling_pct,
)
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, FrontierBasis
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.local.files import encode_json
from hyperloom_multibranch.notebook.rsi.archive import (
    SearchTree,
    TreeBranch,
    longest_root_test_seconds,
    session_start,
    similar_trees,
    tree_from_snapshot,
)
from hyperloom_multibranch.notebook.rsi.similarity import MIN_SIMILARITY, TaskConditions

log = logging.getLogger(__name__)

FRONTIER_LOG_FORMAT = "multibranch.frontier-log.v1"
HOUR = 3600.0
WINDOW_SECONDS = HOUR / 2
OBSERVATION_SECONDS = HOUR
OBSERVATION_FLOOR_PCT = 0.5
NEW_BRANCH_FLOOR_PCT = OBSERVATION_FLOOR_PCT
MIN_EFFECTIVE_SESSIONS = 1.5
#: The session length the default window and observation suit; a shorter session scales both down in proportion.
REFERENCE_BUDGET_SECONDS = 4 * HOUR
#: How many of the longest root tests a window spans at least, so that it holds an explore and its confirm.
WINDOW_TESTS = 4
#: How many judged tests end a branch's new-branch floor; the judge's grace reads it too, so the grace never ends
#: before the floor does.
SETTLED_TESTS = 10


@dataclass(frozen=True)
class FrontierPolicy:
    """The scoring's parameters; ``metric`` is the recorded-value metric gains are read off."""

    metric: str
    window_seconds: float = WINDOW_SECONDS
    observation_seconds: float = OBSERVATION_SECONDS
    observation_floor_pct: float = OBSERVATION_FLOOR_PCT
    new_branch_floor_pct: float = NEW_BRANCH_FLOOR_PCT
    min_effective_sessions: float = MIN_EFFECTIVE_SESSIONS
    min_similarity: float = MIN_SIMILARITY

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


@dataclass(frozen=True)
class RemainingPrior:
    remaining_pct: float
    effective_sessions: float


def remaining_prior(
    history: tuple[SearchTree, ...], task: TaskConditions, tests: int, *, min_similarity: float = MIN_SIMILARITY
) -> RemainingPrior | None:
    """What similar archived branches still gained after their ``tests``-th test; ``None`` when none got that far.

    Only trees at least ``min_similarity`` similar count. A tree's weight is its similarity, shared among its branches
    that count, so a wide tree weighs as much as a narrow one; the prior's ``effective_sessions`` is the sum of those
    weights, so three sessions a tenth similar count as a third of one.
    """

    total = weighted = 0.0
    for item in similar_trees(history, task, minimum=min_similarity):
        reached = [branch for branch in _specialists(item.tree.branches) if len(branch.spent_times) >= tests]
        for branch in reached:
            since = sorted(branch.spent_times)[tests - 1] if tests else branch.opened_s
            total += item.weight / len(reached)
            weighted += item.weight / len(reached) * (branch.final_gain_pct - branch.gain_at(since))
    return RemainingPrior(weighted / total, total) if total else None


def _specialists(branches: tuple[TreeBranch, ...]) -> list[TreeBranch]:
    return [branch for branch in branches if branch.name != INTEGRATION_BRANCH]


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
    """What a branch is scored against: its session's ``task``, the archived ``history`` and the ``policy``."""

    task: TaskConditions
    history: tuple[SearchTree, ...]
    policy: FrontierPolicy


@dataclass(frozen=True)
class Readings:
    """What the scoring reads about a branch besides its tree: whether it is ``observing`` and its two bounds.

    ``observing`` is whether a kernel-layer direction the branch tested within the observation period may still be
    paying off, ``headroom_pct`` how far its head is below the roofline bound (:func:`headroom_pct`),
    ``amdahl_pct`` the ceiling of the head's largest op (:func:`largest_ceiling_pct`), and ``pending_pct`` the gain of
    its best candidate waiting for its confirm, in points over the baseline.
    """

    observing: bool = False
    headroom_pct: float | None = None
    amdahl_pct: float | None = None
    pending_pct: float | None = None


def _young(branch: TreeBranch, t_s: float, policy: FrontierPolicy) -> bool:
    """Too new to have a trend: fewer than :data:`SETTLED_TESTS` tests judged, or, in a tree archived before judgements
    were kept, open less than a window and without a commit."""

    judged = branch.judged_by(t_s)
    if judged is None:
        return t_s - branch.opened_s < policy.window_seconds and not branch.commits
    return judged < SETTLED_TESTS


def score_at(branch: TreeBranch, t_s: float, context: ScoringContext, readings: Readings = Readings()) -> FrontierScore:
    """The frontier score of ``branch`` as it stood at ``t_s`` on its session's clock: the one formula.

    The scoring reads the ``readings`` off the notebook; the calibration reads them, and the branch as of ``t_s``, off
    an archived tree.
    """

    policy = context.policy
    observing = readings.observing
    trend = branch.gain_at(t_s) - branch.gain_at(t_s - policy.window_seconds)
    tests = len(branch.spent_times)
    components = {
        "trend_pct": trend,
        "tests": float(tests),
        "gpu_hours": branch.gpu_seconds / HOUR,
        "observing": float(observing),
    }
    expected, basis = trend, FrontierBasis.TREND
    prior = remaining_prior(context.history, context.task, tests, min_similarity=policy.min_similarity)
    if prior is not None and prior.effective_sessions >= policy.min_effective_sessions:
        components.update(prior_remaining_pct=prior.remaining_pct, prior_effective_sessions=prior.effective_sessions)
        expected, basis = prior.remaining_pct, FrontierBasis.PRIOR
    if observing and expected < policy.observation_floor_pct:
        expected, basis = policy.observation_floor_pct, FrontierBasis.OBSERVING
    if basis is FrontierBasis.TREND and _young(branch, t_s, policy) and expected < policy.new_branch_floor_pct:
        expected, basis = policy.new_branch_floor_pct, FrontierBasis.NEW
    if readings.pending_pct is not None:
        components["pending_pct"] = readings.pending_pct
        if readings.pending_pct > expected:
            expected, basis = readings.pending_pct, FrontierBasis.PENDING
    for key, bound, capped in (
        ("headroom_pct", readings.headroom_pct, FrontierBasis.HEADROOM),
        ("amdahl_ceiling_pct", readings.amdahl_pct, FrontierBasis.AMDAHL),
    ):
        if bound is not None:
            components[key] = bound
            if bound < expected:
                expected, basis = bound, capped
    return FrontierScore(expected_gain_pct=max(0.0, expected), components=components, basis=basis)


def score_branch(
    snapshot: NotebookSnapshot,
    row: BranchProgress,
    tree: SearchTree,
    history: tuple[SearchTree, ...],
    policy: FrontierPolicy,
) -> FrontierScore:
    """The frontier score of one running specialist branch of ``tree``, the session's own search tree."""

    branch = next(item for item in tree.branches if item.name == row.branch)
    ceiling = snapshot.profile.roofline_ceiling.get(policy.metric) if snapshot.profile is not None else None
    return score_at(
        branch,
        snapshot.taken_at - session_start(snapshot),
        ScoringContext(tree.task, history, policy),
        Readings(
            observing=_observing(snapshot, row.branch, snapshot.taken_at - policy.observation_seconds),
            headroom_pct=headroom_pct(ceiling, row.head_value.get(policy.metric)),
            amdahl_pct=largest_ceiling_pct(snapshot.op_shares.get(row.head_commit)),
            pending_pct=_pending_pct(snapshot, row, policy.metric),
        ),
    )


def _pending_pct(snapshot: NotebookSnapshot, row: BranchProgress, metric: str) -> float | None:
    """The best of ``row``'s candidates waiting for their confirm, in points over the baseline, the score's unit."""

    anchor = snapshot.baseline.values.get(metric) if snapshot.baseline is not None else None
    head = row.head_value.get(metric)
    if not anchor or not head:
        return None
    gains = [(item.value - head) / anchor * 100.0 for item in row.pending if item.metric == metric]
    return max(gains, default=None)


def score_progress(
    snapshot: NotebookSnapshot, history: tuple[SearchTree, ...], policy: FrontierPolicy
) -> tuple[BranchProgress, ...]:
    """The notebook's progress, each running specialist branch scored; unscored without a profile or a baseline."""

    try:
        tree = tree_from_snapshot(snapshot, policy.metric)
    except (NotRecorded, ValueError):
        return snapshot.progress
    policy = paced(policy, tree.longest_root_test_seconds)
    return tuple(
        replace(row, frontier=score_branch(snapshot, row, tree, history, policy))
        if row.running and row.branch != INTEGRATION_BRANCH
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
        "window_seconds": policy.window_seconds,
        "observation_seconds": policy.observation_seconds,
        "new_branch_floor_pct": policy.new_branch_floor_pct,
        "min_similarity": policy.min_similarity,
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
    """The notebook's progress with frontier scores, against the archived ``history``; ``log`` keeps every round."""

    notebook: SnapshotQuery
    policy: FrontierPolicy
    history: tuple[SearchTree, ...] = ()
    log: Path | None = None

    async def progress(self) -> tuple[BranchProgress, ...]:
        snapshot = await self.notebook.snapshot()
        rows = score_progress(snapshot, self.history, self.policy)
        if self.log is not None:
            line = frontier_log_line(snapshot, rows, paced(self.policy, longest_root_test_seconds(snapshot)))
            if line is not None:
                self._append(self.log, encode_json(line))
        return rows

    @staticmethod
    def _append(path: Path, data: bytes) -> None:
        # Losing a round of the log costs the calibration one sample; the scheduler's decision must not wait on it.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(data)
        except OSError as exc:
            log.warning("could not append the frontier scores to %s: %s", path, exc)

    async def lineage(self) -> LineageView:
        return (await self.notebook.snapshot()).lineage


__all__ = [
    "FRONTIER_LOG_FORMAT",
    "MIN_EFFECTIVE_SESSIONS",
    "NEW_BRANCH_FLOOR_PCT",
    "OBSERVATION_FLOOR_PCT",
    "OBSERVATION_SECONDS",
    "REFERENCE_BUDGET_SECONDS",
    "SETTLED_TESTS",
    "WINDOW_SECONDS",
    "WINDOW_TESTS",
    "FrontierPolicy",
    "FrontierScoring",
    "Readings",
    "RemainingPrior",
    "ScoringContext",
    "frontier_log_line",
    "headroom_pct",
    "largest_ceiling_pct",
    "paced",
    "remaining_prior",
    "score_at",
    "score_branch",
    "score_progress",
]
