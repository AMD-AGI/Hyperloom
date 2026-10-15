# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Frontier scoring, a prototype: how much one more unit of budget is expected to gain on each specialist branch.

:class:`FrontierScoring` is a ``ProgressQuery`` over the notebook's: the same rows, each running specialist branch's
``frontier`` filled with ``FrontierScore(expected_gain_pct, components, basis)``. The components:

* ``trend_pct``: the cumulative gain the branch reached over the last window, in points over the baseline;
* ``tests`` and ``gpu_hours``: what the branch has spent;
* ``headroom_pct``: how far the head's recorded value is from the roofline bound PRELUDE recorded for the metric;
* ``prior_remaining_pct``: what similar archived branches still gained after spending as many tests, weighted by task
  similarity, with ``prior_effective_sessions``; left out while the similar sessions count for fewer than 1.5;
* ``observing``: 1 while a kernel-layer direction the branch tested lately may still be paying off.

The expected gain is the prior where there is one, else the trend; a branch under observation keeps at least
``observation_floor_pct``; the headroom caps it, and it is never negative. ``basis`` names which of them the score
rests on. The formula is a prototype: what the score means is the scheduler's call (gongzheng's), and it stays here
until then.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, NotRecorded
from hyperloom_multibranch.contracts.notebook.interfaces import ProgressQuery, SnapshotQuery
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import (
    BranchProgress,
    FrontierScore,
    LineageView,
    NotebookSnapshot,
)
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, FrontierBasis
from hyperloom_multibranch.notebook.rsi.archive import (
    SearchTree,
    TreeBranch,
    session_start,
    similar_trees,
    tree_from_snapshot,
)
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions, effective_sessions

WINDOW_SECONDS = 1800.0
OBSERVATION_SECONDS = 3600.0
OBSERVATION_FLOOR_PCT = 0.5
MIN_EFFECTIVE_SESSIONS = 1.5
#: The session length the default window and observation suit; a shorter session scales both down in proportion.
REFERENCE_BUDGET_SECONDS = 4 * 3600.0


@dataclass(frozen=True)
class FrontierPolicy:
    """The scoring's parameters; ``metric`` is the recorded-value metric gains are read off."""

    metric: str
    window_seconds: float = WINDOW_SECONDS
    observation_seconds: float = OBSERVATION_SECONDS
    observation_floor_pct: float = OBSERVATION_FLOOR_PCT
    min_effective_sessions: float = MIN_EFFECTIVE_SESSIONS

    @classmethod
    def for_budget(cls, metric: str, budget_seconds: float) -> FrontierPolicy:
        """The policy of a session that may run ``budget_seconds``: the defaults, shrunk with a shorter session.

        A two-hour session looks back 15 minutes for its trend and observes a kernel direction for 30.
        """

        if budget_seconds <= 0:
            raise ValueError(f"a session budget must be positive, got {budget_seconds}")
        scale = min(1.0, budget_seconds / REFERENCE_BUDGET_SECONDS)
        return cls(metric, window_seconds=WINDOW_SECONDS * scale, observation_seconds=OBSERVATION_SECONDS * scale)


@dataclass(frozen=True)
class RemainingPrior:
    remaining_pct: float
    effective_sessions: float


def remaining_prior(history: tuple[SearchTree, ...], task: TaskConditions, tests: int) -> RemainingPrior | None:
    """What similar archived branches still gained after their ``tests``-th test; ``None`` when none got that far.

    A tree's weight is its similarity, shared among its branches that count, so a wide tree weighs as much as a
    narrow one.
    """

    total = weighted = 0.0
    weights = []
    for item in similar_trees(history, task):
        reached = [branch for branch in _specialists(item.tree.branches) if len(branch.test_times) >= tests]
        if not reached:
            continue
        weights.append(item.weight)
        for branch in reached:
            since = sorted(branch.test_times)[tests - 1] if tests else branch.opened_s
            total += item.weight / len(reached)
            weighted += item.weight / len(reached) * (branch.final_gain_pct - branch.gain_at(since))
    return RemainingPrior(weighted / total, effective_sessions(weights)) if total else None


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


def score_branch(
    snapshot: NotebookSnapshot,
    row: BranchProgress,
    tree: SearchTree,
    history: tuple[SearchTree, ...],
    policy: FrontierPolicy,
) -> FrontierScore:
    """The frontier score of one running specialist branch of ``tree``, the session's own search tree."""

    branch = next(item for item in tree.branches if item.name == row.branch)
    now = snapshot.taken_at - session_start(snapshot)
    trend = branch.gain_at(now) - branch.gain_at(now - policy.window_seconds)
    components = {
        "trend_pct": trend,
        "tests": float(len(branch.test_times)),
        "gpu_hours": branch.gpu_seconds / 3600.0,
        "observing": float(_observing(snapshot, row.branch, snapshot.taken_at - policy.observation_seconds)),
    }
    expected, basis = trend, FrontierBasis.TREND
    prior = remaining_prior(history, tree.task, len(branch.test_times))
    if prior is not None and prior.effective_sessions >= policy.min_effective_sessions:
        components.update(prior_remaining_pct=prior.remaining_pct, prior_effective_sessions=prior.effective_sessions)
        expected, basis = prior.remaining_pct, FrontierBasis.PRIOR
    if components["observing"] and expected < policy.observation_floor_pct:
        expected, basis = policy.observation_floor_pct, FrontierBasis.OBSERVING
    ceiling = snapshot.profile.roofline_ceiling.get(policy.metric) if snapshot.profile is not None else None
    head = row.head_value.get(policy.metric)
    if ceiling and head:
        components["headroom_pct"] = max(0.0, (ceiling / head - 1.0) * 100.0)
        if components["headroom_pct"] < expected:
            expected, basis = components["headroom_pct"], FrontierBasis.HEADROOM
    return FrontierScore(expected_gain_pct=max(0.0, expected), components=components, basis=basis)


def score_progress(
    snapshot: NotebookSnapshot, history: tuple[SearchTree, ...], policy: FrontierPolicy
) -> tuple[BranchProgress, ...]:
    """The notebook's progress, each running specialist branch scored; unscored without a profile or a baseline."""

    try:
        tree = tree_from_snapshot(snapshot, policy.metric)
    except (NotRecorded, ValueError):
        return snapshot.progress
    return tuple(
        replace(row, frontier=score_branch(snapshot, row, tree, history, policy))
        if row.running and row.branch != INTEGRATION_BRANCH
        else row
        for row in snapshot.progress
    )


@dataclass(frozen=True)
class FrontierScoring(ProgressQuery):
    """The notebook's progress with frontier scores, against the archived ``history``."""

    notebook: SnapshotQuery
    policy: FrontierPolicy
    history: tuple[SearchTree, ...] = ()

    async def progress(self) -> tuple[BranchProgress, ...]:
        return score_progress(await self.notebook.snapshot(), self.history, self.policy)

    async def lineage(self) -> LineageView:
        return (await self.notebook.snapshot()).lineage


__all__ = [
    "MIN_EFFECTIVE_SESSIONS",
    "OBSERVATION_FLOOR_PCT",
    "OBSERVATION_SECONDS",
    "REFERENCE_BUDGET_SECONDS",
    "WINDOW_SECONDS",
    "FrontierPolicy",
    "FrontierScoring",
    "RemainingPrior",
    "remaining_prior",
    "score_branch",
    "score_progress",
]
