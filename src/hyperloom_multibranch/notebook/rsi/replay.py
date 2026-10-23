# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replay: how past sessions like this one would have fared under a candidate stop threshold and branch cap.

A replay can only cut what a session recorded; it never invents a branch the session did not open. Under a candidate
policy, a branch opened while ``branch_cap`` others were open is pruned whole; any other branch stops at the first
tick its gain over the last tick falls under ``stop_pct``, losing the commits after that tick and saving the rest of
its GPU time. A branch opened from a branch after that branch was cut goes with it. The integration branch is never
cut, and neither is a branch an autonomous agent drives, which the judge does not steer; it still takes one of the
slots the cap counts.

The utility of a replay is the best gain its kept commits reach, plus the GPU time it saved reinvested at half the
rate its kept time earned. Each archived session counts in proportion to its task similarity.

The two parameters are the scheduler's stop threshold and branch cap; this module only evaluates them. Choosing them
is the scheduler's, and later the outer loop's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.vocab import BranchDriver, LineageKind
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch, similar_trees
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions, effective_sessions

TICK_SECONDS = 1800.0
REINVEST = 0.5


@dataclass(frozen=True)
class ReplayPolicy:
    stop_pct: float
    branch_cap: int


@dataclass(frozen=True)
class ReplayOutcome:
    gain_pct: float
    gpu_hours: float
    saved_gpu_hours: float

    @property
    def utility(self) -> float:
        rate = self.gain_pct / self.gpu_hours if self.gpu_hours > 0 else 0.0
        return self.gain_pct + REINVEST * rate * self.saved_gpu_hours


def _is_integration(branch: TreeBranch) -> bool:
    return branch.name == INTEGRATION_BRANCH or branch.origin is LineageKind.ROOT


def _natural_end(branch: TreeBranch, tree: SearchTree) -> float:
    return branch.closed_s if branch.closed_s is not None else max(tree.budget_seconds, branch.opened_s)


def _stop_time(branch: TreeBranch, end: float, policy: ReplayPolicy, tick_seconds: float) -> float:
    tick = branch.opened_s + tick_seconds
    while tick < end:
        if branch.gain_at(tick) - branch.gain_at(tick - tick_seconds) < policy.stop_pct:
            return tick
        tick += tick_seconds
    return end


def cut_times(tree: SearchTree, policy: ReplayPolicy, tick_seconds: float = TICK_SECONDS) -> dict[str, float]:
    """When each branch stops under ``policy``; a branch pruned whole stops when it opened."""

    cuts: dict[str, float] = {}
    ends = {branch.name: _natural_end(branch, tree) for branch in tree.branches}
    for branch in sorted(tree.branches, key=lambda item: item.opened_s):
        if _is_integration(branch) or branch.driver is BranchDriver.AUTONOMOUS:
            cuts[branch.name] = ends[branch.name]
            continue
        parent_end = ends.get(branch.parent_branch, float("inf"))
        parent_cut = cuts.get(branch.parent_branch, parent_end)
        orphaned = parent_cut < parent_end and parent_cut <= branch.opened_s
        open_now = sum(
            1
            for other in tree.branches
            if other.name in cuts
            and not _is_integration(other)
            and other.opened_s <= branch.opened_s < cuts[other.name]
        )
        if orphaned or open_now >= policy.branch_cap:
            cuts[branch.name] = branch.opened_s
        else:
            cuts[branch.name] = _stop_time(branch, ends[branch.name], policy, tick_seconds)
    return cuts


def replay(tree: SearchTree, policy: ReplayPolicy, *, tick_seconds: float = TICK_SECONDS) -> ReplayOutcome:
    """Replay one archived session under ``policy``."""

    cuts = cut_times(tree, policy, tick_seconds)
    best, used, spent = 0.0, 0.0, 0.0
    for branch in tree.branches:
        if _is_integration(branch):
            continue
        cut, span = cuts[branch.name], _natural_end(branch, tree) - branch.opened_s
        kept = max(0.0, cut - branch.opened_s) / span if span > 0 else 1.0
        used += branch.gpu_seconds * kept
        spent += branch.gpu_seconds
        if cut > branch.opened_s:
            best = max([best, *(commit.gain_pct for commit in branch.commits if commit.t_s <= cut)])
    return ReplayOutcome(gain_pct=best, gpu_hours=used / 3600.0, saved_gpu_hours=(spent - used) / 3600.0)


@dataclass(frozen=True)
class PolicyEvaluation:
    """A policy's utility over the archived sessions similar to this one, and how much evidence it rests on."""

    policy: ReplayPolicy
    utility: float
    effective_sessions: float
    sessions: tuple[str, ...]


def evaluate(
    policy: ReplayPolicy,
    history: Sequence[SearchTree],
    task: TaskConditions,
    *,
    tick_seconds: float = TICK_SECONDS,
) -> PolicyEvaluation:
    """The similarity-weighted utility of ``policy``; zero, with no sessions, when none is similar."""

    weighted = similar_trees(history, task)
    total = sum(item.weight for item in weighted)
    utility = (
        sum(item.weight * replay(item.tree, policy, tick_seconds=tick_seconds).utility for item in weighted) / total
        if total
        else 0.0
    )
    return PolicyEvaluation(
        policy=policy,
        utility=utility,
        effective_sessions=effective_sessions(item.weight for item in weighted),
        sessions=tuple(item.tree.session_id for item in weighted),
    )


__all__ = [
    "REINVEST",
    "TICK_SECONDS",
    "PolicyEvaluation",
    "ReplayOutcome",
    "ReplayPolicy",
    "cut_times",
    "evaluate",
    "replay",
]
