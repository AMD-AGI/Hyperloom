# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AdaEvolve's island bandit (SkyDiscover; arXiv 2602.20133, section 3.3) on archived branches: a frontier scorer
beside ``frontier-v1``, and baselines the frontier calibration sets beside both.

A specialist branch is an island and each of its tests a visit. A visit's reward is the step its commit took, the
gain the rule gate confirmed: the commit's cumulative gain over the branch's previous one, divided by one plus the
session's best gain by then (AdaEvolve's global normalization, in points over the baseline), then by a ``scale`` that
puts a large step near 1, as UCB's bonus expects. A test that landed nothing earns 0, and the gain a fill branch
forked with is its start, not a reward. Rewards and visits decay by ``decay`` at each visit of the branch, so
``R / V`` is its recent gain per test and an early breakthrough fades.

UCB adds ``exploration * sqrt(ln N / n)``, ``N`` the tests of every specialist branch by then and ``n`` the branch's
own. A branch not tested yet counts as tested once: a visit here is an end-to-end test, too dear to force. The
calibration measures how well a score foretells the gains that followed, while the bonus is a choice of where to
spend, so only the decayed mean, :class:`BanditScorer` (``bandit-rv``), is a frontier scorer that may be advice; the
UCB rows show what the bonus costs it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.vocab import FrontierBasis
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch
from hyperloom_multibranch.notebook.rsi.calibration import Baseline, scorer_baseline
from hyperloom_multibranch.notebook.rsi.frontier import Estimate, Readings, ScoringContext

#: SkyDiscover's ``gamma_decay``.
DECAY = 0.95
BANDIT_LABEL = "bandit-rv"
EXPLORATIONS = (0.1, 0.5, math.sqrt(2))
#: The quantile of the archive's steps that the scale puts at 1.
SCALE_QUANTILE = 0.9
_PERCENT = 100.0


def _best_gain(tree: SearchTree, t_s: float) -> float:
    return max((0.0, *(branch.gain_at(t_s) for branch in tree.branches if branch.opened_s <= t_s)))


def _steps(tree: SearchTree, branch: TreeBranch) -> list[tuple[float, float]]:
    """When each commit of ``branch`` landed and the normalized step it took, oldest first."""

    previous, steps = branch.fork_gain_pct, []
    for commit in sorted(branch.commits, key=lambda item: item.t_s):
        step = max(0.0, commit.gain_pct - previous) / (_PERCENT + _best_gain(tree, commit.t_s))
        previous = commit.gain_pct
        steps.append((commit.t_s, step))
    return steps


def visit_rewards(tree: SearchTree, branch: TreeBranch, scale: float = 1.0) -> tuple[float, ...]:
    """Each test's reward, oldest first: the steps its commits took, a commit counting for the last test before it."""

    times = sorted(branch.test_times)
    rewards = [0.0] * len(times)
    for landed, step in _steps(tree, branch):
        earlier = [index for index, when in enumerate(times) if when <= landed]
        if earlier:
            rewards[earlier[-1]] = min(1.0, rewards[earlier[-1]] + step / scale)
    return tuple(rewards)


def decayed_mean(rewards: Sequence[float], decay: float = DECAY) -> float:
    """``R / V``: the rewards, the latest weighing most, over the visits decayed alike; 0 before any visit."""

    weights = [decay ** (len(rewards) - 1 - index) for index in range(len(rewards))]
    return sum(weight * reward for weight, reward in zip(weights, rewards)) / sum(weights) if weights else 0.0


def ucb(rewards: Sequence[float], total_visits: int, exploration: float, decay: float = DECAY) -> float:
    """The decayed mean plus the exploration bonus, the branch counting as visited at least once."""

    bonus = exploration * math.sqrt(math.log(max(total_visits, 1)) / max(len(rewards), 1))
    return decayed_mean(rewards, decay) + bonus


def reward_scale(trees: Sequence[SearchTree], quantile: float = SCALE_QUANTILE) -> float:
    """The archive's step at ``quantile`` of all its specialist branches' steps; 1 when no branch committed."""

    steps = sorted(
        step
        for tree in trees
        for branch in tree.branches
        if branch.name != INTEGRATION_BRANCH
        for _, step in _steps(tree, branch)
        if step > 0
    )
    return steps[min(len(steps) - 1, int(quantile * len(steps)))] if steps else 1.0


def _visits(tree: SearchTree, t_s: float) -> int:
    return sum(branch.tests_by(t_s) for branch in tree.branches if branch.name != INTEGRATION_BRANCH)


@dataclass(frozen=True)
class BanditScorer:
    """The bandit's exploitation term, ``R / V``, as a frontier scorer: the branch's recent gain per test, back in
    points over the baseline, its reward scale read off the archived sessions it is scored against. It sets the branch
    beside the session's best gain by then, so it scores only where the scoring passes the session's tree."""

    decay: float = DECAY
    _scales: dict[int, tuple[tuple[SearchTree, ...], float]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @property
    def label(self) -> str:
        return BANDIT_LABEL

    def estimate(self, branch: TreeBranch, t_s: float, context: ScoringContext, readings: Readings) -> Estimate:
        tree = context.tree
        if tree is None:
            raise ValueError(f"{BANDIT_LABEL} scores a branch beside its session's tree, and none was passed")
        scale = self._scale(context.history)
        mean = decayed_mean(visit_rewards(tree, branch, scale), self.decay)
        points = mean * scale * (_PERCENT + _best_gain(tree, t_s))
        return Estimate(points, {"bandit_rv": mean, "reward_scale": scale}, FrontierBasis.CUSTOM)

    def _scale(self, history: tuple[SearchTree, ...]) -> float:
        """:func:`reward_scale` of ``history``, read once per history: the calibration scores every round against one."""

        held = self._scales.get(id(history))
        if held is not None and held[0] is history:
            return held[1]
        scale = reward_scale(history)
        self._scales.clear()
        self._scales[id(history)] = (history, scale)
        return scale


@dataclass(frozen=True)
class UpperConfidence:
    """The bandit's whole rule, ``R / V`` plus the bonus, as a score of ``branch`` at ``t_s``."""

    scale: float
    exploration: float
    decay: float = DECAY

    def __call__(self, tree: SearchTree, branch: TreeBranch, t_s: float, history: tuple[SearchTree, ...]) -> float:
        rewards = visit_rewards(tree, branch.as_of(t_s), self.scale)
        return ucb(rewards, _visits(tree, t_s), self.exploration, self.decay)


def bandit_baselines(
    trees: Sequence[SearchTree], *, decay: float = DECAY, explorations: Sequence[float] = EXPLORATIONS
) -> tuple[Baseline, ...]:
    """``bandit-rv``, which may be advice, then the whole rule at each of ``explorations``, scaled on ``trees``."""

    scale = reward_scale(trees)
    return (
        scorer_baseline(BanditScorer(decay)),
        *(Baseline(f"bandit UCB, C={c:.2g}", UpperConfidence(scale, c, decay)) for c in explorations),
    )


__all__ = [
    "BANDIT_LABEL",
    "DECAY",
    "EXPLORATIONS",
    "SCALE_QUANTILE",
    "BanditScorer",
    "UpperConfidence",
    "bandit_baselines",
    "decayed_mean",
    "reward_scale",
    "ucb",
    "visit_rewards",
]
