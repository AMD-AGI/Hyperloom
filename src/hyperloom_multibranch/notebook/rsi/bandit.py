# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AdaEvolve's island bandit (SkyDiscover; arXiv 2602.20133, section 3.3) on archived branches: a baseline the
frontier calibration sets beside the frontier score, never a policy.

A specialist branch is an island and each of its tests a visit. A visit's reward is the step its commit took, the
gain the rule gate confirmed: the commit's cumulative gain over the branch's previous one, divided by one plus the
session's best gain by then (AdaEvolve's global normalization, in points over the baseline), then by a ``scale`` that
puts a large step near 1, as UCB's bonus expects. A test that landed nothing earns 0, and the gain a fill branch
forked with is its start, not a reward. Rewards and visits decay by ``decay`` at each visit of the branch, so
``R / V`` is its recent gain per test and an early breakthrough fades.

UCB adds ``exploration * sqrt(ln N / n)``, ``N`` the tests of every specialist branch by then and ``n`` the branch's
own. A branch not tested yet counts as tested once: a visit here is an end-to-end test, too dear to force. The
calibration measures how well a score foretells the gains that followed, while the bonus is a choice of where to
spend, so only the decayed mean can be advice for the frontier score; the UCB rows show what the bonus costs it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch
from hyperloom_multibranch.notebook.rsi.calibration import Baseline

#: SkyDiscover's ``gamma_decay``.
DECAY = 0.95
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
class DecayedMean:
    """The bandit's exploitation term, ``R / V``, as a score of ``branch`` at ``t_s``."""

    scale: float
    decay: float = DECAY

    def __call__(self, tree: SearchTree, branch: TreeBranch, t_s: float, history: tuple[SearchTree, ...]) -> float:
        return decayed_mean(visit_rewards(tree, branch.as_of(t_s), self.scale), self.decay)


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
    """The decayed mean, which may be advice, and the whole rule at each of ``explorations``, scaled on ``trees``."""

    scale = reward_scale(trees)
    return (
        Baseline(f"bandit R/V, decay {decay:g}", DecayedMean(scale, decay), advisable=True),
        *(Baseline(f"bandit UCB, C={c:.2g}", UpperConfidence(scale, c, decay)) for c in explorations),
    )


__all__ = [
    "DECAY",
    "EXPLORATIONS",
    "SCALE_QUANTILE",
    "DecayedMean",
    "UpperConfidence",
    "bandit_baselines",
    "decayed_mean",
    "reward_scale",
    "ucb",
    "visit_rewards",
]
