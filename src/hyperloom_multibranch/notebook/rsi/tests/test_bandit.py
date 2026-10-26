# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AdaEvolve's island bandit on archived branches, as a baseline beside the frontier score."""

from __future__ import annotations

import dataclasses
import math

import pytest

from hyperloom_multibranch.contracts.notebook.vocab import BranchDriver
from hyperloom_multibranch.notebook.rsi.archive import SearchTree
from hyperloom_multibranch.notebook.rsi.bandit import (
    BANDIT_LABEL,
    BanditScorer,
    bandit_baselines,
    decayed_mean,
    reward_scale,
    ucb,
    visit_rewards,
)
from hyperloom_multibranch.notebook.rsi.calibration import CURRENT, Variation, calibrate
from hyperloom_multibranch.notebook.rsi.frontier import SETTLED_TESTS, FrontierPolicy, Readings, ScoringContext
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, TASK, branch, tree


def test_a_tests_reward_is_the_step_its_commit_took_and_a_forked_gain_is_none() -> None:
    worked = branch("s1", [(650.0, 2.0), (1300.0, 3.0)], test_times=(0.0, 600.0, 1200.0))
    filled = branch("s2", [], opened_s=1800.0, fork_gain_pct=5.0, test_times=(1800.0, 2400.0))
    searched = tree("a", worked, filled)

    assert visit_rewards(searched, worked) == pytest.approx((0.0, 2.0 / 102.0, 1.0 / 103.0))
    assert visit_rewards(searched, worked, scale=1.0 / 102.0) == pytest.approx((0.0, 1.0, 102.0 / 103.0))
    assert visit_rewards(searched, filled) == (0.0, 0.0)


def test_the_decayed_mean_lets_an_early_breakthrough_fade() -> None:
    early = [1.0] + [0.0] * 19
    late = [0.0, 0.0, 0.0, 0.15]

    assert sum(early) / len(early) > sum(late) / len(late)
    assert decayed_mean(early) < decayed_mean(late)
    assert decayed_mean([]) == 0.0


def test_ucb_favours_a_branch_tested_less_and_counts_an_untested_one_as_tested_once() -> None:
    assert ucb([], 10, 1.0) == pytest.approx(math.sqrt(math.log(10)))
    assert ucb([0.2] * 4, 10, 0.0) == pytest.approx(0.2)
    assert ucb([0.0], 10, 1.0) > ucb([0.0] * 4, 10, 1.0)


def test_the_scale_puts_the_archives_large_steps_near_one() -> None:
    steps = [(600.0 * step, 1.0 * step) for step in range(1, 11)]
    searched = tree("a", branch("s1", steps))

    assert reward_scale([]) == 1.0
    assert reward_scale([searched]) == pytest.approx(1.0 / 101.0)


def test_the_scale_leaves_out_the_steps_of_a_branch_an_autonomous_agent_drove() -> None:
    steps = [(600.0 * step, 1.0 * step) for step in range(1, 11)]
    # Each agent catches up with s1 in one step, so the session's best gain by then stays s1's.
    agents = (
        branch("agent-1", [(5400.0, 9.0)], driver=BranchDriver.AUTONOMOUS),
        branch("agent-2", [(6000.0, 10.0)], driver=BranchDriver.AUTONOMOUS),
    )
    alone = tree("a", branch("s1", steps))

    assert reward_scale([dataclasses.replace(alone, branches=(*alone.branches, *agents))]) == reward_scale([alone])


def _mature(n: int) -> SearchTree:
    """Two settled branches, one that committed early and late and one that never did: after the early commit leaves
    the window, the trend ties them while the bandit still tells them apart."""

    tested = tuple(600.0 * step for step in range(12))
    settled = (0.0,) * SETTLED_TESTS + tested
    early = branch("early", [(600.0, 2.0), (6600.0, 5.0)], test_times=tested, ended_times=settled)
    never = branch("never", [], test_times=tested, ended_times=settled)
    unlike = TaskConditions(labels={"model": f"model-{n}"})
    return dataclasses.replace(tree(f"sess-{n}", early, never, task=unlike), budget_seconds=2 * HOUR)


def test_bandit_rv_may_be_advice_and_the_ucb_rows_never_are() -> None:
    sessions = tuple(_mature(n) for n in range(3))
    baselines = bandit_baselines(sessions, explorations=(0.0,))

    report = calibrate(sessions, variations=(Variation(CURRENT),), baselines=baselines)
    current, decayed, upper = report.rows
    (compared,) = report.comparisons

    assert (decayed.label, upper.label) == (BANDIT_LABEL, "bandit UCB, C=0")
    # The trend reaches back over the branches' last four ended tests, so it tells them apart in the three rounds
    # after the early commit, and ties them in the other seven.
    assert (current.rank_correlation, current.compared_rounds) == (pytest.approx(1.0), 9)
    assert (compared.label, compared.rounds, compared.sessions) == (BANDIT_LABEL, 30, 3)
    assert compared.rank_correlation == pytest.approx(1.0) and compared.current_correlation == pytest.approx(0.3)
    assert report.recommended == decayed and report.note.startswith(f"Consider '{BANDIT_LABEL}'")


def test_bandit_rv_is_the_recent_gain_per_test_in_points_and_needs_the_sessions_tree() -> None:
    tested = (0.0, 600.0, 1200.0)
    worked = branch("s1", [(650.0, 2.0)], test_times=tested)
    searched = tree("a", worked)
    context = ScoringContext(TASK, (), FrontierPolicy("m"), searched)

    estimate = BanditScorer().estimate(worked, 1200.0, context, Readings())

    mean = decayed_mean(visit_rewards(searched, worked))
    assert estimate.expected_gain_pct == pytest.approx(mean * 102.0)
    assert estimate.components == {"bandit_rv": pytest.approx(mean), "reward_scale": 1.0}
    with pytest.raises(ValueError, match="session's tree"):
        BanditScorer().estimate(worked, 1200.0, dataclasses.replace(context, tree=None), Readings())
