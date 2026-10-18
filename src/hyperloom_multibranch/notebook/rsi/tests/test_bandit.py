# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AdaEvolve's island bandit on archived branches, as a baseline beside the frontier score."""

from __future__ import annotations

import dataclasses
import math

import pytest

from hyperloom_multibranch.notebook.rsi.archive import SearchTree
from hyperloom_multibranch.notebook.rsi.bandit import (
    bandit_baselines,
    decayed_mean,
    reward_scale,
    ucb,
    visit_rewards,
)
from hyperloom_multibranch.notebook.rsi.calibration import CURRENT, Variation, calibrate
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, branch, tree

STEADY = [(600.0 * step, 0.2 * step) for step in range(1, 13)]


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


def _session(n: int) -> SearchTree:
    tested = tuple(600.0 * step - 60.0 for step in range(1, 13))
    steady = branch("steady", STEADY, test_times=tested)
    fresh = branch("fresh", [], opened_s=5400.0, test_times=(5400.0, 6000.0, 6600.0))
    unlike = TaskConditions(labels={"model": f"model-{n}"})
    return dataclasses.replace(tree(f"sess-{n}", steady, fresh, task=unlike), budget_seconds=2 * HOUR)


def test_the_decayed_mean_may_be_advice_and_the_ucb_rows_never_are() -> None:
    sessions = tuple(_session(n) for n in range(7))
    baselines = bandit_baselines(sessions, explorations=(0.0,))

    report = calibrate(sessions, variations=(Variation(CURRENT),), baselines=baselines)
    current, decayed, upper = report.rows

    assert (decayed.label, upper.label) == ("bandit R/V, decay 0.95", "bandit UCB, C=0")
    assert current.rank_correlation == pytest.approx(-1 / 3)
    assert decayed.rank_correlation == pytest.approx(1.0) and upper.rank_correlation == pytest.approx(1.0)
    assert report.recommended == decayed and report.note.startswith("Consider 'bandit R/V, decay 0.95'")
