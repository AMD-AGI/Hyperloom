# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The frontier score of each running specialist branch: trend, spend, headroom, history prior, observation."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import FrontierBasis
from hyperloom_multibranch.notebook.rsi.frontier import (
    OBSERVATION_SECONDS,
    WINDOW_SECONDS,
    FrontierPolicy,
    FrontierScoring,
    remaining_prior,
    score_progress,
)
from hyperloom_multibranch.notebook.rsi.similarity import task_conditions
from hyperloom_multibranch.notebook.rsi.tests.trees import branch, tree
from hyperloom_multibranch.notebook.tests.factories import FakeClock, profile
from hyperloom_multibranch.notebook.tests.sessions import METRIC, scripted_session

POLICY = FrontierPolicy(METRIC)
TASK = task_conditions(profile())
PAYS_LATE = tree(
    "late", branch("s1", [(600.0, 2.0), (1800.0, 6.0)], test_times=(0.0, 600.0, 1200.0, 1800.0)), task=TASK
)
PAYS_SOON = tree("soon", branch("s1", [(1800.0, 3.0)], test_times=(0.0, 900.0, 1800.0)), task=TASK)


def _scores(snapshot: NotebookSnapshot, *history: object, policy: FrontierPolicy = POLICY) -> dict[str, object]:
    return {row.branch: row.frontier for row in score_progress(snapshot, tuple(history), policy)}


def test_a_running_specialist_branch_is_scored_and_the_others_are_not(snapshot: NotebookSnapshot) -> None:
    scores = _scores(snapshot)

    s1 = scores["s1"]
    assert scores["s2"] is None and scores[INTEGRATION_BRANCH] is None
    assert s1 is not None and s1.expected_gain_pct == pytest.approx(4.0)
    assert s1.basis is FrontierBasis.TREND
    assert s1.components["trend_pct"] == pytest.approx(4.0)
    assert (s1.components["tests"], s1.components["gpu_hours"], s1.components["observing"]) == (2.0, 1200 / 3600, 1.0)
    assert s1.components["headroom_pct"] == pytest.approx((1500.0 / 1040.0 - 1.0) * 100.0)
    assert "prior_remaining_pct" not in s1.components


def test_the_history_prior_replaces_the_trend_once_enough_similar_sessions_back_it(snapshot: NotebookSnapshot) -> None:
    backed = _scores(snapshot, PAYS_LATE, PAYS_SOON)["s1"]
    alone = _scores(snapshot, PAYS_LATE)["s1"]

    assert backed is not None and backed.expected_gain_pct == pytest.approx(3.5)
    assert backed.basis is FrontierBasis.PRIOR
    assert backed.components["prior_effective_sessions"] == pytest.approx(2.0)
    assert alone is not None and alone.expected_gain_pct == pytest.approx(4.0)
    assert alone.basis is FrontierBasis.TREND


def test_the_prior_counts_only_branches_that_spent_as_many_tests() -> None:
    assert remaining_prior((PAYS_LATE, PAYS_SOON), TASK, 4) == remaining_prior((PAYS_LATE,), TASK, 4)
    assert remaining_prior((PAYS_SOON,), TASK, 4) is None
    from_the_start = remaining_prior((PAYS_SOON,), TASK, 0)
    assert from_the_start is not None and from_the_start.remaining_pct == pytest.approx(3.0)


def test_the_headroom_caps_the_score_and_observation_keeps_a_floor(snapshot: NotebookSnapshot) -> None:
    tight = dataclasses.replace(snapshot, profile=profile(roofline_ceiling={METRIC: 1050.0}))
    capped = _scores(tight)["s1"]
    stalled = _scores(snapshot, policy=dataclasses.replace(POLICY, window_seconds=0.0))["s1"]
    unwatched = _scores(snapshot, policy=dataclasses.replace(POLICY, window_seconds=0.0, observation_seconds=0.0))["s1"]

    assert capped is not None and capped.expected_gain_pct == pytest.approx((1050.0 / 1040.0 - 1.0) * 100.0)
    assert capped.basis is FrontierBasis.HEADROOM
    assert stalled is not None and stalled.expected_gain_pct == POLICY.observation_floor_pct
    assert stalled.basis is FrontierBasis.OBSERVING
    assert unwatched is not None and unwatched.expected_gain_pct == 0.0
    assert unwatched.basis is FrontierBasis.TREND


def test_a_shorter_session_looks_back_and_observes_for_less_and_a_longer_one_keeps_the_defaults() -> None:
    two_hours = FrontierPolicy.for_budget(METRIC, 2 * 3600.0)
    eight_hours = FrontierPolicy.for_budget(METRIC, 8 * 3600.0)

    assert (two_hours.window_seconds, two_hours.observation_seconds) == (900.0, 1800.0)
    assert (eight_hours.window_seconds, eight_hours.observation_seconds) == (WINDOW_SECONDS, OBSERVATION_SECONDS)
    assert two_hours.metric == METRIC and two_hours.observation_floor_pct == POLICY.observation_floor_pct
    with pytest.raises(ValueError, match="positive"):
        FrontierPolicy.for_budget(METRIC, 0.0)


def test_without_a_profile_nothing_is_scored(snapshot: NotebookSnapshot) -> None:
    unscored = score_progress(dataclasses.replace(snapshot, profile=None), (), POLICY)

    assert all(row.frontier is None for row in unscored)


async def test_the_scoring_is_the_notebooks_progress_query(tmp_path: Path) -> None:
    notebook = await scripted_session(tmp_path / "notebook", FakeClock())
    scoring = FrontierScoring(notebook, POLICY, (PAYS_LATE, PAYS_SOON))

    rows = {row.branch: row for row in await scoring.progress()}

    assert rows["s1"].frontier is not None and rows["s1"].frontier.expected_gain_pct == pytest.approx(3.5)
    assert (await scoring.lineage()) == (await notebook.lineage())
