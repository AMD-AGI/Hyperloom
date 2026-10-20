# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Calibrating the frontier scores on archived sessions: how well each policy ranked the gains that followed."""

from __future__ import annotations

import dataclasses

import pytest

from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.notebook.rsi.archive import SearchTree, session_start, tree_from_snapshot
from hyperloom_multibranch.notebook.rsi.calibration import (
    CURRENT,
    LOGGED,
    Variation,
    calibrate,
    gain_after,
    rank_correlation,
    score_archived,
    session_policy,
)
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy, score_progress
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, branch, tree
from hyperloom_multibranch.notebook.tests.sessions import METRIC

BUDGET = 2 * HOUR
# A commit of +0.2% every ten minutes: under the new-branch floor over any 15-minute window.
STEADY = [(600.0 * step, 0.2 * step) for step in range(1, 13)]
NO_FLOOR = Variation("no new-branch floor", new_branch_floor_pct=0.0)


def _session(n: int) -> SearchTree:
    fresh = branch("fresh", [], opened_s=5400.0)
    unlike = TaskConditions(labels={"model": f"model-{n}"})
    return dataclasses.replace(tree(f"sess-{n}", branch("steady", STEADY), fresh, task=unlike), budget_seconds=BUDGET)


SESSIONS = tuple(_session(n) for n in range(7))


def _tied(n: int) -> SearchTree:
    """Two branches alike until one commits at 90 minutes: before, their scores tie while their later gains differ;
    after, their gains tie."""

    unlike = TaskConditions(labels={"model": f"tied-{n}"})
    branches = (branch("a", [(5400.0, 2.0)]), branch("b", []))
    return dataclasses.replace(tree(f"tied-{n}", *branches, task=unlike), budget_seconds=BUDGET)


def test_an_archived_branch_scores_as_the_live_scoring_scored_it(snapshot: NotebookSnapshot) -> None:
    policy = FrontierPolicy(METRIC)
    live = {row.branch: row.frontier for row in score_progress(snapshot, (), policy)}["s1"]
    searched = tree_from_snapshot(snapshot, METRIC)
    s1 = next(item for item in searched.branches if item.name == "s1")

    archived = score_archived(searched, s1, snapshot.taken_at - session_start(snapshot), (), policy)

    assert live is not None and archived.basis is live.basis
    assert archived.expected_gain_pct == pytest.approx(live.expected_gain_pct)
    assert archived.components == pytest.approx(live.components)


def test_a_variation_that_ranks_the_gains_better_is_recommended() -> None:
    report = calibrate(SESSIONS, variations=(Variation(CURRENT), NO_FLOOR))
    current, unfloored = report.rows

    # Two of each session's three rounds rank the fresh branch, floored at 0.5%, above the steady one.
    assert (current.rounds, current.sessions) == (21, 7)
    assert current.rank_correlation == pytest.approx(-1 / 3) and current.top_hit_rate == pytest.approx(1 / 3)
    assert unfloored.rank_correlation == pytest.approx(1.0) and unfloored.top_hit_rate == pytest.approx(1.0)
    assert report.recommended == unfloored and report.note.startswith("Consider 'no new-branch floor'")


def test_no_variation_better_by_enough_keeps_the_current_policy() -> None:
    report = calibrate(SESSIONS, variations=(dataclasses.replace(NO_FLOOR, label=CURRENT), Variation("floored")))

    assert report.recommended is None and report.note.startswith("Keep the current policy")


def test_rounds_whose_scores_and_gains_do_not_both_differ_are_no_evidence_for_advice() -> None:
    # sess-0 has 3 rounds that compare its two branches; each tied session has 11 rounds that compare none, 8 of them
    # with later gains that differ.
    report = calibrate((SESSIONS[0], *(_tied(n) for n in range(6))), variations=(Variation(CURRENT), NO_FLOOR))
    current = report.rows[0]

    assert (current.rounds, current.compared_rounds, current.sessions) == (69, 3, 7)
    assert report.recommended is None and report.note.startswith("Too little evidence")


def test_too_few_sessions_or_rounds_give_no_advice() -> None:
    report = calibrate(SESSIONS[:2])

    assert report.recommended is None and report.note.startswith("Too little evidence")
    assert [row.label for row in report.rows][:2] == [CURRENT, "window x0.5"]


def test_the_scores_a_session_logged_are_set_beside_the_gains_its_tree_went_on_to_make() -> None:
    line = {
        "t_s": 5400.0,
        "branches": [
            {"branch": "steady", "frontier": {"expected_gain_pct": 0.4}},
            {"branch": "fresh", "frontier": {"expected_gain_pct": 0.5}},
            {"branch": "gone", "frontier": {"expected_gain_pct": 9.0}},
        ],
    }

    report = calibrate(SESSIONS, variations=(Variation(CURRENT),), logs={"sess-0": [line], "other": [line]})
    logged = report.rows[-1]

    assert (logged.label, logged.rank_correlation, logged.rounds, logged.sessions) == (LOGGED, -1.0, 1, 1)
    assert report.recommended is None


def test_the_gain_that_followed_runs_to_the_stop_or_the_horizon() -> None:
    steady = SESSIONS[0].branches[1]

    assert gain_after(SESSIONS[0], steady, 5400.0) == pytest.approx(0.6)
    assert gain_after(SESSIONS[0], steady, 5400.0, 600.0) == pytest.approx(0.2)
    assert session_policy(SESSIONS[0]).window_seconds == 900.0
    assert session_policy(dataclasses.replace(SESSIONS[0], longest_root_test_seconds=600.0)).window_seconds == 2400.0


def test_rank_correlation_ranks_ties_alike_and_needs_a_spread() -> None:
    assert rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert rank_correlation([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.8660254)
    assert rank_correlation([1.0, 1.0], [1.0, 2.0]) is None


def test_the_report_renders_a_table_and_the_advice() -> None:
    text = calibrate(SESSIONS[:2], variations=(Variation(CURRENT),)).render()

    assert text.splitlines()[0] == "Frontier calibration, against the gains that followed until each branch stopped:"
    assert "| policy | rank correlation | top hit rate | rounds | compared | sessions |" in text
    assert "| current | -0.33 | 0.33 | 6 | 6 | 2 |" in text and text.endswith(
        "2 sessions and 6 rounds whose scores and later gains both differ, of 6 with two or more branches open; "
        "3 and 20 needed."
    )
