# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The stop audit: how long each stopped branch had stalled, what its probations came to, what the stops were worth."""

from __future__ import annotations

import dataclasses

import pytest

from hyperloom_multibranch.contracts.blackboard.vocab import CommitOrigin
from hyperloom_multibranch.blackboard.rsi.archive import TreeCommit, TreeProbation
from hyperloom_multibranch.blackboard.rsi.history import MAX_POSITION, Odds, StallOdds
from hyperloom_multibranch.blackboard.rsi.stops import EARLY_STOP_TESTS, stall_study, stop_audit
from hyperloom_multibranch.blackboard.rsi.tests.trees import TASK, branch, tree

MINUTE = 60.0
# Every test commits one time in ten past a stall, and a new branch's one time in five: p(k) is 34% and q 59%.
ODDS = StallOdds(
    1,
    1,
    hazard={position: Odds(1.0, 10.0) for position in range(1, MAX_POSITION + 1)},
    new_branch={position: Odds(2.0, 10.0) for position in range(1, 5)},
)


def _minutes(*values: float) -> tuple[float, ...]:
    return tuple(value * MINUTE for value in values)


def _commits(*times: float) -> list[tuple[float, float]]:
    return [(when * MINUTE, 1.0 + n) for n, when in enumerate(times)]


STALLED = branch(
    "b1",
    _commits(30, 70),
    judged_times=_minutes(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120),
    closed_s=125 * MINUTE,
    stop_reason="judge: lowest expected gain",
)
EARLY = branch(
    "b2",
    _commits(40),
    judged_times=_minutes(10, 20, 30, 40, 50),
    closed_s=52 * MINUTE,
    stop_reason="judge: behind integration",
)
HELD = branch(
    "b3",
    [],
    judged_times=_minutes(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    closed_s=105 * MINUTE,
    stop_reason="judge: stalled",
    probations=(TreeProbation(65 * MINUTE, 4),),
)
PAID = branch(
    "b4", _commits(40), judged_times=_minutes(10, 20, 30, 40, 50), probations=(TreeProbation(25 * MINUTE, 4),)
)
ENDED = branch("b5", [], judged_times=_minutes(10), closed_s=240 * MINUTE, stop_reason="session_end: time_exhausted")
SESSION = tree("s", STALLED, EARLY, HELD, PAID, ENDED)


def test_a_stop_counts_the_judged_tests_and_the_minutes_since_the_branchs_last_commit() -> None:
    stops = {stop.branch: stop for stop in stop_audit(SESSION).stops}

    stalled = stops["b1"]
    assert (stalled.cause, stalled.reason) == ("judge", "lowest expected gain")
    assert (stalled.stalled_tests, stalled.stalled_min, stalled.commits) == (5, 55.0, 2)
    assert (stops["b3"].stalled_tests, stops["b3"].stalled_min, stops["b3"].commits) == (10, 105.0, 0)
    assert stops["b5"].cause == "session_end" and not stops["b5"].by_judge
    assert list(stops) == ["b2", "b3", "b1", "b5"]


def test_a_commit_whose_merge_failed_does_not_end_the_stall() -> None:
    failed = (STALLED.commits[0], dataclasses.replace(STALLED.commits[1], merge_failed=True))
    (stop,) = stop_audit(tree("s", dataclasses.replace(STALLED, commits=failed))).stops

    assert (stop.stalled_tests, stop.stalled_min, stop.commits) == (9, 95.0, 1)


def test_a_stop_at_most_two_tests_past_a_commit_is_early() -> None:
    audit = stop_audit(SESSION)
    stops = {stop.branch: stop for stop in audit.stops}

    assert stops["b2"].stalled_tests == 1 <= EARLY_STOP_TESTS and stops["b2"].early
    assert not stops["b1"].early and not stops["b3"].early
    assert (audit.early_stops, audit.median_stall) == (1, 5)


def test_a_probation_says_how_many_of_its_tests_ran_and_whether_a_commit_came_within_them() -> None:
    audit = stop_audit(SESSION)
    held, paid = (next(item for item in audit.probations if item.branch == name) for name in ("b3", "b4"))
    cut = stop_audit(tree("s", dataclasses.replace(HELD, closed_s=85 * MINUTE))).probations[0]
    split = TreeCommit(40 * MINUTE, 1.0, CommitOrigin.SPLIT)
    unowned = stop_audit(tree("s", dataclasses.replace(PAID, commits=(split,)))).probations[0]

    assert (held.started_min, held.tests, held.ran, held.committed, held.stopped) == (65.0, 4, 4, False, False)
    assert (paid.ran, paid.committed, paid.stopped) == (3, True, False)
    assert (cut.ran, cut.stopped) == (4, True)
    assert not unowned.committed
    assert {stop.branch: stop.probation for stop in audit.stops}["b3"] == held


def test_the_stops_are_worth_q_less_p_of_k_over_the_judges_stops() -> None:
    audit = stop_audit(SESSION, ODDS)
    kept, new = 1 - 0.9**4, 1 - 0.8**4

    assert audit.new_branch_odds == pytest.approx(new)
    assert all(stop.kept_odds == pytest.approx(kept) for stop in audit.stops)
    assert audit.stop_value == pytest.approx(3 * (new - kept))
    assert stop_audit(SESSION).stop_value is None
    assert audit.counts() == {
        "stops": 4,
        "judge_stops": 3,
        "early_stops": 1,
        "below_bar": 2,
        "probations": 2,
        "probations_committed": 1,
    }
    assert stop_audit(SESSION).below_bar == 0


def test_the_audit_reads_as_markdown_with_a_row_for_each_stop_and_each_probation() -> None:
    audit = stop_audit(SESSION, ODDS)
    text = audit.render()

    assert audit.note() == (
        "the judge stopped 3 branches at a median of 5 judged tests past their last commit, 1 of them at most 2 past "
        "one and 2 before the stop bar of 6; 1 of 2 probations committed; the stops were worth +0.74 commits by the "
        "stall odds"
    )
    assert text.startswith("# Stop audit of s\n\nThe judge stopped 3 branches")
    assert "| b1 | 125 | judge | 5 | 55 | 2 | 34% | n/a | lowest expected gain |" in text
    assert "| b3 | 105 | judge | 10 | 105 | 0 | 34% | 4/4, none | stalled |" in text
    assert "| b4 | 25 | 4 | 3 | yes | no |" in text
    assert stop_audit(tree("s", PAID)).note() == "the judge stopped no branch"


def test_the_study_gives_the_odds_of_each_kind_of_session_and_every_sessions_audit() -> None:
    moe = dataclasses.replace(TASK, labels={**TASK.labels, "model_type": "qwen3_next"})
    study = stall_study((SESSION, dataclasses.replace(tree("t", STALLED), task=moe)))
    text = study.render()

    assert list(study.odds) == ["any model type, inferencex", "qwen3_next, inferencex"]
    assert [audit.session_id for audit in study.audits] == ["s", "t"]
    assert "Over the archive the judge stopped 4 branches at a median of 5.0 judged tests" in text
    assert "| s | 3 | 5.0 | 1 | 2 | 50% (1/2) |" in text
