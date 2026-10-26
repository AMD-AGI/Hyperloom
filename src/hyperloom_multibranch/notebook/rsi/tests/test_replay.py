# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduling replay: an archived session under a playbook, cut where it stops earlier and sampled where later."""

from __future__ import annotations

import dataclasses

import pytest

from hyperloom_multibranch.notebook.rsi.archive import SearchTree, TreeBranch
from hyperloom_multibranch.notebook.rsi.history import MAX_POSITION
from hyperloom_multibranch.notebook.rsi.playbook import H0, replaced
from hyperloom_multibranch.notebook.rsi.replay import (
    ARCHIVE,
    OPTIMISTIC,
    PESSIMISTIC,
    Pace,
    archive_cap,
    compare,
    pace_for,
    simulate,
)
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, TASK, branch, tree

MINUTE = 60.0
NEVER = Pace((0.0,) * MAX_POSITION, (0.0,) * MAX_POSITION, (2.0,), 6, 40 * MINUTE)
ALWAYS = Pace((1.0,) * MAX_POSITION, (1.0,) * MAX_POSITION, (2.0,), 6, 40 * MINUTE)
#: A bar of four tests past a commit, a grace of three tests, no probation, and no branch opened after the start.
STRICT = replaced(H0, stall_tests=4, grace_tests=3, probation_share=0.0, last_open_minutes=240.0)
#: As STRICT, opening into a freed slot up to the end.
REFILLING = replaced(STRICT, last_open_minutes=0.0)


def _judged(*minutes: float) -> tuple[float, ...]:
    return tuple(value * MINUTE for value in minutes)


def _tree(*branches: TreeBranch, budget: float = 4 * HOUR) -> SearchTree:
    return dataclasses.replace(tree("s", *branches), budget_seconds=budget)


# Commits at its 2nd test, then stalls 8 tests until the judge stops it at 100 min.
STALLED = branch(
    "b1",
    [(20 * MINUTE, 3.0)],
    judged_times=_judged(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    closed_s=100 * MINUTE,
    stop_reason="judge: lowest expected gain",
    gpu_seconds=10 * HOUR,
)
# Stopped by the judge after two stalled tests, at 30 min.
EARLY = branch(
    "b2",
    [(10 * MINUTE, 1.0)],
    judged_times=_judged(10, 20, 30),
    closed_s=30 * MINUTE,
    stop_reason="judge: lowest expected gain",
    gpu_seconds=3 * HOUR,
)


def test_following_the_archive_replays_it_exactly() -> None:
    later = branch(
        "b3", [(130 * MINUTE, 5.0)], opened_s=110 * MINUTE, judged_times=_judged(120, 130), gpu_seconds=2 * HOUR
    )
    session = _tree(STALLED, EARLY, later)

    kept = simulate(session, None, NEVER)

    assert (kept.best_gain_pct, kept.gpu_hours, kept.stops, kept.opened) == (5.0, 15.0, 2, 3)
    assert archive_cap(session) == 2


def test_a_branch_stopped_earlier_loses_what_it_did_after_and_frees_its_gpu() -> None:
    cut = simulate(_tree(STALLED), STRICT, NEVER)

    # Its stall reaches four tests at 60 min: six of its ten tests ran.
    assert (cut.best_gain_pct, cut.stops, cut.opened) == (3.0, 1, 1)
    assert cut.gpu_hours == pytest.approx(6.0)


def test_a_branch_the_judge_stopped_and_the_playbook_keeps_goes_on_with_sampled_tests() -> None:
    kept = simulate(_tree(EARLY), STRICT, ALWAYS)
    dropped = simulate(_tree(EARLY), STRICT, NEVER)
    gave_up = dataclasses.replace(EARLY, stop_reason="gave_up: nothing left to test")

    assert kept.best_gain_pct > 1.0 + 2.0 and kept.stops == 0
    assert (dropped.best_gain_pct, dropped.stops) == (1.0, 1)
    assert simulate(_tree(gave_up), STRICT, ALWAYS).best_gain_pct == 1.0


def test_a_freed_slot_opens_the_archives_next_branch_shifted_from_the_replays_integration_gain() -> None:
    late_commit = dataclasses.replace(
        STALLED, commits=(*STALLED.commits, dataclasses.replace(STALLED.commits[0], t_s=90 * MINUTE, gain_pct=6.0))
    )
    later = branch("b3", [(120 * MINUTE, 8.0)], opened_s=95 * MINUTE, judged_times=_judged(110, 120), fork_gain_pct=6.0)
    session = _tree(late_commit, later)

    replayed = simulate(session, replaced(REFILLING, max_open=1), NEVER)

    # b1 stops at 60 min, before its +6.0; b3 opens then, 3.0 below where the archive forked it, and keeps its +2.0.
    assert simulate(session, None, NEVER).best_gain_pct == 8.0
    assert replayed.best_gain_pct == pytest.approx(3.0 + 2.0) and replayed.opened >= 2


def test_once_the_archive_runs_out_a_freed_slot_opens_a_sampled_branch_but_not_near_the_end() -> None:
    refilled = simulate(_tree(EARLY), replaced(REFILLING, max_open=1), NEVER)
    late = simulate(_tree(STALLED), replaced(REFILLING, max_open=1, last_open_minutes=200.0), NEVER)

    assert refilled.opened > 1 and refilled.best_gain_pct == 1.0
    assert late.opened == 1


def test_the_same_run_meets_the_same_draws_and_another_run_its_own() -> None:
    coin = dataclasses.replace(ALWAYS, hazard=(0.5,) * MAX_POSITION, new_branch=(0.5,) * MAX_POSITION)
    session = _tree(EARLY, budget=10 * HOUR)

    runs = [simulate(session, STRICT, coin, run).best_gain_pct for run in range(8)]

    assert runs == [simulate(session, STRICT, coin, run).best_gain_pct for run in range(8)]
    assert len(set(runs)) > 1


def test_the_pace_of_a_session_is_read_off_the_others_and_held_or_dropped_beyond_their_data() -> None:
    # Ten sessions without a model type, each weighing half: five observations a position, enough to read.
    paid = branch("b1", [(1200.0, 2.0)], judged_times=_judged(10, 20, 30))
    others = [tree(f"o{n}", paid, task=TASK) for n in range(10)]
    own = tree("s", branch("b1", [(600.0, 9.0)], judged_times=_judged(10)), task=TASK)

    held, dropped = pace_for(own, (own, *others), OPTIMISTIC), pace_for(own, (own, *others), PESSIMISTIC)

    assert held.gains == dropped.gains == (2.0,) * 10
    assert held.hazard[0] == 0.0 and held.new_branch[1] == 1.0
    assert held.new_branch[2:] == (1.0,) * (MAX_POSITION - 2) and dropped.new_branch[2:] == (0.0,) * (MAX_POSITION - 2)


def test_a_comparison_reads_each_playbook_beside_the_archive_session_by_session() -> None:
    sessions = [_tree(STALLED, EARLY), dataclasses.replace(_tree(STALLED), session_id="t")]

    found = compare(sessions, {"strict": STRICT}, runs=3, draws=50)
    rows = {row.label: row for row in found.rows}

    against = compare(sessions, {"h0": H0, "strict": STRICT}, runs=3, draws=50)

    assert list(rows) == [ARCHIVE, "strict"] and found.sessions == ("s", "t") and found.baseline == ARCHIVE
    assert (rows[ARCHIVE].gain_difference, rows[ARCHIVE].difference_low) == (0.0, 0.0)
    assert rows["strict"].gpu_hours < rows[ARCHIVE].gpu_hours
    assert rows["strict"].difference_low <= rows["strict"].gain_difference <= rows["strict"].difference_high
    assert found.render().startswith("Scheduling replay over 2 sessions, 3 runs each, h(k) optimistic")
    assert against.baseline == "h0" and {row.label: row for row in against.rows}["h0"].gain_difference == 0.0
    assert "| playbook | best gain % | vs h0 (95%) |" in against.render()
