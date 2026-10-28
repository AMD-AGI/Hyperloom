# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduling replay: an archived session under a playbook, cut where it stops earlier and sampled where later."""

from __future__ import annotations

import dataclasses
import json

import pytest

from hyperloom_multibranch.blackboard.rsi.archive import SearchTree, TreeBranch
from hyperloom_multibranch.blackboard.rsi.history import MAX_POSITION
from hyperloom_multibranch.blackboard.rsi.playbook import H0, replaced
from hyperloom_multibranch.blackboard.rsi.replay import (
    ARCHIVE,
    OPTIMISTIC,
    PESSIMISTIC,
    STOP_REPLY,
    Draws,
    Gains,
    Pace,
    archive_cap,
    check,
    compare,
    cut_check,
    pace_for,
    simulate,
    stops_in,
    variants,
)
from hyperloom_multibranch.blackboard.rsi.tests.trees import HOUR, TASK, branch, tree

MINUTE = 60.0
#: Every commit adds 2 points.
TWO = Gains.of([(0.0, 2.0)])
NEVER = Pace((0.0,) * MAX_POSITION, (0.0,) * MAX_POSITION, (0.0,) * MAX_POSITION, TWO, 6, 40 * MINUTE)
ALWAYS = Pace((1.0,) * MAX_POSITION, (1.0,) * MAX_POSITION, (1.0,) * MAX_POSITION, TWO, 6, 40 * MINUTE)
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


def _past_the_bar(prompt: str) -> str:
    """A judge that stops every branch its view shows past the stop bar."""

    lines = [line[2:] for line in prompt.splitlines() if line.startswith("- b")]
    named = [line.split(":")[0] for line in lines if not line.endswith(("grace", "below the stop bar"))]
    return json.dumps({"stop": [{"branch": name, "reason": "stalled"} for name in named]})


def test_a_judged_replay_asks_the_judge_every_round_and_refuses_the_stop_of_a_branch_on_probation() -> None:
    asked: list[str] = []

    def never(prompt: str) -> str:
        asked.append(prompt)
        return "no branch to stop"

    kept = simulate(_tree(STALLED), STRICT, NEVER, drawn=Draws(decide=never))
    stopped = simulate(_tree(STALLED), STRICT, NEVER, drawn=Draws(decide=_past_the_bar))
    held = simulate(
        _tree(STALLED),
        replaced(STRICT, probation_share=1.0, probation_tests=8),
        NEVER,
        drawn=Draws(decide=_past_the_bar),
    )

    assert kept.stops == 0 and kept.gpu_hours == pytest.approx(24.0)
    assert asked[0].startswith(
        "Session clock 10 of 240 minutes; the best gain so far +0.00%.\nRunning branches:\n- b1: "
    )
    assert "in its grace" in asked[0] and "Scheduling playbook" in asked[0] and asked[0].endswith(STOP_REPLY)
    # Its stall reaches the bar of four with its test at 60 min; the round at 70 min stops it before its seventh.
    assert (stopped.stops, stopped.best_gain_pct, stopped.gpu_hours) == (1, 3.0, pytest.approx(6.0))
    assert held.stops == 1 and held.gpu_hours > stopped.gpu_hours


def test_a_judges_reply_names_the_branches_to_stop_in_its_json_object() -> None:
    assert stops_in('Stop these: {"stop": [{"branch": "b1", "reason": "stalled"}, "b2"]} done') == ("b1",)
    assert stops_in("{not json}") == () == stops_in('["b1"]') == stops_in('{"stop": null}')


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
    half = (0.5,) * MAX_POSITION
    coin = dataclasses.replace(ALWAYS, hazard=half, new_branch=half, opening_branch=half)
    session = _tree(EARLY, budget=10 * HOUR)

    runs = [simulate(session, STRICT, coin, run=run).best_gain_pct for run in range(8)]

    assert runs == [simulate(session, STRICT, coin, run=run).best_gain_pct for run in range(8)]
    assert len(set(runs)) > 1


def test_the_pace_of_a_session_is_read_off_the_others_and_held_or_dropped_beyond_their_data() -> None:
    # Ten sessions without a model type, each weighing half: five observations a position, enough to read. Each opens
    # one branch with the session, which commits at its 2nd test, and one an hour in, which commits at its 3rd.
    opening = branch("b1", [(1200.0, 2.0)], judged_times=_judged(10, 20, 30))
    later = branch("b2", [(90 * MINUTE, 2.0)], opened_s=HOUR, judged_times=_judged(70, 80, 90))
    others = [tree(f"o{n}", opening, later, task=TASK) for n in range(10)]
    own = tree("s", branch("b1", [(600.0, 9.0)], judged_times=_judged(10)), task=TASK)

    held, dropped = pace_for(own, (own, *others), OPTIMISTIC), pace_for(own, (own, *others), PESSIMISTIC)

    steps = [(0.0, 2.0)] * 20
    assert held.gains == Gains.of(steps, OPTIMISTIC) and dropped.gains == Gains.of(steps, PESSIMISTIC)
    assert held.hazard[0] == 0.0 and held.opening_branch[1] == 1.0 and held.new_branch[:3] == (0.0, 0.0, 1.0)
    assert held.new_branch[3:] == (1.0,) * (MAX_POSITION - 3) and dropped.new_branch[3:] == (0.0,) * (MAX_POSITION - 3)
    assert held.first_commit_seconds == 30 * MINUTE


def test_a_commit_draws_its_gain_from_the_commits_that_started_at_least_as_high() -> None:
    # Twelve commits starting from 0, 10, ..., 110 points, adding 1, 2, ..., 12.
    steps = [(10.0 * n, n + 1.0) for n in range(12)]
    held, dropped = Gains.of(steps, OPTIMISTIC), Gains.of(steps, PESSIMISTIC)

    assert (held.drawn(0.0, 0.0), held.drawn(0.0, 0.99)) == (1.0, 12.0)
    # Nine started at 25 or higher, fewer than ten: optimism reads the ten that started highest.
    assert (held.drawn(25.0, 0.0), dropped.drawn(25.0, 0.0)) == (3.0, 4.0)
    assert (held.drawn(115.0, 0.0), dropped.drawn(115.0, 0.0)) == (3.0, None)
    assert Gains().drawn(5.0, 0.5) is None


def test_a_cut_ranks_the_real_outcome_among_replays_that_sample_every_outcome_after_it() -> None:
    session = _tree(STALLED, EARLY)

    after_all = cut_check(session, ALWAYS, 4 * HOUR, runs=5)
    lucky, unlucky = cut_check(session, ALWAYS, 15 * MINUTE, runs=5), cut_check(session, NEVER, 15 * MINUTE, runs=5)

    assert (after_all.real_gain_pct, after_all.median_gain_pct, after_all.rank) == (3.0, 3.0, 0.5)
    assert lucky.rank == 0.0 and lucky.median_gain_pct > 3.0
    # Before 15 min only b2 committed, +1.0: without b1's +3.0 every replay falls short of the real outcome.
    assert (unlucky.median_gain_pct, unlucky.rank) == (1.0, 1.0)


def test_the_checks_say_which_sessions_the_archive_replay_reproduces_and_where_each_cut_falls() -> None:
    sessions = [_tree(STALLED, EARLY), dataclasses.replace(_tree(STALLED), session_id="t")]

    found = check(sessions, runs=3, draws=20)

    assert found.reproduced == ("s", "t") and found.not_reproduced == ()
    assert [item.session_id for item in found.cuts] == ["s", "t"] and found.mean_rank is not None
    assert set(found.agreeing) | set(found.disagreeing) == set(variants()) - {"h0"}
    assert found.render().startswith("Following the archive reproduces 2 of 2 sessions.")


def test_a_comparison_reads_each_playbook_beside_the_archive_session_by_session() -> None:
    sessions = [_tree(STALLED, EARLY), dataclasses.replace(_tree(STALLED), session_id="t")]

    found = compare(sessions, {"strict": STRICT}, drawn=Draws(runs=3, draws=50))
    rows = {row.label: row for row in found.rows}

    against = compare(sessions, {"h0": H0, "strict": STRICT}, drawn=Draws(runs=3, draws=50))

    assert list(rows) == [ARCHIVE, "strict"] and found.sessions == ("s", "t") and found.baseline == ARCHIVE
    assert (rows[ARCHIVE].gain_difference, rows[ARCHIVE].difference_low) == (0.0, 0.0)
    assert rows["strict"].gpu_hours < rows[ARCHIVE].gpu_hours
    assert rows["strict"].difference_low <= rows["strict"].gain_difference <= rows["strict"].difference_high
    assert found.render().startswith("Scheduling replay over 2 sessions, 3 runs each, h(k) optimistic")
    assert against.baseline == "h0" and {row.label: row for row in against.rows}["h0"].gain_difference == 0.0
    assert "| playbook | best gain % | vs h0 (95%) |" in against.render()
