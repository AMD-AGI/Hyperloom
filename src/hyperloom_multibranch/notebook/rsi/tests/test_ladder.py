# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ladder a playbook climbs: the static check, the replay, the judge's decision points, the challenger sessions."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from hyperloom_multibranch.notebook.rsi.ladder import (
    BETTER,
    GAIN_PER_GPU_HOUR,
    MAX_TRIALS,
    PAIR_SECONDS,
    UNCLEAR,
    DecisionPoint,
    Difference,
    ReplayStanding,
    StandingRow,
    Trial,
    decision_points,
    moved,
    noise_band,
    pairs,
    r0,
    r1,
    r2,
    r3,
    record_standing,
    record_trial,
    replay_standing,
    session_measures,
    session_task,
    settle,
    standing_report,
    standings,
    trial_of,
    trials,
)
from hyperloom_multibranch.notebook.rsi.playbook import H0, Governs, Playbook, PlaybookPool, PlaybookRule, replaced
from hyperloom_multibranch.notebook.rsi.replay import OLD_RULE
from hyperloom_multibranch.notebook.rsi.similarity import TaskConditions, model_name
from hyperloom_multibranch.notebook.rsi.stops import stop_audit
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, branch, tree

MINUTE = 60.0
PATIENT = Playbook("patient", replaced(H0, stall_tests=8).numbers, parent=H0.label, measure="early_stops")
ALIKE = PlaybookRule(Governs.STOP, "Of two branches whose changes are alike, stop the one further behind integration.")


def _judged(*minutes: float) -> tuple[float, ...]:
    return tuple(value * MINUTE for value in minutes)


SESSION = dataclasses.replace(
    tree(
        "s",
        branch(
            "b1",
            [(20 * MINUTE, 3.0)],
            judged_times=_judged(10, 20, 30),
            closed_s=35 * MINUTE,
            stop_reason="judge: lowest expected gain",
        ),
        branch("b2", [], judged_times=_judged(10, 20, 30, 40, 50, 60, 70), closed_s=75 * MINUTE),
        branch("b3", [], opened_s=40 * MINUTE, judged_times=_judged(50, 60, 70, 80)),
    ),
    budget_seconds=4 * HOUR,
    started_at=1000.0,
)


def test_r0_is_the_static_check() -> None:
    assert r0(PATIENT, H0).passed
    found = r0(replaced(H0, stall_tests=20), H0)
    assert not found.passed and found.reason.startswith("stall_tests 20 is outside")


def test_r1_replays_a_number_change_both_ways_and_leaves_a_rule_change_to_r2() -> None:
    ruled = Playbook("ruled", rules=(ALIKE,), parent=H0.label)

    replayed = r1(PATIENT, H0, (SESSION,), runs=3)
    skipped = r1(ruled, H0, (SESSION,), runs=3)

    assert replayed.rung == "R1" and "optimistic" in replayed.reason and "pessimistic" in replayed.reason
    assert skipped.skipped and skipped.passed and skipped.reason.startswith("a rule change")


def _call(minutes: float, *stopped: str) -> dict[str, object]:
    reply = json.dumps({"stop": [{"branch": name, "reason": "lowest"} for name in stopped]})
    return {"at": 1000.0 + minutes * MINUTE, "prompt": f"view at {minutes:g} min", "reply": reply}


def test_the_decision_points_are_the_judges_stops_whose_outcome_is_known() -> None:
    calls = [_call(35, "b1"), _call(75, "b2"), _call(60, "b3"), {"at": "late", "prompt": 1, "reply": "{}"}]

    points = decision_points(SESSION, calls, bar=6)

    # b1: one test past its commit, a stop to undo; b2: seven tests stalled, a stop to keep; b3: two, too few to say.
    assert [(point.branch, point.stalled, point.keep) for point in points] == [("b1", 1, True), ("b2", 7, False)]
    assert points[0].prompt == "view at 35 min" and points[0].at == 35 * MINUTE


async def test_r2_counts_the_points_each_playbook_gets_right_asked_through_the_judge() -> None:
    undo = DecisionPoint("s", "b1", 60.0, 1, True, "view 1")
    keep = DecisionPoint("s", "b2", 120.0, 7, False, "view 2")
    asked: list[str] = []

    async def judge(system: str, prompt: str) -> str:
        asked.append(prompt)
        stops = [] if "Scheduling playbook patient" in prompt else [{"branch": "b1"}, {"branch": "b2"}]
        return json.dumps({"stop": stops})

    tie = await r2(PATIENT, H0, (undo, keep), judge, "the judge's system prompt")
    ahead = await r2(PATIENT, H0, (undo, undo, keep), judge, "the judge's system prompt")
    nothing = await r2(PATIENT, H0, (), judge, "system")

    assert (tie.passed, tie.reason) == (False, "right on 1 of 2 points, h0 on 1")
    assert ahead.passed and ahead.reason == "right on 2 of 3 points, h0 on 1"
    assert nothing.skipped and len(asked) == 10
    assert asked[0].startswith(
        "view 1\n\nThe scheduling playbook to follow in this round:\nScheduling playbook patient"
    )


TASK_KEY = "workload=inferencex model=m budget=240m"
REPEATS = ((3, 1.0), (4, 1.2), (5, 1.4))


def _trial(
    session: str,
    played: str,
    early: float,
    rate: float,
    *,
    at: float = 0.0,
    task: str = TASK_KEY,
    fell_back: bool = False,
    champion: str = H0.label,
) -> Trial:
    return Trial(session, played, champion, task, at, fell_back, {"early_stops": early, GAIN_PER_GPU_HOUR: rate})


def _repeats() -> list[Trial]:
    return [_trial(f"c{n}", H0.label, early, rate, at=n * HOUR) for n, (early, rate) in enumerate(REPEATS)]


def test_r3_crowns_a_playbook_whose_paired_sessions_beat_the_champions_by_more_than_their_noise(
    tmp_path: Path,
) -> None:
    pool = PlaybookPool(tmp_path)
    pool.register(PATIENT)
    for trial in (*_repeats(), *(_trial(f"p{n}", PATIENT.label, 1.0, 1.2, at=n * HOUR + 60) for n in range(2))):
        assert record_trial(pool, trial)
    assert not record_trial(pool, _repeats()[0])

    early = r3(PATIENT, H0, trials(pool))
    record_trial(pool, _trial("p2", PATIENT.label, 1.0, 1.2, at=2 * HOUR + 60))
    ready = r3(PATIENT, H0, trials(pool))
    done = settle(pool, 50.0)

    assert early.skipped and early.reason == "2 of its 3 paired sessions so far"
    assert ready.passed
    assert ready.reason == "early_stops better by 3 over 3 paired sessions, band 1; gain per GPU hour +0, band 0.2"
    assert done == (f"crowned patient: {ready.reason}",) and pool.champion() == PATIENT


def test_a_challenger_session_pairs_once_with_the_nearest_champion_session_of_its_task() -> None:
    kept = [
        _trial("c0", H0.label, 3, 1.0, at=0.0),
        _trial("c1", H0.label, 4, 1.0, at=10 * HOUR),
        _trial("x0", H0.label, 4, 1.0, at=HOUR, task="workload=agentx model=m budget=240m"),
        _trial("p0", PATIENT.label, 1, 1.0, at=HOUR),
        _trial("p1", PATIENT.label, 1, 1.0, at=2 * HOUR),
        _trial("p2", PATIENT.label, 1, 1.0, at=3 * HOUR, fell_back=True),
        _trial("p3", PATIENT.label, 1, 1.0, at=4 * HOUR, champion="older"),
        _trial("p4", PATIENT.label, 1, 1.0, at=10 * HOUR + PAIR_SECONDS + 1),
    ]

    found = pairs(PATIENT.label, H0.label, kept)

    assert [(pair.challenger.session_id, pair.champion.session_id) for pair in found] == [("p0", "c0"), ("p1", "c1")]
    assert [pair.difference("early_stops") for pair in found] == [2.0, 3.0]
    assert (
        r3(PATIENT, H0, kept).reason == "2 of its 3 paired sessions so far; 1 of its sessions fell back to the champion"
    )


def test_the_noise_band_is_the_spread_within_a_task_pooled_over_the_tasks() -> None:
    one = [_trial(f"a{n}", H0.label, value, 1.0, task="a") for n, value in enumerate((3, 4, 5))]
    two = [_trial(f"b{n}", H0.label, value, 1.0, task="b") for n, value in enumerate((10, 12))]

    assert noise_band(one[:2], "early_stops") is None
    assert noise_band(one, "early_stops") == pytest.approx(1.0)
    assert noise_band([*one, *two], "early_stops") == pytest.approx(math.sqrt((2 * 1.0 + 1 * 2.0) / 3))
    assert noise_band(one, "stop_value") is None


def test_a_playbook_retires_once_its_sessions_ran_out_without_beating_the_champion(tmp_path: Path) -> None:
    pool = PlaybookPool(tmp_path)
    pool.register(PATIENT)
    for trial in _repeats():
        record_trial(pool, trial)
    for n in range(MAX_TRIALS):
        record_trial(pool, _trial(f"p{n}", PATIENT.label, 4.0, 1.2, at=n * HOUR + 60))

    (done,) = settle(pool, 50.0)

    assert done.startswith("retired patient: early_stops better by 0 over 3 paired sessions") and pool.on_trial() == ()


def test_a_sessions_task_is_its_workload_model_image_sizes_and_budget_and_its_trial_carries_it() -> None:
    model = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    task = TaskConditions(
        labels={
            "model": model,
            "gpu": "MI355X",
            "framework": "sglang",
            "framework_version": "0.5.4",
            "model_type": "x",
        },
        sizes={"tp": 2.0, "conc": 64.0, "isl": 1024.0, "osl": 1024.0},
    )
    played = dataclasses.replace(SESSION, task=task)

    trial = trial_of(played, stop_audit(played), PATIENT.label, H0.label, fell_back=True)

    assert session_task(played) == (
        f"workload=inferencex model={model_name(model)} gpu=mi355x framework=sglang framework_version=0.5.4 "
        "tp=2 conc=64 isl=1024 osl=1024 budget=240m"
    )
    assert (trial.task, trial.started_at, trial.fell_back) == (session_task(played), 1000.0, True)


def test_the_replay_standing_reads_every_playbook_against_the_champion_and_keeps_once(tmp_path: Path) -> None:
    pool = PlaybookPool(tmp_path)
    pool.register(PATIENT)

    standing, comparisons = replay_standing((SESSION,), pool, session_id="s", at=10.0, runs=2)

    labels = {row.label for row in standing.rows}
    assert (standing.champion, standing.sessions, standing.runs, len(comparisons)) == (H0.label, 1, 2, 2)
    assert {OLD_RULE, "stall-4", PATIENT.label, "archive"} <= labels and H0.label not in labels
    assert record_standing(pool, standing) and not record_standing(pool, standing)
    assert standings(pool) == (standing,)


def test_what_moved_since_the_last_close_is_read_against_the_same_champion() -> None:
    sure = Difference(1.0, 0.5, 1.5)
    first = ReplayStanding("s", 10.0, H0.label, 1, 2, (StandingRow(OLD_RULE, sure, Difference(1.0, -0.5, 2.0)),))
    second = dataclasses.replace(
        first, session_id="t", at=20.0, rows=(StandingRow(OLD_RULE, sure, Difference(1.0, 0.2, 2.0)),)
    )

    report = standing_report(second, (), first)

    assert (first.readings(), second.readings()) == ({OLD_RULE: UNCLEAR}, {OLD_RULE: BETTER})
    assert moved(first, second) == (f"{OLD_RULE}: unclear -> better",)
    assert moved(dataclasses.replace(first, champion="other"), second) == () == moved(None, second)
    assert f"- better both ways: {OLD_RULE}" in report and "- worse both ways: none" in report
    assert f"Since the CLOSE of s: {OLD_RULE}: unclear -> better." in report
    assert "No earlier CLOSE read the replay against h0." in standing_report(second, (), None)


def test_a_sessions_measures_are_its_decisions_counted() -> None:
    measures = session_measures(SESSION, stop_audit(SESSION))

    assert measures["early_stops"] == 1.0 and measures["opened_committed"] == pytest.approx(1 / 3)
    assert measures["first_commit_minutes"] == 20.0 and "probation_commits" not in measures
    assert measures[GAIN_PER_GPU_HOUR] == pytest.approx(3.0 / 12.0)
