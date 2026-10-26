# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ladder a playbook climbs: the static check, the replay, the judge's decision points, the challenger sessions."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.notebook.rsi.ladder import (
    GAIN_PER_GPU_HOUR,
    MAX_TRIALS,
    DecisionPoint,
    Trial,
    decision_points,
    noise_band,
    r0,
    r1,
    r2,
    r3,
    record_trial,
    session_measures,
    settle,
    trials,
)
from hyperloom_multibranch.notebook.rsi.playbook import H0, Governs, Playbook, PlaybookPool, PlaybookRule, replaced
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


def _trial(session: str, played: str, early: float, rate: float) -> Trial:
    return Trial(session, played, H0.label, "qwen3_next", {"early_stops": early, GAIN_PER_GPU_HOUR: rate})


def test_r3_crowns_a_playbook_whose_sessions_beat_the_champions_by_more_than_their_noise(tmp_path: Path) -> None:
    pool = PlaybookPool(tmp_path)
    pool.register(PATIENT)
    champion = [
        _trial(f"c{n}", H0.label, early, rate) for n, (early, rate) in enumerate(((3, 1.0), (4, 1.2), (5, 1.4)))
    ]
    for trial in (*champion, *(_trial(f"p{n}", PATIENT.label, 1.0, 1.2) for n in range(2))):
        assert record_trial(pool, trial)
    assert not record_trial(pool, champion[0])

    early = r3(PATIENT, H0, trials(pool))
    record_trial(pool, _trial("p2", PATIENT.label, 1.0, 1.2))
    ready = r3(PATIENT, H0, trials(pool))
    done = settle(pool, 50.0)

    assert noise_band(champion, "early_stops") == pytest.approx(1.0)
    assert early.skipped and early.reason == "2 of its 3 trial sessions so far"
    assert ready.passed and ready.reason.startswith("early_stops 1 against 4 over 3 and 3 sessions, band 1")
    assert done == (f"crowned patient: {ready.reason}",) and pool.champion() == PATIENT


def test_a_playbook_retires_once_its_sessions_ran_out_without_beating_the_champion(tmp_path: Path) -> None:
    pool = PlaybookPool(tmp_path)
    pool.register(PATIENT)
    for n, (early, rate) in enumerate(((3, 1.0), (4, 1.2), (5, 1.4))):
        record_trial(pool, _trial(f"c{n}", H0.label, early, rate))
    for n in range(MAX_TRIALS):
        record_trial(pool, _trial(f"p{n}", PATIENT.label, 4.0, 1.2))

    (done,) = settle(pool, 50.0)

    assert done.startswith("retired patient: early_stops 4 against 4") and pool.on_trial() == ()


def test_a_sessions_measures_are_its_decisions_counted() -> None:
    measures = session_measures(SESSION, stop_audit(SESSION))

    assert measures["early_stops"] == 1.0 and measures["opened_committed"] == pytest.approx(1 / 3)
    assert measures["first_commit_minutes"] == 20.0 and "probation_commits" not in measures
    assert measures[GAIN_PER_GPU_HOUR] == pytest.approx(3.0 / 12.0)
