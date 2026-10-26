# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Offline generations: the LLM changes the playbook on the evolution set, and the result meets H0 held out."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from hyperloom_multibranch.notebook.rsi.generations import Evolving, generation_prompt, generations, split
from hyperloom_multibranch.notebook.rsi.ladder import JUDGE_SYSTEM_PROMPT
from hyperloom_multibranch.notebook.rsi.playbook import H0, PlaybookPool
from hyperloom_multibranch.notebook.rsi.tests.trees import HOUR, TASK, branch, tree

MINUTE = 60.0
MOE = dataclasses.replace(TASK, labels={**TASK.labels, "model_type": "qwen3_next"})


def _session(name: str, task=TASK):
    stalled = branch(
        "b1",
        [(20 * MINUTE, 3.0)],
        judged_times=tuple(value * MINUTE for value in range(10, 110, 10)),
        closed_s=100 * MINUTE,
        stop_reason="judge: lowest expected gain",
    )
    return dataclasses.replace(tree(name, stalled, task=task), budget_seconds=4 * HOUR)


SESSIONS = [_session(f"s{n}") for n in range(5)] + [_session(f"m{n}", MOE) for n in range(3)]


def test_the_held_out_sessions_hold_one_of_each_model_type_and_the_evolution_set_the_rest() -> None:
    evolution, held = split(SESSIONS, 4, seed="x")

    assert len(held) == 4 and len(evolution) == 4
    assert {tree.task.labels.get("model_type", "") for tree in held} == {"", "qwen3_next"}
    assert split(SESSIONS, 4, seed="x") == (evolution, held)
    assert len(split(SESSIONS[:3], 4)[1]) == 1


def _answers(*documents: object):
    """The author's replies in turn, and a judge that stops nothing; what the author was asked."""

    replies = iter(json.dumps(document) for document in documents)
    asked: list[str] = []

    async def complete(system: str, prompt: str) -> str:
        if system == JUDGE_SYSTEM_PROMPT:
            return '{"stop": []}'
        asked.append(prompt)
        return next(replies)

    return complete, asked


async def test_each_generation_proposes_one_change_that_the_replay_crowns_or_retires(tmp_path: Path) -> None:
    complete, asked = _answers(
        {
            "label": "early-stop",
            "change": {"number": "stall_tests", "value": 4},
            "hypothesis": "h",
            "measure": "stop_value",
        },
        {
            "label": "alike",
            "change": {"add_rule": {"governs": "stop", "text": "Stop the later of two alike branches."}},
            "hypothesis": "h",
            "measure": "early_stops",
        },
        {"none": "nothing else to try"},
    )

    report = await generations(SESSIONS, complete, tmp_path, Evolving(model="m", count=3, runs=2))

    first, second, third = report.generations
    assert (first.proposed, first.changed) == ("early-stop", "stall_tests")
    assert "R1 best gain against h0" in first.verdict
    assert (second.proposed, second.crowned) == ("alike", False)
    assert (
        "R2 no archived decision point with a known outcome; judged replay of 3 sessions, 5 runs each: best gain "
        "+0.00 points against"
    ) in second.verdict
    assert (third.proposed, third.verdict) == ("", "the LLM proposed no change: nothing else to try")
    assert report.champion.label == ("early-stop" if first.crowned else H0.label)
    assert len(report.held_out) == 4 and "retired: R0" in asked[2]
    assert report.render().startswith("Offline generations: 4 evolution sessions, held out ")


async def test_a_rule_change_is_asked_at_the_archived_decision_points_before_its_judged_replay(tmp_path: Path) -> None:
    complete, _ = _answers(
        {
            "label": "alike",
            "change": {"add_rule": {"governs": "stop", "text": "Stop the later of two alike branches."}},
            "hypothesis": "h",
            "measure": "early_stops",
        }
    )
    stop = json.dumps({"stop": [{"branch": "b1", "reason": "lowest"}]})
    calls = [{"at": minutes * MINUTE, "prompt": f"view at {minutes} min", "reply": stop} for minutes in (30, 100)]
    decisions = {tree.session_id: calls for tree in SESSIONS}

    report = await generations(SESSIONS, complete, tmp_path, Evolving(model="m", count=1, runs=2), decisions=decisions)

    # Each evolution session's stop one test past b1's commit should go, the one eight tests stalled stay; a judge that
    # stops nothing gets the first right under either playbook, so the rule ties and never reaches the judged replay.
    (only,) = report.generations
    assert not only.crowned and only.verdict.endswith("R2 right on 4 of 8 points, h0 on 4")


def test_a_generation_reads_the_champion_its_replay_the_stop_audits_and_the_ledger(tmp_path: Path) -> None:
    prompt = generation_prompt(SESSIONS[:2], H0, PlaybookPool(tmp_path), 2, "")

    assert prompt.startswith("You are improving the playbook offline")
    assert "- stall_tests: default (bounds 4 to 10)" in prompt and H0.render() in prompt
    assert "Scheduling replay over 2 sessions" in prompt and "## Stops" in prompt
    assert prompt.endswith("- nothing tried yet")
