# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE asks the LLM for one change to the scheduling playbook and pools what passes the static check."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot
from hyperloom_multibranch.blackboard.rsi.archive import SKIPPED, TreeArchive, tree_from_snapshot
from hyperloom_multibranch.blackboard.rsi.authoring import (
    MEASURES,
    NO_CHANGE,
    NOT_ARCHIVED,
    PLAYBOOK_SYSTEM_PROMPT,
    author_playbook,
    playbook_prompt,
    read_change,
)
from hyperloom_multibranch.blackboard.rsi.playbook import (
    H0,
    Governs,
    Playbook,
    PlaybookPool,
    PlaybookRule,
    Retired,
    replaced,
)
from hyperloom_multibranch.blackboard.rsi.session import FAILED, NO_PROFILE, WRITTEN, CloseStep
from hyperloom_multibranch.blackboard.rsi.stops import stop_audit
from hyperloom_multibranch.blackboard.tests.sessions import METRIC

WHO = {"model": "claude", "session_id": "sess-1"}
LATER = {"label": "later-stops", "change": {"number": "stall_tests", "value": 8}, "hypothesis": "h(5..8) holds up"}
ALIKE = "Of two branches whose changes are alike, stop the one further behind integration."


def _answer(**document: object) -> str:
    return "My proposal:\n" + json.dumps(document) + "\nThat is all."


def test_an_answer_gives_one_change_to_the_champion_as_a_playbook_on_trial() -> None:
    later, why = read_change(_answer(**LATER, measure="probation_commits"), H0, WHO, 100.0)
    ruled = Playbook("ruled", rules=(PlaybookRule(Governs.STOP, ALIKE),), parent=H0.label)
    added, _ = read_change(
        _answer(
            label="alike", change={"add_rule": {"governs": "stop", "text": ALIKE}}, hypothesis="h", measure="stop_value"
        ),
        H0,
        WHO,
        1.0,
    )
    removed, _ = read_change(
        _answer(label="plain", change={"remove_rule": 1}, hypothesis="h", measure="early_stops"), ruled, WHO, 1.0
    )

    assert why == "" and later == Playbook(
        "later-stops",
        replaced(H0, stall_tests=8).numbers,
        parent=H0.label,
        hypothesis="h(5..8) holds up",
        measure="probation_commits",
        registered_at=100.0,
        provenance=WHO,
    )
    assert added is not None and added.rules == (PlaybookRule(Governs.STOP, ALIKE),)
    assert removed is not None and (removed.rules, removed.parent) == ((), "ruled")


def test_an_answer_labelled_as_the_pool_would_refuse_is_registered_under_a_label_it_takes() -> None:
    proposed, why = read_change(_answer(**{**LATER, "label": "Stall 0.25 -> 8"}, measure="stop_value"), H0, WHO, 1.0)

    assert why == "" and proposed is not None and proposed.label == "stall-0.25-8"


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        (_answer(none="the audit shows no stop to regret"), f"{NO_CHANGE}: the audit shows no stop to regret"),
        ("no JSON here", "the answer holds no JSON object"),
        (" \n", "the answer is empty"),
        (_answer(**{**LATER, "label": "!!!"}, measure="stop_value"), "'!!!' gives no label"),
        (_answer(**LATER, measure="gain"), "'gain' is no decision measure"),
        (
            _answer(**{**LATER, "change": {"number": "stall_tests", "value": 12}}, measure="stop_value"),
            "stall_tests 12",
        ),
        (
            _answer(**{**LATER, "change": {"number": "stall_tests", "value": 6.5}}, measure="stop_value"),
            "stall_tests takes",
        ),
        (_answer(**{**LATER, "change": {"number": "speed", "value": 1}}, measure="stop_value"), "'speed' is no number"),
        (_answer(**{**LATER, "change": {"remove_rule": 3}}, measure="stop_value"), "a change is a number"),
        (_answer(**{**LATER, "change": {"number": "stall_tests", "value": None}}, measure="stop_value"), "it changes"),
    ],
)
def test_an_answer_that_is_no_valid_single_change_says_why(answer: str, why: str) -> None:
    proposed, found = read_change(answer, H0, WHO, 1.0)

    assert proposed is None and found.startswith(why)


def test_the_author_reads_the_numbers_the_champion_the_audit_the_new_branches_and_the_ledger(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    tree = tree_from_snapshot(snapshot, METRIC)
    pool = PlaybookPool(tmp_path)
    tried = Playbook(
        "later-stops", replaced(H0, stall_tests=8).numbers, parent=H0.label, hypothesis="later", measure="stop_value"
    )
    pool.register(tried)
    pool.retire(Retired(tried.label, 5.0, "no more probations committed"))

    prompt = playbook_prompt(tree, stop_audit(tree), H0, H0.label, pool)

    assert "- stall_tests: default (bounds 4 to 10)" in prompt and H0.render() in prompt
    assert f"This session, {tree.session_id}, played h0." in prompt and "# Stop audit of" in prompt
    assert "How its new branches opened:" in prompt
    assert (
        "- later-stops (of h0, changing stall_tests): later; aims at stop_value; retired: no more probations committed"
        in prompt
    )
    assert all(measure in PLAYBOOK_SYSTEM_PROMPT for measure in MEASURES)


async def test_close_pools_the_change_once_and_refuses_one_that_repeats_a_retired_playbook(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    archive = TreeArchive(tmp_path / "archive")
    archive.put(tree_from_snapshot(snapshot, METRIC))
    pool = PlaybookPool.beside(archive)
    asked: list[tuple[str, str]] = []
    answer = _answer(**LATER, measure="stop_value")

    async def complete(system: str, prompt: str) -> str:
        asked.append((system, prompt))
        return answer

    done = await author_playbook(snapshot, complete, model="claude", archive=archive)
    (written,) = pool.playbooks()
    pool.retire(Retired(written.label, 9.0, "behind h0"))
    answer = _answer(**{**LATER, "label": "later-again"}, measure="stop_value")
    again = await author_playbook(snapshot, complete, model="claude", archive=archive)

    assert done == CloseStep(WRITTEN, "later-stops: stall_tests of h0", {"written": 1, "refused": 0})
    assert written.provenance == {"model": "claude", "session_id": "sess-1"} and asked[0][0] == PLAYBOOK_SYSTEM_PROMPT
    assert again == CloseStep(SKIPPED, "it repeats later-stops, retired: behind h0", {"written": 0, "refused": 1})


async def test_close_says_why_it_wrote_no_change(snapshot: BlackboardSnapshot, tmp_path: Path) -> None:
    archive = TreeArchive(tmp_path / "archive")

    async def unreachable(system: str, prompt: str) -> str:
        raise RuntimeError("the gateway is down")

    async def nothing(system: str, prompt: str) -> str:
        return _answer(none="nothing to try")

    assert (await author_playbook(snapshot, nothing, model="m", archive=archive)).reason == NOT_ARCHIVED
    archive.put(tree_from_snapshot(snapshot, METRIC))
    unprofiled = dataclasses.replace(snapshot, profile=None)
    assert await author_playbook(unprofiled, nothing, model="m", archive=archive) == CloseStep(SKIPPED, NO_PROFILE)
    assert await author_playbook(snapshot, unreachable, model="m", archive=archive) == CloseStep(
        FAILED, "RuntimeError: the gateway is down"
    )
    assert await author_playbook(snapshot, nothing, model="m", archive=archive) == CloseStep(
        SKIPPED, f"{NO_CHANGE}: nothing to try", {"written": 0, "refused": 0}
    )
