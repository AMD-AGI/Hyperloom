# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE asks the LLM for one change to the scheduling playbook and pools what passes the static check."""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot
from hyperloom_multibranch.blackboard.rsi import authoring
from hyperloom_multibranch.blackboard.rsi.archive import SKIPPED, TreeArchive, tree_from_snapshot
from hyperloom_multibranch.blackboard.rsi.authoring import (
    MEASURES,
    NO_CHANGE,
    NO_MEASURE,
    NOT_ARCHIVED,
    PLAYBOOK_SYSTEM_PROMPT,
    author_playbook,
    measures_line,
    playbook_prompt,
    read_change,
)
from hyperloom_multibranch.blackboard.rsi.ladder import Difference, StandingRow, Trial, record_trial, session_task
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
        (
            _answer(**{**LATER, "change": {"number": "stall_tests", "value": math.inf}}, measure="stop_value"),
            "stall_tests takes a finite number",
        ),
        (
            _answer(**{**LATER, "change": {"number": "stall_tests", "value": -math.inf}}, measure="stop_value"),
            "stall_tests takes a finite number",
        ),
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
    assert measures_line(("stop_value", "early_stops")) == (
        "The decision measures R3 can read on this task, one of which a change aims at: stop_value, early_stops."
    )


def test_an_answer_aiming_at_a_measure_r3_cannot_read_here_says_so() -> None:
    proposed, why = read_change(_answer(**LATER, measure="stop_value"), H0, WHO, 1.0, ("first_commit_minutes",))

    assert proposed is None
    assert why == "'stop_value' is no decision measure R3 can read here; one of first_commit_minutes"


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
    assert asked[0][1].endswith(f"\n\n{measures_line(MEASURES)}")
    assert again == CloseStep(SKIPPED, "it repeats later-stops, retired: behind h0", {"written": 0, "refused": 1})


async def test_close_asks_for_no_change_when_no_measure_varied_in_the_champions_sessions(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    archive = TreeArchive(tmp_path / "archive")
    tree = tree_from_snapshot(snapshot, METRIC)
    archive.put(tree)
    pool = PlaybookPool.beside(archive)
    for n in range(2):
        record_trial(
            pool, Trial(f"c{n}", H0.label, H0.label, session_task(tree), float(n), False, {"early_stops": 0.0})
        )

    async def unasked(system: str, prompt: str) -> str:
        raise AssertionError("with no measure to aim at, nothing is asked")

    done = await author_playbook(snapshot, unasked, model="m", archive=archive)

    assert done == CloseStep(SKIPPED, NO_MEASURE, {"written": 0, "refused": 0})


async def test_close_pools_a_change_the_replay_reads_worse_both_ways_as_retired(
    snapshot: BlackboardSnapshot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = TreeArchive(tmp_path / "archive")
    archive.put(tree_from_snapshot(snapshot, METRIC))
    pool = PlaybookPool.beside(archive)
    worse = StandingRow("later-stops", Difference(-2.0, -3.0, -1.0), Difference(-1.0, -1.5, -0.5))
    monkeypatch.setattr(authoring, "r1_reading", lambda candidate, champion, trees: worse)

    async def complete(system: str, prompt: str) -> str:
        return _answer(**LATER, measure="stop_value")

    done = await author_playbook(snapshot, complete, model="claude", archive=archive)

    evidence = (
        "the scheduling replay of the archive reads it worse both ways: "
        "optimistic -2.00 (-3.00 to -1.00); pessimistic -1.00 (-1.50 to -0.50)"
    )
    assert done == CloseStep(SKIPPED, f"later-stops: {evidence}", {"written": 0, "refused": 1})
    assert pool.on_trial() == ()
    assert [(item.label, item.evidence) for item in pool.retired()] == [("later-stops", evidence)]


def _noting(threads: list[int], call: Callable[..., object]) -> Callable[..., object]:
    def noted(*args: object) -> object:
        threads.append(threading.get_ident())
        return call(*args)

    return noted


async def test_close_reads_the_archive_and_replays_the_change_off_the_event_loop(
    snapshot: BlackboardSnapshot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = TreeArchive(tmp_path / "archive")
    archive.put(tree_from_snapshot(snapshot, METRIC))
    threads: list[int] = []
    for name in ("archived_history", "r1_reading"):
        monkeypatch.setattr(authoring, name, _noting(threads, getattr(authoring, name)))

    async def complete(system: str, prompt: str) -> str:
        return _answer(**LATER, measure="stop_value")

    done = await author_playbook(snapshot, complete, model="claude", archive=archive)

    assert done == CloseStep(WRITTEN, "later-stops: stall_tests of h0", {"written": 1, "refused": 0})
    assert len(threads) == 2 and threading.get_ident() not in threads


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
