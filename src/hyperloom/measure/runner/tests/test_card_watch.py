# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's watch on its cards: who may hold them, and what the signed result says of what it saw."""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import CARD_CLEAN, CARD_FOREIGN, CARD_UNKNOWN
from hyperloom.measure.runner.card_watch import AFTER_TEST, BEFORE_TEST, DURING_TEST, CardStatement, CardWatch
from hyperloom.measure.runner.service import RunnerFailure
from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.card_occupancy import METHOD_PID_MATCH, CardGuard, CardReading, OwnProcesses
from hyperloom.sandbox.card_occupancy import read_holders as read_the_machine
from hyperloom.sandbox.tests.kernel_tree import KernelTree

RUNNER = 100
SERVER = 101


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KernelTree:
    """A machine whose list the watch reads: the runner (pid 100) with its server below it, and nothing else."""
    machine = KernelTree(tmp_path)
    machine.process(RUNNER)
    machine.process(SERVER, parent=RUNNER, command=("python3", "-m", "sglang.launch_server"))
    monkeypatch.setattr(
        card_occupancy, "read_holders", lambda cards: read_the_machine(cards, kfd=machine.kfd, procfs=machine.procfs)
    )
    return machine


def _watch(
    tree: KernelTree, cards: Sequence[int] = (0, 1), guard: CardGuard = CardGuard.ENFORCE, **kwargs: object
) -> CardWatch:
    return CardWatch(cards, OwnProcesses(frozenset({RUNNER}), procfs=tree.procfs), guard, **kwargs)  # type: ignore[arg-type]


def _unix(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def test_cards_nobody_but_the_test_holds_are_clean_and_the_result_says_how_often_it_looked(tree: KernelTree) -> None:
    tree.holds(SERVER, 0)
    watch = _watch(tree)

    watch.check(BEFORE_TEST)
    watch.check(BEFORE_TEST)
    watch.enforce_guard()

    assert watch.statement() == CardStatement(CARD_CLEAN, None, 2, METHOD_PID_MATCH)
    assert watch.statement().to_document() == {
        "state": "clean",
        "reason": None,
        "checks": 2,
        "method": "pid-match",
    }  # type: ignore[union-attr]


def test_a_process_outside_the_test_that_holds_a_card_before_it_starts_refuses_the_result(tree: KernelTree) -> None:
    tree.process(900, command=("python3", "-c", "import torch; torch.zeros(1).cuda()"))
    tree.holds(900, 1)
    watch = _watch(tree, clock=lambda: _unix("2026-10-24T04:05:06Z"))

    watch.check(BEFORE_TEST)

    with pytest.raises(RunnerFailure) as refused:
        watch.enforce_guard()
    assert refused.value.error_class == "foreign_process_on_card"
    assert str(refused.value) == (
        "1 held the test's cards: pid 900 on card 1, first seen 2026-10-24T04:05:06Z before the test: "
        "python3 -c import torch; torch.zeros(1).cuda()"
    )


def test_a_process_that_arrives_while_the_test_runs_is_seen_by_the_poll_and_remembered_after_it_left(
    tree: KernelTree,
) -> None:
    watch = _watch(tree, poll_sec=0.01)

    with watch.watching():
        tree.process(901, command=("python3", "intruder.py"))
        tree.holds(901, 0)
        deadline = time.monotonic() + 10
        while not watch._sightings and time.monotonic() < deadline:  # the poll thread finds it on its own
            time.sleep(0.01)
        tree.exit(901)

    with pytest.raises(RunnerFailure, match=r"pid 901 on card 0, first seen \S+ during the test: python3 intruder.py"):
        watch.enforce_guard()


def test_a_process_that_is_still_there_when_the_test_ends_is_seen_by_the_last_look(tree: KernelTree) -> None:
    watch = _watch(tree, poll_sec=3600)

    with watch.watching():
        tree.process(902)
        tree.holds(902, 1)

    with pytest.raises(RunnerFailure, match=rf"pid 902 on card 1, first seen \S+ {AFTER_TEST}"):
        watch.enforce_guard()


def test_the_time_a_process_was_first_seen_is_kept_and_it_is_one_process_per_card(tree: KernelTree) -> None:
    tree.process(903)
    tree.holds(903, 0)
    clock = itertools.count(1_700_000_000)
    watch = _watch(tree, clock=lambda: float(next(clock)))

    watch.check(BEFORE_TEST)
    watch.check(DURING_TEST)
    watch.check(AFTER_TEST)

    with pytest.raises(
        RunnerFailure, match=r"^1 held the test's cards: pid 903 on card 0, first seen 2023-11-14T22:13:20Z before"
    ):
        watch.enforce_guard()


def test_the_failure_names_the_earliest_two_and_counts_the_rest_and_cuts_long_commands(tree: KernelTree) -> None:
    clock = itertools.count(100)
    watch = _watch(tree, clock=lambda: float(next(clock)))
    for pid in (910, 911, 912):
        tree.process(pid, command=("python3", "x" * 300))
    tree.holds(910, 0)
    watch.check(BEFORE_TEST)
    tree.holds(911, 0)
    watch.check(DURING_TEST)
    tree.holds(912, 1)
    watch.check(DURING_TEST)

    with pytest.raises(RunnerFailure) as refused:
        watch.enforce_guard()

    text = str(refused.value)
    assert text.startswith("3 held the test's cards: pid 910 on card 0")
    assert "pid 911 on card 0" in text and "pid 912" not in text and text.endswith("; and 1 more")
    assert text.count("x" * 72) == 2 and "x" * 73 not in text, (
        "a command line is cut at 80 characters, program included"
    )


def test_a_probe_that_cannot_say_once_leaves_the_result_unknown_with_its_reason_and_refuses_nothing(
    tree: KernelTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch = _watch(tree)
    answers = iter([CardReading(), CardReading(unavailable="no AMD GPU driver"), CardReading()])
    monkeypatch.setattr(card_occupancy, "read_holders", lambda cards: next(answers))

    watch.check(BEFORE_TEST)
    watch.check(DURING_TEST)
    watch.check(AFTER_TEST)
    watch.enforce_guard()

    assert watch.statement() == CardStatement(CARD_UNKNOWN, "no AMD GPU driver", 2, METHOD_PID_MATCH)


def test_a_probe_that_fails_outright_is_unknown_not_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(cards: Sequence[int]) -> CardReading:
        raise RuntimeError("the list went away")

    monkeypatch.setattr(card_occupancy, "read_holders", broken)
    watch = CardWatch((0,), OwnProcesses(frozenset({1})), CardGuard.ENFORCE)

    watch.check(BEFORE_TEST)

    assert watch.statement() == CardStatement(CARD_UNKNOWN, "a look at the cards failed (RuntimeError)", 0)


def test_a_watch_that_never_looked_does_not_call_the_cards_clean(tree: KernelTree) -> None:
    assert _watch(tree).statement() == CardStatement(CARD_UNKNOWN, "the cards were not looked at", 0)


def test_a_test_with_no_card_has_nothing_to_watch_and_states_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def never(cards: Sequence[int]) -> CardReading:
        raise AssertionError("no card, no look")

    monkeypatch.setattr(card_occupancy, "read_holders", never)
    watch = CardWatch((), OwnProcesses(frozenset({1})), CardGuard.ENFORCE)

    with watch.watching():
        watch.check(BEFORE_TEST)
    watch.enforce_guard()

    assert watch.statement() is None


def test_the_poll_thread_ends_with_the_block(tree: KernelTree) -> None:
    before = {t.name for t in threading.enumerate()}

    with _watch(tree, poll_sec=0.01).watching():
        assert "hl-card-watch" in {t.name for t in threading.enumerate()}

    assert {t.name for t in threading.enumerate()} == before


# --- a guard that only observes ---------------------------------------------------------------------------------------


def _intruder_seen_by(watch: CardWatch, tree: KernelTree) -> None:
    tree.process(950, command=("python3", "-c", "import torch"))
    tree.holds(950, 1)
    watch.check(BEFORE_TEST)
    watch.check(DURING_TEST)


def test_a_guard_that_observes_refuses_nothing_and_the_result_says_who_was_seen(tree: KernelTree) -> None:
    watch = _watch(tree, guard=CardGuard.OBSERVE, clock=lambda: _unix("2026-10-24T04:05:06Z"))
    _intruder_seen_by(watch, tree)

    watch.enforce_guard()

    statement = watch.statement()
    assert statement == CardStatement(
        CARD_FOREIGN,
        "1 held the test's cards: pid 950 on card 1, first seen 2026-10-24T04:05:06Z before the test: "
        "python3 -c import torch",
        2,
        METHOD_PID_MATCH,
    )
    assert statement.to_document()["state"] == "foreign"  # type: ignore[union-attr]


def test_a_guard_that_observes_leaves_a_note_that_says_the_result_was_kept(tree: KernelTree) -> None:
    watch = _watch(tree, guard=CardGuard.OBSERVE)
    _intruder_seen_by(watch, tree)

    note = watch.observation()

    assert note is not None
    assert note.startswith(
        "CARD GUARD OBSERVES ONLY (HYPERLOOM_CARD_GUARD=observe): this result was kept although 1 held"
    )
    assert "pid 950 on card 1" in note


def test_a_guard_that_observes_leaves_no_note_for_cards_nobody_but_the_test_held(tree: KernelTree) -> None:
    watch = _watch(tree, guard=CardGuard.OBSERVE)
    watch.check(BEFORE_TEST)

    assert watch.observation() is None and watch.statement() == CardStatement(CARD_CLEAN, None, 1, METHOD_PID_MATCH)


def test_a_guard_that_enforces_refuses_and_leaves_no_note(tree: KernelTree) -> None:
    watch = _watch(tree, guard=CardGuard.ENFORCE)
    _intruder_seen_by(watch, tree)

    assert watch.observation() is None
    with pytest.raises(RunnerFailure, match="pid 950 on card 1"):
        watch.enforce_guard()


def test_a_process_seen_wins_over_a_look_that_could_not_say(tree: KernelTree, monkeypatch: pytest.MonkeyPatch) -> None:
    watch = _watch(tree, guard=CardGuard.OBSERVE)
    _intruder_seen_by(watch, tree)
    monkeypatch.setattr(card_occupancy, "read_holders", lambda cards: CardReading(unavailable="no AMD GPU driver"))
    watch.check(AFTER_TEST)

    statement = watch.statement()

    assert statement is not None and statement.state == CARD_FOREIGN and "pid 950" in (statement.reason or "")
