# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's watch on its cards in a pod, whose pid namespace is its own: it counts what holds them (the numbers are the
ones a GPU pod measured) and says in the signed result how it told a foreign holder from the test's own."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import CARD_CLEAN, CARD_FOREIGN, CARD_UNKNOWN
from hyperloom.measure.runner.card_watch import AFTER_TEST, BEFORE_TEST, DURING_TEST, CardWatch
from hyperloom.measure.runner.service import RunnerFailure
from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.card_occupancy import METHOD_CENSUS, METHOD_CENSUS_FDINFO, CardGuard, OwnProcesses
from hyperloom.sandbox.card_occupancy import read_holders as read_the_machine
from hyperloom.sandbox.tests.kernel_tree import KernelTree

CARD = 5
SERVER_VRAM = 249_506_392 << 10
FOREIGN_VRAM = 2_759_956 << 10


@pytest.fixture
def pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KernelTree:
    machine = KernelTree(tmp_path, pid_namespace="pid:[4026552279]")
    machine.opened_only(1678127)
    monkeypatch.setattr(
        card_occupancy, "read_holders", lambda cards: read_the_machine(cards, kfd=machine.kfd, procfs=machine.procfs)
    )
    return machine


def _watch(pod: KernelTree, guard: CardGuard) -> CardWatch:
    return CardWatch((CARD,), OwnProcesses(frozenset({100}), procfs=pod.procfs), guard)


def _server(pod: KernelTree) -> None:
    pod.holds(1827025, CARD, vram=SERVER_VRAM, queues=6)
    pod.process(6495)
    pod.renders(6495, CARD, kib=SERVER_VRAM >> 10)


def test_a_pod_whose_card_only_the_server_holds_is_clean_and_the_result_says_how_that_was_told(pod: KernelTree) -> None:
    watch = _watch(pod, CardGuard.ENFORCE)
    watch.check(BEFORE_TEST)
    _server(pod)
    watch.check(DURING_TEST)
    watch.check(AFTER_TEST)
    watch.enforce_guard()

    statement = watch.statement()
    assert statement is not None and (statement.state, statement.reason, statement.checks) == (CARD_CLEAN, None, 3)
    assert statement.method == METHOD_CENSUS_FDINFO
    assert statement.to_document()["method"] == "census+fdinfo"


def test_a_holder_of_the_card_before_the_test_starts_refuses_it_under_enforce_and_is_only_noted_under_observe(
    pod: KernelTree,
) -> None:
    pod.holds(1845543, CARD, vram=FOREIGN_VRAM, queues=2)
    enforcing, observing = _watch(pod, CardGuard.ENFORCE), _watch(pod, CardGuard.OBSERVE)
    for watch in (enforcing, observing):
        watch.check(BEFORE_TEST)

    with pytest.raises(RunnerFailure) as refused:
        enforcing.enforce_guard()
    assert refused.value.error_class == "foreign_process_on_card"
    assert "host pid 1845543 on card 5 (2.6 GiB resident, 2 queues), first seen" in str(refused.value)
    observing.enforce_guard()
    assert observing.observation() is not None and "CARD GUARD OBSERVES ONLY" in observing.observation()  # type: ignore[operator]
    statement = observing.statement()
    assert statement is not None and statement.state == CARD_FOREIGN and statement.method == METHOD_CENSUS_FDINFO


def test_a_second_holder_that_appears_while_the_test_runs_is_caught_and_one_that_has_gone_again_stays_on_record(
    pod: KernelTree,
) -> None:
    _server(pod)
    watch = _watch(pod, CardGuard.ENFORCE)
    watch.check(DURING_TEST)
    pod.holds(1845543, CARD, vram=FOREIGN_VRAM, queues=2)
    watch.check(DURING_TEST)
    shutil.rmtree(pod.kfd / "proc" / "1845543")
    watch.check(AFTER_TEST)

    with pytest.raises(RunnerFailure) as refused:
        watch.enforce_guard()
    assert "2 processes with memory on card 5 at once" in str(refused.value) and "during the test" in str(refused.value)


def test_memory_no_process_of_the_pod_accounts_for_is_caught_while_the_server_is_between_two_starts(
    pod: KernelTree,
) -> None:
    pod.holds(1845543, CARD, vram=FOREIGN_VRAM, queues=2)
    watch = _watch(pod, CardGuard.ENFORCE)

    watch.check(DURING_TEST)

    with pytest.raises(RunnerFailure) as refused:
        watch.enforce_guard()
    assert "2.6 GiB on card 5 that no process this container sees accounts for" in str(refused.value)


def test_a_look_that_could_not_set_the_memory_against_the_pods_processes_says_the_weaker_method(
    pod: KernelTree,
) -> None:
    _server(pod)
    (pod.procfs / "6495" / "fdinfo" / "3").write_text("drm-memory-vram: 1 furlongs\n")
    watch = _watch(pod, CardGuard.ENFORCE)

    watch.check(DURING_TEST)

    statement = watch.statement()
    assert statement is not None and statement.state == CARD_CLEAN and statement.method == METHOD_CENSUS


def test_a_look_the_probe_cannot_make_leaves_the_result_unknown_with_no_method(pod: KernelTree) -> None:
    shutil.rmtree(pod.kfd / "proc")
    watch = _watch(pod, CardGuard.ENFORCE)

    watch.check(BEFORE_TEST)

    statement = watch.statement()
    assert statement is not None and statement.state == CARD_UNKNOWN and statement.method is None
    assert statement.reason is not None and "no AMD GPU driver" in statement.reason
