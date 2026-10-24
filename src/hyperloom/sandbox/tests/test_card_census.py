# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who holds a card when the kernel's process ids cannot be matched: holders are counted, and the memory the kernel lists
on a card is set against what the processes of this container hold (fixtures modelled on what a GPU pod measured).

A pod has a pid namespace of its own, so the tree below is read as such: the kernel's list names host ids (1827025, 1845543)
that are not processes of the tree's ``/proc``. The numbers are the measured ones: the server holds 237.94 GiB on its card
with six queues, a foreign process 2.63 GiB with two, and a host process that only opened the interface holds nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.sandbox import card_occupancy, drm_fdinfo
from hyperloom.sandbox.card_occupancy import (
    METHOD_CENSUS,
    METHOD_CENSUS_FDINFO,
    CardReading,
    Intrusion,
    OwnProcesses,
    read_holders,
)
from hyperloom.sandbox.tests.kernel_tree import KernelTree

_GIB = 1 << 30
_SERVER_VRAM = 249_506_392 << 10  # what the server's descriptor said, in bytes: 237.95 GiB
_FOREIGN_VRAM = 2_759_956 << 10  # 2.63 GiB
_CARD = 5
_NOBODY = OwnProcesses(frozenset())


@pytest.fixture
def pod(tmp_path: Path) -> KernelTree:
    """A pod's view: a pid namespace of its own, idle host processes in the kernel's list on every card."""
    tree = KernelTree(tmp_path, pid_namespace="pid:[4026552279]")
    for host_pid in (1678127, 1744260, 1747860):
        tree.opened_only(host_pid)
    return tree


def _serve(tree: KernelTree, *, host_pid: int = 1827025, pod_pid: int = 6495, kib: int | None = None) -> None:
    """The test's server: one holder with memory and queues on the card, seen in the pod through its render node."""
    tree.holds(host_pid, _CARD, vram=_SERVER_VRAM, queues=6)
    tree.process(pod_pid)
    tree.renders(pod_pid, _CARD, kib=kib if kib is not None else _SERVER_VRAM // 1024)


def _read(tree: KernelTree, cards: list[int] | None = None) -> CardReading:
    return read_holders(cards or [_CARD], kfd=tree.kfd, procfs=tree.procfs)


def _whats(reading: CardReading, *, before_test: bool) -> list[str]:
    return [i.what for i in reading.intrusions(_NOBODY, before_test=before_test)]


def test_idle_host_processes_with_no_memory_and_no_queue_hold_nothing_and_the_card_is_clean(pod: KernelTree) -> None:
    reading = _read(pod)

    assert reading.unavailable is None and reading.method == METHOD_CENSUS_FDINFO
    assert reading.holders == ()
    assert _whats(reading, before_test=True) == [] and _whats(reading, before_test=False) == []


def test_before_the_test_starts_any_holder_of_its_card_is_foreign_and_named_by_its_host_id(pod: KernelTree) -> None:
    pod.holds(1845543, _CARD, vram=_FOREIGN_VRAM, queues=2)

    found = _read(pod).intrusions(_NOBODY, before_test=True)

    assert found == (
        Intrusion(("host-pid", 1845543, _CARD), "host pid 1845543 on card 5 (2.6 GiB resident, 2 queues)"),
    )


def test_before_the_test_a_holder_with_only_a_queue_is_foreign_and_during_it_is_not(pod: KernelTree) -> None:
    pod.holds(1845543, _CARD, vram=0, queues=2)
    reading = _read(pod)

    assert [i.key for i in reading.intrusions(_NOBODY, before_test=True)] == [("host-pid", 1845543, _CARD)]
    assert _whats(reading, before_test=False) == [], "a queue with no memory is not a second holder of the card"


def test_the_one_server_holder_of_a_card_with_its_memory_seen_in_the_pod_is_not_foreign_while_it_runs(
    pod: KernelTree,
) -> None:
    _serve(pod)

    reading = _read(pod)

    assert reading.method == METHOD_CENSUS_FDINFO
    assert [h.pid for h in reading.holders] == [1827025] and reading.holders[0].queues == 6
    assert reading.unaccounted == {_CARD: 0}
    assert _whats(reading, before_test=False) == []


def test_a_second_holder_with_memory_on_the_card_is_foreign_while_the_test_runs(pod: KernelTree) -> None:
    _serve(pod)
    pod.holds(1845543, _CARD, vram=_FOREIGN_VRAM, queues=2)
    pod.process(44094)
    pod.renders(44094, _CARD, kib=_FOREIGN_VRAM // 1024, descriptor=4)  # visible in the pod: only the count catches it

    found = _read(pod).intrusions(_NOBODY, before_test=False)

    assert [i.key for i in found] == [("crowded", _CARD)]
    assert found[0].what == (
        "2 processes with memory on card 5 at once (host pid 1827025 with 237.9 GiB, host pid 1845543 with 2.6 GiB)"
    )


def test_memory_that_no_process_of_the_pod_accounts_for_is_foreign_even_with_no_other_holder_of_the_card(
    pod: KernelTree,
) -> None:
    """The server is between two starts; a foreign process outside the pod holds the card alone."""
    pod.holds(1845543, _CARD, vram=_FOREIGN_VRAM, queues=2)

    reading = _read(pod)

    assert [h.pid for h in reading.holders] == [1845543]
    found = reading.intrusions(_NOBODY, before_test=False)
    assert [(i.key, i.what) for i in found] == [
        (("unaccounted", _CARD), "2.6 GiB on card 5 that no process this container sees accounts for")
    ]


def test_memory_the_pod_accounts_for_to_within_the_tolerance_is_not_foreign(pod: KernelTree) -> None:
    _serve(pod, kib=(_SERVER_VRAM - (100 << 20)) // 1024)  # the pod's processes say 100 MiB less than the kernel lists

    reading = _read(pod)

    assert reading.unaccounted == {_CARD: 100 << 20}
    assert _whats(reading, before_test=False) == []


def test_an_open_drm_file_shared_by_two_descriptors_counts_once(pod: KernelTree) -> None:
    """A forked worker inherits the server's render node: counted twice, the pod would seem to hold what it does not."""
    _serve(pod)
    pod.holds(1845543, _CARD, vram=_FOREIGN_VRAM, queues=2)
    pod.process(6496)
    pod.renders(
        6496, _CARD, kib=_SERVER_VRAM // 1024, client="6495-3"
    )  # the same client id the server's descriptor has
    pod.renders(6495, _CARD, kib=_SERVER_VRAM // 1024, client="6495-3", descriptor=9)

    reading = _read(pod)

    assert reading.unaccounted[_CARD] == _FOREIGN_VRAM
    assert [i.key for i in reading.intrusions(_NOBODY, before_test=False)] == [
        ("crowded", _CARD),
        ("unaccounted", _CARD),
    ]


def test_cards_that_are_not_the_tests_are_not_counted(pod: KernelTree) -> None:
    _serve(pod)
    pod.holds(1845543, 7, vram=_FOREIGN_VRAM, queues=2)

    reading = _read(pod, [_CARD])

    assert [h.card for h in reading.holders] == [_CARD]
    assert _whats(reading, before_test=False) == []


def test_processes_whose_descriptors_cannot_be_listed_leave_the_holders_counted_and_say_the_memory_was_not_set_against(
    pod: KernelTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(pod)
    real = os.scandir

    def refuse(path: object):
        if str(path).endswith("/6495/fd"):
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(drm_fdinfo.os, "scandir", refuse)

    reading = _read(pod)

    assert reading.method == METHOD_CENSUS
    assert reading.accounting_gap is not None and "cannot be listed: Permission denied" in reading.accounting_gap
    assert reading.unaccounted == {} and [h.pid for h in reading.holders] == [1827025]


def test_a_memory_size_with_a_unit_the_reader_does_not_know_is_not_taken_for_zero(pod: KernelTree) -> None:
    _serve(pod)
    (pod.procfs / "6495" / "fdinfo" / "3").write_text("drm-client-id: 1\ndrm-memory-vram: 5 petabytes\n")

    reading = _read(pod)

    assert reading.method == METHOD_CENSUS
    assert reading.accounting_gap is not None and "not a size it knows" in reading.accounting_gap


def test_an_excess_that_is_gone_at_the_second_read_was_a_process_that_ended_between_the_two(
    pod: KernelTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(pod)
    real = drm_fdinfo.vram_by_render_minor
    answers = iter(
        [{TREE_MINOR: 0}]
    )  # the first read of the processes finds none of the server's memory: it was ending

    def reads(procfs: Path, minors):
        return next(answers, None) or real(procfs, minors)

    monkeypatch.setattr(card_occupancy.drm_fdinfo, "vram_by_render_minor", reads)

    reading = _read(pod)

    assert reading.unaccounted == {_CARD: 0}
    assert _whats(reading, before_test=False) == []


TREE_MINOR = KernelTree.render_minor(_CARD)
