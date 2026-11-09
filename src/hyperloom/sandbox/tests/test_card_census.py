# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who holds a card when the kernel's process ids cannot be matched: holders are counted, and the memory the kernel lists
on a card is set against what the processes of this container hold (fixtures shaped like what a GPU pod reads).

A pod has a pid namespace of its own, so the tree below is read as such: the kernel's list names host ids (3001, 3002)
that are not processes of the tree's ``/proc``, whose own ids are in the hundreds. The numbers are synthetic: the server
holds most of its card's memory with six queues, a foreign process a few GiB with two, and a host process that only
opened the interface holds nothing.
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

_SERVER_VRAM = 200 << 30  # what the server's descriptor says, in bytes
_FOREIGN_VRAM = 3 << 30
_CARD = 5
_NOBODY = OwnProcesses(frozenset())


@pytest.fixture
def pod(tmp_path: Path) -> KernelTree:
    """A pod's view: a pid namespace of its own, idle host processes in the kernel's list on every card."""
    tree = KernelTree(tmp_path, pid_namespace="pid:[4026532001]")
    for host_pid in (2001, 2002, 2003):
        tree.opened_only(host_pid)
    return tree


class _Clock:
    """Stands in for ``time`` in the probe: a sleep moves the clock on at once, and the sleeps are kept."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(card_occupancy, "time", fake)
    return fake


def _serve(tree: KernelTree, *, host_pid: int = 3001, pod_pid: int = 101, kib: int | None = None) -> None:
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
    pod.holds(3002, _CARD, vram=_FOREIGN_VRAM, queues=2)

    found = _read(pod).intrusions(_NOBODY, before_test=True)

    assert found == (Intrusion(("host-pid", 3002, _CARD), "host pid 3002 on card 5 (3.0 GiB resident, 2 queues)"),)


def test_before_the_test_a_holder_with_only_a_queue_is_foreign_and_during_it_is_not(pod: KernelTree) -> None:
    pod.holds(3002, _CARD, vram=0, queues=2)
    assert [i.key for i in _read(pod).intrusions(_NOBODY, before_test=True)] == [("host-pid", 3002, _CARD)]

    _serve(pod)  # the server's memory is the one holder of the card with memory: a queue with none is not a second one
    assert _whats(_read(pod), before_test=False) == []


def test_the_one_server_holder_of_a_card_with_its_memory_seen_in_the_pod_is_not_foreign_while_it_runs(
    pod: KernelTree,
) -> None:
    _serve(pod)

    reading = _read(pod)

    assert reading.method == METHOD_CENSUS_FDINFO
    assert [h.pid for h in reading.holders] == [3001] and reading.holders[0].queues == 6
    assert reading.unaccounted == {_CARD: 0}
    assert _whats(reading, before_test=False) == []


def test_a_second_holder_with_memory_on_the_card_is_foreign_while_the_test_runs(pod: KernelTree) -> None:
    _serve(pod)
    pod.holds(3002, _CARD, vram=_FOREIGN_VRAM, queues=2)
    pod.process(201)
    pod.renders(201, _CARD, kib=_FOREIGN_VRAM // 1024, descriptor=4)  # visible in the pod: only the count catches it

    found = _read(pod).intrusions(_NOBODY, before_test=False)

    assert [i.key for i in found] == [("crowded", _CARD)]
    assert found[0].what == (
        "2 processes with memory on card 5 at once (host pid 3001 with 200.0 GiB, host pid 3002 with 3.0 GiB), 1 explained; "
        "this container's processes with memory on its render node: "
        "pid 101 (not the test) with 200.0 GiB: python3 work.py; pid 201 (not the test) with 3.0 GiB: python3 work.py"
    )


def test_memory_that_no_process_of_the_pod_accounts_for_is_foreign_even_with_no_other_holder_of_the_card(
    pod: KernelTree,
) -> None:
    """The server is between two starts; a foreign process outside the pod holds the card alone."""
    pod.holds(3002, _CARD, vram=_FOREIGN_VRAM, queues=2)

    reading = _read(pod)

    assert [h.pid for h in reading.holders] == [3002]
    found = reading.intrusions(_NOBODY, before_test=False)
    assert [(i.key, i.what) for i in found] == [
        (("unaccounted", _CARD), "3.0 GiB on card 5 that no process this container sees accounts for")
    ]


def test_memory_the_pod_accounts_for_to_within_the_tolerance_is_not_foreign(pod: KernelTree) -> None:
    _serve(pod, kib=(_SERVER_VRAM - (100 << 20)) // 1024)  # the pod's processes say 100 MiB less than the kernel lists

    reading = _read(pod)

    assert reading.unaccounted == {_CARD: 100 << 20}
    assert _whats(reading, before_test=False) == []


def test_an_open_drm_file_shared_by_two_descriptors_counts_once(pod: KernelTree) -> None:
    """A forked worker inherits the server's render node: counted twice, the pod would seem to hold what it does not."""
    _serve(pod)
    pod.holds(3002, _CARD, vram=_FOREIGN_VRAM, queues=2)
    pod.process(102)
    pod.renders(102, _CARD, kib=_SERVER_VRAM // 1024, client="101-3")  # the same client id the server's descriptor has
    pod.renders(101, _CARD, kib=_SERVER_VRAM // 1024, client="101-3", descriptor=9)

    reading = _read(pod)

    assert reading.unaccounted[_CARD] == _FOREIGN_VRAM
    assert [i.key for i in reading.intrusions(_NOBODY, before_test=False)] == [
        ("crowded", _CARD),
        ("unaccounted", _CARD),
    ]


def test_cards_that_are_not_the_tests_are_not_counted(pod: KernelTree) -> None:
    _serve(pod)
    pod.holds(3002, 7, vram=_FOREIGN_VRAM, queues=2)

    reading = _read(pod, [_CARD])

    assert [h.card for h in reading.holders] == [_CARD]
    assert _whats(reading, before_test=False) == []


def test_processes_whose_descriptors_cannot_be_listed_leave_the_holders_counted_and_say_the_memory_was_not_set_against(
    pod: KernelTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(pod)
    real = os.scandir

    def refuse(path: object):
        if str(path).endswith("/101/fd"):
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(drm_fdinfo.os, "scandir", refuse)

    reading = _read(pod)

    assert reading.method == METHOD_CENSUS
    assert reading.accounting_gap is not None and "cannot be listed: Permission denied" in reading.accounting_gap
    assert reading.unaccounted == {} and [h.pid for h in reading.holders] == [3001]


def test_a_memory_size_with_a_unit_the_reader_does_not_know_is_not_taken_for_zero(pod: KernelTree) -> None:
    _serve(pod)
    (pod.procfs / "101" / "fdinfo" / "3").write_text("drm-client-id: 1\ndrm-memory-vram: 5 petabytes\n")

    reading = _read(pod)

    assert reading.method == METHOD_CENSUS
    assert reading.accounting_gap is not None and "not a size it knows" in reading.accounting_gap


def test_an_excess_that_is_gone_at_the_next_read_was_a_process_that_ended_between_the_two(
    pod: KernelTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(pod)
    real = drm_fdinfo.clients_by_render_minor
    answers = iter(
        [{TREE_MINOR: ()}]
    )  # the first read of the processes finds none of the server's memory: it was ending

    def reads(procfs: Path, minors):
        return next(answers, None) or real(procfs, minors)

    monkeypatch.setattr(card_occupancy.drm_fdinfo, "clients_by_render_minor", reads)

    reading = _read(pod)

    assert reading.unaccounted == {_CARD: 0}
    assert _whats(reading, before_test=False) == []


TREE_MINOR = KernelTree.render_minor(_CARD)


def test_a_server_that_exited_keeps_its_memory_listed_for_seconds_and_is_not_foreign_once_the_kernel_frees_it(
    pod: KernelTree, monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    """A server that exited is gone from ``/proc`` while the kernel still lists its memory, with no queue, for seconds;
    an immediate second read would still see it and call the test's own server foreign."""
    pod.holds(3001, _CARD, vram=_SERVER_VRAM, queues=0)  # the server is gone from the pod; the kernel still lists it
    real = card_occupancy._census_once
    reads: list[float] = []

    def census_once(kfd: Path, procfs: Path, selected):
        reads.append(clock.now)
        if clock.now >= 3.0:  # the driver has freed the memory and dropped the entry
            pod.holds(3001, _CARD, vram=0, queues=0)
        return real(kfd, procfs, selected)

    monkeypatch.setattr(card_occupancy, "_census_once", census_once)

    reading = _read(pod)

    assert _whats(reading, before_test=False) == []
    assert reading.unaccounted == {_CARD: 0}
    assert reads == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def test_an_excess_that_outlasts_the_settle_time_is_foreign_and_is_read_no_longer_than_that(
    pod: KernelTree, clock: _Clock
) -> None:
    pod.holds(3002, _CARD, vram=_FOREIGN_VRAM, queues=2)

    reading = _read(pod)

    assert _whats(reading, before_test=False) == ["3.0 GiB on card 5 that no process this container sees accounts for"]
    assert sum(clock.sleeps) == pytest.approx(card_occupancy.EXCESS_SETTLE_SEC)


def test_the_holders_of_a_card_whose_memory_the_pod_accounts_for_are_remembered_and_not_those_of_one_with_an_excess(
    pod: KernelTree,
) -> None:
    _serve(pod)
    pod.holds(3002, 7, vram=_FOREIGN_VRAM, queues=2)

    reading = _read(pod, [_CARD, 7])

    assert reading.accounted_holders() == frozenset({(_CARD, 3001)})


def test_the_memory_of_a_holder_found_accounted_for_before_is_not_foreign_once_its_process_ended_and_another_is(
    pod: KernelTree,
) -> None:
    """The test's server ended: its descriptors are closed and the kernel still lists its memory, with no queue, past the
    settle time. A holder that was never found accounted for, on the same card, is foreign all the same."""
    pod.holds(3001, _CARD, vram=_SERVER_VRAM, queues=0)
    remembered = frozenset({(_CARD, 3001)})

    reading = _read(pod)

    assert reading.unaccounted == {_CARD: _SERVER_VRAM}
    assert reading.intrusions(_NOBODY, before_test=False, accounted=remembered) == ()
    assert [i.key for i in reading.intrusions(_NOBODY, before_test=False)] == [("unaccounted", _CARD)]
    pod.holds(3002, _CARD, vram=_FOREIGN_VRAM, queues=2)
    found = _read(pod).intrusions(_NOBODY, before_test=False, accounted=remembered)
    assert [i.key for i in found] == [("crowded", _CARD), ("unaccounted", _CARD)]


def test_read_listed_counts_every_process_the_kernel_lists_on_the_cards_running_or_ended(pod: KernelTree) -> None:
    pod.holds(3001, _CARD, vram=_SERVER_VRAM, queues=0)  # ended, its memory not yet freed
    pod.holds(3002, 7, vram=_FOREIGN_VRAM, queues=2)

    reading = card_occupancy.read_listed([_CARD, 7], kfd=pod.kfd)

    assert reading.unavailable is None and reading.method == METHOD_CENSUS
    assert [(h.card, h.pid, h.vram) for h in reading.holders] == [
        (_CARD, 3001, _SERVER_VRAM),
        (7, 3002, _FOREIGN_VRAM),
    ]
    assert (
        card_occupancy.read_listed([9], kfd=pod.kfd).unavailable
        == "card(s) [9] are not among the 8 GPUs the kernel lists"
    )


def _test_server(tree: KernelTree, pids: list[int]) -> OwnProcesses:
    """The test's processes are those under the runner (pod pid 100): ``pids`` are made its children."""
    tree.process(100, command=("python3", "-m", "hyperloom.measure.runner"))
    for pid in pids:
        tree.process(pid, parent=100, command=("python3", "-m", "server", f"--rank={pid}"))
    return OwnProcesses(frozenset({100}), procfs=tree.procfs)


def test_two_holders_of_a_card_that_are_both_the_tests_own_processes_with_memory_on_it_are_not_crowded(
    pod: KernelTree,
) -> None:
    """The server and a helper of it that initialised the GPU: two fresh holders, one with almost no memory, both seen in
    the pod with memory on the card's render node and both the test's. Not a foreign holder."""
    own = _test_server(pod, [101, 102])
    pod.holds(3001, _CARD, vram=_SERVER_VRAM, queues=6)
    pod.holds(3002, _CARD, vram=4 << 20, queues=0)
    pod.renders(101, _CARD, kib=_SERVER_VRAM // 1024)
    pod.renders(102, _CARD, kib=(4 << 20) // 1024)

    reading = _read(pod)

    assert reading.method == METHOD_CENSUS_FDINFO and reading.unaccounted == {_CARD: 0}
    assert reading.explained(_CARD, own) == 2
    assert reading.intrusions(own, before_test=False) == ()
    assert [i.key for i in reading.intrusions(_NOBODY, before_test=False)] == [("crowded", _CARD)]


def test_a_holder_beyond_what_the_tests_own_processes_explain_is_crowded_and_the_record_names_each_one(
    pod: KernelTree,
) -> None:
    """Two of the test's processes have memory on the card, and the kernel lists a third holder with memory: a foreign
    one, whose memory the pod's processes do not account for either."""
    own = _test_server(pod, [101, 102])
    pod.holds(3001, _CARD, vram=_SERVER_VRAM, queues=6)
    pod.holds(3002, _CARD, vram=4 << 20, queues=0)
    pod.holds(3003, _CARD, vram=_FOREIGN_VRAM, queues=2)
    pod.renders(101, _CARD, kib=_SERVER_VRAM // 1024)
    pod.renders(102, _CARD, kib=(4 << 20) // 1024)

    found = _read(pod).intrusions(own, before_test=False)

    assert [i.key for i in found] == [("crowded", _CARD), ("unaccounted", _CARD)]
    assert found[0].what == (
        "3 processes with memory on card 5 at once (host pid 3001 with 200.0 GiB, host pid 3002 with 0.0 GiB, "
        "host pid 3003 with 3.0 GiB), 2 explained; this container's processes with memory on its render node: "
        "pid 101 (the test) with 200.0 GiB: python3 -m server --rank=101; "
        "pid 102 (the test) with 0.0 GiB: python3 -m server --rank=102"
    )


def test_a_process_of_the_pod_that_is_not_the_tests_explains_no_holder(pod: KernelTree) -> None:
    """A second test's server on the same card, in the same container: visible, and still foreign."""
    own = _test_server(pod, [101])
    pod.holds(3001, _CARD, vram=_SERVER_VRAM, queues=6)
    pod.renders(101, _CARD, kib=_SERVER_VRAM // 1024)
    pod.holds(3002, _CARD, vram=_FOREIGN_VRAM, queues=2)
    pod.process(201)
    pod.renders(201, _CARD, kib=_FOREIGN_VRAM // 1024)

    reading = _read(pod)

    assert reading.explained(_CARD, own) == 1
    found = reading.intrusions(own, before_test=False)
    assert [i.key for i in found] == [("crowded", _CARD)]
    assert "pid 201 (not the test) with 3.0 GiB: python3 work.py" in found[0].what


def test_a_forked_child_sharing_its_parents_drm_file_explains_no_second_holder(pod: KernelTree) -> None:
    own = _test_server(pod, [101, 102])
    pod.holds(3001, _CARD, vram=_SERVER_VRAM, queues=6)
    pod.holds(3002, _CARD, vram=4 << 20, queues=0)
    pod.renders(101, _CARD, kib=_SERVER_VRAM // 1024, client="101-3")
    pod.renders(102, _CARD, kib=_SERVER_VRAM // 1024, client="101-3")

    reading = _read(pod)

    assert reading.explained(_CARD, own) == 1
    assert [i.key for i in reading.intrusions(own, before_test=False)] == [("crowded", _CARD)]


def test_plain_census_still_allows_one_holder_with_memory_per_card_and_says_the_descriptors_were_not_read(
    pod: KernelTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the render nodes' descriptors cannot be read, two holders are one too many, however many the test has."""
    own = _test_server(pod, [101, 102])
    pod.holds(3001, _CARD, vram=_SERVER_VRAM, queues=6)
    pod.holds(3002, _CARD, vram=4 << 20, queues=0)
    pod.renders(101, _CARD, kib=_SERVER_VRAM // 1024)
    pod.renders(102, _CARD, kib=(4 << 20) // 1024)
    real = os.scandir

    def refuse(path: object):
        if str(path).endswith("/102/fd"):
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(drm_fdinfo.os, "scandir", refuse)

    reading = _read(pod)

    assert reading.method == METHOD_CENSUS and reading.render_clients == ()
    assert reading.explained(_CARD, own) == 1
    found = reading.intrusions(own, before_test=False)
    assert [i.key for i in found] == [("crowded", _CARD)]
    assert found[0].what.startswith(
        "2 processes with memory on card 5 at once (host pid 3001 with 200.0 GiB, host pid 3002 with 0.0 GiB), "
        "1 explained; the render node's descriptors were not read ("
    )
    assert "cannot be listed: Permission denied" in found[0].what
