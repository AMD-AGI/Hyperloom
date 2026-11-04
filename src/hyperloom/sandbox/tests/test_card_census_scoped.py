# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who holds a card when the reader is a container shown only some GPUs: the kernel refuses it the ``gpu_id`` and
``properties`` of every other GPU node (permission denied), as a one-GPU Agent Sandbox pod measured. The probe numbers
the cards among the nodes it can read, as the runtime does, and still lists every holder of those; a view that can read
no GPU node, not the asked card's or no process list is ``denied``, never a reading in which nobody holds the cards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.card_occupancy import METHOD_CENSUS_FDINFO, CardReading, OwnProcesses, read_holders, read_listed
from hyperloom.sandbox.tests.kernel_tree import KernelTree

_GIB = 1 << 30
_SERVER_VRAM = 200 * _GIB
_FOREIGN_VRAM = 3 * _GIB
#: The machine's GPU the pod is given, and so the one it numbers 0.
_GIVEN = 3
_OTHER = 5
_NOBODY = OwnProcesses(frozenset())


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(card_occupancy, "time", _Clock())


@pytest.fixture
def pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KernelTree:
    """A one-GPU pod: a pid namespace of its own, the machine's GPU ``_GIVEN`` the only GPU node it may read."""
    tree = KernelTree(tmp_path, pid_namespace="pid:[4026552279]")
    tree.opened_only(1678127)
    tree.show_only(_GIVEN)
    tree.refuse_denied(monkeypatch)
    return tree


def _serve(tree: KernelTree) -> None:
    tree.holds(1827025, _GIVEN, vram=_SERVER_VRAM, queues=6)
    tree.process(6495)
    tree.renders(6495, _GIVEN, kib=_SERVER_VRAM >> 10)


def _read(tree: KernelTree, cards: tuple[int, ...] = (0,)) -> CardReading:
    return read_holders(cards, kfd=tree.kfd, procfs=tree.procfs)


def test_a_pod_numbers_its_card_among_the_nodes_it_can_read_and_counts_its_server_there(pod: KernelTree) -> None:
    _serve(pod)
    pod.holds(1900001, _OTHER, vram=_FOREIGN_VRAM, queues=2)  # another pod's GPU: not this card

    reading = _read(pod)

    assert (reading.unavailable, reading.denied, reading.method) == (None, False, METHOD_CENSUS_FDINFO)
    assert [(h.card, h.pid, h.vram) for h in reading.holders] == [(0, 1827025, _SERVER_VRAM)]
    assert reading.unaccounted == {0: 0}
    assert reading.intrusions(_NOBODY, before_test=False) == ()


def test_a_pod_catches_a_foreign_holder_of_its_card_before_and_while_the_test_runs(pod: KernelTree) -> None:
    pod.holds(1845543, _GIVEN, vram=_FOREIGN_VRAM, queues=2)
    assert [i.key for i in _read(pod).intrusions(_NOBODY, before_test=True)] == [("host-pid", 1845543, 0)]

    _serve(pod)
    during = _read(pod).intrusions(_NOBODY, before_test=False)

    assert [i.key for i in during] == [("crowded", 0), ("unaccounted", 0)]


def test_the_lease_census_numbers_a_pods_card_among_the_nodes_it_can_read(pod: KernelTree) -> None:
    pod.holds(1845543, _GIVEN, vram=_FOREIGN_VRAM, queues=2)

    listed = read_listed((0,), kfd=pod.kfd)

    assert (listed.unavailable, [(h.card, h.pid) for h in listed.holders]) == (None, [(0, 1845543)])


def test_a_view_refused_every_gpu_node_is_denied_never_read_as_nobody_holding_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = KernelTree(tmp_path, pid_namespace="pid:[4026552279]")
    tree.show_only()
    tree.refuse_denied(monkeypatch)

    reading = _read(tree)

    assert reading.denied and reading.holders == ()
    assert reading.unavailable is not None and "every one of the 8 GPU nodes" in reading.unavailable
    assert "permission denied" in reading.unavailable


def test_a_card_beyond_the_nodes_a_pod_can_read_is_denied_while_one_beyond_the_machine_is_only_unavailable(
    pod: KernelTree, tmp_path: Path
) -> None:
    beyond_view = _read(pod, (1,))
    whole = KernelTree(tmp_path / "host")

    beyond_machine = _read(whole, (8,))

    assert beyond_view.denied and beyond_view.unavailable is not None
    assert "not among the 1 GPUs this view can read (7 more are refused to it" in beyond_view.unavailable
    assert (
        not beyond_machine.denied
        and beyond_machine.unavailable == "card(s) [8] are not among the 8 GPUs the kernel lists"
    )


def test_a_view_refused_the_process_list_or_a_process_entry_is_denied(
    pod: KernelTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod.holds(1845543, _GIVEN, vram=_FOREIGN_VRAM, queues=2)
    pod.denied.add(pod.kfd / "proc" / "1845543")
    entry = _read(pod)
    pod.denied.add(pod.kfd / "proc")
    listing = _read(pod)

    assert (
        entry.denied
        and entry.unavailable is not None
        and entry.unavailable.endswith("cannot be listed: permission denied")
    )
    assert listing.denied and listing.unavailable is not None and "process list" in listing.unavailable
