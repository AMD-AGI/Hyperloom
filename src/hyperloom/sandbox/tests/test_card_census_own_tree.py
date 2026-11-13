# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test's own process tree is one resident of a card, however many of its processes hold memory there.

The fixtures are shaped like what a GPU pod with a pid namespace of its own reads while a TP4 server runs: the kernel lists
the server's launcher (``python -m sglang.launch_server``, the parent of every rank) with 0.7 GiB on the first rank's card
once that rank has its memory, and the rank itself with most of the card. The pod sees both through their render
descriptors, under the test's runner. The numbers are the ones such a pod read when its card guard refused every TP4 test.

The whole server (:func:`_tp4_round`) is laid out as such a pod read it with all four ranks up: each rank is listed by the
kernel on its own card only, and reports its whole memory (about 169.5 GiB) on the render node of every card of the server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.card_occupancy import METHOD_CENSUS_FDINFO, CardReading, OwnProcesses, read_holders
from hyperloom.sandbox.tests.kernel_tree import KernelTree

_CARD = 4
_LAUNCHER_HOST_VRAM = 739987456  # what the kernel lists for the launcher on the first rank's card
_RANK_HOST_VRAM = int(162.4 * (1 << 30))
_LAUNCHER_POD_KIB = int(0.2 * (1 << 20))  # what the launcher's render descriptor says
_RANK_POD_KIB = int(170.8 * (1 << 20))
_FOREIGN_VRAM = 3 << 30
_TP4_CARDS = (4, 5, 6, 7)
_TP4_RANKS = (642890, 642891, 642892, 642893)
_TP4_RANK_HOSTS = (3658436, 3658437, 3658438, 3658439)
_PEER_POD_KIB = int(169.5 * (1 << 20))  # what each rank's descriptor of each render node says
_LIVE_FOREIGN_VRAM = int(2.5 * (1 << 30))  # the separate process put beside a TP4 server


@pytest.fixture(autouse=True)
def _no_settling(monkeypatch: pytest.MonkeyPatch) -> None:
    """An excess is read again for seconds before it counts; these readings have none worth waiting for."""
    monkeypatch.setattr(card_occupancy, "EXCESS_SETTLE_SEC", 0.0)


def _pod(root: Path) -> KernelTree:
    return KernelTree(root, pid_namespace="pid:[4026532001]")


def _round(
    tree: KernelTree, *, runner: int, launcher: int, rank: int, launcher_host: int, rank_host: int
) -> OwnProcesses:
    """A round's server on ``_CARD``: its runner, the launcher under it and the first rank under the launcher, each listed
    by the kernel under a host id and seen in the pod under its own id; the round owns what is under its runner."""
    tree.process(runner, command=("python3", "-m", "hyperloom.measure.runner"))
    tree.process(launcher, parent=runner, command=("/opt/venv/bin/python", "-m", "sglang.launch_server", "--tp", "4"))
    tree.process(rank, parent=launcher, command=("sglang::scheduler_TP0_EP0",))
    tree.holds(launcher_host, _CARD, vram=_LAUNCHER_HOST_VRAM, queues=0)
    tree.holds(rank_host, _CARD, vram=_RANK_HOST_VRAM, queues=6)
    tree.renders(launcher, _CARD, kib=_LAUNCHER_POD_KIB)
    tree.renders(rank, _CARD, kib=_RANK_POD_KIB)
    return OwnProcesses(frozenset({runner}), procfs=tree.procfs)


def _own_round(tree: KernelTree) -> OwnProcesses:
    return _round(tree, runner=641900, launcher=641919, rank=642890, launcher_host=3656589, rank_host=3658436)


def _keys(reading: CardReading, own: OwnProcesses) -> list[tuple[str | int, ...]]:
    assert reading.unavailable is None and reading.method == METHOD_CENSUS_FDINFO
    return [i.key for i in reading.intrusions(own, before_test=False)]


def test_the_launcher_and_its_first_rank_both_the_tests_own_are_one_resident_and_the_card_is_clean(
    tmp_path: Path,
) -> None:
    tree = _pod(tmp_path)
    own = _own_round(tree)

    reading = read_holders([_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert sorted((h.pid, h.vram, h.queues) for h in reading.holders) == [
        (3656589, _LAUNCHER_HOST_VRAM, 0),
        (3658436, _RANK_HOST_VRAM, 6),
    ]
    assert _keys(reading, own) == []


def test_a_foreign_holder_with_memory_beside_the_tests_own_tree_still_crowds_the_card(tmp_path: Path) -> None:
    """A process outside the pod with memory and queues on the card: the pod's descriptors already say more than the
    kernel lists, so no memory is unaccounted for, and only the count of residents finds it."""
    tree = _pod(tmp_path)
    own = _own_round(tree)
    tree.holds(3700001, _CARD, vram=_FOREIGN_VRAM, queues=2)

    reading = read_holders([_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert reading.unaccounted == {_CARD: 0}
    found = reading.intrusions(own, before_test=False)
    assert [i.key for i in found] == [("crowded", _CARD)]
    assert found[0].what.startswith(
        "3 processes with memory on card 4 at once (host pid 3656589 with 0.7 GiB, host pid 3658436 with 162.4 GiB, "
        "host pid 3700001 with 3.0 GiB); this container's processes with memory on its render node: "
        "pid 641919 (the test) with 0.2 GiB: /opt/venv/bin/python -m sglang.launch_server --tp 4; "
        "pid 642890 (the test) with 170.8 GiB: sglang::scheduler_TP0_EP0"
    )


def test_two_foreign_holders_with_memory_crowd_the_card(tmp_path: Path) -> None:
    """Two holders, neither of them the test's: the pod's descriptors account for their memory (another round's
    launcher and rank in the same pod), and they are still two residents of a card the test is on."""
    tree = _pod(tmp_path)
    _own_round(tree)
    nobody = OwnProcesses(frozenset({999}), procfs=tree.procfs)

    reading = read_holders([_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert _keys(reading, nobody) == [("crowded", _CARD)]


def test_another_rounds_launcher_on_this_rounds_card_is_not_this_rounds_own_and_crowds_it(tmp_path: Path) -> None:
    """Two rounds in one pod: round A's launcher holds memory on a card that is round B's, beside B's rank. A's process
    tree is A's, not B's, so on B's card it is foreign, as before."""
    tree = _pod(tmp_path)
    _round(tree, runner=500, launcher=501, rank=502, launcher_host=3500001, rank_host=3500002)
    tree.process(600, command=("python3", "-m", "hyperloom.measure.runner"))
    round_b = OwnProcesses(frozenset({600}), procfs=tree.procfs)

    reading = read_holders([_CARD], kfd=tree.kfd, procfs=tree.procfs)

    found = reading.intrusions(round_b, before_test=False)
    assert [i.key for i in found] == [("crowded", _CARD)]
    assert "pid 501 (not the test)" in found[0].what and "pid 502 (not the test)" in found[0].what


def _tp4_round(tree: KernelTree) -> OwnProcesses:
    """The round of :func:`_own_round` with its server's three other ranks up as well: TP1 to TP3 under the launcher, each
    listed by the kernel on its own card with its memory and queues, and every rank's descriptor of every render node of
    the server saying its whole memory. The launcher has every render node open, with almost nothing on the others."""
    own = _own_round(tree)
    for index, (rank, host, card) in enumerate(zip(_TP4_RANKS, _TP4_RANK_HOSTS, _TP4_CARDS)):
        if index:
            tree.process(rank, parent=641919, command=(f"sglang::scheduler_TP{index}_EP{index}",))
            tree.holds(host, card, vram=_RANK_HOST_VRAM, queues=6)
        for node in _TP4_CARDS:
            if (index, node) != (0, _CARD):
                tree.renders(rank, node, kib=_RANK_POD_KIB if index == 0 else _PEER_POD_KIB, descriptor=10 + node)
    for node in _TP4_CARDS[1:]:
        tree.renders(641919, node, kib=4 << 10, descriptor=10 + node)
    return own


def _read_tp4(tree: KernelTree) -> CardReading:
    return read_holders(list(_TP4_CARDS), kfd=tree.kfd, procfs=tree.procfs)


def test_a_tp4_servers_ranks_listed_on_every_render_node_leave_each_card_clean(tmp_path: Path) -> None:
    tree = _pod(tmp_path)
    own = _tp4_round(tree)

    reading = _read_tp4(tree)

    assert [(h.card, h.pid) for h in reading.holders] == [
        (4, 3656589),
        (4, 3658436),
        (5, 3658437),
        (6, 3658438),
        (7, 3658439),
    ]
    on_first = [c for c in reading.render_clients if c.card == _CARD and c.vram > card_occupancy.NEGLIGIBLE_VRAM]
    assert sorted(pid for c in on_first for pid, _ in c.processes) == [641919, *_TP4_RANKS]
    assert dict(reading.unaccounted) == dict.fromkeys(_TP4_CARDS, 0)
    assert _keys(reading, own) == []


def test_a_foreign_holder_beside_a_tp4_servers_launcher_and_first_rank_crowds_the_first_card(tmp_path: Path) -> None:
    """A process outside the pod with 2.5 GiB and two queues on the first card: the five open DRM files of the test on its
    render node do not let it pass, the launcher and the rank do."""
    tree = _pod(tmp_path)
    own = _tp4_round(tree)
    tree.holds(3700001, _CARD, vram=_LIVE_FOREIGN_VRAM, queues=2)

    assert _keys(_read_tp4(tree), own) == [("crowded", _CARD)]


@pytest.mark.parametrize(
    ("vram", "queues"), [(_LIVE_FOREIGN_VRAM, 2), (512 << 20, 0)], ids=["computing", "launcher-sized"]
)
def test_a_foreign_holder_on_a_tp4_servers_second_card_crowds_it(tmp_path: Path, vram: int, queues: int) -> None:
    """Card 5 is TP1's alone: a second holder there is foreign, even one with no more than a launcher's memory, as the
    test's launcher holds its memory on the first card."""
    tree = _pod(tmp_path)
    own = _tp4_round(tree)
    tree.holds(3700001, 5, vram=vram, queues=queues)

    assert _keys(_read_tp4(tree), own) == [("crowded", 5)]


def test_a_launcher_seen_with_memory_on_every_render_node_still_has_one_place_on_the_first_card(tmp_path: Path) -> None:
    """Were the launcher's descriptors to say its memory on every render node of the server, it is still one launcher:
    its place goes to the first card, and a launcher-sized holder on card 5 beside TP1 is foreign."""
    tree = _pod(tmp_path)
    own = _tp4_round(tree)
    for node in _TP4_CARDS[1:]:
        tree.renders(641919, node, kib=_LAUNCHER_POD_KIB, descriptor=20 + node)
    tree.holds(3700001, 5, vram=512 << 20, queues=0)

    assert _keys(_read_tp4(tree), own) == [("crowded", 5)]


def test_a_launcher_sized_holder_on_a_card_where_the_launcher_holds_nothing_is_foreign(tmp_path: Path) -> None:
    """While the ranks load, the launcher has almost nothing on its first card yet, so no card needs its place; a
    launcher-sized holder on card 5, whose render node the launcher holds almost nothing on, is still foreign."""
    tree = _pod(tmp_path)
    own = _tp4_round(tree)
    (tree.kfd / "proc" / "3656589" / f"vram_{tree.gpu_id(_CARD)}").write_text(f"{2 << 20}\n")
    tree.holds(3700001, 5, vram=512 << 20, queues=0)

    assert _keys(_read_tp4(tree), own) == [("crowded", 5)]


def test_a_launcher_sized_foreign_holder_beside_the_launcher_and_its_rank_still_crowds_the_card(tmp_path: Path) -> None:
    """The launcher is allowed beside its rank once: a third holder on the first card with no more memory than it has is
    foreign."""
    tree = _pod(tmp_path)
    own = _tp4_round(tree)
    tree.holds(3700001, _CARD, vram=512 << 20, queues=0)

    assert _keys(_read_tp4(tree), own) == [("crowded", _CARD)]


def test_a_holder_with_more_than_a_launchers_memory_beside_the_rank_is_not_the_launcher(tmp_path: Path) -> None:
    """The launcher's place on its card is for a holder with no more than :data:`LAUNCHER_VRAM`: a process outside the pod
    with more, while the launcher has almost nothing yet, is a second resident."""
    tree = _pod(tmp_path)
    tree.process(641900, command=("python3", "-m", "hyperloom.measure.runner"))
    tree.process(641919, parent=641900, command=("/opt/venv/bin/python", "-m", "sglang.launch_server", "--tp", "4"))
    tree.process(642890, parent=641919, command=("sglang::scheduler_TP0_EP0",))
    tree.holds(3656589, _CARD, vram=2 << 20, queues=0)
    tree.holds(3658436, _CARD, vram=_RANK_HOST_VRAM, queues=6)
    tree.holds(3700001, _CARD, vram=_LIVE_FOREIGN_VRAM, queues=0)
    tree.renders(641919, _CARD, kib=_LAUNCHER_POD_KIB)
    tree.renders(642890, _CARD, kib=_RANK_POD_KIB)
    own = OwnProcesses(frozenset({641900}), procfs=tree.procfs)

    reading = read_holders([_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert _keys(reading, own) == [("crowded", _CARD)]


def test_a_second_rank_of_the_tests_server_is_the_one_resident_of_its_own_card(tmp_path: Path) -> None:
    """TP1 of the server, listed by the kernel on card 5 and seen on both render nodes: it is card 5's one resident, and
    on card 4 it does not count as one of the test's holders."""
    tree = _pod(tmp_path)
    own = _own_round(tree)
    tree.process(642891, parent=641919, command=("sglang::scheduler_TP1_EP1",))
    tree.holds(3658437, 5, vram=_RANK_HOST_VRAM, queues=6)
    tree.renders(642891, 5, kib=_PEER_POD_KIB, descriptor=4)
    tree.renders(642891, _CARD, kib=_PEER_POD_KIB, descriptor=5)
    tree.renders(642890, 5, kib=_RANK_POD_KIB, descriptor=5)

    reading = read_holders([_CARD, 5], kfd=tree.kfd, procfs=tree.procfs)

    assert _keys(reading, own) == []
    tree.holds(3700001, _CARD, vram=_FOREIGN_VRAM, queues=2)
    assert _keys(read_holders([_CARD, 5], kfd=tree.kfd, procfs=tree.procfs), own) == [("crowded", _CARD)]


def test_a_process_of_the_test_that_started_no_holder_has_no_launchers_place(tmp_path: Path) -> None:
    """Two processes of the test side by side under its runner, both holding memory on the card: neither started the
    other, so neither is a launcher, and the card has one resident too many."""
    tree = _pod(tmp_path)
    tree.process(641900, command=("python3", "-m", "hyperloom.measure.runner"))
    tree.process(641919, parent=641900, command=("python3", "-m", "helper"))
    tree.process(642890, parent=641900, command=("sglang::scheduler_TP0_EP0",))
    tree.holds(3656589, _CARD, vram=_LAUNCHER_HOST_VRAM, queues=0)
    tree.holds(3658436, _CARD, vram=_RANK_HOST_VRAM, queues=6)
    tree.renders(641919, _CARD, kib=_LAUNCHER_POD_KIB)
    tree.renders(642890, _CARD, kib=_RANK_POD_KIB)
    own = OwnProcesses(frozenset({641900}), procfs=tree.procfs)

    reading = read_holders([_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert _keys(reading, own) == [("crowded", _CARD)]


def test_a_child_sharing_its_parents_render_file_adds_no_allowance_and_a_foreign_holder_still_crowds_the_card(
    tmp_path: Path,
) -> None:
    """A child the rank forked after opening the GPU shares the rank's open DRM file and has no holder of its own in the
    kernel's list: the test's own render clients are still two, so a process outside the pod with memory and queues on
    the card is a third resident."""
    tree = _pod(tmp_path)
    own = _own_round(tree)
    tree.process(642999, parent=642890, command=("python3", "-c", "worker"))
    tree.renders(642999, _CARD, kib=_RANK_POD_KIB, client="642890-3")
    tree.holds(3700001, _CARD, vram=_FOREIGN_VRAM, queues=2)

    reading = read_holders([_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert [c.processes for c in reading.render_clients if c.card == _CARD and len(c.processes) > 1] == [
        ((642890, "sglang::scheduler_TP0_EP0"), (642999, "python3 -c worker"))
    ]
    assert _keys(reading, own) == [("crowded", _CARD)]


def test_a_rank_whose_forked_child_shares_its_render_files_is_not_a_launcher(tmp_path: Path) -> None:
    """TP1 forks a worker after opening the GPU, which shares TP1's open DRM files: TP1 started no process with a file of
    its own, so it has no launcher's place on its card, and a launcher-sized holder beside it there is foreign."""
    tree = _pod(tmp_path)
    own = _tp4_round(tree)
    tree.process(642998, parent=642891, command=("python3", "-c", "worker"))
    for node in _TP4_CARDS:
        tree.renders(642998, node, kib=_PEER_POD_KIB, client=f"642891-{10 + node}", descriptor=10 + node)
    tree.holds(3700001, 5, vram=512 << 20, queues=0)

    assert _keys(_read_tp4(tree), own) == [("crowded", 5)]


def test_a_child_sharing_its_parents_render_file_alone_leaves_the_card_clean(tmp_path: Path) -> None:
    """The same forked child without a foreign holder: two kernel holders, two own render clients."""
    tree = _pod(tmp_path)
    own = _own_round(tree)
    tree.process(642999, parent=642890, command=("python3", "-c", "worker"))
    tree.renders(642999, _CARD, kib=_RANK_POD_KIB, client="642890-3")

    reading = read_holders([_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert _keys(reading, own) == []


def test_before_the_test_starts_the_tests_own_tree_on_its_card_is_still_named_holder_by_holder(tmp_path: Path) -> None:
    """Before a test starts nothing may hold its card, its own processes included: that rule is not this one."""
    tree = _pod(tmp_path)
    own = _own_round(tree)

    reading = read_holders([_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert [i.key for i in reading.intrusions(own, before_test=True)] == [
        ("host-pid", 3656589, _CARD),
        ("host-pid", 3658436, _CARD),
    ]
