# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test's own process tree may have one holder with memory on a card, a second in a launcher's place, and more only
when each pairs with a process of the test by what that process's DRM files on that card's render node say it holds.

Each launcher of the test (a process of the test that holds memory and started one that holds its own) has one place,
for a holder with no more than :data:`LAUNCHER_VRAM` on a card where the launcher holds memory on the render node.

The fixtures are shaped like what a GPU pod with a pid namespace of its own reads while a TP4 server runs: the kernel lists
the server's launcher (``python -m sglang.launch_server``, the parent of every rank) with 0.7 GiB on the first rank's card
once that rank has its memory, and the rank itself with most of the card. The pod sees both through their render
descriptors, under the test's runner. The numbers are the ones such a pod read when its card guard refused every TP4 test.

The whole server (:func:`_tp4_round`) is laid out as such a pod read it with all four ranks up: each rank is listed by the
kernel on its own card only, and reports its whole memory (about 169.5 GiB) on the render node of every card of the server.

The Kimi round (:func:`_kimi_round`) is the eight-card round production refused for a rank and a spawn child of its
server on card 0, with the numbers of that record.
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


def test_the_launcher_and_its_first_rank_both_the_tests_own_leave_the_card_clean(tmp_path: Path) -> None:
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
    kernel's list: the card still has the rank and the launcher beside it, so a process outside the pod with memory and
    queues on it is a third holder, one more than a card may have."""
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
    """The same forked child without a foreign holder: the card has only the rank and the launcher beside it."""
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


_GIB = 1 << 30
_SPAWN_CARD = 0
#: The Kimi round the guard refused in production, eight cards, read on card 0 while its ranks were loading: the kernel
#: listed host pids 607948 with 1.7 GiB and 609774 with 1.2 GiB there, and the pod's processes 1153 with 1.7 GiB and 1336
#: with 1.2 GiB on its render node, both spawned by the server (``multiprocessing.spawn``). 1153 is a rank of the server.
_KIMI_CARDS = tuple(range(8))
_KIMI_RUNNER, _KIMI_SERVER = 1100, 1120
_KIMI_RANKS = tuple(range(1153, 1161))
_KIMI_RANK_HOSTS = (607948, *range(607949, 607956))  # only card 0's, 607948, is in the record
_KIMI_CHILD, _KIMI_CHILD_HOST = 1336, 609774
_RANK_LOADING_VRAM = int(1.7 * _GIB) >> 10 << 10  # whole KiB, as a descriptor says it
_CHILD_VRAM = int(1.2 * _GIB) >> 10 << 10
#: What a rank's two open DRM files said once loaded, read in production on its render nodes: 171686536 KiB and
#: 172560516 KiB. The record does not say which is its own card's; as in the TP4 reading above, the fixture takes the
#: larger for its own card and the other for every other card of the server.
_RANK_OWN_KIB = 172560516
_RANK_PEER_KIB = 171686536
_SPAWN_COMMAND = (
    "python3",
    "-c",
    "from multiprocessing.spawn import spawn_main; spawn_main()",
    "--multiprocessing-fork",
)
_SOLO_HOST, _SOLO_CHILD_HOST, _OTHER_HOST = 3600001, 3600002, 3700002


def _kimi_round(tree: KernelTree, *, loaded: bool = False) -> OwnProcesses:
    """The production Kimi round: its runner, the server under it, the eight ranks the server spawned (1153 on card 0 to
    1160 on card 7) and the spawn child 1336. Each rank is listed by the kernel on its own card only and has a DRM file on
    the render node of every card of the server, each saying its memory; 1336 holds its memory on card 0 through a file
    of its own. While loading (the record) every rank has 1.7 GiB, as rank 0 did; ``loaded``, every rank's files say the
    production figures and the kernel lists the rank with what its own card's file says (the kernel's figure for a loaded
    rank is not in the record). The round owns what is under its runner."""
    tree.process(_KIMI_RUNNER, command=("python3", "-m", "hyperloom.measure.runner"))
    tree.process(_KIMI_SERVER, parent=_KIMI_RUNNER, command=("python3", "-m", "sglang.launch_server", "--tp", "8"))
    own_kib, peer_kib = (_RANK_OWN_KIB, _RANK_PEER_KIB) if loaded else (_RANK_LOADING_VRAM >> 10,) * 2
    for rank, host, card in zip(_KIMI_RANKS, _KIMI_RANK_HOSTS, _KIMI_CARDS):
        tree.process(rank, parent=_KIMI_SERVER, command=_SPAWN_COMMAND)
        tree.holds(host, card, vram=own_kib << 10, queues=4)
        for node in _KIMI_CARDS:
            tree.renders(rank, node, kib=own_kib if node == card else peer_kib, descriptor=10 + node)
    tree.process(_KIMI_CHILD, parent=_KIMI_SERVER, command=_SPAWN_COMMAND)
    tree.holds(_KIMI_CHILD_HOST, _SPAWN_CARD, vram=_CHILD_VRAM, queues=2)
    tree.renders(_KIMI_CHILD, _SPAWN_CARD, kib=_CHILD_VRAM >> 10)
    return OwnProcesses(frozenset({_KIMI_RUNNER}), procfs=tree.procfs)


def _read_kimi(tree: KernelTree) -> CardReading:
    return read_holders(list(_KIMI_CARDS), kfd=tree.kfd, procfs=tree.procfs)


@pytest.mark.parametrize("loaded", [False, True], ids=["loading-as-recorded", "loaded"])
def test_the_production_kimi_rounds_rank_and_spawn_child_on_card_0_leave_every_card_clean(
    tmp_path: Path, loaded: bool
) -> None:
    """Card 0 has two holders, rank 0 and the 1.2 GiB spawn child, more than a launcher's memory: each pairs with what the
    test's process holds on card 0 by its files there, rank 0's among files on all eight render nodes, so no card is
    crowded. The other ranks' files on card 0's node are left over and do not count against it."""
    tree = _pod(tmp_path)
    own = _kimi_round(tree, loaded=loaded)

    reading = _read_kimi(tree)

    assert sorted((h.pid, h.vram) for h in reading.holders if h.card == _SPAWN_CARD) == sorted(
        [(607948, _RANK_LOADING_VRAM if not loaded else _RANK_OWN_KIB << 10), (_KIMI_CHILD_HOST, _CHILD_VRAM)]
    )
    assert _CHILD_VRAM > card_occupancy.LAUNCHER_VRAM
    assert dict(reading.unaccounted) == dict.fromkeys(_KIMI_CARDS, 0)
    assert _keys(reading, own) == []


@pytest.mark.parametrize("case", ["child-on-two-cards", "child-with-two-files-on-card-0"])
def test_a_process_of_the_test_pairs_by_what_it_holds_on_each_card(tmp_path: Path, case: str) -> None:
    """What a process holds on a card is the sum of its files on that card's render node, and it pairs on every card it
    holds memory on: the spawn child holding 1.2 GiB on cards 0 and 1, listed by the kernel on both, pairs on each (beside
    rank 0 on card 0 and rank 1 on card 1); the child's 1.2 GiB on card 0 split over two files of 0.7 and 0.5 GiB pairs
    as one process with its one holder."""
    tree = _pod(tmp_path)
    own = _kimi_round(tree)
    if case == "child-on-two-cards":
        tree.holds(_KIMI_CHILD_HOST, 1, vram=_CHILD_VRAM, queues=2)
        tree.renders(_KIMI_CHILD, 1, kib=_CHILD_VRAM >> 10, descriptor=4)
    else:
        (tree.procfs / str(_KIMI_CHILD) / "fd" / "3").unlink()
        tree.renders(_KIMI_CHILD, _SPAWN_CARD, kib=int(0.7 * _GIB) >> 10, descriptor=4)
        tree.renders(_KIMI_CHILD, _SPAWN_CARD, kib=(_CHILD_VRAM - int(0.7 * _GIB)) >> 10, descriptor=5)

    assert _keys(_read_kimi(tree), own) == []


@pytest.mark.parametrize(
    ("loaded", "card", "vram"),
    [(False, _SPAWN_CARD, 20 * _GIB), (False, 3, int(0.6 * _GIB)), (False, 3, int(2.8 * _GIB)), (True, 3, 20 * _GIB)],
    ids=["loading-card-0-20GiB", "loading-card-3-0.6GiB", "loading-card-3-2.8GiB", "loaded-card-3-20GiB"],
)
def test_a_foreign_holder_of_other_memory_than_any_process_of_the_test_on_the_card_still_crowds_it(
    tmp_path: Path, loaded: bool, card: int, vram: int
) -> None:
    """A process outside the pod on a card of the Kimi round, its memory more than the tolerance away from what every
    process of the test holds on that card (1.2 and 1.7 GiB while loading, about 164 GiB loaded), is a holder no process
    of the test pairs with: that card is crowded, and only that card (20 GiB beside the loading ranks is also memory no
    process of the pod accounts for)."""
    tree = _pod(tmp_path)
    own = _kimi_round(tree, loaded=loaded)
    tree.holds(_OTHER_HOST, card, vram=vram, queues=2)

    assert [key for key in _keys(_read_kimi(tree), own) if key[0] == "crowded"] == [("crowded", card)]


def test_accepted_limitation_a_foreign_holder_of_a_ranks_memory_beside_another_rank_reads_as_the_tests(
    tmp_path: Path,
) -> None:
    """The bound of pairing by memory: every rank reports its memory on every card's render node, so on card 3 seven
    ranks' 1.7 GiB stand for no holder of card 3. A process outside the pod with 1.7 GiB beside rank 3 pairs with one of
    them and passes; one of the same memory as a process of the test cannot be told from the test's own."""
    tree = _pod(tmp_path)
    own = _kimi_round(tree)
    tree.holds(_OTHER_HOST, 3, vram=_RANK_LOADING_VRAM, queues=2)

    assert _keys(_read_kimi(tree), own) == []


def _solo_server(tree: KernelTree, *, child_owner: int = 2100) -> OwnProcesses:
    """A made-up single-card server under the test's runner that started a ``multiprocessing`` spawn child, each holding
    memory of its own on the card through a DRM file of its own (1.7 GiB and 1.2 GiB): the kernel lists both on the card.
    ``child_owner`` is the process the child is under: the server, or a process of another round."""
    tree.process(2100, command=("python3", "-m", "hyperloom.measure.runner"))
    tree.process(2153, parent=2100, command=("python3", "-m", "server"))
    tree.process(2336, parent=child_owner, command=_SPAWN_COMMAND)
    tree.holds(_SOLO_HOST, _SPAWN_CARD, vram=_RANK_LOADING_VRAM, queues=4)
    tree.holds(_SOLO_CHILD_HOST, _SPAWN_CARD, vram=_CHILD_VRAM, queues=2)
    tree.renders(2153, _SPAWN_CARD, kib=_RANK_LOADING_VRAM >> 10)
    tree.renders(2336, _SPAWN_CARD, kib=_CHILD_VRAM >> 10)
    return OwnProcesses(frozenset({2100}), procfs=tree.procfs)


@pytest.mark.parametrize(
    "case",
    [
        "third-holder",
        "child-not-the-tests",
        "child-without-a-file",
        "child-file-with-less-memory",
    ],
)
def test_a_holder_not_matched_to_the_tests_own_files_still_crowds_the_card(tmp_path: Path, case: str) -> None:
    """A 1.2 GiB holder the test's files do not account for one to one is foreign as before: a third holder beside the
    server and its child; the same child under another round's process; a child the pod has no file of (a process outside
    the pod); and a child whose file says much less than the holder's 1.2 GiB."""
    tree = _pod(tmp_path)
    tree.process(999, command=("sleep", "infinity"))
    own = _solo_server(tree, child_owner=999 if case == "child-not-the-tests" else 2153)
    if case == "third-holder":
        tree.holds(_OTHER_HOST, _SPAWN_CARD, vram=_CHILD_VRAM, queues=2)
    elif case == "child-without-a-file":
        (tree.procfs / "2336" / "fd" / "3").unlink()
    elif case == "child-file-with-less-memory":
        (tree.procfs / "2336" / "fdinfo" / "3").write_text(
            f"pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t2336-3\ndrm-memory-vram:\t{(512 << 20) >> 10} KiB\n"
        )

    reading = read_holders([_SPAWN_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert [i.key for i in reading.intrusions(own, before_test=False) if i.key[0] == "crowded"] == [
        ("crowded", _SPAWN_CARD)
    ]


@pytest.mark.parametrize(
    "files", [(int(1.2 * (1 << 30)), 2 << 30), (2 << 30, int(1.2 * (1 << 30)))], ids=["1.2-2.0", "2.0-1.2"]
)
def test_each_holder_is_paired_with_a_file_of_its_own_whichever_file_it_would_take_first(
    tmp_path: Path, files: tuple[int, int]
) -> None:
    """A holder whose memory is near both files' does not keep the only file the other holder's memory is near; both have
    more than a launcher's memory, so only the pairing lets them pass."""
    tree = _pod(tmp_path)
    tree.process(2100, command=("python3", "-m", "hyperloom.measure.runner"))
    tree.process(2153, parent=2100, command=("python3", "-m", "server"))
    tree.process(2336, parent=2153, command=_SPAWN_COMMAND)
    tree.holds(_SOLO_HOST, _SPAWN_CARD, vram=int(1.6 * (1 << 30)), queues=4)
    tree.holds(_SOLO_CHILD_HOST, _SPAWN_CARD, vram=int(1.1 * (1 << 30)), queues=2)
    tree.renders(2153, _SPAWN_CARD, kib=files[0] >> 10)
    tree.renders(2336, _SPAWN_CARD, kib=files[1] >> 10)
    own = OwnProcesses(frozenset({2100}), procfs=tree.procfs)

    assert _keys(read_holders([_SPAWN_CARD], kfd=tree.kfd, procfs=tree.procfs), own) == []


@pytest.mark.parametrize(
    "case", ["file-shared-with-a-process-not-the-tests", "file-on-another-cards-node", "negligible-file"]
)
def test_a_file_pairs_only_when_all_its_processes_are_the_tests_it_is_on_this_cards_node_and_holds_memory(
    tmp_path: Path, case: str
) -> None:
    """A second holder on the server's card with more than a launcher's memory pairs with none of the test's files: the
    child's file is also held by a process of the pod outside the test's tree (one process of the test is not enough);
    the test's other process holds its file on another card's render node only; or that file holds almost no memory
    (:data:`NEGLIGIBLE_VRAM`), which would otherwise pair with any holder of up to the tolerance."""
    tree = _pod(tmp_path)
    tree.process(2100, command=("python3", "-m", "hyperloom.measure.runner"))
    tree.process(2153, parent=2100, command=("python3", "-m", "server"))
    tree.process(2336, parent=2100, command=("python3", "-m", "helper"))
    tree.process(999, command=("sleep", "infinity"))
    tree.holds(_SOLO_HOST, _SPAWN_CARD, vram=_RANK_LOADING_VRAM, queues=4)
    tree.renders(2153, _SPAWN_CARD, kib=_RANK_LOADING_VRAM >> 10)
    cards = [_SPAWN_CARD]
    if case == "file-shared-with-a-process-not-the-tests":
        tree.holds(_SOLO_CHILD_HOST, _SPAWN_CARD, vram=_CHILD_VRAM, queues=2)
        tree.renders(2336, _SPAWN_CARD, kib=_CHILD_VRAM >> 10, client="2336-3")
        tree.renders(999, _SPAWN_CARD, kib=_CHILD_VRAM >> 10, client="2336-3")
    elif case == "file-on-another-cards-node":
        tree.holds(_OTHER_HOST, _SPAWN_CARD, vram=_CHILD_VRAM, queues=2)
        tree.holds(_SOLO_CHILD_HOST, 1, vram=_CHILD_VRAM, queues=2)
        tree.renders(2336, 1, kib=_CHILD_VRAM >> 10)
        cards = [_SPAWN_CARD, 1]
    else:
        tree.holds(_OTHER_HOST, _SPAWN_CARD, vram=400 << 20, queues=2)
        tree.renders(2336, _SPAWN_CARD, kib=(4 << 20) >> 10)
    own = OwnProcesses(frozenset({2100}), procfs=tree.procfs)

    reading = read_holders(cards, kfd=tree.kfd, procfs=tree.procfs)

    assert reading.method == METHOD_CENSUS_FDINFO
    assert [i.key for i in reading.intrusions(own, before_test=False) if i.key[0] == "crowded"] == [
        ("crowded", _SPAWN_CARD)
    ]


def _gib(value: float) -> int:
    return int(value * (1 << 30))


@pytest.mark.parametrize(
    ("files", "listed", "clean"),
    [
        ((_gib(2.7), _gib(2.2)), (_gib(2.2), _gib(1.7)), True),
        ((_gib(1.7), _gib(1.3)), (_gib(1.3), _gib(2.1)), True),
        ((_gib(1.7), _gib(1.2)), (_gib(1.7), _gib(2.8)), False),
    ],
    ids=["within-bound-of-the-listed-childs-file", "within-bound-of-a-listed-file", "beyond-bound-of-every-file"],
)
def test_accepted_limitation_a_foreign_holder_near_a_file_of_the_test_reads_as_the_tests_while_one_is_not_listed(
    tmp_path: Path, files: tuple[int, int], listed: tuple[int, int], clean: bool
) -> None:
    """The accepted limitation, at the bound the pairing has: with one of the test's processes not yet listed by the
    kernel, a foreign holder passes when it is within :data:`UNACCOUNTED_VRAM_TOLERANCE` of any file of the test on the
    card and the listed holder pairs with the remaining file. Files of 2.7 and 2.2 GiB, the 2.2 GiB child listed and a
    foreign 1.7 GiB holder (1.0 GiB from the unlisted file): clean. Files of 1.7 and 1.3 GiB, the server listed at
    1.3 GiB and a foreign 2.1 GiB holder: clean. A foreign 2.8 GiB holder beside files of 1.7 and 1.2 GiB is within the
    tolerance of none of them and crowds the card."""
    tree = _pod(tmp_path)
    tree.process(2100, command=("python3", "-m", "hyperloom.measure.runner"))
    tree.process(2153, parent=2100, command=("python3", "-m", "server"))
    tree.process(2336, parent=2153, command=_SPAWN_COMMAND)
    tree.renders(2153, _SPAWN_CARD, kib=files[0] >> 10)
    tree.renders(2336, _SPAWN_CARD, kib=files[1] >> 10)
    tree.holds(_SOLO_HOST, _SPAWN_CARD, vram=listed[0], queues=4)
    tree.holds(_OTHER_HOST, _SPAWN_CARD, vram=listed[1], queues=2)
    own = OwnProcesses(frozenset({2100}), procfs=tree.procfs)

    reading = read_holders([_SPAWN_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert reading.method == METHOD_CENSUS_FDINFO
    crowded = [i.key for i in reading.intrusions(own, before_test=False) if i.key[0] == "crowded"]
    if clean:
        assert _keys(reading, own) == []
    else:
        assert crowded == [("crowded", _SPAWN_CARD)]


def test_a_file_of_the_test_with_no_listed_holder_does_not_count_against_the_card(tmp_path: Path) -> None:
    """A server and two spawn children with files of 1.7, 1.2 and 0.9 GiB, the third child not listed by the kernel yet:
    the two listed holders (1.7 and 1.2 GiB) pair with two of the files, and the file left over does not count against
    the card, so it is clean."""
    tree = _pod(tmp_path)
    own = _solo_server(tree, child_owner=2153)
    tree.process(2337, parent=2153, command=_SPAWN_COMMAND)
    tree.renders(2337, _SPAWN_CARD, kib=_gib(0.9) >> 10)

    reading = read_holders([_SPAWN_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert sorted(h.vram for h in reading.holders) == [_CHILD_VRAM, _RANK_LOADING_VRAM]
    assert _keys(reading, own) == []
