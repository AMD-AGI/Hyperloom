# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test's own process tree is one resident of a card, however many of its processes hold memory there.

The fixtures are shaped like what a GPU pod with a pid namespace of its own reads while a TP4 server runs: the kernel lists
the server's launcher (``python -m sglang.launch_server``, the parent of every rank) with 0.7 GiB on the first rank's card
once that rank has its memory, and the rank itself with most of the card. The pod sees both through their render
descriptors, under the test's runner. The numbers are the ones such a pod read when its card guard refused every TP4 test.
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


def test_a_second_rank_of_the_tests_server_on_the_card_is_still_part_of_one_resident(tmp_path: Path) -> None:
    """A second rank of the same server (a descendant of the launcher) with its own open DRM file and memory on the card
    too: three residents against three own render clients, all the test's, still one."""
    tree = _pod(tmp_path)
    own = _own_round(tree)
    tree.process(642891, parent=641919, command=("sglang::scheduler_TP1_EP1",))
    tree.holds(3658437, _CARD, vram=1 << 30, queues=1)
    tree.renders(642891, _CARD, kib=1 << 20)

    reading = read_holders([_CARD], kfd=tree.kfd, procfs=tree.procfs)

    assert _keys(reading, own) == []


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
