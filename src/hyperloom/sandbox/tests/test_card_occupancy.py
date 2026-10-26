# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which processes hold a card: read from the kernel's list, never guessed, never "nobody" for a list that was not read."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.sandbox.card_occupancy import (
    COMMAND_CHARS,
    METHOD_CENSUS_FDINFO,
    CardGuard,
    CardReading,
    Holder,
    OwnProcesses,
    card_guard,
    own_sandbox_processes,
    read_holders,
)
from hyperloom.sandbox.launcher import LAUNCHER_SOURCE
from hyperloom.sandbox.tests.kernel_tree import KernelTree


def _read(tree: KernelTree, cards: list[int]) -> CardReading:
    return read_holders(cards, kfd=tree.kfd, procfs=tree.procfs)


@pytest.fixture
def tree(tmp_path: Path) -> KernelTree:
    return KernelTree(tmp_path)


def test_a_list_that_was_read_and_holds_nothing_says_clean(tree: KernelTree) -> None:
    reading = _read(tree, [0, 1, 2, 3])

    assert reading == CardReading(holders=(), unavailable=None)


def test_a_process_with_a_queue_and_memory_on_a_card_holds_it_and_is_named(tree: KernelTree) -> None:
    tree.process(4242, command=("python3", "-c", "import torch"))
    tree.holds(4242, 3)

    reading = _read(tree, [2, 3])

    assert reading.unavailable is None
    assert reading.holders == (Holder(card=3, pid=4242, command="python3 -c import torch", vram=1 << 30, queues=1),)
    assert reading.held_cards() == frozenset({3})


def test_a_card_is_the_nth_gpu_node_not_the_nth_node_of_the_topology(tmp_path: Path) -> None:
    """Two CPU nodes come first in the topology; card 0 is still the first GPU, as the runtime numbers its devices."""
    tree = KernelTree(tmp_path, gpus=4, cpus=2)
    tree.process(7)
    tree.holds(7, 0)

    assert _read(tree, [0]).holders == (Holder(0, 7, "python3 work.py", 1 << 30, 1),)
    assert _read(tree, [1, 2, 3]).holders == ()


def test_a_card_nobody_asked_about_is_not_reported(tree: KernelTree) -> None:
    tree.process(10)
    tree.holds(10, 5)

    assert _read(tree, [0, 1]).holders == ()


def test_memory_alone_or_a_queue_alone_holds_a_card(tree: KernelTree) -> None:
    tree.process(11)
    tree.holds(11, 1, vram=4096, queues=0)
    tree.process(12)
    tree.holds(12, 2, vram=0, queues=1)

    assert {(h.pid, h.card) for h in _read(tree, [1, 2]).holders} == {(11, 1), (12, 2)}


def test_a_process_that_only_opened_the_interface_holds_nothing(tree: KernelTree) -> None:
    """``rocminfo`` or a monitor opens the interface on every GPU and leaves no queue and no memory."""
    tree.process(13)
    tree.opened_only(13)

    assert _read(tree, list(range(8))).holders == ()


def test_one_process_on_two_cards_is_two_holders_in_card_order(tree: KernelTree) -> None:
    tree.process(20)
    tree.holds(20, 4)
    tree.holds(20, 2)

    assert [(h.card, h.pid) for h in _read(tree, [2, 4]).holders] == [(2, 20), (4, 20)]


def test_a_dead_process_holds_nothing_however_long_the_kernels_entry_lingers(tree: KernelTree) -> None:
    tree.process(30, state="Z")
    tree.holds(30, 0)

    assert _read(tree, [0]).holders == ()


def test_a_process_that_left_proc_while_the_kernel_still_lists_it_holds_nothing(tree: KernelTree) -> None:
    tree.process(31)
    tree.holds(31, 0)
    tree.exit(31)

    assert _read(tree, [0]).holders == ()


def test_a_command_line_is_cleaned_masked_and_shortened_and_never_carries_the_environment(tree: KernelTree) -> None:
    tree.process(
        40,
        command=("python3", "serve.py", "--token", "ghp_abcdefghijklmnop", "line\nbreak", "x" * 400),
        environ=("SECRET_ENV=do-not-leak",),
    )
    tree.holds(40, 0)

    (holder,) = _read(tree, [0]).holders

    assert "do-not-leak" not in holder.command
    assert "ghp_abcdefghijklmnop" not in holder.command and "ghp_[REDACTED]" in holder.command
    assert "\n" not in holder.command and "line break" in holder.command
    assert len(holder.command) == COMMAND_CHARS


def test_a_process_with_no_command_line_is_named_by_its_program(tree: KernelTree) -> None:
    tree.process(41)
    (tree.procfs / "41" / "cmdline").write_bytes(b"")
    (tree.procfs / "41" / "comm").write_text("kworker/7:1\n")
    tree.holds(41, 0)

    assert _read(tree, [0]).holders[0].command == "[kworker/7:1]"


# --- the probe says so when it cannot say -------------------------------------------------------------------------


def test_without_the_kernels_list_the_probe_cannot_say_and_lists_nobody(tmp_path: Path) -> None:
    tree = KernelTree(tmp_path)
    (tree.kfd / "proc").rmdir()

    reading = _read(tree, [0])

    assert reading.holders == ()
    assert reading.unavailable is not None and "no AMD GPU driver" in reading.unavailable


def test_in_a_pid_namespace_of_its_own_the_probe_counts_the_holders_and_does_not_name_them(tmp_path: Path) -> None:
    tree = KernelTree(tmp_path, pid_namespace="pid:[4026532550]")
    tree.process(50, command=("python3", "pod_process.py"))  # an unrelated process of the pod that has the host's id 50
    tree.holds(4242, 0)

    reading = _read(tree, [0])

    assert reading.unavailable is None
    assert reading.method == METHOD_CENSUS_FDINFO
    assert reading.holders == (Holder(0, 4242, "", 1 << 30, 1),), "the host's id is kept, and no command is read for it"


def test_where_proc_hides_other_accounts_the_probe_cannot_say(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = KernelTree(tmp_path, proc_options="rw,hidepid=2")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)

    reading = _read(tree, [0])

    assert reading.unavailable is not None and "hides the processes of other accounts" in reading.unavailable


def test_root_sees_every_process_whatever_proc_is_mounted_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = KernelTree(tmp_path, proc_options="rw,hidepid=2")
    monkeypatch.setattr(os, "geteuid", lambda: 0)

    assert _read(tree, [0]).unavailable is None


@pytest.mark.parametrize("options", ["rw", "rw,hidepid=0", "rw,hidepid=off"])
def test_a_proc_that_hides_nothing_does_not_stop_the_probe(
    tmp_path: Path, options: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / options.replace(",", "_").replace("=", "-")
    root.mkdir()
    tree = KernelTree(root, proc_options=options)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)

    assert _read(tree, [0]).unavailable is None


def test_a_card_the_kernel_does_not_list_is_not_a_card_nobody_holds(tmp_path: Path) -> None:
    tree = KernelTree(tmp_path, gpus=2)

    reading = _read(tree, [1, 5])

    assert reading.holders == ()
    assert reading.unavailable is not None and "[5] are not among the 2 GPUs" in reading.unavailable


def test_a_topology_node_that_holds_no_number_makes_the_probe_unable_to_say(tmp_path: Path) -> None:
    tree = KernelTree(tmp_path)
    (tree.kfd / "topology" / "nodes" / "3" / "gpu_id").write_text("not a number\n")

    reading = _read(tree, [0])

    assert reading.unavailable is not None and "does not hold a number" in reading.unavailable


def test_a_queue_file_that_holds_no_number_makes_the_probe_unable_to_say(tree: KernelTree) -> None:
    tree.process(60)
    tree.holds(60, 0)
    (tree.kfd / "proc" / "60" / "queues" / "0" / "gpuid").write_text("?\n")

    assert "does not hold a number" in (_read(tree, [0]).unavailable or "")


def test_the_real_machine_reads_without_raising() -> None:
    """On a machine with no GPU driver the probe says so; on one with it, it lists holders; it never raises."""
    reading = read_holders([0])

    assert reading.unavailable is None or isinstance(reading.unavailable, str)
    assert not (reading.unavailable and reading.holders)


# --- whose a process is ------------------------------------------------------------------------------------------------


def test_a_process_under_a_root_is_own_however_deep_and_one_outside_is_not(tree: KernelTree) -> None:
    tree.process(100)
    tree.process(101, parent=100)
    tree.process(102, parent=101)
    tree.process(200)
    own = OwnProcesses(frozenset({100}), procfs=tree.procfs)

    assert [own.owns(pid) for pid in (100, 101, 102, 200)] == [True, True, True, False]


def test_a_process_that_detached_and_was_adopted_by_a_subreaper_root_is_own(tree: KernelTree) -> None:
    """A launcher is the child subreaper of its sandbox: a detached server is re-parented to it, never to init."""
    tree.process(300)
    tree.process(301, parent=300, command=("sglang::scheduler",))

    assert OwnProcesses(frozenset({300}), procfs=tree.procfs).owns(301)


def test_a_process_with_the_sandbox_tag_is_own_even_outside_the_tree(tree: KernelTree) -> None:
    tree.process(400, environ=("PATH=/usr/bin", "E2B_SANDBOX_ID=lsb-1"))
    tree.process(401, environ=("E2B_SANDBOX_ID=lsb-2",))
    own = OwnProcesses(frozenset(), tag="E2B_SANDBOX_ID=lsb-1", procfs=tree.procfs)

    assert [own.owns(400), own.owns(401)] == [True, False]


def test_init_is_nobodys_ancestor_for_the_purpose_of_owning(tree: KernelTree) -> None:
    """A root that is pid 1 would make every process own; the walk stops before it."""
    tree.process(500, parent=1)

    assert not OwnProcesses(frozenset({1}), procfs=tree.procfs).owns(500)


def test_a_parent_loop_in_proc_ends_the_walk(tree: KernelTree) -> None:
    tree.process(600, parent=601)
    tree.process(601, parent=600)

    assert not OwnProcesses(frozenset({700}), procfs=tree.procfs).owns(600)


def test_foreign_keeps_the_holders_that_are_not_own(tree: KernelTree) -> None:
    tree.process(100)
    tree.process(101, parent=100)
    tree.process(900, command=("python3", "intruder.py"))
    tree.holds(101, 0)
    tree.holds(900, 0)
    tree.holds(900, 1)

    reading = _read(tree, [0, 1])
    foreign = reading.foreign(OwnProcesses(frozenset({100}), procfs=tree.procfs))

    assert [(h.pid, h.card) for h in reading.holders] == [(101, 0), (900, 0), (900, 1)]
    assert [(h.pid, h.card, h.command) for h in foreign] == [
        (900, 0, "python3 intruder.py"),
        (900, 1, "python3 intruder.py"),
    ]


# --- the sandbox this process is in ---------------------------------------------------------------------------------


def test_a_runner_under_a_launcher_owns_everything_the_launcher_adopted(tree: KernelTree) -> None:
    """The runner's command may be a shell that starts it: the launcher is further up, and it adopted the detached server."""
    tree.process(10, command=("python3", "-I", "-c", LAUNCHER_SOURCE, "5", "bash", "start.sh"))
    tree.process(11, parent=10, command=("bash", "start.sh"))
    tree.process(12, parent=11, command=("python3", "-m", "runner"))
    tree.process(13, parent=10, command=("sglang::scheduler",))  # detached, then adopted by the launcher
    tree.process(99, command=("python3", "intruder.py"))

    own = own_sandbox_processes({"E2B_SANDBOX_ID": "lsb-7"}, pid=12, procfs=tree.procfs)

    assert own.roots == frozenset({12, 10})
    assert own.tag == "E2B_SANDBOX_ID=lsb-7"
    assert [own.owns(pid) for pid in (12, 13, 99)] == [True, True, False]


def test_a_runner_with_no_launcher_above_it_owns_only_its_own_tree_and_its_tag(tree: KernelTree) -> None:
    tree.process(20, command=("bash",))
    tree.process(21, parent=20, command=("python3", "-m", "runner"))
    tree.process(22, parent=21)
    tree.process(23, parent=20, command=("python3", "other.py"))

    own = own_sandbox_processes({}, pid=21, procfs=tree.procfs)

    assert own.roots == frozenset({21}) and own.tag == ""
    assert [own.owns(pid) for pid in (22, 23)] == [True, False]


def test_a_launcher_is_told_by_the_source_it_runs_not_by_its_name(tree: KernelTree) -> None:
    tree.process(30, command=("python3", "-I", "-c", "print('launcher')", "5", "x"))
    tree.process(31, parent=30)

    assert own_sandbox_processes({}, pid=31, procfs=tree.procfs).roots == frozenset({31})


# --- the guard a process names ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, CardGuard.ENFORCE),
        ({"HYPERLOOM_CARD_GUARD": ""}, CardGuard.ENFORCE),
        ({"HYPERLOOM_CARD_GUARD": "observe"}, CardGuard.OBSERVE),
        ({"HYPERLOOM_CARD_GUARD": " observe\n"}, CardGuard.OBSERVE),
        ({"HYPERLOOM_CARD_GUARD": "enforce"}, CardGuard.ENFORCE),
    ],
)
def test_a_process_enforces_unless_its_environment_says_to_observe(
    environ: dict[str, str], expected: CardGuard
) -> None:
    assert card_guard(environ) is expected


@pytest.mark.parametrize("named", ["enfroce", "ENFORCE", "on", "1"])
def test_a_guard_that_is_not_one_is_refused_and_not_read_as_either(named: str) -> None:
    with pytest.raises(ValueError, match=rf"HYPERLOOM_CARD_GUARD is {named!r}, not one of \['observe', 'enforce'\]"):
        card_guard({"HYPERLOOM_CARD_GUARD": named})
