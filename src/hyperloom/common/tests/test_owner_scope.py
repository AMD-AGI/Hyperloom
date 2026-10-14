# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.common.owner_scope import local_owner_scope, process_started_at, same_environment

BOOT = "c0e93f64-ebe7-4bd1-b11f-2238ee6a7893"


def _proc(root: Path, *, init_stat: str, ns_target: Path) -> Path:
    (root / "sys/kernel/random").mkdir(parents=True)
    (root / "sys/kernel/random/boot_id").write_text(BOOT + "\n", encoding="ascii")
    (root / "self/ns").mkdir(parents=True)
    os.symlink(ns_target, root / "self/ns/pid")
    (root / "1").mkdir()
    (root / "1/stat").write_text(init_stat, encoding="ascii")
    return root


def _stat(start: int, comm: str = "sleep") -> str:
    # pid (comm) state ppid ... with field 22 (starttime) set; 52 fields in all.
    rest = ["S", "0"] + ["0"] * 17 + [str(start)] + ["0"] * 30
    return f"1 ({comm}) " + " ".join(rest) + "\n"


def test_two_containers_of_one_boot_and_namespace_inode_differ(tmp_path: Path) -> None:
    """A container started after another exited on the same node can get the same PID
    namespace inode; the start of its init tells them apart."""
    ns = tmp_path / "ns"
    ns.write_text("")
    first = _proc(tmp_path / "a", init_stat=_stat(1_000), ns_target=ns)
    second = _proc(tmp_path / "b", init_stat=_stat(9_000), ns_target=ns)
    inode = ns.stat().st_ino
    assert local_owner_scope(first) == f"{BOOT}:{inode}:1000"
    assert local_owner_scope(second) == f"{BOOT}:{inode}:9000"


def test_a_command_name_with_spaces_and_parentheses(tmp_path: Path) -> None:
    ns = tmp_path / "ns"
    ns.write_text("")
    root = _proc(tmp_path / "a", init_stat=_stat(4242, comm="a b) (c"), ns_target=ns)
    assert local_owner_scope(root).endswith(":4242")


@pytest.mark.parametrize("init_stat", ["", "1 (init)\n", "1 (init) S 0 x\n"])
def test_an_unreadable_init_leaves_ownership_unobservable(tmp_path: Path, init_stat: str) -> None:
    ns = tmp_path / "ns"
    ns.write_text("")
    assert local_owner_scope(_proc(tmp_path / "a", init_stat=init_stat, ns_target=ns)) == ""


def test_the_real_proc_gives_three_parts() -> None:
    scope = local_owner_scope()
    if not scope:
        pytest.skip("no /proc here")
    boot, namespace, started = scope.split(":")
    assert boot and int(namespace) > 0 and int(started) >= 0


@pytest.mark.parametrize(
    ("recorded", "local", "same"),
    [
        ("b:1:100", "b:1:100", True),
        ("b:1", "b:1:100", True),
        ("b:1:99", "b:1:100", False),
        ("b:2", "b:1:100", False),
        ("c:1", "b:1:100", False),
        ("", "b:1:100", False),
        ("b:1", "", False),
    ],
)
def test_same_environment(recorded: str, local: str, same: bool) -> None:
    assert same_environment(recorded, local) is same


def test_a_running_process_has_a_start_and_a_gone_one_none() -> None:
    started = process_started_at(os.getpid())
    if started is None:
        pytest.skip("no /proc here")
    from datetime import datetime, timezone

    assert started <= datetime.now(timezone.utc)
    assert process_started_at(2**22 + 12345) is None
