# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``descendants`` over a procfs whose parent links are not a tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common import proctree


def _procfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parents: dict[int, int]) -> None:
    for pid in parents:
        (tmp_path / str(pid)).mkdir()
    monkeypatch.setattr(proctree, "_PROC", tmp_path)
    monkeypatch.setattr(proctree, "proc_identity", lambda pid: (parents[pid], float(pid)) if pid in parents else None)


def test_descendants_are_deepest_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _procfs(tmp_path, monkeypatch, {10: 1, 11: 10, 12: 11, 13: 10})

    assert [pid for pid, _ in proctree.descendants(10)] == [12, 11, 13]


def test_a_parent_cycle_ends_the_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A sandbox's PID view can list a process as its own parent, or two as each other's.
    _procfs(tmp_path, monkeypatch, {10: 1, 11: 10, 12: 13, 13: 12, 1: 1, 14: 11})

    assert sorted(pid for pid, _ in proctree.descendants(10)) == [11, 14]
    assert sorted(pid for pid, _ in proctree.descendants(1)) == [10, 11, 14]
    assert sorted(pid for pid, _ in proctree.descendants(12)) == [13]


def test_a_deep_chain_does_not_overflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    depth = 3000
    _procfs(tmp_path, monkeypatch, {100 + i: 99 + i for i in range(depth)})

    assert len(proctree.descendants(99)) == depth
