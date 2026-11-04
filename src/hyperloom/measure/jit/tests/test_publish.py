# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which files of a contribution the host fetches back from the sandbox, and how often it reads the snapshot it extends."""

from __future__ import annotations

from hyperloom.measure.jit import publish
from hyperloom.measure.jit.store import Snapshot, StoredFile

_PARENT = "a" * 64


def _file(n: int) -> StoredFile:
    return StoredFile(f"{n:064x}", n)


class _Store:
    """Answers ``listing`` with the parent snapshot's files and counts how often it is asked."""

    def __init__(self, parent: dict[str, StoredFile] | None) -> None:
        self.parent = parent
        self.reads = 0

    def listing(self, kind, key, sha256):
        self.reads += 1
        assert sha256 == _PARENT
        return self.parent


def test_the_snapshot_a_contribution_extends_is_read_once_whatever_the_number_of_its_files() -> None:
    files = {f"k{n}.bin": _file(n) for n in range(5000)}
    parent = {name: entry for name, entry in files.items() if not name.endswith("7.bin")}
    parent["k3.bin"] = _file(99999)
    store = _Store(parent)

    staged = publish._staged_by_runner(store, Snapshot("triton", "k" * 64, _PARENT, files))

    assert store.reads == 1
    assert staged == {name for name in files if name.endswith("7.bin")} | {"k3.bin"}, (
        "new or changed against the parent"
    )


def test_a_first_snapshot_was_staged_whole_and_one_whose_parent_is_gone_is_taken_from_the_store() -> None:
    files = {f"k{n}.bin": _file(n) for n in range(3)}
    store = _Store(None)

    assert publish._staged_by_runner(store, Snapshot("triton", "k" * 64, None, files)) == set(files)
    assert store.reads == 0
    assert publish._staged_by_runner(store, Snapshot("triton", "k" * 64, _PARENT, files)) == frozenset()
    assert store.reads == 1
