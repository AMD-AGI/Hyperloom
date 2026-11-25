# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which files of a contribution the host fetches back from the sandbox, how often it reads the snapshot it extends,
and that it reads what the sandbox staged as untrusted."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from hyperloom.measure.jit import publish
from hyperloom.measure.jit.store import DirJitStore, Snapshot, StoredFile
from hyperloom.measure.manifest import JitCacheFacts
from hyperloom.measure.jit.tests.dir_sandbox import DirSandbox

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


# -- the publish path reads the sandbox as untrusted -------------------------------------------------------

_KEY = "b" * 64
_MODULE = b"\x7fELF a module"


def _staged(tmp_path: Path, data: bytes, *, listed_size: int | None = None, link: bool = False) -> JitCacheFacts:
    staged = tmp_path / "out" / "jit-cache" / "aiter_jit"
    (staged / "blobs").mkdir(parents=True)
    sha = hashlib.sha256(data).hexdigest()
    if link:
        (tmp_path / "target").write_bytes(data)
        (staged / "blobs" / sha).symlink_to(tmp_path / "target")
    else:
        (staged / "blobs" / sha).write_bytes(data)
    snapshot = Snapshot("aiter_jit", _KEY, None, {"module_a.so": StoredFile(sha, listed_size or len(data))})
    (staged / "manifest.json").write_bytes(snapshot.to_bytes())
    contribution = {"sha256": snapshot.sha256(), "files": 1, "bytes": len(data)}
    kind = {"kind": "aiter_jit", "key": _KEY, "state": "miss", "seed_files": 0, "seed_bytes": 0, "seed_snapshot": None}
    document = {"kinds": [{**kind, "contribution": contribution}], "builds_seen": 0, "jit_compile_sec": 0.0}
    return JitCacheFacts.parse(document, "jit_cache")


@pytest.mark.parametrize("how", ["link", "size"])
def test_a_staged_blob_that_is_a_link_or_not_its_listed_size_is_not_fetched(tmp_path, how) -> None:
    facts = _staged(tmp_path, _MODULE, link=how == "link", listed_size=len(_MODULE) + 5 if how == "size" else None)
    store = DirJitStore(tmp_path / "store")
    (tmp_path / "scratch").mkdir()

    assert publish.publish_contributions(store, DirSandbox(tmp_path), facts, tmp_path / "scratch") == 0
    assert not (tmp_path / "store" / "blobs").exists()


def test_a_staged_plain_blob_of_its_listed_size_is_published(tmp_path) -> None:
    facts = _staged(tmp_path, _MODULE)
    store = DirJitStore(tmp_path / "store")
    (tmp_path / "scratch").mkdir()

    assert publish.publish_contributions(store, DirSandbox(tmp_path), facts, tmp_path / "scratch") == 1
    assert os.path.isdir(tmp_path / "store" / "blobs")
