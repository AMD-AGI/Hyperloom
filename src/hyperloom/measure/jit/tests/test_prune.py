# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pruning the shared compiled-code cache: what no test has used for long, or what is past the size bound, goes; what a
test may still be reading, and what another key still lists, stays.

Every time is given, never read from the clock: a key's time is its directory's, which the tests set as the host's
stamp of its last use would.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from hyperloom.measure.jit.prune import (
    RETIRED_DIR,
    PruneLimits,
    note_used,
    prune,
    prune_logged,
)
from hyperloom.measure.jit.store import DirJitStore, Snapshot, StoredFile
from hyperloom.measure.jit_kinds import AITER_JIT, TRITON

NOW = 1_800_000_000.0
DAY = 24 * 3600.0
LIMITS = PruneLimits(max_idle_sec=14 * DAY, max_bytes=10**9, grace_sec=2 * DAY)
OLD, FRESH, SHARED = b"\x7fELF old module", b"\x7fELF fresh module", b"\x7fELF module both keys hold"
OLD_KEY, FRESH_KEY = "a" * 64, "b" * 64


def _publish(store: DirJitStore, staging: Path, key: str, files: dict[str, bytes], *, kind=AITER_JIT) -> Snapshot:
    """Publish ``files`` under ``key`` as a host does what a test staged."""
    (staging / "blobs").mkdir(parents=True)
    entries = {}
    for relative, data in files.items():
        sha = hashlib.sha256(data).hexdigest()
        (staging / "blobs" / sha).write_bytes(data)
        entries[relative] = StoredFile(sha, len(data))
    snapshot = Snapshot(kind.name, key, None, entries)
    (staging / "manifest.json").write_bytes(snapshot.to_bytes())
    store.publish(staging, snapshot.sha256())
    return snapshot


def _key_dir(store: DirJitStore, key: str, kind=AITER_JIT) -> Path:
    return store.root / "snapshots" / kind.name / key


def _blob(store: DirJitStore, data: bytes) -> Path:
    sha = hashlib.sha256(data).hexdigest()
    return store.root / "blobs" / sha[:2] / sha


def _age(path: Path, at: float) -> None:
    os.utime(path, (at, at), follow_symlinks=False)


@pytest.fixture
def store(tmp_path: Path) -> DirJitStore:
    """Two keys: one last used 20 days ago, one now; each lists a blob of its own and both list one blob, and every blob
    was stored 20 days ago."""
    store = DirJitStore(tmp_path / "store")
    _publish(store, tmp_path / "old", OLD_KEY, {"module_old.so": OLD, "module_shared.so": SHARED})
    _publish(store, tmp_path / "fresh", FRESH_KEY, {"module_fresh.so": FRESH, "module_shared.so": SHARED})
    for data in (OLD, FRESH, SHARED):
        _age(_blob(store, data), NOW - 20 * DAY)
    _age(_key_dir(store, OLD_KEY), NOW - 20 * DAY)
    _age(_key_dir(store, FRESH_KEY), NOW)
    return store


def _fetched(store: DirJitStore, key: str, destination: Path) -> Snapshot | None:
    return store.fetch(AITER_JIT, key, destination).snapshot


def test_a_key_unused_too_long_is_retired_at_once_and_its_files_stay_until_the_grace_has_passed(
    store, tmp_path
) -> None:
    done = prune(store, LIMITS, now=NOW)

    assert (done.retired, done.removed, done.blobs) == (1, 0, 0)
    assert _fetched(store, OLD_KEY, tmp_path / "a") is None, "no new test finds the retired key"
    assert _fetched(store, FRESH_KEY, tmp_path / "b") is not None, "a key in use is left alone"
    assert _blob(store, OLD).is_file(), "a test that read the retired list before still finds its files"
    (retired,) = (store.root / RETIRED_DIR).iterdir()
    assert retired.name.startswith(f"{int(NOW)}-aiter_jit-{OLD_KEY}-")

    later = prune(store, LIMITS, now=NOW + 3 * DAY)

    assert (later.retired, later.removed, later.blobs, later.bytes) == (0, 1, 1, len(OLD))
    assert not _blob(store, OLD).exists(), "the grace has passed: what only the retired key listed goes"
    assert _blob(store, SHARED).is_file() and _blob(store, FRESH).is_file(), "what a live key lists stays"
    assert not any((store.root / RETIRED_DIR).iterdir())
    assert _fetched(store, FRESH_KEY, tmp_path / "c") is not None


def test_a_key_an_accepted_result_named_is_in_use_and_is_kept(store) -> None:
    note_used(store, AITER_JIT.name, OLD_KEY, now=NOW - DAY)

    assert prune(store, LIMITS, now=NOW).retired == 0
    assert _key_dir(store, OLD_KEY).is_dir()


def test_noting_a_key_the_store_does_not_hold_or_a_name_that_is_no_key_writes_nothing(store) -> None:
    before = sorted(path.relative_to(store.root) for path in store.root.rglob("*"))

    note_used(store, AITER_JIT.name, "c" * 64, now=NOW)
    note_used(store, AITER_JIT.name, "../../escape", now=NOW)
    note_used(store, "no_such_kind", OLD_KEY, now=NOW)
    note_used(store, AITER_JIT.name, None, now=NOW)

    assert sorted(path.relative_to(store.root) for path in store.root.rglob("*")) == before


def test_over_the_size_bound_the_least_recently_used_keys_past_the_grace_are_retired_until_the_rest_fit(
    tmp_path,
) -> None:
    store = DirJitStore(tmp_path / "store")
    sizes = {}
    for index, key in enumerate(("1" * 64, "2" * 64, "3" * 64, "4" * 64)):
        module = bytes([index]) * 1000
        _publish(store, tmp_path / key[:1], key, {f"module_{index}.so": module})
        _age(_blob(store, module), NOW - 20 * DAY)
        _age(_key_dir(store, key), NOW - (10 - index) * DAY if index < 3 else NOW - DAY)
        sizes[key] = len(module)

    done = prune(store, PruneLimits(max_idle_sec=30 * DAY, max_bytes=2500, grace_sec=2 * DAY), now=NOW)

    assert done.retired == 2, "two of 1000 bytes each go: 4000 bytes down to 2000"
    live = sorted(path.name[:1] for path in (store.root / "snapshots" / "aiter_jit").iterdir())
    assert live == ["3", "4"], "the two used longest ago go first"


def test_a_key_used_within_the_grace_is_never_retired_for_size(tmp_path) -> None:
    store = DirJitStore(tmp_path / "store")
    _publish(store, tmp_path / "s", OLD_KEY, {"module_a.so": OLD})
    _age(_key_dir(store, OLD_KEY), NOW - DAY)

    assert prune(store, PruneLimits(max_idle_sec=30 * DAY, max_bytes=1, grace_sec=2 * DAY), now=NOW).retired == 0


def test_a_blob_stored_within_the_grace_that_nothing_lists_yet_is_kept_for_the_publisher_about_to_list_it(
    store,
) -> None:
    pending = store.root / "blobs" / "cc" / ("c" * 64)
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_bytes(b"stored, its list not yet renamed into place")
    _age(pending, NOW - 60)

    assert prune(store, LIMITS, now=NOW).blobs == 0 and pending.is_file()
    _age(pending, NOW - 3 * DAY)
    assert prune(store, LIMITS, now=NOW).blobs == 1 and not pending.exists()


def test_a_test_that_read_a_key_before_it_was_retired_publishes_under_it_again_from_the_files_still_stored(
    store, tmp_path
) -> None:
    """Its contribution names the files it took from the cache without staging them: they must still be stored."""
    read = store.fetch(AITER_JIT, OLD_KEY, tmp_path / "read")
    assert read.snapshot is not None
    prune(store, LIMITS, now=NOW)

    (tmp_path / "extend" / "blobs").mkdir(parents=True)
    added = b"\x7fELF built by the test"
    sha = hashlib.sha256(added).hexdigest()
    (tmp_path / "extend" / "blobs" / sha).write_bytes(added)
    files = {**read.snapshot.files, "module_new.so": StoredFile(sha, len(added))}
    extended = Snapshot(AITER_JIT.name, OLD_KEY, read.snapshot.sha256(), files)
    (tmp_path / "extend" / "manifest.json").write_bytes(extended.to_bytes())
    store.publish(tmp_path / "extend", extended.sha256())

    got = store.fetch(AITER_JIT, OLD_KEY, tmp_path / "again")
    assert got.snapshot == extended and (tmp_path / "again" / "module_old.so").read_bytes() == OLD


def test_a_publishers_leftover_is_removed_after_the_grace(store) -> None:
    leftover = store.root / ".incoming-deadbeef"
    leftover.mkdir()
    (leftover / "half").write_bytes(b"x")
    _age(leftover, NOW - 3 * DAY)
    recent = store.root / ".incoming-cafe"
    recent.mkdir()
    _age(recent, NOW)

    prune(store, LIMITS, now=NOW)

    assert not leftover.exists() and recent.is_dir()


def test_kinds_are_kept_apart_and_an_empty_or_missing_store_is_nothing_to_prune(tmp_path) -> None:
    assert prune(DirJitStore(tmp_path / "none"), LIMITS, now=NOW).retired == 0
    store = DirJitStore(tmp_path / "store")
    _publish(store, tmp_path / "t", OLD_KEY, {"abcdefghijklmnop/kernel.hsaco": OLD}, kind=TRITON)
    _age(_key_dir(store, OLD_KEY, TRITON), NOW)

    assert prune(store, LIMITS, now=NOW).retired == 0 and _key_dir(store, OLD_KEY, TRITON).is_dir()


def test_two_hosts_pruning_at_once_leave_one_consistent_store(store, tmp_path) -> None:
    first = prune(store, LIMITS, now=NOW)
    second = prune(store, LIMITS, now=NOW)

    assert first.retired + second.retired == 1
    assert _fetched(store, FRESH_KEY, tmp_path / "f") is not None


def test_a_store_that_cannot_be_pruned_is_logged_and_the_session_goes_on(tmp_path, caplog, monkeypatch) -> None:
    store = DirJitStore(tmp_path / "store")
    (store.root / "snapshots").mkdir(parents=True)

    def unreadable(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("hyperloom.measure.jit.prune._stored_blobs", unreadable)

    assert prune_logged(store, LIMITS) is None
    assert "was not pruned (PermissionError)" in caplog.text
