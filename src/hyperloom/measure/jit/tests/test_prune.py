# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pruning the shared compiled-code cache: what no test has used for long, or what is past the size bound, goes; what a
test may still be reading, and what another key still lists, stays.

Every time is given, never read from the clock, except where the logged prune is run: a key's time is its directory's,
which the tests set as the host's stamp of its last use would.
"""

from __future__ import annotations

import errno
import hashlib
import itertools
import os
import time
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


def _six_idle_keys(tmp_path) -> DirJitStore:
    """Six keys of 1000 bytes each, last used five to ten days ago (the first longest ago), every blob stored long ago."""
    store = DirJitStore(tmp_path / "store")
    for index in range(6):
        key = str(index + 1) * 64
        module = bytes([index]) * 1000
        _publish(store, tmp_path / key[:1], key, {f"module_{index}.so": module})
        _age(_blob(store, module), NOW - 20 * DAY)
        _age(_key_dir(store, key), NOW - (10 - index) * DAY)
    return store


def _live(store: DirJitStore) -> list[str]:
    return sorted(path.name[:1] for path in (store.root / "snapshots" / "aiter_jit").iterdir())


@pytest.mark.parametrize("starts", [2, 3])
def test_starts_inside_the_grace_retire_for_size_only_until_the_live_keys_fit_though_retired_blobs_are_still_stored(
    tmp_path, starts
) -> None:
    """A retired key's blobs stay stored for the grace: the bound is on what the live keys list, so the starts after the
    first, an hour apart, find the store within it and retire nothing more."""
    store = _six_idle_keys(tmp_path)
    limits = PruneLimits(max_idle_sec=30 * DAY, max_bytes=4500, grace_sec=2 * DAY)

    done = [prune(store, limits, now=NOW + start * 3600) for start in range(starts)]

    assert [run.retired for run in done] == [2] + [0] * (starts - 1), "6000 live bytes down to 4000, then nothing"
    assert _live(store) == ["3", "4", "5", "6"], "the two used longest ago go, the rest stay live"
    assert sum(1 for _ in (store.root / "blobs").rglob("*") if _.is_file()) == 6, "retired blobs wait out the grace"


def test_a_blob_two_keys_list_counts_once_and_retiring_one_of_them_frees_none_of_it(tmp_path) -> None:
    store = DirJitStore(tmp_path / "store")
    shared = b"\x01" * 1000
    for index, key in enumerate(("1" * 64, "2" * 64, "3" * 64)):
        own = bytes([index + 2]) * 1000
        files = {f"module_{index}.so": own} | ({"module_shared.so": shared} if index < 2 else {})
        _publish(store, tmp_path / key[:1], key, files)
        _age(_key_dir(store, key), NOW - (10 - index) * DAY)

    done = prune(store, PruneLimits(max_idle_sec=30 * DAY, max_bytes=3000, grace_sec=2 * DAY), now=NOW)

    assert done.retired == 1 and _live(store) == ["2", "3"], "4000 live bytes; retiring key 1 frees only its own 1000"


def test_a_prune_out_of_time_stops_and_says_where_and_removes_no_blob(store, caplog) -> None:
    caplog.set_level("INFO", logger="hyperloom.measure.jit.prune")
    for key in (OLD_KEY, FRESH_KEY):  # the logged prune reads the clock
        _age(_key_dir(store, key), time.time() - 20 * DAY)

    done = prune_logged(store, PruneLimits(max_idle_sec=14 * DAY, max_bytes=10**9, grace_sec=2 * DAY, budget_sec=0))

    assert done is not None and done.stopped_at == "reading the live keys" and done.blobs == 0
    assert done.retired == 1, "the idle key it read before the budget ran out is retired; the other waits"
    assert all(_blob(store, data).is_file() for data in (OLD, FRESH, SHARED))
    assert "stopped reading the live keys after its 0 s budget, the next start goes on from there" in caplog.text
    assert any(
        record.levelname == "WARNING"
        and "before the size bound or the removal of blobs was done" in record.getMessage()
        for record in caplog.records
    ), "a store whose keys outlast the budget is not bounded by size, and says so"


def test_a_prune_that_removes_nothing_still_says_so_in_one_line(tmp_path, caplog) -> None:
    caplog.set_level("INFO", logger="hyperloom.measure.jit.prune")
    store = DirJitStore(tmp_path / "store")
    (store.root / "snapshots").mkdir(parents=True)

    done = prune_logged(store, LIMITS)

    assert done is not None and (done.retired, done.removed, done.blobs, done.stopped_at) == (0, 0, 0, None)
    (line,) = [record.getMessage() for record in caplog.records if "pruned in" in record.getMessage()]
    assert f"{store.root} pruned in" in line and "0 key(s) retired, 0 removed, 0 blob(s) of 0 bytes removed" in line


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


@pytest.mark.parametrize("failing", ["snapshots/aiter_jit", f"snapshots/aiter_jit/{FRESH_KEY}", "blobs"])
def test_a_listing_that_fails_stops_the_prune_and_is_never_read_as_empty(store, monkeypatch, caplog, failing) -> None:
    """A transient error (``ESTALE`` on a network file system) while listing the keys, a key's lists or the blobs: the
    prune stops and says so, and no blob of a key in use is removed."""
    before = sorted(path.relative_to(store.root) for path in store.root.rglob("*"))
    listing = Path.iterdir

    def flaky(directory: Path):
        if directory == store.root / failing:
            raise OSError(errno.ESTALE, "Stale file handle")
        return listing(directory)

    monkeypatch.setattr(Path, "iterdir", flaky)
    _age(_key_dir(store, OLD_KEY), NOW)  # both keys in use, every blob stored long ago

    with pytest.raises(OSError):
        prune(store, LIMITS, now=NOW)
    assert prune_logged(store, LIMITS) is None
    assert "was not pruned (OSError)" in caplog.text

    monkeypatch.setattr(Path, "iterdir", listing)
    assert sorted(path.relative_to(store.root) for path in store.root.rglob("*")) == before
    assert _blob(store, OLD).is_file() and _blob(store, FRESH).is_file() and _blob(store, SHARED).is_file()


def test_a_store_that_cannot_be_pruned_is_logged_and_the_session_goes_on(tmp_path, caplog, monkeypatch) -> None:
    store = DirJitStore(tmp_path / "store")
    (store.root / "snapshots").mkdir(parents=True)

    def unreadable(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("hyperloom.measure.jit.prune._live_names", unreadable)

    assert prune_logged(store, LIMITS) is None
    assert "was not pruned (PermissionError)" in caplog.text


@pytest.fixture
def ticks(monkeypatch) -> None:
    """The budget's clock moves one second each time it is read: a prune with a budget of ``n`` seconds stops at its
    ``n``-th look at the clock after the start."""
    clock = itertools.count()
    monkeypatch.setattr("hyperloom.measure.jit.prune._clock", lambda: float(next(clock)))


def _stored(store: DirJitStore) -> int:
    return sum(1 for path in (store.root / "blobs").rglob("*") if path.is_file())


@pytest.mark.usefixtures("ticks")
def test_starts_after_a_stopped_one_remove_the_retired_keys_past_the_grace_and_then_the_blobs(tmp_path) -> None:
    """Every start does part of the work: the retired keys first, which stay removed, then the blobs from where the last
    start stopped, so the store empties of what nothing lists over a few starts however slow it is."""
    store = _six_idle_keys(tmp_path)
    limits = PruneLimits(max_idle_sec=3 * DAY, max_bytes=10**9, grace_sec=2 * DAY, budget_sec=100)
    assert prune(store, limits, now=NOW).retired == 6, "the six keys, unused five days or more, are retired"
    limits = PruneLimits(max_idle_sec=3 * DAY, max_bytes=10**9, grace_sec=2 * DAY, budget_sec=4)

    later = [prune(store, limits, now=NOW + 3 * DAY + start * 3600) for start in range(4)]

    assert [(run.removed, run.blobs) for run in later] == [(3, 0), (3, 0), (0, 3), (0, 3)]
    assert [run.stopped_at for run in later] == ["removing retired keys", "removing blobs", "removing blobs", None]
    assert not any((store.root / RETIRED_DIR).iterdir()) and _stored(store) == 0
    assert prune(store, limits, now=NOW + 4 * DAY).stopped_at is None


@pytest.mark.usefixtures("ticks")
def test_a_start_after_one_stopped_removing_retired_keys_goes_on_to_the_size_bound(tmp_path) -> None:
    store = _six_idle_keys(tmp_path)
    for index in range(3):  # three keys retired long ago: removing them takes the first start's whole budget
        key = str(index + 7) * 64
        _publish(store, tmp_path / key[:1], key, {f"module_{index + 6}.so": bytes([index + 6]) * 1000})
        os.makedirs(store.root / RETIRED_DIR, exist_ok=True)
        os.rename(_key_dir(store, key), store.root / RETIRED_DIR / f"{int(NOW - 5 * DAY)}-aiter_jit-{key}-{index:012d}")
    limits = PruneLimits(max_idle_sec=30 * DAY, max_bytes=4500, grace_sec=2 * DAY, budget_sec=3)

    first = prune(store, limits, now=NOW)
    assert (first.removed, first.retired, first.stopped_at) == (2, 0, "removing retired keys")

    limits = PruneLimits(max_idle_sec=30 * DAY, max_bytes=4500, grace_sec=2 * DAY, budget_sec=12)
    second = prune(store, limits, now=NOW + 3600)
    assert (second.removed, second.retired) == (1, 2), "the last retired key goes, then the size bound applies"
    assert _live(store) == ["3", "4", "5", "6"]


@pytest.mark.usefixtures("ticks")
def test_the_blob_sweep_goes_on_where_the_last_start_stopped_so_blobs_behind_kept_ones_are_reached(tmp_path) -> None:
    """Eight blobs a live key lists sort before four nothing lists; a start looks at three blobs: without going on from
    where the last stopped, no start would ever reach the four."""
    store = DirJitStore(tmp_path / "store")
    kept = {f"module_{index}.so": b"kept %d" % index for index in range(40)}
    kept = dict(sorted(kept.items(), key=lambda item: hashlib.sha256(item[1]).hexdigest())[:8])
    _publish(store, tmp_path / "k", FRESH_KEY, kept)
    _age(_key_dir(store, FRESH_KEY), NOW)
    last_kept = max(hashlib.sha256(data).hexdigest() for data in kept.values())
    unlisted = [b"orphan %d" % index for index in range(400)]
    unlisted = [data for data in unlisted if hashlib.sha256(data).hexdigest() > last_kept][:4]
    for data in unlisted:
        path = _blob(store, data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for path in (store.root / "blobs").rglob("*"):
        if path.is_file():
            _age(path, NOW - 20 * DAY)
    # one look at the live key and one at the retired ones, then three blobs
    limits = PruneLimits(max_idle_sec=14 * DAY, max_bytes=10**9, grace_sec=2 * DAY, budget_sec=5)

    removed = [prune(store, limits, now=NOW + start).blobs for start in range(5)]

    assert sum(removed) == 4 and _stored(store) == 8, "every blob nothing lists goes, the kept ones stay"
    assert removed[:2] == [0, 0], "the first two starts look only at kept blobs"


@pytest.mark.usefixtures("ticks")
def test_reading_the_live_keys_goes_on_where_the_last_start_stopped_so_idle_keys_behind_used_ones_are_retired(
    tmp_path,
) -> None:
    """Keys 1 to 3 are in use and sort first, 4 to 6 are idle; a start reads two keys: without going on from where the
    last stopped, every start would read keys 1 and 2 only."""
    store = _six_idle_keys(tmp_path)
    for key in ("1" * 64, "2" * 64, "3" * 64):
        _age(_key_dir(store, key), NOW)
    limits = PruneLimits(max_idle_sec=4 * DAY, max_bytes=10**9, grace_sec=2 * DAY, budget_sec=2)

    retired = [prune(store, limits, now=NOW + start).retired for start in range(3)]

    assert retired == [0, 1, 2] and _live(store) == ["1", "2", "3"]
