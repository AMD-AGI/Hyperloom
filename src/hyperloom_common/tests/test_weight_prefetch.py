# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The checkpoint prefetch: which files it reads, in which order, how far memory lets it, what it reads again after the
page cache dropped them, and that nothing it meets stops the round. A fake host stands for the machine, so no test reads
more than a few kilobytes."""

from __future__ import annotations

import json
import logging
import mmap
import os
import threading
from pathlib import Path

import pytest

from hyperloom_common import weight_prefetch as wp
from hyperloom_common.weight_prefetch import (
    WEIGHT_PREFETCH_ENV,
    LinuxHost,
    Memory,
    PrefetchTuning,
    WeightPrefetch,
    checkpoint_files,
    start_weight_prefetch,
)

GIB = 1 << 30
#: Headroom of this limit before anything is cached: 200 - 10 used - 20 margin (a tenth of the limit).
LIMIT, USED = 200 * GIB, 10 * GIB


class FakeHost:
    """A machine whose files are only sizes and whose page cache is a set of uncached spans per file. Reading a span
    caches it and, as file pages do, adds it to the memory in use."""

    def __init__(self, sizes: dict[str, int], *, fs: str | None = "nfs4", used: int = USED) -> None:
        self.sizes = sizes
        self.fs = fs
        self.limit, self.used = LIMIT, used
        self.uncached_spans = {name: [(0, size)] for name, size in sizes.items()}
        self.reads: list[tuple[str, tuple[int, int]]] = []
        self.lock = threading.Lock()
        self.fail_read: set[str] = set()
        self.read_seen = threading.Event()
        #: Whether a read page counts toward this cgroup's use; a page another cgroup already charged does not.
        self.charged = True
        #: Files whose residency the kernel does not tell (another account's, on a read-only mount).
        self.blind: set[str] = set()

    def fs_type(self, path: Path) -> str | None:
        return self.fs

    def memory(self) -> Memory | None:
        with self.lock:
            return Memory(self.limit, self.used)

    def size(self, path: Path) -> int:
        return self.sizes[path.name]

    def uncached(self, path: Path, size: int) -> list[tuple[int, int]] | None:
        if path.name in self.blind:
            return None
        with self.lock:
            return list(self.uncached_spans[path.name])

    def read(self, path: Path, span: tuple[int, int], stop: threading.Event) -> int:
        if path.name in self.fail_read:
            raise OSError(5, "Input/output error")
        with self.lock:
            self.reads.append((path.name, span))
            self.uncached_spans[path.name] = [s for s in self.uncached_spans[path.name] if s != span]
            self.used += (span[1] - span[0]) if self.charged else 0
        self.read_seen.set()
        return span[1] - span[0]

    def drop_cache(self) -> None:
        with self.lock:
            for name, size in self.sizes.items():
                self.used -= size - sum(end - start for start, end in self.uncached_spans[name])
                self.uncached_spans[name] = [(0, size)]


def _model(tmp_path: Path, sizes: dict[str, int], index: bool = True) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    for name in sizes:
        (model / name).write_bytes(b"")
    if index:
        weight_map = {f"layer.{n}.weight": name for n, name in enumerate(reversed(list(sizes)))}
        (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return model


def _run(model: Path, host: FakeHost, **tuning) -> WeightPrefetch:
    prefetch = WeightPrefetch(model, host, PrefetchTuning(**{"streams": 1, "max_refills": 0, **tuning}))
    prefetch.start()
    assert prefetch.join(10.0), "the prefetch ends on its own"
    return prefetch


def _names(n: int, size: int = 60 * GIB) -> dict[str, int]:
    return {f"model-{i:05d}-of-{n:05d}.safetensors": size for i in range(1, n + 1)}


def test_the_files_are_the_ones_the_index_names_in_the_loaders_order(tmp_path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    for name in ("model-10.safetensors", "model-9.safetensors", "model-2.safetensors", "stray.safetensors", "a.bin"):
        (model / name).write_bytes(b"")
    index = {"weight_map": {"z": "model-9.safetensors", "a": "model-10.safetensors", "m": "model-2.safetensors"}}
    (model / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    assert [p.name for p in checkpoint_files(model)] == [
        "model-2.safetensors",
        "model-9.safetensors",
        "model-10.safetensors",
    ]


def test_without_an_index_every_safetensors_file_is_read_in_name_order(tmp_path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    for name in ("b-11.safetensors", "b-3.safetensors", "config.json"):
        (model / name).write_bytes(b"")
    assert [p.name for p in checkpoint_files(model)] == ["b-3.safetensors", "b-11.safetensors"]


def test_it_reads_the_files_in_order_while_they_fit_the_headroom(tmp_path) -> None:
    sizes = _names(4)  # 4 x 60 GiB against 170 GiB of headroom: two fit
    host = FakeHost(sizes)
    prefetch = _run(_model(tmp_path, sizes), host)

    names = list(sizes)
    assert host.reads == [(names[0], (0, 60 * GIB)), (names[1], (0, 60 * GIB))]
    assert prefetch.read_bytes == 120 * GIB and prefetch.passes == 1


def test_the_headroom_at_the_start_bounds_the_reads_even_when_use_does_not_grow(tmp_path) -> None:
    sizes = _names(4)
    host = FakeHost(sizes)
    host.charged = False  # the pages are charged elsewhere, so the live headroom alone would let all four through
    _run(_model(tmp_path, sizes), host)

    assert [name for name, _ in host.reads] == list(sizes)[:2]


def test_only_what_is_not_cached_is_read(tmp_path) -> None:
    sizes = _names(1, 10 * GIB)
    host = FakeHost(sizes)
    name = next(iter(sizes))
    host.uncached_spans[name] = [(0, GIB), (5 * GIB, 6 * GIB)]
    _run(_model(tmp_path, sizes), host)

    assert host.reads == [(name, (0, GIB)), (name, (5 * GIB, 6 * GIB))]


@pytest.mark.parametrize("fs", ["ext4", "xfs", "tmpfs", "overlay", None])
def test_a_checkpoint_on_a_local_or_unknown_file_system_is_not_read(tmp_path, fs, caplog) -> None:
    sizes = _names(2, GIB)
    host = FakeHost(sizes, fs=fs)
    with caplog.at_level(logging.INFO, logger=wp.__name__):
        prefetch = _run(_model(tmp_path, sizes), host)
    assert host.reads == [] and prefetch.passes == 0
    assert "not a network file system" in caplog.text


@pytest.mark.parametrize("fs", ["nfs", "nfs4", "ceph", "fuse.juicefs", "lustre"])
def test_a_checkpoint_on_a_network_file_system_is_read(tmp_path, fs) -> None:
    sizes = _names(1, GIB)
    host = FakeHost(sizes, fs=fs)
    _run(_model(tmp_path, sizes), host)
    assert len(host.reads) == 1


def test_too_little_headroom_reads_nothing(tmp_path, caplog) -> None:
    sizes = _names(1, GIB)
    host = FakeHost(sizes, used=LIMIT - 20 * GIB - 7 * GIB)  # 7 GiB over the margin: under the 8 GiB that matters
    with caplog.at_level(logging.INFO, logger=wp.__name__):
        _run(_model(tmp_path, sizes), host)
    assert host.reads == [] and "too small to matter" in caplog.text


def test_it_stops_when_the_server_takes_the_headroom_and_does_not_read_again(tmp_path, caplog) -> None:
    sizes = _names(3, 50 * GIB)  # all three fit the 170 GiB the start leaves
    host = FakeHost(sizes)
    original = host.read

    def read(path, span, stop):
        got = original(path, span, stop)
        with host.lock:
            host.used += 80 * GIB  # the server allocates while the first file is read
        return got

    host.read = read  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO, logger=wp.__name__):
        prefetch = _run(_model(tmp_path, sizes), host, max_refills=3, refill_interval_sec=0.01)

    assert [name for name, _ in host.reads] == [next(iter(sizes))]
    assert prefetch.passes == 1, "no refill once the memory ran out"
    assert "headroom ran out" in caplog.text


def test_what_the_loader_drops_from_the_cache_is_read_again_a_bounded_number_of_times(tmp_path) -> None:
    sizes = _names(2, 4 * GIB)
    host = FakeHost(sizes)
    original = host.read

    def read(path, span, stop):
        got = original(path, span, stop)
        if path.name == list(sizes)[-1]:
            host.drop_cache()  # the loader drops the checkpoint's page cache after each pass
        return got

    host.read = read  # type: ignore[method-assign]
    prefetch = _run(_model(tmp_path, sizes), host, max_refills=2, refill_interval_sec=0.01)

    assert prefetch.passes == 3
    assert [name for name, _ in host.reads] == list(sizes) * 3


def test_a_cache_that_stays_warm_is_not_read_again(tmp_path) -> None:
    sizes = _names(1, 4 * GIB)
    host = FakeHost(sizes)
    prefetch = WeightPrefetch(_model(tmp_path, sizes), host, PrefetchTuning(streams=1, refill_interval_sec=0.01))
    prefetch.start()
    assert not prefetch.join(0.2), "it keeps watching the cache until it is stopped"
    prefetch.stop()
    assert prefetch.join(5.0)
    assert prefetch.passes == 1 and len(host.reads) == 1


def test_stop_ends_a_read_in_progress(tmp_path) -> None:
    sizes = _names(2, GIB)
    host = FakeHost(sizes)
    reading = threading.Event()

    def read(path, span, stop):
        reading.set()
        stop.wait(30.0)  # a read the stop interrupts between two chunks
        return 0

    host.read = read  # type: ignore[method-assign]
    prefetch = WeightPrefetch(_model(tmp_path, sizes), host, PrefetchTuning(streams=1))
    prefetch.start()
    assert reading.wait(10.0)
    prefetch.stop()
    assert prefetch.join(5.0)


def test_a_file_that_cannot_be_read_does_not_stop_the_others_and_is_counted(tmp_path, caplog) -> None:
    sizes = _names(3, GIB)
    host = FakeHost(sizes)
    names = list(sizes)
    host.fail_read = {names[1]}
    with caplog.at_level(logging.WARNING, logger=wp.__name__):
        prefetch = _run(_model(tmp_path, sizes), host, streams=2)
    assert sorted(name for name, _ in host.reads) == [names[0], names[2]]
    assert prefetch.outcome.unreadable == 1
    failed = [record for record in caplog.records if "not read" in record.getMessage()]
    assert failed and failed[0].exc_info is not None, "the traceback is logged"
    assert "Input/output error" in caplog.text


def test_a_host_that_cannot_be_read_ends_the_prefetch_quietly(tmp_path, caplog) -> None:
    sizes = _names(1, GIB)
    host = FakeHost(sizes)

    def broken() -> Memory:
        raise PermissionError("cgroupfs is not readable")

    host.memory = broken  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger=wp.__name__):
        prefetch = _run(_model(tmp_path, sizes), host)
    assert host.reads == [] and prefetch.passes == 0
    assert prefetch.outcome.error == "PermissionError: cgroupfs is not readable" and prefetch.outcome.ended
    failed = [record for record in caplog.records if "failed and ends" in record.getMessage()]
    assert failed and failed[0].exc_info is not None, "the traceback is logged"


def test_a_file_whose_residency_the_kernel_hides_is_read_whole_within_the_budget_and_not_refilled(tmp_path) -> None:
    sizes = _names(4, 60 * GIB)
    host = FakeHost(sizes)
    host.blind = set(sizes)
    original = host.read

    def read(path, span, stop):
        got = original(path, span, stop)
        host.drop_cache()  # a refill would read a known file again; a blind one must not be
        return got

    host.read = read  # type: ignore[method-assign]
    prefetch = _run(_model(tmp_path, sizes), host, max_refills=2, refill_interval_sec=0.01)

    names = list(sizes)
    assert host.reads == [(names[0], (0, 60 * GIB)), (names[1], (0, 60 * GIB))], "by size, within 170 GiB"
    assert prefetch.passes == 1
    assert [item["residency_known"] for item in prefetch.outcome.chosen] == [False, False]


def test_the_outcome_names_the_files_the_totals_and_why_nothing_was_read(tmp_path) -> None:
    sizes = _names(4)
    prefetch = _run(_model(tmp_path, sizes), FakeHost(sizes))
    outcome = prefetch.outcome
    assert [item["name"] for item in outcome.chosen] == list(sizes)[:2]
    assert (outcome.files, outcome.total_bytes, outcome.read_bytes) == (4, 240 * GIB, 120 * GIB)
    assert outcome.fs_type == "nfs4" and outcome.skipped is None and outcome.ended

    record = tmp_path / "weight_prefetch.json"
    prefetch.write_outcome(record)
    assert json.loads(record.read_text(encoding="utf-8"))["read_bytes"] == 120 * GIB

    (tmp_path / "local").mkdir()
    skipped = _run(_model(tmp_path / "local", sizes), FakeHost(sizes, fs="ext4")).outcome
    assert skipped.skipped == "on ext4, not a network file system" and skipped.chosen == []


def test_an_outcome_that_cannot_be_written_is_logged(tmp_path, caplog) -> None:
    prefetch = WeightPrefetch(tmp_path, FakeHost({}))
    with caplog.at_level(logging.WARNING, logger=wp.__name__):
        prefetch.write_outcome(tmp_path / "missing" / "weight_prefetch.json")
    assert caplog.records and caplog.records[0].exc_info is not None


@pytest.fixture
def fake_linux_host(monkeypatch) -> dict[str, FakeHost]:
    """``start_weight_prefetch`` builds a :class:`LinuxHost`; this hands it the fake of the model a test registers."""
    hosts: dict[str, FakeHost] = {}
    monkeypatch.setattr(wp, "LinuxHost", lambda: hosts["host"])
    return hosts


def test_it_is_off_unless_the_environment_turns_it_on(tmp_path, fake_linux_host) -> None:
    sizes = _names(1, GIB)
    model = _model(tmp_path, sizes)
    host = fake_linux_host["host"] = FakeHost(sizes)
    assert start_weight_prefetch(str(model), {}) is None
    assert start_weight_prefetch(str(model), {WEIGHT_PREFETCH_ENV: "0"}) is None
    assert start_weight_prefetch(str(model), {WEIGHT_PREFETCH_ENV: "bogus"}) is None

    prefetch = start_weight_prefetch(str(model), {WEIGHT_PREFETCH_ENV: "1"})
    assert prefetch is not None
    assert host.read_seen.wait(10.0)
    prefetch.stop()
    assert host.reads == [(next(iter(sizes)), (0, GIB))]


def test_a_model_that_is_not_a_directory_reads_nothing_and_says_so(tmp_path, fake_linux_host) -> None:
    on = {WEIGHT_PREFETCH_ENV: "1"}
    host = fake_linux_host["host"] = FakeHost({})
    for model in ("org/model-name", str(tmp_path / "missing")):
        prefetch = start_weight_prefetch(model, on)
        assert prefetch is not None and prefetch.join(10.0)
        assert prefetch.outcome.skipped == "the model is not a local directory"
    assert host.reads == []


def test_a_thread_that_cannot_start_starts_nothing(tmp_path, fake_linux_host, monkeypatch, caplog) -> None:
    fake_linux_host["host"] = FakeHost({})

    def refuse(self) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    with caplog.at_level(logging.WARNING, logger=wp.__name__):
        assert start_weight_prefetch(str(tmp_path), {WEIGHT_PREFETCH_ENV: "1"}) is None
    assert caplog.records[0].exc_info is not None


# --- the real machine, on files of a few kilobytes -----------------------------------------------------------------


def test_the_mount_a_path_is_on_is_the_longest_mount_point_it_starts_with() -> None:
    mountinfo = "\n".join(
        [
            "1 0 8:1 / / rw - ext4 /dev/sda1 rw",
            "2 1 0:50 / /mnt/models rw - nfs4 server:/export rw",
            "3 2 0:51 / /mnt/models/local\\040disk rw - xfs /dev/sdb rw",
            "4 1 0:52 / /mnt/modelsx rw - tmpfs tmpfs rw",
            "5 1 0:53 / /mnt/a\\134b rw - ceph ceph rw",
            "garbage",
        ]
    )
    assert wp._mount_fs_type(mountinfo, "/mnt/models/llama") == "nfs4"
    assert wp._mount_fs_type(mountinfo, "/mnt/models/local disk/llama") == "xfs"
    assert wp._mount_fs_type(mountinfo, "/mnt/modelsx/a") == "tmpfs"
    assert wp._mount_fs_type(mountinfo, "/home/x") == "ext4"
    assert wp._mount_fs_type(mountinfo, "/mnt/a\\b/c") == "ceph"
    assert wp._mount_fs_type("", "/x") is None


def test_the_cgroup_limit_is_the_processs_own_and_no_limit_falls_through(tmp_path) -> None:
    own = tmp_path / "pod" / "c1"
    own.mkdir(parents=True)
    (own / "memory.max").write_text("1000\n")
    (own / "memory.current").write_text("300\n")
    assert wp._cgroup_memory(tmp_path, "0::/pod/c1\n") == Memory(1000, 300)

    (own / "memory.max").write_text("max\n")
    assert wp._cgroup_memory(tmp_path, "0::/pod/c1\n") is None
    (tmp_path / "memory.max").write_text("2000\n")
    (tmp_path / "memory.current").write_text("100\n")
    assert wp._cgroup_memory(tmp_path, "0::/\n") == Memory(2000, 100)


def test_meminfo_stands_in_without_a_cgroup_limit() -> None:
    meminfo = "MemTotal:       1000 kB\nMemFree:  10 kB\nMemAvailable:    600 kB\n"
    assert wp._meminfo_memory(meminfo) == Memory(1000 * 1024, 400 * 1024)
    assert wp._meminfo_memory("MemTotal: 1 kB\n") is None


def test_uncached_pages_become_byte_ranges() -> None:
    page = mmap.PAGESIZE
    flags = bytes([1, 0, 0, 1, 0x81, 0, 2])  # bits above bit 0 do not mean resident
    assert list(wp._uncached_runs(flags, 10 * page, 10 * page + 6 * page + 5)) == [
        (11 * page, 13 * page),
        (15 * page, 16 * page + 5),
    ]
    assert wp._merge([(0, 4), (4, 8), (9, 10)]) == [(0, 8), (9, 10)]


def test_the_linux_host_reads_a_file_into_the_cache_and_then_finds_nothing_uncached(tmp_path) -> None:
    path = tmp_path / "w.safetensors"
    size = 3 * mmap.PAGESIZE + 17
    path.write_bytes(b"x" * size)
    host = LinuxHost()

    spans = host.uncached(path, size)
    assert all(0 <= start < end <= size for start, end in spans)
    assert host.read(path, (0, size), threading.Event()) == size
    assert host.uncached(path, size) == []
    assert host.uncached(path, 0) == []

    stopped = threading.Event()
    stopped.set()
    assert host.read(path, (0, size), stopped) == 0
    assert host.size(path) == size
    assert host.memory() is not None
    assert host.fs_type(tmp_path) is not None


@pytest.mark.parametrize(
    ("euid", "owner", "writable", "visible"),
    [(0, 5, False, True), (7, 7, False, True), (7, 5, True, True), (7, 5, False, False)],
)
def test_the_kernel_tells_residency_only_to_the_owner_a_writer_or_root(
    monkeypatch, tmp_path, euid, owner, writable, visible
) -> None:
    monkeypatch.setattr(wp.os, "geteuid", lambda: euid)
    monkeypatch.setattr(wp.os, "access", lambda path, mode, **kwargs: writable)
    assert wp._residency_visible(tmp_path, owner) is visible


class _NetworkHost(LinuxHost):
    """The real page cache and reads, on a file system said to be a network one, with room to spare."""

    def fs_type(self, path: Path) -> str | None:
        return "nfs4"

    def memory(self) -> Memory | None:
        return Memory(1 << 40, 0)


def _evicted(path: Path, size: int) -> None:
    path.write_bytes(os.urandom(size))
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="needs posix_fadvise")
@pytest.mark.parametrize("visible", [True, False])
def test_an_evicted_file_is_found_uncached_and_read_back_into_the_cache(tmp_path, monkeypatch, visible) -> None:
    model = tmp_path / "model"
    model.mkdir()
    path = model / "model.safetensors"
    size = 8 << 20
    _evicted(path, size)
    host = LinuxHost()
    try:
        spans = host.uncached(path, size)
    except OSError as exc:
        pytest.skip(f"mincore is not available here: {exc}")
    if not spans:
        pytest.skip("this file system kept the pages after POSIX_FADV_DONTNEED")
    assert all(0 <= start < end <= size for start, end in spans)

    if not visible:  # another account's file: the kernel's answer is not taken, the file is read by its size
        monkeypatch.setattr(wp, "_residency_visible", lambda path, owner: False)
        assert host.uncached(path, size) is None
    prefetch = _run(model, _NetworkHost())

    assert prefetch.read_bytes == (size if not visible else sum(end - start for start, end in spans))
    monkeypatch.setattr(wp, "_residency_visible", lambda path, owner: True)
    assert host.uncached(path, size) == [], "the read left the file in the page cache"
