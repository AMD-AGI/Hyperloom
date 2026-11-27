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
    WeightPrefetch,
    checkpoint_files,
    prefetch_order,
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

    def cached_share(self, path: Path, size: int) -> float | None:
        if path.name in self.blind:
            return None
        with self.lock:
            return 1.0 - sum(end - start for start, end in self.uncached_spans[path.name]) / size if size else 1.0

    def read(self, path: Path, span: tuple[int, int], stop: threading.Event) -> int:
        if path.name in self.fail_read:
            raise OSError(5, "Input/output error")
        with self.lock:
            self.reads.append((path.name, span))
            self.uncached_spans[path.name] = _subtract(self.uncached_spans[path.name], span)
            self.used += (span[1] - span[0]) if self.charged else 0
        self.read_seen.set()
        return span[1] - span[0]

    def cached(self, name: str) -> bool:
        with self.lock:
            return not self.uncached_spans[name]

    def read_bytes(self) -> dict[str, int]:
        """The bytes read of each file, in the order the files were first read."""
        with self.lock:
            totals: dict[str, int] = {}
            for name, (start, end) in self.reads:
                totals[name] = totals.get(name, 0) + end - start
            return totals

    def drop_cache(self) -> None:
        with self.lock:
            for name, size in self.sizes.items():
                self.used -= size - sum(end - start for start, end in self.uncached_spans[name])
                self.uncached_spans[name] = [(0, size)]


def _subtract(spans: list[tuple[int, int]], cut: tuple[int, int]) -> list[tuple[int, int]]:
    kept = []
    for start, end in spans:
        if end <= cut[0] or start >= cut[1]:
            kept.append((start, end))
            continue
        if start < cut[0]:
            kept.append((start, cut[0]))
        if end > cut[1]:
            kept.append((cut[1], end))
    return kept


def _model(tmp_path: Path, sizes: dict[str, int], index: bool = True) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    for name in sizes:
        (model / name).write_bytes(b"")
    if index:
        weight_map = {f"layer.{n}.weight": name for n, name in enumerate(reversed(list(sizes)))}
        (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return model


@pytest.fixture
def machine(monkeypatch):
    """Make ``host`` the machine every :class:`WeightPrefetch` built after it reads, as ``start_weight_prefetch`` builds
    it, with one stream and no refill unless the test says otherwise."""

    def install(host, *, streams: int = 1, max_refills: int = 0, poll_interval_sec: float = 0.01) -> None:
        monkeypatch.setattr(wp, "LinuxHost", lambda: host)
        monkeypatch.setattr(wp, "_STREAMS", streams)
        monkeypatch.setattr(wp, "_MAX_REFILLS", max_refills)
        monkeypatch.setattr(wp, "_POLL_INTERVAL_SEC", poll_interval_sec)

    return install


@pytest.fixture
def run(machine):
    """Run a prefetch of ``model`` on ``host`` until it ends on its own or, while it only watches the cache, until
    ``watch_sec`` later the server is ready and stops it."""

    def run(model: Path, host, *, watch_sec: float = 0.3, **tuning) -> WeightPrefetch:
        machine(host, **tuning)
        prefetch = WeightPrefetch(model)
        prefetch.start()
        if not prefetch.join(watch_sec):
            prefetch.stop()
        assert prefetch.join(10.0), "the prefetch ends once stopped"
        return prefetch

    return run


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


def test_files_of_one_size_are_read_in_the_loaders_order_while_they_fit_the_headroom(tmp_path, run) -> None:
    sizes = _names(4)  # 4 x 60 GiB against 170 GiB of headroom: two fit
    host = FakeHost(sizes)
    prefetch = run(_model(tmp_path, sizes), host)

    names = list(sizes)
    assert host.read_bytes() == {names[0]: 60 * GIB, names[1]: 60 * GIB}
    assert all(end - start <= GIB for _, (start, end) in host.reads), "in pieces several streams can share"
    assert prefetch.outcome.read_bytes == 120 * GIB and prefetch.outcome.passes == 1


def test_the_largest_files_come_first_and_files_of_one_size_keep_the_loaders_order() -> None:
    sized = [(Path(f"model-{n}.safetensors"), size) for n, size in enumerate([5, 7, 101, 5, 101, 2], start=1)]
    assert [path.name for path, _ in prefetch_order(sized)] == [
        "model-3.safetensors",
        "model-5.safetensors",
        "model-2.safetensors",
        "model-1.safetensors",
        "model-4.safetensors",
        "model-6.safetensors",
    ]


def test_a_large_file_last_in_the_loaders_order_is_read_first_and_one_that_does_not_fit_is_left_out(
    tmp_path, run
) -> None:
    sizes = {"model-1.safetensors": 30 * GIB, "model-2.safetensors": 200 * GIB, "model-3.safetensors": 120 * GIB}
    host = FakeHost(sizes)  # 170 GiB of headroom: the 200 GiB file never fits, the 120 GiB one goes first
    prefetch = run(_model(tmp_path, sizes), host)

    assert host.read_bytes() == {"model-3.safetensors": 120 * GIB, "model-1.safetensors": 30 * GIB}
    assert [item["name"] for item in prefetch.outcome.chosen] == ["model-3.safetensors", "model-1.safetensors"]


def test_the_headroom_at_the_start_bounds_the_reads_even_when_use_does_not_grow(tmp_path, run) -> None:
    sizes = _names(4)
    host = FakeHost(sizes)
    host.charged = False  # the pages are charged elsewhere, so the live headroom alone would let all four through
    run(_model(tmp_path, sizes), host)

    assert list(host.read_bytes()) == list(sizes)[:2]


def test_only_what_is_not_cached_is_read(tmp_path, run) -> None:
    sizes = _names(1, 10 * GIB)
    host = FakeHost(sizes)
    name = next(iter(sizes))
    host.uncached_spans[name] = [(0, GIB), (5 * GIB, 6 * GIB)]
    run(_model(tmp_path, sizes), host)

    assert host.reads == [(name, (0, GIB)), (name, (5 * GIB, 6 * GIB))]


@pytest.mark.parametrize("fs", ["ext4", "xfs", "tmpfs", "overlay", None])
def test_a_checkpoint_on_a_local_or_unknown_file_system_is_not_read(tmp_path, fs, caplog, run) -> None:
    sizes = _names(2, GIB)
    host = FakeHost(sizes, fs=fs)
    with caplog.at_level(logging.INFO, logger=wp.__name__):
        prefetch = run(_model(tmp_path, sizes), host)
    assert host.reads == [] and prefetch.outcome.passes == 0
    assert "not a network file system" in caplog.text


@pytest.mark.parametrize("fs", ["nfs", "nfs4", "ceph", "fuse.juicefs", "lustre"])
def test_a_checkpoint_on_a_network_file_system_is_read(tmp_path, fs, run) -> None:
    sizes = _names(1, GIB)
    host = FakeHost(sizes, fs=fs)
    run(_model(tmp_path, sizes), host)
    assert len(host.reads) == 1


def test_too_little_headroom_reads_nothing(tmp_path, caplog, run) -> None:
    sizes = _names(1, GIB)
    host = FakeHost(sizes, used=LIMIT - 20 * GIB - 7 * GIB)  # 7 GiB over the margin: under the 8 GiB that matters
    with caplog.at_level(logging.INFO, logger=wp.__name__):
        run(_model(tmp_path, sizes), host)
    assert host.reads == [] and "too small to matter" in caplog.text


def test_it_stops_when_the_server_takes_the_headroom_and_does_not_read_again(tmp_path, caplog, run) -> None:
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
        prefetch = run(_model(tmp_path, sizes), host, max_refills=3)

    assert list(host.read_bytes()) == [next(iter(sizes))]
    assert prefetch.outcome.out_of_room
    assert prefetch.outcome.passes == 1, "a cache nothing dropped is not read again"
    assert "headroom ran out" in caplog.text


def test_what_the_loader_drops_from_the_cache_is_read_again_a_bounded_number_of_times(tmp_path, run) -> None:
    sizes = _names(2, 4 * GIB)
    host = FakeHost(sizes)
    original = host.read
    last = (list(sizes)[-1], (3 * GIB, 4 * GIB))

    def read(path, span, stop):
        got = original(path, span, stop)
        if (path.name, span) == last:
            host.drop_cache()  # the loader drops the checkpoint's page cache right after each pass
        return got

    host.read = read  # type: ignore[method-assign]
    prefetch = run(_model(tmp_path, sizes), host, max_refills=2, watch_sec=10.0)

    assert prefetch.outcome.passes == 3, "it ends on its own once the refills are spent"
    assert [name for name, _ in host.reads] == ([list(sizes)[0]] * 4 + [list(sizes)[1]] * 4) * 3


def test_a_drop_seen_while_a_pass_reads_ends_it_and_reads_the_largest_file_again_first(tmp_path, run) -> None:
    sizes = {"model-1.safetensors": 2 * GIB, "model-2.safetensors": 8 * GIB}
    host = FakeHost(sizes)
    original = host.read
    interrupted = threading.Event()

    def read(path, span, stop):
        if path.name == "model-1.safetensors" and not interrupted.is_set():
            stop.wait(0.2)  # long enough for a look at the cache to see the large file whole
            host.drop_cache()  # the loader drops the cache while the small file is read
            if stop.wait(10.0):
                interrupted.set()
            return 0
        return original(path, span, stop)

    host.read = read  # type: ignore[method-assign]
    prefetch = run(_model(tmp_path, sizes), host, max_refills=1, watch_sec=3.0)

    assert interrupted.is_set(), "the drop ended the read in progress"
    assert prefetch.outcome.passes == 2
    reads = [name for name, _ in host.reads]
    assert reads == ["model-2.safetensors"] * 8 + ["model-2.safetensors"] * 8 + ["model-1.safetensors"] * 2
    assert host.cached("model-1.safetensors") and host.cached("model-2.safetensors")


def test_a_small_or_partial_loss_from_the_cache_is_not_a_drop(tmp_path, run) -> None:
    sizes = _names(2, 4 * GIB)
    host = FakeHost(sizes)
    names = list(sizes)
    original = host.read

    def read(path, span, stop):
        got = original(path, span, stop)
        if path.name == names[1] and span[1] == 4 * GIB:
            with host.lock:  # a third of one file reclaimed: under half of it, so not a drop
                host.uncached_spans[names[0]] = [(0, GIB + GIB // 3)]
        return got

    host.read = read  # type: ignore[method-assign]
    prefetch = run(_model(tmp_path, sizes), host, max_refills=3)
    assert prefetch.outcome.passes == 1


def test_after_the_servers_anonymous_memory_the_engram_tables_of_deepseek_v4_still_fit(tmp_path, run) -> None:
    """DeepSeek V4.1 Flash on a 448 GiB sandbox: 46 shards of 286.1 GiB, then two engram embedding files of 101.54 GB
    each. The loader drops the cache when it allocates ~200 GiB of anonymous memory for its host tables; the headroom
    left, 448 - 200 - 44.8 = 203.2 GiB, holds both engram files (189.1 GiB), which the refill reads first."""
    engram = 101_540_000_000
    shards = {f"model-{n:05d}-of-00048.safetensors": int(286.1 * GIB) // 46 for n in range(1, 47)}
    sizes = {**shards, "model-00047-of-00048.safetensors": engram, "model-00048-of-00048.safetensors": engram}
    host = FakeHost(sizes, used=4 * GIB)
    host.limit = 448 * GIB
    anon = 200 * GIB
    assert Memory(host.limit, anon).headroom() > 2 * engram > 189 * GIB
    original = host.read
    dropped, seen = threading.Event(), threading.Event()
    started = iter(range(10**6))
    refill_order: list[str] = []

    def read(path, span, stop):
        if stop.is_set():
            return 0  # a read the drop interrupted
        if seen.is_set():
            refill_order.append(path.name)
            return original(path, span, stop)
        n = next(started)
        if n < 250:  # the first pass is well on its way, the engram files read whole
            return original(path, span, stop)
        if n == 250:
            stop.wait(0.2)  # a look at the cache sees them whole
            host.drop_cache()  # the loader drops the cache and allocates its host tables
            with host.lock:
                host.used += anon
            dropped.set()
        if stop.wait(5.0):  # the first pass reads on, slowly, until the drop is seen
            seen.set()
        return 0

    host.read = read  # type: ignore[method-assign]
    prefetch = run(_model(tmp_path, sizes), host, streams=8, max_refills=3, watch_sec=2.0)

    engram_files = list(sizes)[-2:]
    assert dropped.is_set() and prefetch.outcome.passes >= 2
    assert all(host.cached(name) for name in engram_files), "both engram files are whole in the cache"
    first_refill = refill_order[: 2 * 95]
    assert set(first_refill) == set(engram_files), "the refill reads the engram files before any shard"
    assert prefetch.outcome.out_of_room, "the shards fill what is left"
    assert host.used <= host.limit - 44.8 * GIB + 8 * GIB, "never past the margin (less the pieces in flight)"


def test_a_cache_that_stays_warm_is_not_read_again(tmp_path, machine) -> None:
    sizes = _names(1, 4 * GIB)
    host = FakeHost(sizes)
    machine(host, max_refills=3)
    prefetch = WeightPrefetch(_model(tmp_path, sizes))
    prefetch.start()
    assert not prefetch.join(0.2), "it keeps watching the cache until it is stopped"
    prefetch.stop()
    assert prefetch.join(5.0)
    assert prefetch.outcome.passes == 1 and host.read_bytes() == {next(iter(sizes)): 4 * GIB}


def test_stop_ends_a_read_in_progress(tmp_path, machine) -> None:
    sizes = _names(2, GIB)
    host = FakeHost(sizes)
    reading = threading.Event()

    def read(path, span, stop):
        reading.set()
        stop.wait(30.0)  # a read the stop interrupts between two chunks
        return 0

    host.read = read  # type: ignore[method-assign]
    machine(host)
    prefetch = WeightPrefetch(_model(tmp_path, sizes))
    prefetch.start()
    assert reading.wait(10.0)
    prefetch.stop()
    assert prefetch.join(5.0)


def test_a_file_that_cannot_be_read_does_not_stop_the_others_and_is_counted(tmp_path, caplog, run) -> None:
    sizes = _names(3, GIB)
    host = FakeHost(sizes)
    names = list(sizes)
    host.fail_read = {names[1]}
    with caplog.at_level(logging.WARNING, logger=wp.__name__):
        prefetch = run(_model(tmp_path, sizes), host, streams=2)
    assert sorted(name for name, _ in host.reads) == [names[0], names[2]]
    assert prefetch.outcome.unreadable == 1
    failed = [record for record in caplog.records if "not read" in record.getMessage()]
    assert failed and failed[0].exc_info is not None, "the traceback is logged"
    assert "Input/output error" in caplog.text


def test_a_host_that_cannot_be_read_ends_the_prefetch_quietly(tmp_path, caplog, run) -> None:
    sizes = _names(1, GIB)
    host = FakeHost(sizes)

    def broken() -> Memory:
        raise PermissionError("cgroupfs is not readable")

    host.memory = broken  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger=wp.__name__):
        prefetch = run(_model(tmp_path, sizes), host)
    assert host.reads == [] and prefetch.outcome.passes == 0
    assert prefetch.outcome.error == "PermissionError: cgroupfs is not readable" and prefetch.outcome.ended
    failed = [record for record in caplog.records if "failed and ends" in record.getMessage()]
    assert failed and failed[0].exc_info is not None, "the traceback is logged"


def test_a_file_whose_residency_the_kernel_hides_is_read_whole_within_the_budget_and_not_refilled(
    tmp_path, run
) -> None:
    sizes = _names(4, 60 * GIB)
    host = FakeHost(sizes)
    host.blind = set(sizes)
    original = host.read

    def read(path, span, stop):
        got = original(path, span, stop)
        host.drop_cache()  # a refill would read a known file again; a blind one must not be
        return got

    host.read = read  # type: ignore[method-assign]
    prefetch = run(_model(tmp_path, sizes), host, max_refills=2)

    names = list(sizes)
    assert host.read_bytes() == {names[0]: 60 * GIB, names[1]: 60 * GIB}, "by size, within 170 GiB"
    assert prefetch.outcome.passes == 1
    assert [item["residency_known"] for item in prefetch.outcome.chosen] == [False, False]


def test_the_outcome_names_the_files_the_totals_and_why_nothing_was_read(tmp_path, run) -> None:
    sizes = _names(4)
    prefetch = run(_model(tmp_path, sizes), FakeHost(sizes))
    outcome = prefetch.outcome
    assert [item["name"] for item in outcome.chosen] == list(sizes)[:2]
    assert (outcome.files, outcome.total_bytes, outcome.read_bytes) == (4, 240 * GIB, 120 * GIB)
    assert outcome.fs_type == "nfs4" and outcome.skipped is None and outcome.ended

    record = tmp_path / "weight_prefetch.json"
    prefetch.write_outcome(record)
    assert json.loads(record.read_text(encoding="utf-8"))["read_bytes"] == 120 * GIB

    (tmp_path / "local").mkdir()
    skipped = run(_model(tmp_path / "local", sizes), FakeHost(sizes, fs="ext4")).outcome
    assert skipped.skipped == "on ext4, not a network file system" and skipped.chosen == []


def test_an_outcome_that_cannot_be_written_is_logged(tmp_path, caplog) -> None:
    prefetch = WeightPrefetch(tmp_path)
    with caplog.at_level(logging.WARNING, logger=wp.__name__):
        prefetch.write_outcome(tmp_path / "missing" / "weight_prefetch.json")
    assert caplog.records and caplog.records[0].exc_info is not None


def test_it_is_off_unless_the_environment_turns_it_on(tmp_path, machine) -> None:
    sizes = _names(1, GIB)
    model = _model(tmp_path, sizes)
    host = FakeHost(sizes)
    machine(host)
    assert start_weight_prefetch(str(model), {}) is None
    assert start_weight_prefetch(str(model), {WEIGHT_PREFETCH_ENV: "0"}) is None
    assert start_weight_prefetch(str(model), {WEIGHT_PREFETCH_ENV: "bogus"}) is None

    prefetch = start_weight_prefetch(str(model), {WEIGHT_PREFETCH_ENV: "1"})
    assert prefetch is not None
    assert host.read_seen.wait(10.0)
    prefetch.stop()
    assert host.reads == [(next(iter(sizes)), (0, GIB))]


def test_a_model_that_is_not_a_directory_reads_nothing_and_says_so(tmp_path, machine) -> None:
    on = {WEIGHT_PREFETCH_ENV: "1"}
    host = FakeHost({})
    machine(host)
    for model in ("org/model-name", str(tmp_path / "missing")):
        prefetch = start_weight_prefetch(model, on)
        assert prefetch is not None and prefetch.join(10.0)
        assert prefetch.outcome.skipped == "the model is not a local directory"
    assert host.reads == []


def test_a_thread_that_cannot_start_starts_nothing(tmp_path, machine, monkeypatch, caplog) -> None:
    machine(FakeHost({}))

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
    [(7, 7, False, True), (7, 5, True, True), (7, 5, False, False), (0, 5, False, False)],
)
def test_the_kernel_tells_residency_only_to_the_owner_or_a_writer(
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
def test_an_evicted_file_is_found_uncached_and_read_back_into_the_cache(tmp_path, monkeypatch, visible, run) -> None:
    model = tmp_path / "model"
    model.mkdir()
    path = model / "model.safetensors"
    size = 8 << 20
    _evicted(path, size)
    host = LinuxHost()
    spans: list[tuple[int, int]] | None = None
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
    prefetch = run(model, _NetworkHost())

    assert prefetch.outcome.read_bytes == (size if not visible else sum(end - start for start, end in spans))
    monkeypatch.setattr(wp, "_residency_visible", lambda path, owner: True)
    assert host.uncached(path, size) == [], "the read left the file in the page cache"


def test_a_pass_whose_streams_cannot_start_ends(tmp_path, run, monkeypatch) -> None:
    sizes = _names(1, 4 * GIB)
    host = FakeHost(sizes)

    class NoStreams(threading.Thread):
        def start(self) -> None:
            if self.name.startswith("weight-prefetch-"):
                raise RuntimeError("can't start new thread")
            super().start()

    monkeypatch.setattr(wp.threading, "Thread", NoStreams)
    prefetch = run(_model(tmp_path, sizes), host, max_refills=1)

    assert prefetch.outcome.ended and prefetch.outcome.passes == 1 and not host.reads
