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
    it, with one stream and no refill unless the test says otherwise, a short pause after a drop, and a drop seen from
    1 GiB cached (the files of most tests are small)."""

    def install(
        host,
        *,
        streams: int = 1,
        max_refills: int = 0,
        poll_interval_sec: float = 0.01,
        min_collapse: int = GIB,
    ) -> None:
        monkeypatch.setattr(wp, "LinuxHost", lambda: host)
        monkeypatch.setattr(wp, "_STREAMS", streams)
        monkeypatch.setattr(wp, "_MAX_REFILLS", max_refills)
        monkeypatch.setattr(wp, "_POLL_INTERVAL_SEC", poll_interval_sec)
        monkeypatch.setattr(wp, "_COLLAPSE_PAUSE_SEC", 0.01)
        monkeypatch.setattr(wp, "_MIN_COLLAPSE", min_collapse)

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


def test_a_file_that_does_not_fit_is_left_out_and_a_later_one_that_fits_is_read(tmp_path, run) -> None:
    sizes = {"model-1.safetensors": 30 * GIB, "model-2.safetensors": 200 * GIB, "model-3.safetensors": 120 * GIB}
    host = FakeHost(sizes)  # 170 GiB of headroom: the 200 GiB file never fits, the 120 GiB one after it does
    prefetch = run(_model(tmp_path, sizes), host)

    assert host.read_bytes() == {"model-1.safetensors": 30 * GIB, "model-3.safetensors": 120 * GIB}
    assert [item["name"] for item in prefetch.outcome.chosen] == ["model-1.safetensors", "model-3.safetensors"]


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


def test_after_each_drop_the_whole_checkpoint_is_read_again_a_bounded_number_of_times(tmp_path, run) -> None:
    sizes = _names(4)  # 240 GiB against 170 GiB of headroom: the first pass reads two files, each refill all four
    host = FakeHost(sizes)
    original = host.read

    def read(path, span, stop):
        got = original(path, span, stop)
        with host.lock:
            whole = sum(1 for spans in host.uncached_spans.values() if not spans)
        if whole == (2 if len(host.reads) == 120 else 4):
            threading.Timer(0.1, host.drop_cache).start()  # the loader drops the page cache a moment after each pass
        return got

    host.read = read  # type: ignore[method-assign]
    prefetch = run(_model(tmp_path, sizes), host, max_refills=2, watch_sec=10.0)

    names = list(sizes)
    assert prefetch.outcome.passes == 3, "it ends on its own once the refills are spent"
    assert [entry["reason"] for entry in prefetch.outcome.pass_log] == ["first", "collapse", "collapse"]
    assert [entry["read_bytes"] for entry in prefetch.outcome.pass_log] == [120 * GIB, 240 * GIB, 240 * GIB]
    assert [drop["rule"] for drop in prefetch.outcome.drops] == ["collapse"] * 3
    assert all(drop["cached_after"] < drop["cached_before"] / 4 for drop in prefetch.outcome.drops)
    assert prefetch.outcome.cache_samples_gib == [], "samples are kept only while no drop was found"
    assert [name for name, _ in host.reads] == [n for n in names[:2] for _ in range(60)] + [
        n for n in names for _ in range(60)
    ] * 2, "a refill reads every file in the loader's order, past the headroom the first pass kept to"


def test_a_drop_seen_while_a_pass_reads_ends_it_and_reads_the_checkpoint_again_from_its_first_file(
    tmp_path, run
) -> None:
    sizes = {"model-1.safetensors": 8 * GIB, "model-2.safetensors": 2 * GIB}
    host = FakeHost(sizes)
    original = host.read
    interrupted = threading.Event()

    def read(path, span, stop):
        if path.name == "model-2.safetensors" and not interrupted.is_set():
            stop.wait(0.2)  # long enough for a look at the cache to see the first file whole
            host.drop_cache()  # the loader drops the cache while the second file is read
            if stop.wait(10.0):
                interrupted.set()
            return 0
        return original(path, span, stop)

    host.read = read  # type: ignore[method-assign]
    prefetch = run(_model(tmp_path, sizes), host, max_refills=1, watch_sec=3.0)

    assert interrupted.is_set(), "the drop ended the read in progress"
    assert prefetch.outcome.passes == 2
    reads = [name for name, _ in host.reads]
    assert reads == ["model-1.safetensors"] * 8 + ["model-1.safetensors"] * 8 + ["model-2.safetensors"] * 2
    assert host.cached("model-1.safetensors") and host.cached("model-2.safetensors")


def test_reclaim_of_the_oldest_pages_mid_pass_neither_ends_the_pass_nor_is_read_again(tmp_path, run) -> None:
    """As measured on DeepSeek V4 (two servers on one host): 111 GiB read, then half of the first file, 47.3 GiB, gone
    between two looks. That is the kernel reclaiming the oldest pages, not a loader's drop."""
    sizes = _names(3, 50 * GIB)
    host = FakeHost(sizes)
    names = list(sizes)
    original = host.read
    reclaimed = threading.Event()

    def read(path, span, stop):
        got = original(path, span, stop)
        if path.name == names[2] and span[0] == 0:
            with host.lock:  # the first half of the first file, the oldest pages, reclaimed at once
                host.uncached_spans[names[0]] = [(0, 25 * GIB)]
                host.used -= 25 * GIB
            reclaimed.set()
            stop.wait(0.2)  # several looks at the cache see the loss while the pass reads on
        return got

    host.read = read  # type: ignore[method-assign]
    prefetch = run(_model(tmp_path, sizes), host, max_refills=3, watch_sec=0.5)

    assert reclaimed.is_set()
    assert prefetch.outcome.passes == 1, "neither ended early nor refilled"
    assert host.read_bytes() == dict.fromkeys(names, 50 * GIB)
    assert not host.cached(names[0]), "what reclaim took is left to the loader"


def _deepseek_v4(used: int) -> FakeHost:
    """DeepSeek V4.1 Flash on a 448 GiB sandbox: 46 shards of 286.1 GiB, then two engram files of 101.54 GB each."""
    engram = 101_540_000_000
    shards = {f"model-{n:05d}-of-00048.safetensors": int(286.1 * GIB) // 46 for n in range(1, 47)}
    host = FakeHost(
        {**shards, "model-00047-of-00048.safetensors": engram, "model-00048-of-00048.safetensors": engram}, used=used
    )
    host.limit = 448 * GIB
    return host


def test_deepseek_v4_reclaim_before_the_engram_drop_is_ignored_and_the_drop_rereads_everything_in_order(
    tmp_path, run
) -> None:
    """The measured timeline of one DeepSeek V4 server: the first pass reads (in the loader's order here) while the
    server allocates; reclaim takes ~50 GiB of the oldest pages ~90 s in; ~180 s in, the engram host table drops the
    cache of all 48 files and allocates ~200 GiB of huge pages. The reclaim is no drop; the drop ends the pass, and
    the pass after it reads every file in the loader's order, the engram files last, past the headroom."""
    host = _deepseek_v4(used=GIB)
    names = list(host.sizes)
    original = host.read
    reclaimed, dropped, seen = threading.Event(), threading.Event(), threading.Event()
    lock = threading.Lock()
    count = iter(range(10**6))
    after_drop: list[str] = []

    def read(path, span, stop):
        if stop.is_set():
            return 0  # a read the drop interrupted
        if seen.is_set():
            after_drop.append(path.name)
            return original(path, span, stop)
        with lock:
            n = next(count)
        if n == 100:
            with host.lock:  # the oldest 50 GiB reclaimed while the server allocates
                for name in names[:8]:
                    host.uncached_spans[name] = [(0, host.sizes[name])]
                host.used += 30 * GIB
            reclaimed.set()
            stop.wait(0.2)
        if n < 250:
            return original(path, span, stop)
        if n == 250:
            stop.wait(0.2)  # a look at the cache sees what is there
            host.drop_cache()
            with host.lock:
                host.used += 200 * GIB
            dropped.set()
        if stop.wait(5.0):  # the first pass reads on, slowly, until the drop is seen
            seen.set()
        return 0

    host.read = read  # type: ignore[method-assign]
    prefetch = run(_model(tmp_path, host.sizes), host, streams=8, max_refills=2, watch_sec=3.0, min_collapse=16 * GIB)

    assert reclaimed.is_set() and dropped.is_set(), "the reclaim did not end the first pass"
    assert [item["name"] for item in prefetch.outcome.chosen] == names[:47], "the loader's order, as far as it fits"
    assert prefetch.outcome.passes == 2
    assert list(dict.fromkeys(after_drop)) == names, "after the drop, every file in the loader's order"
    assert all(host.cached(name) for name in names), "the engram files too, past the headroom"


def test_only_a_collapse_of_the_cached_total_is_a_drop() -> None:
    """The sampled total against the most of the last 10 s, replaying the measured DeepSeek V4 timelines."""
    host = _deepseek_v4(used=GIB)
    names = list(host.sizes)
    items = [wp._Chosen(Path(name), size, True) for name, size in host.sizes.items()]
    now = [0.0]
    watch = wp._Watch(host, items, clock=lambda: now[0])

    def look(seconds: float = 1.0) -> int:
        now[0] += seconds
        return watch.collapsed()

    def cache(name: str, start: int = 0, end: int | None = None) -> None:
        end = host.sizes[name] if end is None else end
        host.uncached_spans[name] = _subtract(host.uncached_spans[name], (start, end))

    for name in names[:30]:  # ~186 GiB read over a minute and a half
        cache(name)
        assert look(3.0) == 0
    for name in names[:8]:  # reclaim takes the oldest ~50 GiB in one look: no drop
        host.uncached_spans[name] = [(0, host.sizes[name])]
    assert look() == 0
    for _ in range(10):  # then a shard a second for ten seconds, 62 GiB of 136: no drop either
        name = next(n for n in names if not host.uncached_spans[n])
        host.uncached_spans[name] = [(0, host.sizes[name])]
        assert look() == 0
    for name in names[8:]:
        cache(name)
    assert look() == 0
    cached = watch.cached()
    host.drop_cache()  # the engram host table drops every file
    assert look() == cached
    assert look() == 0, "the same drop is not found twice"


def _after_each_pass(tmp_path, machine, host: FakeHost, after_pass, *, passes: int) -> WeightPrefetch:
    """Run a prefetch of DeepSeek V4 on ``host``; once each pass has ended and the warm cache was looked at,
    ``after_pass`` changes the cache while the prefetch watches, up to ``passes`` times; the server is ready a moment
    after the last."""
    machine(host, streams=8, max_refills=2, min_collapse=16 * GIB)
    prefetch = WeightPrefetch(_model(tmp_path, host.sizes))
    prefetch.start()
    done = 0
    for _ in range(500):
        log = prefetch.outcome.pass_log
        if len(log) > done and "end" in log[done]:
            threading.Event().wait(0.1)  # a few looks at the warm cache
            after_pass(done)
            done += 1
        if done == passes or prefetch.join(0.02):
            break
    threading.Event().wait(0.3)
    prefetch.stop()
    assert prefetch.join(10.0)
    return prefetch


def _drop_unmapped(host: FakeHost, kept: float) -> None:
    """A loader's drop on a host other servers share: of what the cache held of each file, only the share ``kept``
    stays, the pages the other servers have mapped. Files here are read whole, so a file is cached whole or not at all."""
    with host.lock:
        for name, size in host.sizes.items():
            cached = size - sum(end - start for start, end in host.uncached_spans[name])
            gone = int(cached * (1 - kept))
            if gone:
                host.uncached_spans[name] = [(0, size - cached + gone)]
                host.used -= gone


def test_a_drop_other_servers_on_the_host_hide_is_read_again_once(tmp_path, machine) -> None:
    """As measured with three servers on one host: the drop took only what the other servers had not mapped, under the
    three quarters the collapse rule looks for, and the latest server never read again (main weights 1930 s). The
    level rule reads once: the cache fell below half of the most it held, and this cgroup has its headroom back. A
    second such fall after that pass is not read again: only a collapse may use the second refill."""

    host = _deepseek_v4(used=GIB)
    prefetch = _after_each_pass(tmp_path, machine, host, lambda n: _drop_unmapped(host, kept=0.45), passes=2)

    outcome = prefetch.outcome
    assert outcome.passes == 2, "one read after the hidden drop, none after the second"
    assert [entry["reason"] for entry in outcome.pass_log] == ["first", "level"]
    assert outcome.pass_log[1]["files_read"] == len(host.sizes), "the whole checkpoint"
    assert [drop["rule"] for drop in outcome.drops] == ["level"]
    drop = outcome.drops[0]
    assert drop["cached_before"] == outcome.pass_log[0]["read_bytes"], "the most it held: what the first pass read"
    assert drop["cached_before"] / 4 < drop["cached_after"] < drop["cached_before"] / 2, "not a collapse"
    record = tmp_path / "weight_prefetch.json"
    prefetch.write_outcome(record)
    written = json.loads(record.read_text(encoding="utf-8"))
    assert [entry["reason"] for entry in written["pass_log"]] == ["first", "level"]
    assert record.stat().st_size < 16 * 1024


def _look_at(machine, host: FakeHost) -> tuple[WeightPrefetch, list[float], wp._Watch]:
    machine(host, min_collapse=16 * GIB)
    prefetch = WeightPrefetch(Path("/nonexistent"))
    items = [wp._Chosen(Path(name), size, True) for name, size in host.sizes.items()]
    now = [0.0]
    return prefetch, now, wp._Watch(host, items, clock=lambda: now[0])


def test_reclaim_at_the_cgroup_limit_after_the_first_pass_is_no_drop(machine) -> None:
    """The measured single-server timeline, with the first pass over before reclaim: the server's memory and the cache
    meet the limit, reclaim takes the oldest pages a few GB a second until most of what was read is gone. No headroom
    is left the whole time, so the level rule does not read again; the engram drop after it is a collapse."""
    host = _deepseek_v4(used=GIB)
    names = list(host.sizes)
    prefetch, now, watch = _look_at(machine, host)

    def look(level: bool = True) -> str | None:
        now[0] += 1.0
        return prefetch._look(watch, level=level)

    for name in names[:40]:  # ~250 GiB read, the first pass bounded by the headroom
        host.uncached_spans[name] = []
        assert look(level=False) is None
    host.used = host.limit - 40 * GIB  # out of room: under a GiB of headroom past the 44.8 GiB margin
    assert prefetch._host.memory().headroom() < GIB
    for name in names[:30]:  # reclaim at the limit, a shard a second: the server grows, the cache shrinks
        host.uncached_spans[name] = [(0, host.sizes[name])]
        assert look() is None
    assert watch.last <= watch.peak / 4, "most of what was read is gone, gradually"
    for name in names[:30]:
        host.uncached_spans[name] = []
    for _ in range(11):
        assert look() is None
    host.drop_cache()
    host.used = 60 * GIB
    assert look() == "collapse"
    assert [drop["rule"] for drop in prefetch.outcome.drops] == ["collapse"]


def test_a_fall_below_half_with_headroom_free_is_a_drop_only_when_the_level_rule_may_look(machine) -> None:
    host = _deepseek_v4(used=GIB)
    names = list(host.sizes)
    prefetch, now, watch = _look_at(machine, host)
    for name in names[:40]:
        host.uncached_spans[name] = []
    now[0] += 1.0
    assert prefetch._look(watch, level=True) is None
    for name in names[:24]:  # 60 % gone over 24 s: no collapse
        host.uncached_spans[name] = [(0, host.sizes[name])]
        now[0] += 1.0
        assert prefetch._look(watch, level=False) is None
    host.used = 100 * GIB
    now[0] += 1.0
    assert prefetch._look(watch, level=False) is None, "during a pass, or after a read again, only a collapse counts"
    now[0] += 1.0
    assert prefetch._look(watch, level=True) == "level"
    assert prefetch.outcome.cache_peak_bytes == watch.peak and prefetch.outcome.cache_peak_at


def test_a_small_cache_that_empties_is_not_a_drop() -> None:
    sizes = _names(2, 4 * GIB)
    host = FakeHost(sizes)
    items = [wp._Chosen(Path(name), size, True) for name, size in sizes.items()]
    host.uncached_spans = {name: [] for name in sizes}
    watch = wp._Watch(host, items, clock=lambda: 0.0)
    assert watch.collapsed() == 0
    host.drop_cache()
    assert watch.collapsed() == 0, "8 GiB is under the 16 GiB a drop is looked for in"


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
    outcome = prefetch.outcome
    assert outcome.drops == [] and outcome.cache_peak_bytes == 4 * GIB
    assert outcome.cache_samples_gib and outcome.cache_samples_gib[0][1:] == [4.0, 166.0]
    assert outcome.pass_log[0]["reason"] == "first" and outcome.pass_log[0]["files_read"] == 1


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
