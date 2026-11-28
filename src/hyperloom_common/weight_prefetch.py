# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read a model's checkpoint into the page cache while its server starts, so the server's page faults hit memory.

A server that maps its safetensors files and reads them by page faults reads a network file system one fault at a time,
tens of MB/s per rank, where a few sequential streams read the same files at gigabytes per second.
:class:`WeightPrefetch` reads the files the loader reads with a few streams in background threads, from the moment the
server is spawned until it is ready, so that its faults find the pages cached. It reads them in the loader's order
(:func:`checkpoint_files`), the order the loader consumes them in.

It runs alongside the server rather than before it. The server's own start (process start, distributed and device
set-up, shard loading) overlaps the reads instead of waiting for them. A loader may drop the checkpoint's page cache
while it starts (SGLang's engram host table calls ``POSIX_FADV_DONTNEED`` on every checkpoint file before it pre-faults
its shard, and may again in ``finish_load``), which discards what was read before. So until the server is ready it
samples, every ``_POLL_INTERVAL_SEC``, how much of the checkpoint is cached (``mincore`` on a few pages spread over each
file), and when the cached total collapses, at least ``_COLLAPSE_SHARE`` of the most it held in the last
``_COLLAPSE_WINDOW_SEC`` gone, it ends the pass in progress, waits ``_COLLAPSE_PAUSE_SEC`` (every rank drops, each a
moment after the other) and reads the whole checkpoint once more in the loader's order; at most ``_MAX_REFILLS`` times.
A gradual loss is not a drop: the kernel reclaiming the oldest pages, when the server's own memory and the reads meet
the cgroup limit, takes a few GB a second, never most of a checkpoint at once, and reading those pages again would only
evict others.

The page cache is the host's, so servers on one host share the checkpoint's: a drop by one server takes only what no
other server has mapped, while the others' loaders and reads keep the rest, and the cached total of a later server then
falls by less than ``_COLLAPSE_SHARE`` (measured: the third of three servers on one host never read again after its
drop). So after its first pass, until it has read again once, it also reads again when the cached total has fallen
below ``_LEVEL_SHARE`` of the most it ever held while this cgroup has at least ``_MIN_COLLAPSE`` of headroom left. The
headroom tells the two losses apart: reclaim in this cgroup happens only at its limit, where no headroom is left, while
a drop frees the pages this cgroup was charged for.

The pass after a drop is not bounded by the headroom. The server allocates most of its memory right after the drop (the
engram host tables, ~200 GiB of anonymous huge pages for DeepSeek V4), after which what is left does not hold the
checkpoint; the loader then consumes it in this order, and the kernel reclaims the oldest pages first, those it has
consumed by then. Clean file pages never make a cgroup run out of memory: the server's allocations reclaim them.

``mincore`` tells the truth only about a file this process owns or may open for writing; for any other file Linux
reports every page cached. Such a file is read whole, by its size, within the same budget, and its pages are not counted
in the cached total.

Its first pass is bounded by memory. File pages count toward the cgroup's ``memory.current``, so it reads at most
``memory.max - memory.current`` less a margin. At the start it chooses, in the loader's order, the files that fit that
headroom; then before each piece (``_PIECE``) it reads, it looks at the live headroom again, less what its other streams
are reading, and ends the pass when the piece no longer fits, rather than evict the pages it cached or crowd the server
(whose memory grows as it starts). Without a cgroup limit, the host's ``MemTotal`` and ``MemAvailable`` stand in. A
checkpoint on a local file system is left alone (a local disk is not the bottleneck), as is one whose headroom is too
small to matter.

It is off unless ``HYPERLOOM_WEIGHT_PREFETCH`` is ``1`` in the environment of the process that boots the server, for
every kind of run: a hosted deployment sets it in its ``measurement`` template's environment, and a local sandbox takes it
from the environment that starts the session. What it did (the files it chose, why it skipped, what it read, the files
it could not read) is :attr:`WeightPrefetch.outcome`, which the server's lifecycle writes into the round's run_dir. A
prefetch never fails the round: every error is logged and ends it.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import json
import logging
import mmap
import os
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from hyperloom_common.env import is_truthy

log = logging.getLogger(__name__)

__all__ = [
    "WEIGHT_PREFETCH_ENV",
    "Host",
    "LinuxHost",
    "Memory",
    "PrefetchOutcome",
    "WeightPrefetch",
    "checkpoint_files",
    "start_weight_prefetch",
]

#: ``1`` reads the checkpoint into the page cache while the server starts; unset or ``0`` (the default everywhere)
#: leaves it to the server.
WEIGHT_PREFETCH_ENV = "HYPERLOOM_WEIGHT_PREFETCH"

_GIB = 1 << 30
#: How many pieces are read at once. The field measurement this rests on read ~0.5 GB/s with one stream and ~2.3 GB/s
#: with four, still close to linear, so four had not reached what the storage serves; eight keeps going up that line at
#: a cost of eight 16 MiB buffers. Not an operator setting: a hosted round that shows otherwise changes it here.
_STREAMS = 8
#: How often the cache of the checkpoint is sampled while the server starts, during a pass and after it.
_POLL_INTERVAL_SEC = 1.0
#: A drop is the checkpoint's cached total falling by at least this share of the most it held within the window. A
#: loader's ``POSIX_FADV_DONTNEED`` takes all of it within a few seconds; reclaim under the cgroup limit takes a few GB a
#: second (measured on DeepSeek V4: half of one 94.6 GiB file of 230 GiB read, over about a minute), never most of it.
_COLLAPSE_SHARE = 0.75
_COLLAPSE_WINDOW_SEC = 10.0
#: How long after a drop the next pass starts: every rank drops the cache, each a moment after the other.
_COLLAPSE_PAUSE_SEC = 5.0
#: How many times the checkpoint is read again after a drop. SGLang's engram host table drops it once before it
#: pre-faults its shard and once more in ``finish_load`` when the shard lacks huge pages.
_MAX_REFILLS = 2
#: A drop other servers on the host hide: the cached total below this share of the most it ever held, after the first
#: pass and before any pass that read again, with at least ``_MIN_COLLAPSE`` of this cgroup's headroom free.
_LEVEL_SHARE = 0.5
#: What a pass was read for (the run_dir record names it): the first pass, or the rule that found the drop before it.
_FIRST = "first"
_COLLAPSE = "collapse"
_LEVEL = "level"
#: How often a sample of the cache is kept for the run_dir record, and how many of the last are kept (30 minutes).
_HISTORY_EVERY_SEC = 10.0
_HISTORY_SAMPLES = 180
#: Bytes one stream reads of a file before it takes the next piece, so several streams share one large file.
_PIECE = _GIB
#: How many places of a file a sample looks at, and how many pages at each: a drop (``POSIX_FADV_DONTNEED``) or a reclaim
#: takes whole files or large runs of them, which a sample of a few hundred pages per file sees at a cost of about a
#: hundred syscalls per file per look, instead of a page vector of every page of a checkpoint of hundreds of GiB.
_SAMPLES = 32
_SAMPLE_PAGES = 16
#: Bytes one ``preadv`` asks for; a stop is seen between two of them.
_CHUNK = _GIB // 64  # 16 MiB
#: Headroom kept free for the server: a tenth of the limit, and never less than this.
_MARGIN_FLOOR = 16 * _GIB
_MARGIN_SHARE = 10
#: Below this much headroom the reads would not cover enough of a checkpoint to matter.
_MIN_BUDGET = _MARGIN_FLOOR // 2  # 8 GiB
#: Below this much cached, a loss is not looked at: too little for a drop to matter, or to tell it from reclaim.
_MIN_COLLAPSE = _MARGIN_FLOOR  # 16 GiB
#: How much of a file one ``mincore`` call maps, so its page vector stays small.
_RESIDENCY_WINDOW = _GIB
#: How long :meth:`WeightPrefetch.stop` waits for the reads to end.
_STOP_WAIT_SEC = 5.0
_INDEX = "model.safetensors.index.json"
_SUFFIX = ".safetensors"
_UTF8 = "utf-8"
#: File systems whose reads cross the network; a checkpoint anywhere else is not prefetched.
_NETWORK_FS = frozenset(
    {"nfs", "nfs4", "ceph", "cifs", "smb3", "smbfs", "lustre", "gpfs", "beegfs", "glusterfs", "9p", "virtiofs", "afs"}
)
_FUSE_PREFIX = "fuse"
_DIGITS = re.compile(r"(\d+)")
_UNCACHED_RUN = re.compile(rb"\x00+")
#: ``mincore`` sets bit 0 of a page's byte when the page is resident; this maps each byte to that bit alone.
_RESIDENT_BIT = bytes(value & 1 for value in range(256))

Span = tuple[int, int]


@dataclass(frozen=True)
class Memory:
    """The memory the prefetch may fill: the limit, and what is in use under it (file pages included)."""

    limit: int
    used: int

    def headroom(self) -> int:
        """What can still be cached before the margin the server keeps."""
        return self.limit - self.used - max(_MARGIN_FLOOR, self.limit // _MARGIN_SHARE)


class Host(Protocol):
    """What the prefetch reads of the machine; :class:`LinuxHost` is the real one, a test gives a fake."""

    def fs_type(self, path: Path) -> str | None:
        """The type of the file system ``path`` is on, ``None`` when it cannot be told."""

    def memory(self) -> Memory | None:
        """This process's memory limit and use, ``None`` when neither can be read."""

    def size(self, path: Path) -> int:
        """The file's size in bytes."""

    def uncached(self, path: Path, size: int) -> list[Span] | None:
        """The byte ranges of the file that are not in the page cache, as ``(start, end)``; ``None`` when the kernel
        does not say which pages of this file are cached."""

    def cached_share(self, path: Path, size: int) -> float | None:
        """The share of the file's pages in the page cache, sampled; ``None`` when the kernel does not tell."""

    def read(self, path: Path, span: Span, stop: threading.Event) -> int:
        """Read ``span`` of the file into the page cache, until it is read or ``stop`` is set; the bytes read."""


@dataclass
class PrefetchOutcome:
    """What one prefetch did, as the round's run_dir records it: ``skipped`` says why nothing was read, ``chosen`` the
    files it read with their sizes (``residency_known`` false for a file read whole by size), ``unreadable`` how many
    file reads failed, ``out_of_room`` whether the headroom ran out, ``error`` what ended it early.

    ``pass_log`` has one entry per pass (at most ``1 + _MAX_REFILLS``): its UTC start and end, ``reason`` (``first``, or
    the rule that found the drop before it: ``collapse`` or ``level``), the bytes and files it read, whether the
    headroom ran out and whether a drop ended it. ``drops`` has one entry per drop found: when, by which rule, and the
    sampled cache of the checkpoint before (the most of the window, or the most ever for ``level``) and after.
    ``cache_peak_bytes`` is the most the sampled cache held and when; while no drop was found, ``cache_samples_gib``
    keeps the last samples, one every ``_HISTORY_EVERY_SEC``, as ``[seconds since start, cached GiB, headroom GiB]``."""

    model_dir: str
    fs_type: str | None = None
    skipped: str | None = None
    files: int = 0
    total_bytes: int = 0
    budget_bytes: int | None = None
    chosen: list[dict[str, object]] = field(default_factory=list)
    streams: int = _STREAMS
    passes: int = 0
    read_bytes: int = 0
    unreadable: int = 0
    out_of_room: bool = False
    error: str | None = None
    elapsed_sec: float = 0.0
    ended: bool = False
    pass_log: list[dict[str, object]] = field(default_factory=list)
    drops: list[dict[str, object]] = field(default_factory=list)
    cache_peak_bytes: int = 0
    cache_peak_at: str | None = None
    cache_samples_gib: list[list[float | None]] = field(default_factory=list)


@dataclass(frozen=True)
class _Chosen:
    path: Path
    size: int
    #: Whether ``mincore`` answers for the file; a file it does not answer for is read whole by its size, and is not
    #: counted in the cached total a drop is seen in.
    residency_known: bool


def _natural(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part for part in _DIGITS.split(path.name)]


def checkpoint_files(model_dir: Path) -> list[Path]:
    """The safetensors files the loader reads, in its order: the files ``model.safetensors.index.json`` names (every
    ``*.safetensors`` file when there is no index), sorted by name with numbers compared as numbers."""
    index = model_dir / _INDEX
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding=_UTF8)).get("weight_map") or {}
        named = {model_dir / str(name) for name in weight_map.values()}
        files = [path for path in named if path.is_file()]
    else:
        files = [path for path in model_dir.glob(f"*{_SUFFIX}") if path.is_file()]
    return sorted(files, key=_natural)


def _span_total(spans: Iterable[Span]) -> int:
    return sum(end - start for start, end in spans)


def _pieces(spans: Iterable[Span]) -> Iterable[Span]:
    """``spans`` cut into pieces of at most ``_PIECE`` bytes, so that several streams share one large file."""
    for start, end in spans:
        for offset in range(start, end, _PIECE):
            yield offset, min(offset + _PIECE, end)


def _stamp() -> str:
    """Now in UTC to the second, as the run_dir record spells times."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gib(value: int) -> float:
    return round(value / _GIB, 1)


def _is_network(fs_type: str | None) -> bool:
    return fs_type is not None and (fs_type in _NETWORK_FS or fs_type.startswith(_FUSE_PREFIX))


@dataclass
class _PassTally:
    """What one pass read, under the lock its streams share; ``pending`` is the bytes the streams are reading now, which
    the memory in use does not count yet."""

    files: set[Path] = field(default_factory=set)
    read: int = 0
    unreadable: int = 0
    out_of_room: bool = False
    pending: int = 0
    running: int = 0
    #: Whether each piece must fit the live headroom.
    bounded: bool = True
    #: Files a read of failed; their other pieces are not read in this pass.
    failed: set[Path] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    done: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)


class _Watch:
    """The sampled bytes of the watched files in the page cache, against the most seen in the last
    ``_COLLAPSE_WINDOW_SEC``: a fall by ``_COLLAPSE_SHARE`` of that is a drop of the checkpoint's cache. It also keeps
    the most ever seen (``peak``), the last sample (``last``) and a sample every ``_HISTORY_EVERY_SEC`` (``history``)."""

    def __init__(self, host: Host, watched: Sequence[_Chosen], clock: Callable[[], float] = time.monotonic) -> None:
        self._host = host
        self._watched = watched
        self._clock = clock
        self._recent: deque[tuple[float, int]] = deque()
        self._started = self._next_history = clock()
        self.peak = self.last = 0
        self.peak_at: str | None = None
        self.history: deque[list[float | None]] = deque(maxlen=_HISTORY_SAMPLES)

    def cached(self) -> int:
        """The watched files' bytes in the page cache now, sampled."""
        total = 0.0
        for item in self._watched:
            share = self._host.cached_share(item.path, item.size)
            total += (share or 0.0) * item.size
        return int(total)

    def collapsed(self) -> int:
        """Sample the cache; the bytes a drop took, ``0`` when there was none. A drop found starts the window again,
        so the same drop is not found twice."""
        now, total = self._clock(), self.cached()
        self.last = total
        if total > self.peak:
            self.peak, self.peak_at = total, _stamp()
        if now >= self._next_history:
            self._next_history = now + _HISTORY_EVERY_SEC
            memory = self._host.memory()
            headroom = None if memory is None else _gib(memory.headroom())
            self.history.append([round(now - self._started), _gib(total), headroom])
        while self._recent and now - self._recent[0][0] > _COLLAPSE_WINDOW_SEC:
            self._recent.popleft()
        peak = max([total, *(cached for _, cached in self._recent)])
        self._recent.append((now, total))
        lost = peak - total
        if peak < _MIN_COLLAPSE or lost < _COLLAPSE_SHARE * peak:
            return 0
        self._recent.clear()
        return lost

    def faded(self) -> int:
        """The bytes the last sample is below the most ever seen, when it is below ``_LEVEL_SHARE`` of it; else ``0``."""
        if self.peak < _MIN_COLLAPSE or self.last >= _LEVEL_SHARE * self.peak:
            return 0
        return self.peak - self.last

    def reset(self) -> None:
        self._recent.clear()


class WeightPrefetch:
    """Reads one checkpoint into the page cache in background threads from :meth:`start` until :meth:`stop`."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._host: Host = LinuxHost()
        self._stop = threading.Event()
        #: Ends the reads of the pass in progress: set by :meth:`stop`, and by a drop seen while the pass reads.
        self._interrupt = threading.Event()
        self._thread: threading.Thread | None = None
        #: Every checkpoint file in the loader's order, as :meth:`_choose` found it.
        self._files: list[_Chosen] = []
        self.outcome = PrefetchOutcome(str(model_dir), streams=_STREAMS)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="weight-prefetch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the reads to stop and wait a few seconds for them; a read still blocked is left to finish alone."""
        self._stop.set()
        self._interrupt.set()
        self.join(_STOP_WAIT_SEC)

    def join(self, timeout_sec: float) -> bool:
        """Wait up to ``timeout_sec`` for the prefetch to end; whether it has."""
        if self._thread is None:
            return True
        self._thread.join(timeout_sec)
        return not self._thread.is_alive()

    def write_outcome(self, path: Path) -> None:
        """Write :attr:`outcome` as JSON to ``path``; a write that fails is logged, never raised."""
        try:
            path.write_text(json.dumps(asdict(self.outcome), indent=1) + "\n", encoding=_UTF8)
        except OSError:
            log.warning("weight prefetch: its outcome could not be written to %s", path, exc_info=True)

    def _run(self) -> None:
        started = time.monotonic()
        try:
            self._prefetch()
        except Exception as exc:  # isolation boundary: the prefetch only saves time, never fails a round
            self.outcome.error = f"{type(exc).__name__}: {exc}"
            log.warning("weight prefetch of %s failed and ends", self._model_dir, exc_info=True)
        self.outcome.elapsed_sec = round(time.monotonic() - started, 3)
        self.outcome.ended = True
        if self.outcome.passes:
            log.info(
                "weight prefetch of %s ended: %.1f GiB read in %d pass(es) over %.0f s, %d file read(s) failed",
                self._model_dir,
                self.outcome.read_bytes / _GIB,
                self.outcome.passes,
                self.outcome.elapsed_sec,
                self.outcome.unreadable,
            )

    def _prefetch(self) -> None:
        """The first pass over the chosen files, then, until :meth:`stop`, a look at the cache every
        ``_POLL_INTERVAL_SEC``; after a drop, seen while a pass reads (which ends it) or after, a pause and a pass over
        the whole checkpoint, at most ``_MAX_REFILLS`` of them."""
        chosen = self._choose()
        if not chosen:
            return
        watched = [item for item in self._files if item.residency_known]
        watch = _Watch(self._host, watched)
        rule = self._read_pass(chosen, watch, bounded=True, reason=_FIRST)
        if not watched:
            return
        refills = 0
        while not self._stop.is_set():
            if rule is None:
                if self._stop.wait(_POLL_INTERVAL_SEC):
                    return
                rule = self._look(watch, level=refills == 0)
                continue
            if refills >= _MAX_REFILLS:
                log.info("weight prefetch: the checkpoint's page cache was dropped again; no refill is left")
                return
            refills += 1
            log.info("weight prefetch: reading the whole checkpoint again in %.0f s", _COLLAPSE_PAUSE_SEC)
            if self._stop.wait(_COLLAPSE_PAUSE_SEC):
                return
            watch.reset()
            rule = self._read_pass(self._files, watch, bounded=False, reason=rule)

    def _look(self, watch: _Watch, *, level: bool) -> str | None:
        """Sample the cache; the rule that found a drop (``collapse``, or ``level`` when ``level`` allows it), else
        ``None``. A drop found is recorded in :attr:`outcome`, and logged."""
        lost = watch.collapsed()
        rule, before = (_COLLAPSE, watch.last + lost) if lost else (None, 0)
        if rule is None and level and watch.faded():
            memory = self._host.memory()
            if memory is not None and memory.headroom() >= _MIN_COLLAPSE:
                rule, before = _LEVEL, watch.peak
        outcome = self.outcome
        outcome.cache_peak_bytes, outcome.cache_peak_at = watch.peak, watch.peak_at
        if rule is None:
            if not outcome.drops:
                outcome.cache_samples_gib = list(watch.history)
            return None
        outcome.drops.append({"at": _stamp(), "rule": rule, "cached_before": before, "cached_after": watch.last})
        outcome.cache_samples_gib = []
        log.info(
            "weight prefetch: the checkpoint's sampled page cache fell from %.1f to %.1f GiB (%s): a drop",
            before / _GIB,
            watch.last / _GIB,
            "at once" if rule == _COLLAPSE else "below half of the most it held, with headroom free",
        )
        return rule

    def _skip(self, reason: str) -> list[_Chosen]:
        self.outcome.skipped = reason
        log.info("weight prefetch of %s skipped: %s", self._model_dir, reason)
        return []

    def _choose(self) -> list[_Chosen]:
        """The files to read with their sizes, in the loader's order: each whose uncached bytes still fit the headroom
        once the files before it are counted, a file that does not fit left out. Every file is kept in ``_files``."""
        if not self._model_dir.is_dir():
            return self._skip("the model is not a local directory")
        files = checkpoint_files(self._model_dir)
        if not files:
            return self._skip("no safetensors checkpoint")
        fs_type = self.outcome.fs_type = self._host.fs_type(self._model_dir)
        if not _is_network(fs_type):
            return self._skip(f"on {fs_type}, not a network file system")
        memory = self._host.memory()
        budget = None if memory is None else memory.headroom()
        self.outcome.budget_bytes = budget
        if budget is None or budget < _MIN_BUDGET:
            shown = "unknown" if budget is None else f"{budget / _GIB:.1f} GiB"
            return self._skip(f"memory headroom {shown} is too small to matter")
        sized = [(path, self._host.size(path)) for path in files]
        total = sum(size for _, size in sized)
        chosen: list[_Chosen] = []
        wanted = 0
        for path, size in sized:
            spans = self._host.uncached(path, size)
            need = size if spans is None else _span_total(spans)
            item = _Chosen(path, size, spans is not None)
            self._files.append(item)
            if wanted + need > budget:
                continue
            wanted += need
            chosen.append(item)
        self.outcome.files, self.outcome.total_bytes = len(files), total
        self.outcome.chosen = [
            {"name": item.path.name, "size": item.size, "residency_known": item.residency_known} for item in chosen
        ]
        log.info(
            "weight prefetch: %s on %s, %d files of %.1f GiB; reading %.1f GiB of %d in the loader's order "
            "(headroom %.1f GiB) with %d streams",
            self._model_dir,
            fs_type,
            len(files),
            total / _GIB,
            wanted / _GIB,
            len(chosen),
            budget / _GIB,
            _STREAMS,
        )
        return chosen

    def _read_pass(self, chosen: Sequence[_Chosen], watch: _Watch, *, bounded: bool, reason: str) -> str | None:
        """Read what of ``chosen`` is not cached, in pieces in that order, ``streams`` pieces at a time (each within the
        live headroom when ``bounded``), looking at the cache every ``_POLL_INTERVAL_SEC`` meanwhile for a collapse;
        the rule that found a drop meanwhile (the pass then ends early), else ``None``. ``reason`` is what the pass is
        read for, as :attr:`outcome` records it."""
        self.outcome.passes += 1
        record: dict[str, object] = {"pass": self.outcome.passes, "reason": reason, "start": _stamp()}
        self.outcome.pass_log.append(record)
        self._interrupt.clear()
        if self._stop.is_set():
            self._interrupt.set()
        work: queue.SimpleQueue[tuple[_Chosen, Span]] = queue.SimpleQueue()
        pieces = 0
        for item in chosen:
            spans = self._host.uncached(item.path, item.size)
            for piece in _pieces([(0, item.size)] if spans is None else spans):
                work.put((item, piece))
                pieces += 1
        tally = _PassTally(running=min(_STREAMS, pieces), bounded=bounded)
        started = time.monotonic()
        if not pieces:
            tally.done.set()
        for n in range(tally.running):
            try:
                threading.Thread(
                    target=self._stream, args=(work, tally), name=f"weight-prefetch-{n}", daemon=True
                ).start()
            except RuntimeError:  # a stream that did not start is not waited for, or the pass would never end
                log.warning("weight prefetch: stream %d did not start", n, exc_info=True)
                with tally.lock:
                    tally.running -= 1
                    if tally.running <= 0:
                        tally.done.set()
        rule = None
        try:
            while not tally.done.wait(_POLL_INTERVAL_SEC):
                if self._interrupt.is_set():
                    continue  # ending: no look at the cache while a read blocked on the storage returns
                rule = self._look(watch, level=False)
                if rule:
                    self._interrupt.set()
        except BaseException:
            self._interrupt.set()  # a look at the cache that fails ends the reads with it
            raise
        self._end_pass(record, tally, len(chosen), time.monotonic() - started, dropped=rule is not None)
        return rule

    def _end_pass(
        self, record: dict[str, object], tally: _PassTally, files: int, elapsed: float, *, dropped: bool
    ) -> None:
        """Add what a pass over ``files`` files read to :attr:`outcome`, its ``record`` (the last of ``pass_log``) completed
        in a new entry, and log it."""
        elapsed = max(elapsed, 1e-9)
        self.outcome.read_bytes += tally.read
        self.outcome.unreadable += tally.unreadable
        self.outcome.out_of_room = self.outcome.out_of_room or tally.out_of_room
        # A new entry in place of the old, never the old one changed: write_outcome may copy the record meanwhile.
        self.outcome.pass_log[-1] = {
            **record,
            "end": _stamp(),
            "read_bytes": tally.read,
            "files_read": len(tally.files),
            "out_of_room": tally.out_of_room,
            "ended_by_drop": dropped,
        }
        log.info(
            "weight prefetch pass %d: read %.1f GiB from %d of %d files in %.1f s, %.0f MB/s, %d unreadable%s%s",
            record["pass"],
            tally.read / _GIB,
            len(tally.files),
            files,
            elapsed,
            tally.read / elapsed / 1e6,
            tally.unreadable,
            "; stopped: the memory headroom ran out" if tally.out_of_room else "",
            "; ended early: the checkpoint's page cache was dropped" if dropped else "",
        )

    def _stream(self, work: queue.SimpleQueue[tuple[_Chosen, Span]], tally: _PassTally) -> None:
        try:
            while not self._interrupt.is_set() and not tally.out_of_room:
                try:
                    item, piece = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    self._read_piece(item, piece, tally)
                except Exception:  # isolation boundary: one unreadable file ends no other read
                    with tally.lock:
                        first = item.path not in tally.failed
                        tally.failed.add(item.path)
                        tally.unreadable += first
                    if first:
                        log.warning("weight prefetch: %s not read", item.path.name, exc_info=True)
        finally:
            with tally.lock:
                tally.running -= 1
                if tally.running <= 0:
                    tally.done.set()

    def _read_piece(self, item: _Chosen, piece: Span, tally: _PassTally) -> None:
        """Read one piece if the live headroom, less what the other streams are reading now, still holds it (or the pass
        is not bounded)."""
        need = piece[1] - piece[0]
        memory = self._host.memory() if tally.bounded else None
        with tally.lock:
            if item.path in tally.failed:
                return
            if tally.bounded and (memory is None or need > memory.headroom() - tally.pending):
                tally.out_of_room = True
                return
            tally.pending += need
        read = 0
        try:
            read = self._host.read(item.path, piece, self._interrupt)
        finally:
            with tally.lock:
                tally.pending -= need
                tally.read += read
                if read:
                    tally.files.add(item.path)


def start_weight_prefetch(model_path: str, environ: Mapping[str, str]) -> WeightPrefetch | None:
    """Start reading the checkpoint at ``model_path`` when ``environ`` turns the prefetch on; ``None`` when it is off
    or its thread could not start. Why a started prefetch read nothing is in its :attr:`~WeightPrefetch.outcome`."""
    if not is_truthy(environ.get(WEIGHT_PREFETCH_ENV)):
        return None
    prefetch = WeightPrefetch(Path(model_path))
    try:
        prefetch.start()
    except RuntimeError:
        log.warning("weight prefetch of %s did not start", model_path, exc_info=True)
        return None
    return prefetch


# --- the real machine --------------------------------------------------------------------------------------------

_PROC_SELF = Path("/proc/self")
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_NO_LIMIT = "max"
_MOUNTINFO_SEPARATOR = " - "
_MOUNT_POINT_FIELD = 4
_KIB = 1024
#: The characters the kernel escapes in a mountinfo path (``seq_path``'s set), as it spells them.
_MOUNTINFO_ESCAPES = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
_MOUNTINFO_ESCAPE = re.compile("|".join(re.escape(escape) for escape in _MOUNTINFO_ESCAPES))


def _unescape(field: str) -> str:
    """A mountinfo path with the kernel's escapes (``\\040`` for a space) undone."""
    return _MOUNTINFO_ESCAPE.sub(lambda match: _MOUNTINFO_ESCAPES[match.group(0)], field)


def _mount_fs_type(mountinfo: str, path: str) -> str | None:
    """The file-system type of the mount the absolute ``path`` is under: the longest mount point it starts with."""
    best, best_type = "", None
    for line in mountinfo.splitlines():
        head, separator, tail = line.partition(_MOUNTINFO_SEPARATOR)
        fields = head.split()
        if not separator or len(fields) <= _MOUNT_POINT_FIELD or not tail.split():
            continue
        point = _unescape(fields[_MOUNT_POINT_FIELD])
        inside = path == point or path.startswith(point.rstrip("/") + "/")
        if inside and len(point) >= len(best):
            best, best_type = point, tail.split()[0]
    return best_type


def _cgroup_memory(cgroup_root: Path, proc_cgroup: str) -> Memory | None:
    """The cgroup v2 limit and use of the process whose ``/proc/<pid>/cgroup`` is ``proc_cgroup``, ``None`` without one."""
    own = next((line[3:] for line in proc_cgroup.splitlines() if line.startswith("0::")), None)
    candidates = [cgroup_root / own.lstrip("/")] if own else []
    candidates.append(cgroup_root)
    for directory in candidates:
        try:
            limit = (directory / "memory.max").read_text(encoding="ascii").strip()
            used = (directory / "memory.current").read_text(encoding="ascii").strip()
        except OSError:
            continue
        if limit != _NO_LIMIT:
            return Memory(int(limit), int(used))
    return None


def _meminfo_memory(meminfo: str) -> Memory | None:
    values = {}
    for line in meminfo.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[name] = int(parts[0]) * _KIB
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    return None if total is None or available is None else Memory(total, total - available)


def _uncached_runs(flags: bytes, offset: int, end: int) -> Iterable[Span]:
    """The byte ranges of a window starting at ``offset`` whose pages ``flags`` (one byte each) marks uncached."""
    for match in _UNCACHED_RUN.finditer(flags.translate(_RESIDENT_BIT)):
        yield offset + match.start() * mmap.PAGESIZE, min(offset + match.end() * mmap.PAGESIZE, end)


def _merge(spans: Iterable[Span]) -> list[Span]:
    merged: list[Span] = []
    for start, end in spans:
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _residency_visible(path: Path, owner: int) -> bool:
    """Whether ``mincore`` reports which pages of the file at ``path`` (owned by ``owner``) are cached. Linux answers
    only for a file the caller owns or may open for writing; for any other file, a file of another account or on a
    read-only mount, it reports every page cached. Root is no exception: root of a user namespace that does not map the
    file's owner, or root without ``CAP_FOWNER`` and ``CAP_DAC_OVERRIDE``, gets the same answer, and ``os.access`` says
    so too. Real root over a read-only mount is then told hidden, and reads the file whole."""
    if os.geteuid() == owner:
        return True
    return os.access(path, os.W_OK, effective_ids=os.access in os.supports_effective_ids)


class _Libc:
    """``mmap``, ``mincore`` and ``munmap`` through ctypes: Python's own ``mmap`` hands out no address to ask about."""

    _PROT_READ = 1
    _MAP_SHARED = 1

    def __init__(self) -> None:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        self._mmap = libc.mmap
        self._mmap.restype = ctypes.c_void_p
        self._mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        self._mincore = libc.mincore
        self._mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        self._munmap = libc.munmap
        self._munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._failed = ctypes.c_void_p(-1).value

    def residency(self, fd: int, offset: int, length: int) -> bytes:
        """One byte per page of ``length`` bytes at ``offset`` (a page multiple) of ``fd``; bit 0 set when cached."""
        address = self._mmap(None, length, self._PROT_READ, self._MAP_SHARED, fd, offset)
        if address in (None, self._failed):
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            vector = ctypes.create_string_buffer(-(-length // mmap.PAGESIZE))
            if self._mincore(address, length, vector) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return vector.raw
        finally:
            self._munmap(address, length)


class LinuxHost:
    """The machine this process runs on, read through procfs, cgroupfs, ``mincore`` and ``preadv``."""

    def __init__(self) -> None:
        self._libc: _Libc | None = None

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding=_UTF8, errors="replace")

    def fs_type(self, path: Path) -> str | None:
        return _mount_fs_type(self._read_text(_PROC_SELF / "mountinfo"), os.path.realpath(path))

    def memory(self) -> Memory | None:
        with contextlib.suppress(OSError):
            limited = _cgroup_memory(_CGROUP_ROOT, self._read_text(_PROC_SELF / "cgroup"))
            if limited is not None:
                return limited
        with contextlib.suppress(OSError):
            return _meminfo_memory(self._read_text(Path("/proc/meminfo")))
        return None

    def size(self, path: Path) -> int:
        return path.stat().st_size

    def uncached(self, path: Path, size: int) -> list[Span] | None:
        if size <= 0:
            return []
        if self._libc is None:
            self._libc = _Libc()
        fd = os.open(path, os.O_RDONLY)
        try:
            if not _residency_visible(path, os.fstat(fd).st_uid):
                return None
            spans: list[Span] = []
            for offset in range(0, size, _RESIDENCY_WINDOW):
                length = min(_RESIDENCY_WINDOW, size - offset)
                spans.extend(_uncached_runs(self._libc.residency(fd, offset, length), offset, offset + length))
            return _merge(spans)
        finally:
            os.close(fd)

    def cached_share(self, path: Path, size: int) -> float | None:
        pages = -(-size // mmap.PAGESIZE)
        if pages <= 0:
            return 1.0
        if self._libc is None:
            self._libc = _Libc()
        fd = os.open(path, os.O_RDONLY)
        try:
            if not _residency_visible(path, os.fstat(fd).st_uid):
                return None
            seen = resident = 0
            for first in sorted({n * pages // _SAMPLES for n in range(min(_SAMPLES, pages))}):
                count = min(_SAMPLE_PAGES, pages - first)
                length = min(count * mmap.PAGESIZE, size - first * mmap.PAGESIZE)
                flags = self._libc.residency(fd, first * mmap.PAGESIZE, length)
                seen += len(flags)
                resident += sum(flags.translate(_RESIDENT_BIT))
            return resident / seen
        finally:
            os.close(fd)

    def read(self, path: Path, span: Span, stop: threading.Event) -> int:
        start, end = span
        fd = os.open(path, os.O_RDONLY)
        try:
            with contextlib.suppress(OSError):
                os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_SEQUENTIAL)
            buffer = memoryview(bytearray(_CHUNK))
            offset = start
            while offset < end and not stop.is_set():
                got = os.preadv(fd, [buffer[: min(_CHUNK, end - offset)]], offset)
                if got <= 0:
                    break
                offset += got
            return offset - start
        finally:
            os.close(fd)
