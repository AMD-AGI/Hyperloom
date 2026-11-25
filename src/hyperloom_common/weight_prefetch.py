# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read a model's checkpoint into the page cache while its server starts, so the server's page faults hit memory.

A server that maps its safetensors files and reads them by page faults reads a network file system one fault at a time,
tens of MB/s per rank, where a few sequential streams read the same files at gigabytes per second. :class:`WeightPrefetch`
reads the files the loader reads with a few streams in background threads, from the moment the server is spawned until it
is ready, so that its faults find the pages cached. It reads the largest files first (:func:`prefetch_order`): a loader
reads its small shards quickly with its own threads but faults a very large file in one stream per rank, and those are
the files a cached page saves the most time on.

It runs alongside the server rather than before it. The server's own start (process start, distributed and device set-up,
shard loading) overlaps the reads instead of waiting for them, and a loader that drops the checkpoint's page cache while it
starts (SGLang's engram host table does, before it pre-faults its shard) discards what was read before. So until the
server is ready it samples, every ``_POLL_INTERVAL_SEC``, how much of each file it chose is still cached (``mincore`` on
a few pages spread over the file), and as soon as files it had cached lost most of their pages it reads what is missing
again, largest first, ending a pass in progress to do so; at most ``_MAX_REFILLS`` times.

``mincore`` tells the truth only about a file this process owns or may open for writing; for any other file Linux reports
every page cached. Such a file is read whole, by its size, within the same budget, and is not read again by a refill.

What it reads is bounded by memory. File pages count toward the cgroup's ``memory.current``, so it reads at most
``memory.max - memory.current`` less a margin. At the start it chooses, largest first, the files that fit that headroom;
then before each piece (``_PIECE``) it reads, it looks at the live headroom again, less what its other streams are reading,
and ends the pass when the piece no longer fits, rather than evict the pages it cached or crowd the server (whose memory
grows as it starts). Without a cgroup limit, the host's ``MemTotal`` and ``MemAvailable`` stand in. A checkpoint on a local
file system is left alone (a local disk is not the bottleneck), as is one whose headroom is too small to matter.

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
from collections.abc import Iterable, Mapping, Sequence
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
    "prefetch_order",
    "start_weight_prefetch",
]

#: ``1`` reads the checkpoint into the page cache while the server starts; unset or ``0`` (the default everywhere)
#: leaves it to the server.
WEIGHT_PREFETCH_ENV = "HYPERLOOM_WEIGHT_PREFETCH"

_GIB = 1 << 30
#: How many files are read at once. The field measurement this rests on read ~0.5 GB/s with one stream and ~2.3 GB/s
#: with four, still close to linear, so four had not reached what the storage serves; eight keeps going up that line at
#: a cost of eight 16 MiB buffers. Not an operator setting: a hosted round that shows otherwise changes it here.
_STREAMS = 8
#: How often the cache of the chosen files is sampled while the server starts, during a pass and after it, and how many
#: refill passes there are at most. Short, because a pass that has not yet seen a drop keeps reading files further down
#: the order into the headroom the files at its top need again: a second at a few GB/s is a few GB. A loader that drops the checkpoint's page cache does it once or twice in a start
#: (SGLang's engram host table: before it pre-faults its shard, and again when the shard lacks huge pages); the bound
#: only keeps a cache that is reclaimed over and over from being read over and over.
_POLL_INTERVAL_SEC = 1.0
_MAX_REFILLS = 8
#: A file that lost at least this share of itself from the cache since its highest share has been dropped.
_DROPPED_SHARE = 0.5
#: Bytes one stream reads of a file before it takes the next piece, so several streams share one large file.
_PIECE = _GIB
#: How many places of a file a sample looks at, and how many pages at each: a drop (``POSIX_FADV_DONTNEED``) or a reclaim
#: takes whole files or large runs of them, which a sample of a few hundred pages per file sees at a cost of about a
#: hundred syscalls per file per look, instead of a page vector of every page of a checkpoint of hundreds of GiB.
_SAMPLES = 32
_SAMPLE_PAGES = 16
#: Bytes one ``preadv`` asks for; a stop is seen between two of them.
_CHUNK = 16 << 20
#: Headroom kept free for the server: a tenth of the limit, and never less than this.
_MARGIN_FLOOR = 16 * _GIB
_MARGIN_SHARE = 10
#: Below this much headroom the reads would not cover enough of a checkpoint to matter.
_MIN_BUDGET = 8 * _GIB
#: A refill pass runs only when at least this much of the chosen files left the cache.
_MIN_REFILL = _GIB
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
    file reads failed, ``out_of_room`` whether the headroom ran out, ``error`` what ended it early."""

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


@dataclass(frozen=True)
class _Chosen:
    path: Path
    size: int
    #: Whether ``mincore`` answers for the file; a file it does not answer for is read whole and never refilled.
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


def prefetch_order(sized: Sequence[tuple[Path, int]]) -> list[tuple[Path, int]]:
    """The order the prefetch reads ``sized`` (files in the loader's order with their sizes) in: largest first, files of
    one size in the loader's order.

    A loader reads its many small shards with a few threads, from the network as fast as a prefetch would, but faults a
    very large file in through a mapping, one stream per rank; and it reads the largest files last in a checkpoint whose
    biggest tensors sit in its last files (DeepSeek V4's engram embedding tables, two files of 101.5 GB at the end of 48).
    Those are the reads a cached page saves the most time on, and the ones a prefix of the loader's order leaves out when
    the memory does not hold the whole checkpoint. A checkpoint of shards of one size is read in the loader's order."""
    return sorted(sized, key=lambda pair: -pair[1])


def _span_total(spans: Iterable[Span]) -> int:
    return sum(end - start for start, end in spans)


def _pieces(spans: Iterable[Span]) -> Iterable[Span]:
    """``spans`` cut into pieces of at most ``_PIECE`` bytes, so that several streams share one large file."""
    for start, end in spans:
        for offset in range(start, end, _PIECE):
            yield offset, min(offset + _PIECE, end)


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
    #: Files a read of failed; their other pieces are not read in this pass.
    failed: set[Path] = field(default_factory=set)
    #: How many pieces of each file were read whole.
    whole: dict[Path, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    done: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)


class _Watch:
    """The sampled share of each watched file in the page cache, against the highest share seen since the last refill:
    a file that lost at least ``_DROPPED_SHARE`` of itself has been dropped (by a loader, or reclaimed)."""

    def __init__(self, host: Host, watched: Sequence[_Chosen]) -> None:
        self._host = host
        self._watched = watched
        self._highest = {item.path: 0.0 for item in watched}

    def filled(self, paths: Iterable[Path]) -> None:
        """A pass read every page of ``paths`` that was missing: they were whole in the cache, even if a drop comes
        before the next sample."""
        for path in paths:
            if path in self._highest:
                self._highest[path] = 1.0

    def dropped(self) -> int:
        """Sample every watched file; the bytes of the files that dropped, ``0`` when less than ``_MIN_REFILL``. A drop
        found starts the highest shares again from what is cached now, so the same drop is not found twice."""
        shares = {item.path: self._host.cached_share(item.path, item.size) for item in self._watched}
        lost = 0
        for item in self._watched:
            share = shares[item.path]
            if share is None:
                continue
            highest = self._highest[item.path] = max(self._highest[item.path], share)
            if highest - share >= _DROPPED_SHARE:
                lost += int((highest - share) * item.size)
        if lost < _MIN_REFILL:
            return 0
        for path, share in shares.items():
            self._highest[path] = share or 0.0
        return lost


class WeightPrefetch:
    """Reads one checkpoint into the page cache in background threads from :meth:`start` until :meth:`stop`."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._host: Host = LinuxHost()
        self._stop = threading.Event()
        #: Ends the reads of the pass in progress: set by :meth:`stop`, and by a drop seen while the pass reads.
        self._interrupt = threading.Event()
        self._thread: threading.Thread | None = None
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
        ``_POLL_INTERVAL_SEC`` and a refill pass as soon as a drop is seen, at most ``_MAX_REFILLS`` of them. A drop seen
        while a pass reads ends that pass and starts the refill at once, from the top of the order again."""
        chosen = self._choose()
        if not chosen:
            return
        watched = [item for item in chosen if item.residency_known]
        watch = _Watch(self._host, watched)
        lost = self._read_pass(chosen, watch)
        if not watched:
            return
        refills = 0
        while not self._stop.is_set():
            if not lost:
                if self._stop.wait(_POLL_INTERVAL_SEC):
                    return
                lost = watch.dropped()
                continue
            if refills >= _MAX_REFILLS:
                log.info("weight prefetch: %.1f GiB left the page cache again; no refill is left", lost / _GIB)
                return
            refills += 1
            log.info(
                "weight prefetch: %.1f GiB of the chosen files left the page cache; reading them again", lost / _GIB
            )
            lost = self._read_pass(watched, watch)

    def _skip(self, reason: str) -> list[_Chosen]:
        self.outcome.skipped = reason
        log.info("weight prefetch of %s skipped: %s", self._model_dir, reason)
        return []

    def _choose(self) -> list[_Chosen]:
        """The files to read with their sizes, in :func:`prefetch_order`: each whose uncached bytes still fit the
        headroom once the files before it are counted, a file that does not fit left out."""
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
        for path, size in prefetch_order(sized):
            spans = self._host.uncached(path, size)
            need = size if spans is None else _span_total(spans)
            if wanted + need > budget:
                continue
            wanted += need
            chosen.append(_Chosen(path, size, spans is not None))
        self.outcome.files, self.outcome.total_bytes = len(files), total
        self.outcome.chosen = [
            {"name": item.path.name, "size": item.size, "residency_known": item.residency_known} for item in chosen
        ]
        log.info(
            "weight prefetch: %s on %s, %d files of %.1f GiB; reading %.1f GiB of %d, largest first (headroom %.1f GiB) "
            "with %d streams",
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

    def _read_pass(self, chosen: Sequence[_Chosen], watch: _Watch) -> int:
        """Read what of ``chosen`` is not cached, in pieces in that order, ``streams`` pieces at a time, looking at the
        cache every ``_POLL_INTERVAL_SEC`` meanwhile; the bytes a drop seen meanwhile took (the pass then ends early),
        else ``0``."""
        self.outcome.passes += 1
        self._interrupt.clear()
        if self._stop.is_set():
            self._interrupt.set()
        work: queue.SimpleQueue[tuple[_Chosen, Span]] = queue.SimpleQueue()
        queued: dict[Path, int] = {}
        for item in chosen:
            spans = self._host.uncached(item.path, item.size)
            queued[item.path] = 0
            for piece in _pieces([(0, item.size)] if spans is None else spans):
                work.put((item, piece))
                queued[item.path] += 1
        pieces = sum(queued.values())
        tally = _PassTally(running=min(_STREAMS, pieces))
        started = time.monotonic()
        if not pieces:
            tally.done.set()
        for n in range(tally.running):
            threading.Thread(target=self._stream, args=(work, tally), name=f"weight-prefetch-{n}", daemon=True).start()
        lost = 0
        try:
            while not tally.done.wait(_POLL_INTERVAL_SEC):
                lost = lost or watch.dropped()
                if lost:
                    self._interrupt.set()
        except BaseException:
            self._interrupt.set()  # a look at the cache that fails ends the reads with it
            raise
        elapsed = max(time.monotonic() - started, 1e-9)
        if not lost:  # after a drop, what was read before it is no longer whole
            watch.filled(path for path, count in queued.items() if tally.whole.get(path, 0) == count)
        self.outcome.read_bytes += tally.read
        self.outcome.unreadable += tally.unreadable
        self.outcome.out_of_room = self.outcome.out_of_room or tally.out_of_room
        log.info(
            "weight prefetch pass %d: read %.1f GiB from %d of %d files in %.1f s, %.0f MB/s, %d unreadable%s%s",
            self.outcome.passes,
            tally.read / _GIB,
            len(tally.files),
            len(chosen),
            elapsed,
            tally.read / elapsed / 1e6,
            tally.unreadable,
            "; stopped: the memory headroom ran out" if tally.out_of_room else "",
            "; ended early: the page cache dropped what was read" if lost else "",
        )
        return lost

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
        """Read one piece if the live headroom, less what the other streams are reading now, still holds it."""
        need = piece[1] - piece[0]
        memory = self._host.memory()
        with tally.lock:
            if item.path in tally.failed:
                return
            if memory is None or need > memory.headroom() - tally.pending:
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
                if read == need:
                    tally.whole[item.path] = tally.whole.get(item.path, 0) + 1


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
