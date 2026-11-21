# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read a model's checkpoint into the page cache while its server starts, so the server's page faults hit memory.

A server that maps its safetensors files and reads them by page faults reads a network file system one fault at a time,
tens of MB/s per rank, where a few sequential streams read the same files at gigabytes per second. :class:`WeightPrefetch`
reads the files the loader reads, in the loader's order, with a few streams in background threads, from the moment the
server is spawned until it is ready, so that its faults find the pages cached.

It runs alongside the server rather than before it. The server's own start (process start, distributed and device set-up,
shard loading) overlaps the reads instead of waiting for them, and a loader that drops the checkpoint's page cache while it
starts (SGLang's engram host table does, before it pre-faults its shard) would discard what a read before the start had
cached. So after the first pass, every ``refill_interval_sec`` until the server is ready, it asks the kernel which pages
of the files it chose are no longer cached (``mincore``) and reads those again, at most ``max_refills`` times.

What it reads is bounded by memory. File pages count toward the cgroup's ``memory.current``, so it reads at most
``memory.max - memory.current`` less a margin: the files in the loader's order while they fit. Before each file it looks at
the headroom again and stops when the file no longer fits, rather than evict the pages it cached or crowd the server.
Without a cgroup limit, the host's ``MemTotal`` and ``MemAvailable`` stand in. A checkpoint on a local file system is left
alone (a local disk is not the bottleneck), as is one whose headroom is too small to matter.

It is switched by ``HYPERLOOM_WEIGHT_PREFETCH``. A hosted runner turns it on for its sandbox unless the variable is set
there already; a local run leaves it off, since a local machine's checkpoints are usually on local disk and its page cache
is shared with whatever else runs there. A prefetch never fails the round: every error is logged and ends it.
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from hyperloom_common.env import is_truthy

log = logging.getLogger(__name__)

__all__ = [
    "WEIGHT_PREFETCH_ENV",
    "WEIGHT_PREFETCH_STREAMS_ENV",
    "Host",
    "LinuxHost",
    "Memory",
    "PrefetchTuning",
    "WeightPrefetch",
    "checkpoint_files",
    "start_weight_prefetch",
]

#: ``1`` reads the checkpoint into the page cache while the server starts; unset or ``0`` leaves it to the server.
WEIGHT_PREFETCH_ENV = "HYPERLOOM_WEIGHT_PREFETCH"
#: How many files are read at once (default :data:`_DEFAULT_STREAMS`).
WEIGHT_PREFETCH_STREAMS_ENV = "HYPERLOOM_WEIGHT_PREFETCH_STREAMS"

_GIB = 1 << 30
_DEFAULT_STREAMS = 8
_MAX_STREAMS = 64
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
_INDEX = "model.safetensors.index.json"
_SUFFIX = ".safetensors"
#: File systems whose reads cross the network; a checkpoint anywhere else is not prefetched.
_NETWORK_FS = frozenset(
    {"nfs", "nfs4", "ceph", "cifs", "smb3", "smbfs", "lustre", "gpfs", "beegfs", "glusterfs", "9p", "virtiofs", "afs"}
)
_FUSE_PREFIX = "fuse"
_DIGITS = re.compile(r"(\d+)")
_UNCACHED_RUN = re.compile(rb"\x00+")
#: ``mincore`` sets bit 0 of a page's byte when the page is resident; this maps each byte to that bit alone.
_RESIDENT_BIT = bytes(value & 1 for value in range(256))


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

    def uncached(self, path: Path, size: int) -> list[tuple[int, int]]:
        """The byte ranges of the file that are not in the page cache, as ``(start, end)``."""

    def read(self, path: Path, span: tuple[int, int], stop: threading.Event) -> int:
        """Read ``span`` of the file into the page cache, until it is read or ``stop`` is set; the bytes read."""


@dataclass(frozen=True)
class PrefetchTuning:
    """How many files are read at once, how long a refill waits, and how many refills there are at most."""

    streams: int = _DEFAULT_STREAMS
    refill_interval_sec: float = 30.0
    max_refills: int = 3


def _natural(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part for part in _DIGITS.split(path.name)]


def checkpoint_files(model_dir: Path) -> list[Path]:
    """The safetensors files the loader reads, in its order: the files ``model.safetensors.index.json`` names (every
    ``*.safetensors`` file when there is no index), sorted by name with numbers compared as numbers."""
    index = model_dir / _INDEX
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
        named = {model_dir / str(name) for name in weight_map.values()}
        files = [path for path in named if path.is_file()]
    else:
        files = [path for path in model_dir.glob(f"*{_SUFFIX}") if path.is_file()]
    return sorted(files, key=_natural)


def _span_total(spans: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in spans)


def _is_network(fs_type: str | None) -> bool:
    return fs_type is not None and (fs_type in _NETWORK_FS or fs_type.startswith(_FUSE_PREFIX))


@dataclass
class _PassTally:
    """What one pass read, under the lock its streams share."""

    files: int = 0
    read: int = 0
    out_of_room: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


class WeightPrefetch:
    """Reads one checkpoint into the page cache in background threads from :meth:`start` until :meth:`stop`."""

    def __init__(self, model_dir: Path, host: Host | None = None, tuning: PrefetchTuning | None = None) -> None:
        self._model_dir = model_dir
        self._host: Host = LinuxHost() if host is None else host
        self._tuning = PrefetchTuning() if tuning is None else tuning
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.passes = 0
        self.read_bytes = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="weight-prefetch", daemon=True)
        self._thread.start()

    def stop(self, timeout_sec: float = 5.0) -> None:
        """Ask the reads to stop and wait up to ``timeout_sec`` for them; a read still blocked is left to finish alone."""
        self._stop.set()
        self.join(timeout_sec)

    def join(self, timeout_sec: float) -> bool:
        """Wait up to ``timeout_sec`` for the prefetch to end; whether it has."""
        if self._thread is None:
            return True
        self._thread.join(timeout_sec)
        return not self._thread.is_alive()

    def _run(self) -> None:
        started = time.monotonic()
        try:
            self._prefetch()
        except Exception as exc:  # noqa: BLE001 - isolation boundary: the prefetch only saves time, never fails a round
            log.warning("weight prefetch of %s failed and ends: %s: %s", self._model_dir, type(exc).__name__, exc)
        if self.passes:
            elapsed = time.monotonic() - started
            log.info(
                "weight prefetch of %s ended: %.1f GiB read in %d pass(es) over %.0f s",
                self._model_dir,
                self.read_bytes / _GIB,
                self.passes,
                elapsed,
            )

    def _prefetch(self) -> None:
        chosen = self._choose()
        if not chosen:
            return
        if not self._read_pass(chosen):
            return
        refills = 0
        while refills < self._tuning.max_refills and not self._stop.wait(self._tuning.refill_interval_sec):
            uncached = sum(_span_total(self._host.uncached(path, size)) for path, size in chosen)
            if uncached < _MIN_REFILL:
                continue
            refills += 1
            log.info(
                "weight prefetch: %.1f GiB of the chosen files left the page cache; reading them again", uncached / _GIB
            )
            if not self._read_pass(chosen):
                return

    def _choose(self) -> list[tuple[Path, int]]:
        """The files to read with their sizes: a prefix of the loader's order whose uncached bytes fit the headroom."""
        files = checkpoint_files(self._model_dir)
        if not files:
            log.info("weight prefetch: %s holds no safetensors checkpoint; nothing to read", self._model_dir)
            return []
        fs_type = self._host.fs_type(self._model_dir)
        if not _is_network(fs_type):
            log.info("weight prefetch: %s is on %s, not a network file system; not read", self._model_dir, fs_type)
            return []
        memory = self._host.memory()
        budget = None if memory is None else memory.headroom()
        if budget is None or budget < _MIN_BUDGET:
            shown = "unknown" if budget is None else f"{budget / _GIB:.1f} GiB"
            log.info("weight prefetch: memory headroom %s is too small to matter; %s not read", shown, self._model_dir)
            return []
        chosen: list[tuple[Path, int]] = []
        total = wanted = 0
        for path in files:
            size = self._host.size(path)
            total += size
            need = _span_total(self._host.uncached(path, size))
            if wanted + need > budget:
                break
            wanted += need
            chosen.append((path, size))
        log.info(
            "weight prefetch: %s on %s, %d files of %.1f GiB; reading %.1f GiB of the first %d (headroom %.1f GiB) "
            "with %d streams",
            self._model_dir,
            fs_type,
            len(files),
            total / _GIB,
            wanted / _GIB,
            len(chosen),
            budget / _GIB,
            self._tuning.streams,
        )
        return chosen

    def _read_pass(self, chosen: Sequence[tuple[Path, int]]) -> bool:
        """Read what of ``chosen`` is not cached, ``streams`` files at a time; whether the memory still had room."""
        self.passes += 1
        work: queue.SimpleQueue[tuple[Path, int]] = queue.SimpleQueue()
        for item in chosen:
            work.put(item)
        tally = _PassTally()
        started = time.monotonic()
        streams = [
            threading.Thread(target=self._stream, args=(work, tally), name=f"weight-prefetch-{n}", daemon=True)
            for n in range(min(self._tuning.streams, len(chosen)))
        ]
        for stream in streams:
            stream.start()
        for stream in streams:
            stream.join()
        elapsed = max(time.monotonic() - started, 1e-9)
        self.read_bytes += tally.read
        log.info(
            "weight prefetch pass %d: read %.1f GiB from %d of %d files in %.1f s, %.0f MB/s%s",
            self.passes,
            tally.read / _GIB,
            tally.files,
            len(chosen),
            elapsed,
            tally.read / elapsed / 1e6,
            "; stopped: the memory headroom ran out" if tally.out_of_room else "",
        )
        return not tally.out_of_room

    def _stream(self, work: queue.SimpleQueue[tuple[Path, int]], tally: _PassTally) -> None:
        while not self._stop.is_set() and not tally.out_of_room:
            try:
                path, size = work.get_nowait()
            except queue.Empty:
                return
            try:
                self._read_file(path, size, tally)
            except Exception as exc:  # noqa: BLE001 - isolation boundary: one unreadable file ends no other read
                log.warning("weight prefetch: %s not read: %s: %s", path.name, type(exc).__name__, exc)

    def _read_file(self, path: Path, size: int, tally: _PassTally) -> None:
        spans = self._host.uncached(path, size)
        need = _span_total(spans)
        if not need:
            return
        memory = self._host.memory()
        if memory is None or need > memory.headroom():
            tally.out_of_room = True
            return
        read = 0
        for span in spans:
            if self._stop.is_set():
                break
            read += self._host.read(path, span, self._stop)
        with tally.lock:
            tally.files += 1
            tally.read += read


def _streams(environ: Mapping[str, str]) -> int:
    raw = (environ.get(WEIGHT_PREFETCH_STREAMS_ENV) or "").strip()
    if not raw:
        return _DEFAULT_STREAMS
    if raw.isdigit() and 1 <= int(raw) <= _MAX_STREAMS:
        return int(raw)
    log.warning(
        "%s=%r is not a count from 1 to %d; using %d", WEIGHT_PREFETCH_STREAMS_ENV, raw, _MAX_STREAMS, _DEFAULT_STREAMS
    )
    return _DEFAULT_STREAMS


def start_weight_prefetch(
    model_path: str, environ: Mapping[str, str], host: Host | None = None
) -> WeightPrefetch | None:
    """Start reading the checkpoint at ``model_path`` when ``environ`` turns the prefetch on and it is a directory;
    ``None`` when nothing was started. Never raises: a prefetch that cannot start is logged and skipped."""
    try:
        if not is_truthy(environ.get(WEIGHT_PREFETCH_ENV)):
            return None
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            log.info("weight prefetch: the model %s is not a local directory; nothing to read", model_path)
            return None
        prefetch = WeightPrefetch(model_dir, host, PrefetchTuning(streams=_streams(environ)))
        prefetch.start()
    except Exception as exc:  # noqa: BLE001 - isolation boundary: a prefetch that cannot start never fails a round
        log.warning("weight prefetch of %s did not start: %s: %s", model_path, type(exc).__name__, exc)
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


def _uncached_runs(flags: bytes, offset: int, end: int) -> Iterable[tuple[int, int]]:
    """The byte ranges of a window starting at ``offset`` whose pages ``flags`` (one byte each) marks uncached."""
    for match in _UNCACHED_RUN.finditer(flags.translate(_RESIDENT_BIT)):
        yield offset + match.start() * mmap.PAGESIZE, min(offset + match.end() * mmap.PAGESIZE, end)


def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


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

    def __init__(self, read_text: Callable[[Path], str] | None = None) -> None:
        self._read_text = read_text or (lambda path: path.read_text(encoding="utf-8", errors="replace"))
        self._libc: _Libc | None = None

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

    def uncached(self, path: Path, size: int) -> list[tuple[int, int]]:
        if size <= 0:
            return []
        if self._libc is None:
            self._libc = _Libc()
        fd = os.open(path, os.O_RDONLY)
        try:
            spans: list[tuple[int, int]] = []
            for offset in range(0, size, _RESIDENCY_WINDOW):
                length = min(_RESIDENCY_WINDOW, size - offset)
                spans.extend(_uncached_runs(self._libc.residency(fd, offset, length), offset, offset + length))
            return _merge(spans)
        finally:
            os.close(fd)

    def read(self, path: Path, span: tuple[int, int], stop: threading.Event) -> int:
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
