# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted implementation of the ``events`` socket: CloudEvents 1.0 lines in rolling segment files.

Each event is one JSON object on one line of ``<events dir>/<10-digit segment>.jsonl``.  A reader that tails the
directory by byte offset relies on three guarantees kept here:

* a line is written whole, in one ``write`` call, and always ends with ``\\n``, so a line without its newline is
  an unfinished write and never an event;
* ``id`` is a decimal string that strictly increases within the app run, across restarts: on start the last
  complete line of the last segment gives the id to continue after, and a half-written tail left by a crash is
  cut off first (the segment is replaced by a copy of its complete lines, never cut in place); what is no regular file
  at a segment's name is passed over, and so is a newer segment this account cannot read (one only its owner may
  read), after which the id continues past the most it could hold (:data:`_MAX_EVENTS_PER_SEGMENT`);
* each start of the writer begins a new segment, made new (``O_EXCL``, never through a link) with mode ``0664``:
  the next leg of a hosted run may be another account of the directory's group, which reads the earlier segments but
  never opens one of another account's for writing;
* there is exactly one writer per events directory.  The directory and the run id come from the host inputs
  (:class:`hyperloom.hosted.host_info.HostInfo`) the plug host was given, never from the process environment,
  which :func:`hyperloom.hosted.host_info.take_host_info` has already cleared: a child process therefore has
  nothing to write to.

Emitting is best effort: a failure is logged and the run continues.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from hyperloom_common.shared_storage import (
    create_new_shared_file,
    make_shared_dirs,
    open_directory,
    read_directory,
    write_shared_bytes,
)
from hyperloom.plugs.socket import PlugContext

log = logging.getLogger(__name__)

SEGMENT_SUFFIX = ".jsonl"
SEGMENT_DIGITS = 10
_KIB = 1024
SEGMENT_MAX_BYTES = 4 * _KIB * _KIB
EVENT_MAX_BYTES = 16 * _KIB
FSYNC_INTERVAL_SEC = 1.0
SPECVERSION = "1.0"
DATA_CONTENT_TYPE = "application/json"
_OVERSIZE_KEYS = 20
#: How many taken names past the last segment a new one is tried at before the writer gives up.
_NEW_SEGMENT_TRIES = 64
#: Fewer bytes than any event line takes (the fixed fields alone are longer).
_MIN_EVENT_BYTES = 64
#: The most events a segment can hold: it is cut at :data:`SEGMENT_MAX_BYTES`, and no line is longer than
#: :data:`EVENT_MAX_BYTES`, which is less.
_MAX_EVENTS_PER_SEGMENT = SEGMENT_MAX_BYTES // _MIN_EVENT_BYTES
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def segment_name(number: int) -> str:
    """File name of segment ``number``."""
    return f"{number:0{SEGMENT_DIGITS}d}{SEGMENT_SUFFIX}"


def segment_numbers(events_dir: Path) -> list[int]:
    """The segment numbers present in ``events_dir``, ascending."""
    return _segment_numbers(entry.name for entry in events_dir.iterdir())


def _segment_numbers(names: Iterable[str]) -> list[int]:
    found = []
    for name in names:
        stem = name[: -len(SEGMENT_SUFFIX)]
        if name.endswith(SEGMENT_SUFFIX) and len(stem) == SEGMENT_DIGITS and stem.isdigit():
            found.append(int(stem))
    return sorted(found)


def _complete_lines(raw: bytes) -> tuple[list[bytes], int]:
    """The newline-terminated lines of ``raw`` and the length of the prefix they occupy."""
    end = raw.rfind(b"\n") + 1
    return raw[:end].splitlines(), end


def _last_id(lines: list[bytes]) -> int | None:
    for line in reversed(lines):
        try:
            value = json.loads(line)["id"]
            if isinstance(value, str) and value.isascii() and value.isdecimal():
                return int(value)
        except (ValueError, KeyError, TypeError):
            continue
    return None


def _cut_unfinished_tail(path: Path, raw: bytes, keep: int) -> None:
    """Drop the bytes after the last newline: a crashed write that no reader may take for an event.  The segment is
    replaced by a new file of its complete lines, never opened in place: it may be another account's."""
    if keep < len(raw):
        log.warning("events: dropping %d unfinished bytes at the end of %s", len(raw) - keep, path.name)
        write_shared_bytes(path, raw[:keep])


class _Unreadable:
    """What :func:`_read_segment` returns for a segment file this account cannot read."""


_UNREADABLE = _Unreadable()


def _read_segment(directory: int, name: str) -> bytes | _Unreadable | None:
    """What the segment file ``name`` of the directory open as ``directory`` holds; ``None`` when it is not a regular
    file (a link or a directory a member of the group put there, a pipe), which is no segment, and
    :data:`_UNREADABLE` for a regular file this account cannot read (only its owner may: a segment of a writer from
    before segments were the group's to read). Such a name is never read, written or taken for a segment with events."""
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=directory)
    except OSError as exc:
        return _not_read(directory, name, exc)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            log.warning("events: passing over %s, which is not a regular file", name)
            return None
        chunks = []
        while chunk := os.read(descriptor, SEGMENT_MAX_BYTES):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        return _not_read(directory, name, exc)
    finally:
        os.close(descriptor)


def _not_read(directory: int, name: str, exc: OSError) -> _Unreadable | None:
    log.warning("events: passing over %s, which cannot be read: %s", name, exc.strerror or type(exc).__name__)
    try:
        regular = stat.S_ISREG(os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode)
    except OSError:
        return None
    return _UNREADABLE if regular else None


def _resume_point(events_dir: Path) -> tuple[int, int]:
    """``(last segment number, last id)`` to continue after; ``(0, 0)`` for an empty directory.  The newest segment
    that is a regular file has its unfinished tail cut first (:func:`_cut_unfinished_tail`).  For each newer segment
    this account cannot read (:func:`_read_segment`) the id continues after the most events it could hold, so no id it
    may hold is used again."""
    directory = open_directory(events_dir)
    try:
        listing = read_directory(directory)
        try:
            numbers = _segment_numbers(os.listdir(listing))
        finally:
            os.close(listing)
        passed_over = 0
        newest = True
        for number in reversed(numbers):
            raw = _read_segment(directory, segment_name(number))
            if raw is None:
                continue
            if isinstance(raw, _Unreadable):
                passed_over += 1
                continue
            lines, keep = _complete_lines(raw)
            if newest:
                _cut_unfinished_tail(events_dir / segment_name(number), raw, keep)
                newest = False
            last = _last_id(lines)
            if last is not None:
                return numbers[-1], last + passed_over * _MAX_EVENTS_PER_SEGMENT
        return (numbers[-1] if numbers else 0), passed_over * _MAX_EVENTS_PER_SEGMENT
    finally:
        os.close(directory)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class HostedProgressEvents:
    """Appends CloudEvents to the events directory; thread-safe, one instance per directory."""

    def __init__(
        self,
        events_dir: Path,
        run_id: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        segment_max_bytes: int = SEGMENT_MAX_BYTES,
    ) -> None:
        self._dir = Path(events_dir)
        self._source = f"hyperloom/{run_id}"
        self._clock = clock
        self._segment_max = segment_max_bytes
        self._lock = threading.Lock()
        self._fd: int | None = None
        self._size = 0
        self._dirty = False
        self._last_sync = clock()
        self._closed = False
        make_shared_dirs(self._dir)
        last_segment, self._last_id = _resume_point(self._dir)
        self._fd, self._segment = self._new_segment(last_segment)

    @property
    def last_id(self) -> int:
        """The id of the last event written, ``0`` before the first."""
        return self._last_id

    def _new_segment(self, after: int) -> tuple[int, int]:
        """Create the first free segment after ``after``, new; its descriptor and number.  Changes no state, so a
        failure leaves the writer as it was.

        Raises:
            FileExistsError: When every one of the next :data:`_NEW_SEGMENT_TRIES` names is taken.
        """
        for number in range(after + 1, after + 1 + _NEW_SEGMENT_TRIES):
            try:
                return create_new_shared_file(self._dir / segment_name(number)), number
            except FileExistsError:
                continue  # made meanwhile by something else: never written into
        raise FileExistsError(f"events: no free segment name after {segment_name(after)} in {self._dir}")

    def _sync(self) -> None:
        if self._fd is not None and self._dirty:
            os.fsync(self._fd)
        self._dirty = False
        self._last_sync = self._clock()

    def _roll_if_full(self, incoming: int) -> None:
        if self._size and self._size + incoming > self._segment_max:
            self._sync()
            fd, segment = self._new_segment(self._segment)  # the current segment stays in use until this succeeds
            os.close(self._fd)  # type: ignore[arg-type]
            self._fd, self._size, self._segment = fd, 0, segment

    def _line(self, event_id: int, event_type: str, data: Mapping[str, Any]) -> bytes:
        event = {
            "specversion": SPECVERSION,
            "id": str(event_id),
            "source": self._source,
            "type": event_type,
            "time": _utc_now(),
            "datacontenttype": DATA_CONTENT_TYPE,
            "data": dict(data),
        }
        line = (json.dumps(event, separators=(",", ":"), default=str) + "\n").encode("utf-8")
        if len(line) <= EVENT_MAX_BYTES:
            return line
        log.warning(
            "events: %s is %d bytes, over the %d limit; sent without its content",
            event_type,
            len(line),
            EVENT_MAX_BYTES,
        )
        event["data"] = {"oversized": True, "bytes": len(line), "keys": sorted(map(str, data))[:_OVERSIZE_KEYS]}
        return (json.dumps(event, separators=(",", ":"), default=str) + "\n").encode("utf-8")

    def _append(self, line: bytes) -> None:
        self._roll_if_full(len(line))
        before = self._size
        try:
            view = memoryview(line)
            while view:
                view = view[os.write(self._fd, view) :]  # type: ignore[arg-type]
        except OSError:
            try:
                os.ftruncate(self._fd, before)  # type: ignore[arg-type]
            except OSError:
                log.exception("events: could not remove a partial line from %s", segment_name(self._segment))
            raise
        self._size = before + len(line)
        self._dirty = True

    def emit(self, event_type: str, data: Mapping[str, Any]) -> None:
        """Append one event.  A write or encoding failure is logged and the run goes on."""
        try:
            with self._lock:
                if self._closed:
                    return
                line = self._line(self._last_id + 1, event_type, data)
                self._append(line)
                self._last_id += 1
                if self._clock() - self._last_sync >= FSYNC_INTERVAL_SEC:
                    self._sync()
        except (OSError, ValueError):
            log.exception("events: could not write %s", event_type)

    def flush(self) -> None:
        """Make everything emitted so far durable."""
        try:
            with self._lock:
                if not self._closed:
                    self._sync()
        except OSError:
            log.exception("events: could not sync the events directory")

    def close(self) -> None:
        """Sync and close; later emits are ignored."""
        with self._lock:
            if self._closed:
                return
            try:
                self._sync()
            except OSError:
                log.exception("events: could not sync on close")
            self._closed = True
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None


def create(context: PlugContext) -> HostedProgressEvents:
    """Factory the built-in registration points at.  Needs the host inputs; reads nothing from the environment."""
    info = context.host_info
    events_dir = getattr(info, "events_dir", None)
    run_id = getattr(info, "run_id", None)
    if not events_dir or not run_id:
        raise ValueError("the hosted events implementation needs host inputs with an events directory and a run id")
    return HostedProgressEvents(Path(events_dir), str(run_id))
