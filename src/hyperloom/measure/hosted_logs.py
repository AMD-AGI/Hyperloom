# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted attempt keeps of its runner's files when no result lands: the logs, and where the test last was.

A hosted test that fails, is stopped or runs out of time leaves no signed result, and its sandbox, with everything the
runner wrote, ends with it. :func:`save_runner_files` copies the runner's diagnostic files from ``/out/files`` into the
attempt's output directory first: the logs, stamps and small JSON documents of every round, however deep the round's
directory sits (``variant_00_test/suite_run_0/server.log``), each at its relative path below a ``hosted-`` prefix on its
first component, so a top-level ``server.log`` stays ``hosted-server.log``. Profiler traces are left behind, a link is
never followed, and the bytes kept are bounded per file and in all, every round's ``server.log`` first; a file cut to
its tail starts with a line that says so. ``hosted-saved-files.json`` lists every file kept or cut and every entry the
walk read and left behind (a trace directory, a link, a file of another kind), and says where the walk or the copy
stopped early: at the entry bound, or at the save's wall-time deadline.

:class:`SeenProgress` holds the last ``progress.json`` the application read of each running test, and
:func:`write_progress` writes it as ``hosted-progress.json``: a test the host ended says which phase it was in, whether
its server had reported ready and how old its newest log line was.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.sandbox.socket import DirEntry, Sandbox, SandboxError, SandboxGone
from inference_suite.contracts.layout import BOOT_ACTIVE_STAMP, PROFILE_DIR, PROFILE_TRACE_DIRS, READY_STAMP, SERVER_PID

__all__ = [
    "DEADLINE_SEC",
    "DEPTH",
    "ENTRIES",
    "FETCH_BYTES",
    "FILE_BYTES",
    "INDEX_FILE",
    "PREFIX",
    "PROGRESS_FILE",
    "TOTAL_BYTES",
    "SeenProgress",
    "save_runner_files",
    "write_progress",
]


PREFIX = "hosted-"
INDEX_FILE = f"{PREFIX}saved-files.json"
PROGRESS_FILE = f"{PREFIX}progress.json"
#: What a runner writes that tells why a test went the way it did: logs (``server.log``, the client's), the phase marks,
#: ``launch.json``, progress and abort documents. A trace is a ``.json`` too, so the trace directories are left out.
_KEPT_SUFFIXES = (".log", ".json", ".jsonl", ".txt", ".err", ".out")
_KEPT_NAMES = frozenset({READY_STAMP, BOOT_ACTIVE_STAMP, SERVER_PID})
_SKIPPED_DIRS = frozenset({PROFILE_DIR, *PROFILE_TRACE_DIRS})
_KIB = 1024
_MIB = _KIB * _KIB


#: The most kept of one file; a longer one keeps its tail, which is where a test that was ended was.
FILE_BYTES = 64 * _MIB
#: The most of the files' bytes kept in all (a cut file's note line aside); a file that would pass it is left behind and
#: listed, and one that grew past what was left of it since it was listed is cut to that.
TOTAL_BYTES = 256 * _MIB
#: A file listed larger than this is not fetched at all: cutting it would cost the whole transfer.
FETCH_BYTES = _KIB * _MIB
#: How many directories deep below the files root the walk goes.
DEPTH = 8
#: How many directory entries the walk reads in all.
ENTRIES = 5000
#: How long one save may take, from its start: past it, the walk and the copy stop and the index lists what was kept.
#: It runs after the attempt's own deadline and before its sandbox and cards are released.
DEADLINE_SEC = 120.0
_SERVER_LOG = "server.log"
_TOTAL_REACHED = "total_bytes reached"


@dataclass(frozen=True)
class _Found:
    relative: str
    size: int


class SeenProgress:
    """The last ``progress.json`` the application read of each running test, with the wall-clock second it read it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: dict[str, tuple[float, dict[str, Any]]] = {}

    def note(self, request_id: str, progress: Mapping[str, Any] | None, read_unix: float) -> None:
        if progress is not None:
            with self._lock:
                self._seen[request_id] = (read_unix, dict(progress))

    def get(self, request_id: str) -> tuple[float, dict[str, Any]] | None:
        with self._lock:
            return self._seen.get(request_id)

    def forget(self, request_id: str) -> None:
        with self._lock:
            self._seen.pop(request_id, None)


def write_progress(output: Path, seen: tuple[float, Mapping[str, Any]] | None, now_unix: float) -> None:
    """Write ``hosted-progress.json``: the last progress read, when it was read and how long before this write.

    Raises:
        OSError: When the file cannot be written.
    """
    if seen is None:
        document: dict[str, Any] = {"progress": None, "note": "the application read no progress from the runner"}
    else:
        read_unix, progress = seen
        document = {"read_unix": read_unix, "read_sec_ago": round(max(0.0, now_unix - read_unix), 3)}
        document["progress"] = dict(progress)
    output.mkdir(parents=True, exist_ok=True)
    (output / PROGRESS_FILE).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_runner_files(handle: Sandbox, root: str, output: Path) -> None:
    """Copy the runner's diagnostic files below ``root`` into ``output`` within the module's bounds and list them.

    Every round's ``server.log`` is copied first, then the rest in path order, so a failing round's server log is not
    pushed out of the total by other rounds' client logs. A file that went missing or could not be read is listed and
    the rest are still copied; a sandbox that ends during the copy stops it, and what was kept so far is listed.

    Raises:
        FileNotFoundError: When ``root`` does not exist (the runner failed before it wrote anything).
        SandboxError: When the root cannot be listed.
        OSError: When the output directory cannot be written.
    """
    saver = _Saver(handle, root, output)
    found = _Walk(handle, root, saver).files()
    output.mkdir(parents=True, exist_ok=True)
    try:
        for done, item in enumerate(sorted(found, key=_copy_order)):
            if saver.past_deadline():
                saver.stop(f"{len(found) - done} files found were not copied")
                break
            saver.save(item)
    finally:
        saver.index["saved_bytes"] = saver.total
        (output / INDEX_FILE).write_text(json.dumps(saver.index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_order(item: _Found) -> tuple[bool, str]:
    """Each round's ``server.log`` first, then the rest in path order."""
    return item.relative.rpartition("/")[2] != _SERVER_LOG, item.relative


class _Saver:
    """Copies the files found one at a time and lists each in the index: kept, cut or left behind and why."""

    def __init__(self, handle: Sandbox, root: str, output: Path) -> None:
        self.handle, self.root, self.output = handle, root, output
        self.deadline = time.monotonic() + DEADLINE_SEC
        limits = {"file_bytes": FILE_BYTES, "total_bytes": TOTAL_BYTES, "fetch_bytes": FETCH_BYTES}
        limits.update({"depth": DEPTH, "entries": ENTRIES, "deadline_sec": DEADLINE_SEC})
        self.index: dict[str, Any] = {"root": root, "limits": limits, "saved": [], "truncated": [], "skipped": []}
        self.total = 0

    def past_deadline(self) -> bool:
        return time.monotonic() >= self.deadline

    def stop(self, left_behind: str) -> None:
        """The save ran past its deadline and stopped; ``left_behind`` says what it did not reach."""
        self.index["stopped"] = f"stopped at the {DEADLINE_SEC:g} s deadline; {left_behind}"

    def skip(self, path: str, reason: str, size: int) -> None:
        """A file left behind, with the size its listing gave."""
        self.index["skipped"].append({"path": path, "size": size, "reason": reason})

    def pass_over(self, path: str, reason: str) -> None:
        """A directory, link or special file the walk did not enter or copy."""
        self.index["skipped"].append({"path": path, "reason": reason})

    def save(self, item: _Found) -> None:
        if item.size > FETCH_BYTES:
            self.skip(item.relative, "larger than fetch_bytes", item.size)
            return
        left = TOTAL_BYTES - self.total
        if min(item.size, FILE_BYTES) > left:
            self.skip(item.relative, _TOTAL_REACHED, item.size)
            return
        dest = self._dest(item.relative)
        if dest is None:
            self.skip(item.relative, "its saved name is reserved", item.size)
            return
        copied = self._copy(item, dest)
        if copied is None:
            return
        kept = min(copied, FILE_BYTES, left)  # a file still growing since its listing is held to what is left
        if kept == 0 < copied:
            dest.unlink(missing_ok=True)
            self.skip(item.relative, _TOTAL_REACHED, copied)
            return
        if kept < copied:
            _keep_tail(dest, copied, kept)
            self.index["truncated"].append({"path": item.relative, "size": copied, "kept_bytes": kept})
        else:
            self.index["saved"].append({"path": item.relative, "size": copied})
        self.total += kept

    def _dest(self, relative: str) -> Path | None:
        """Where a file is kept: its relative path, with ``hosted-`` before its first component."""
        first, _, rest = relative.partition("/")
        if not rest and f"{PREFIX}{first}" in (INDEX_FILE, PROGRESS_FILE):
            return None
        dest = self.output / f"{PREFIX}{first}" / rest if rest else self.output / f"{PREFIX}{first}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

    def _copy(self, item: _Found, dest: Path) -> int | None:
        try:
            return self.handle.copy_out(f"{self.root}/{item.relative}", dest)
        except SandboxGone:
            dest.unlink(missing_ok=True)
            raise
        except (SandboxError, OSError) as exc:  # FileNotFoundError too: removed since the listing
            dest.unlink(missing_ok=True)
            self.skip(item.relative, f"not read: {exc}", item.size)
            return None


def _keep_tail(dest: Path, size: int, kept: int) -> None:
    """Cut ``dest`` to its last ``kept`` bytes, after a line that says it was cut."""
    with dest.open("rb") as source:
        source.seek(size - kept)
        tail = source.read(kept)
    note = f"[hyperloom: truncated, kept the last {kept} of {size} bytes]\n".encode()
    partial = dest.with_name(dest.name + ".partial")
    partial.write_bytes(note + tail)
    os.replace(partial, dest)


def _kept(name: str) -> bool:
    return name in _KEPT_NAMES or name.endswith(_KEPT_SUFFIXES)


def _plain(name: str) -> bool:
    """A name that is one path component, which every listing should give and a hostile one might not."""
    return bool(name) and name not in (".", "..") and "/" not in name and "\x00" not in name


class _Walk:
    """Every kept regular file below ``root``, breadth first; a link is listed as skipped and never followed."""

    def __init__(self, handle: Sandbox, root: str, saver: _Saver) -> None:
        self.handle, self.root, self.saver = handle, root, saver
        self.found: list[_Found] = []
        self.pending: list[tuple[str, int]] = [("", 0)]
        self.read = 0

    def files(self) -> list[_Found]:
        while self.pending:
            relative, depth = self.pending.pop(0)
            if relative and self.saver.past_deadline():
                self.saver.stop(f"the directories from {relative} on were not listed")
                return self.found
            for entry in self._list(relative):
                self.read += 1
                if self.read > ENTRIES:
                    self.saver.index["listing_truncated"] = f"stopped after {ENTRIES} entries"
                    return self.found
                if _plain(entry.name):
                    self._take(entry, f"{relative}/{entry.name}" if relative else entry.name, depth)
        return self.found

    def _list(self, relative: str) -> list[DirEntry]:
        try:
            return self.handle.list_entries(f"{self.root}/{relative}" if relative else self.root)
        except (FileNotFoundError, SandboxError) as exc:
            if not relative or isinstance(exc, SandboxGone):
                raise
            self.saver.pass_over(relative, f"not listed: {exc}")
            return []

    def _take(self, entry: DirEntry, path: str, depth: int) -> None:
        if entry.kind == "dir":
            self._enter(entry, path, depth)
        elif entry.kind == "link":
            self.saver.pass_over(path, "a link, not followed")
        elif entry.kind != "file":
            self.saver.pass_over(path, "not a regular file")
        elif _kept(entry.name):
            self.found.append(_Found(path, entry.size))
        else:
            self.saver.skip(path, "not a diagnostic file", entry.size)

    def _enter(self, entry: DirEntry, path: str, depth: int) -> None:
        if entry.name in _SKIPPED_DIRS:
            self.saver.pass_over(path, "a profiler trace directory")
        elif depth < DEPTH:
            self.pending.append((path, depth + 1))
        else:
            self.saver.pass_over(path, "deeper than the depth limit")
