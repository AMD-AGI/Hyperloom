# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a hosted attempt keeps of its runner's files when no result lands: the logs, and where the test last was.

A hosted test that fails, is stopped or runs out of time leaves no signed result, and its sandbox, with everything the
runner wrote, ends with it. :func:`save_runner_files` copies the runner's diagnostic files from ``/out/files`` into the
attempt's output directory first: the logs, stamps and small JSON documents of every round, however deep the round's
directory sits (``variant_00_test/suite_run_0/server.log``), each at its relative path below a ``hosted-`` prefix on its
first component, so a top-level ``server.log`` stays ``hosted-server.log``. Profiler traces are left behind, a link is
never followed, and the bytes kept are bounded per file and in all; a file cut to its tail starts with a line that says
so, and ``hosted-saved-files.json`` lists what was kept, cut and left behind.

:class:`SeenProgress` holds the last ``progress.json`` the application read of each running test, and
:func:`write_progress` writes it as ``hosted-progress.json``: a test the host ended says which phase it was in, whether
its server had reported ready and how old its newest log line was.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.sandbox.socket import DirEntry, ListsEntries, Sandbox, SandboxError, SandboxGone
from inference_suite.contracts.layout import BOOT_ACTIVE_STAMP, PROFILE_DIR, PROFILE_TRACE_DIRS, READY_STAMP, SERVER_PID

__all__ = [
    "INDEX_FILE",
    "PREFIX",
    "PROGRESS_FILE",
    "SaveLimits",
    "SeenProgress",
    "save_runner_files",
    "write_progress",
]

log = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class SaveLimits:
    """How much of a runner's files an attempt keeps.

    Attributes:
        file_bytes: The most kept of one file; a longer one keeps its tail, which is where a test that was ended was.
        total_bytes: The most kept in all; a file that would pass it is left behind and listed.
        fetch_bytes: A file listed larger than this is not fetched at all: cutting it would cost the whole transfer.
        depth: How many directories deep below the files root the walk goes.
        entries: How many directory entries the walk reads in all.
    """

    file_bytes: int = 64 * _MIB
    total_bytes: int = 256 * _MIB
    fetch_bytes: int = _KIB * _MIB
    depth: int = 8
    entries: int = 5000


@dataclass(frozen=True)
class _Found:
    relative: str
    size: int | None  # None where the sandbox cannot say before the copy


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


def save_runner_files(handle: Sandbox, root: str, output: Path, limits: SaveLimits | None = None) -> None:
    """Copy the runner's diagnostic files below ``root`` into ``output`` within ``limits`` and list them.

    A file that went missing or could not be read is listed and the rest are still copied; a sandbox that has ended
    stops the copy, and what was kept so far is listed.

    Raises:
        FileNotFoundError: When ``root`` does not exist (the runner failed before it wrote anything).
        SandboxError: When the root cannot be listed.
        OSError: When the output directory cannot be written.
    """
    saver = _Saver(handle, root, output, limits or SaveLimits())
    found = _Walk(handle, root, saver).files() if isinstance(handle, ListsEntries) else _top_level(handle, root)
    output.mkdir(parents=True, exist_ok=True)
    try:
        for item in sorted(found, key=lambda each: (not each.relative.endswith(".log"), each.relative)):
            saver.save(item)
    finally:
        saver.index["saved_bytes"] = saver.total
        (output / INDEX_FILE).write_text(json.dumps(saver.index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _Saver:
    """Copies the files found one at a time and lists each in the index: kept, cut or left behind and why."""

    def __init__(self, handle: Sandbox, root: str, output: Path, limits: SaveLimits) -> None:
        self.handle, self.root, self.output, self.limits = handle, root, output, limits
        self.index: dict[str, Any] = {
            "root": root,
            "limits": limits.__dict__,
            "saved": [],
            "truncated": [],
            "skipped": [],
        }
        self.total = 0

    def skip(self, path: str, reason: str, size: int | None) -> None:
        """A file left behind, with the size its listing gave."""
        self.index["skipped"].append({"path": path, "size": size, "reason": reason})

    def pass_over(self, path: str, reason: str) -> None:
        """A directory or link the walk did not enter."""
        self.index["skipped"].append({"path": path, "reason": reason})

    def save(self, item: _Found) -> None:
        limits = self.limits
        if item.size is not None and item.size > limits.fetch_bytes:
            self.skip(item.relative, "larger than fetch_bytes", item.size)
            return
        expected = min(item.size if item.size is not None else limits.file_bytes, limits.file_bytes)
        if self.total + expected > limits.total_bytes:
            self.skip(item.relative, "total_bytes reached", item.size)
            return
        dest = self._dest(item.relative)
        if dest is None:
            self.skip(item.relative, "its saved name is reserved", item.size)
            return
        copied = self._copy(item, dest)
        if copied is None:
            return
        if copied > limits.file_bytes:
            _keep_tail(dest, copied, limits.file_bytes)
            self.index["truncated"].append({"path": item.relative, "size": copied, "kept_bytes": limits.file_bytes})
            self.total += limits.file_bytes
            return
        self.index["saved"].append({"path": item.relative, "size": copied})
        self.total += copied

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


def _top_level(handle: Sandbox, root: str) -> list[_Found]:
    """For a sandbox that cannot tell a file from a link or a directory: the kept names at the top only, as before."""
    return [_Found(name, None) for name in handle.list_dir(root) if _plain(name) and _kept(name)]


class _Walk:
    """Every kept regular file below ``root``, breadth first; a link is listed as skipped and never followed."""

    def __init__(self, handle: ListsEntries, root: str, saver: _Saver) -> None:
        self.handle, self.root, self.saver, self.limits = handle, root, saver, saver.limits
        self.found: list[_Found] = []
        self.pending: list[tuple[str, int]] = [("", 0)]
        self.read = 0

    def files(self) -> list[_Found]:
        while self.pending:
            relative, depth = self.pending.pop(0)
            for entry in self._list(relative):
                self.read += 1
                if self.read > self.limits.entries:
                    self.saver.index["listing_truncated"] = f"stopped after {self.limits.entries} entries"
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
        if entry.kind == "link":
            self.saver.pass_over(path, "a link, not followed")
        elif entry.kind == "dir" and entry.name not in _SKIPPED_DIRS:
            if depth < self.limits.depth:
                self.pending.append((path, depth + 1))
            else:
                self.saver.pass_over(path, "deeper than the depth limit")
        elif entry.kind == "file" and _kept(entry.name):
            self.found.append(_Found(path, entry.size))
