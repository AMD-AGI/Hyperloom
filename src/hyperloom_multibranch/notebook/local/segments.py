# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single-writer JSONL segments: append-only files every reader merges.

Each notebook handle is one writer and appends to a segment of its own, so no append needs a lock on any host, and a
shared folder's lack of atomic cross-client appends never interleaves two rows. A line still being written is not
valid JSON; readers skip it and see it whole on their next read.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from hyperloom.common.io import append_jsonl
from hyperloom.common.jsonio import read_jsonl
from hyperloom_multibranch.contracts.notebook.vocab import Role
from hyperloom_multibranch.notebook.codec import key_sorted

log = logging.getLogger(__name__)

ENTRY_VERSION = 1
SEGMENT_SUFFIX = ".jsonl"
SUMMARY_LIMIT = 160
_TOKEN = re.compile(r"[A-Za-z0-9_.:@/-]{1,200}")
_HOST_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")
_HOST_LIMIT = 40


class EntryKind(str, Enum):
    """What one notebook row records."""

    REQUEST = "request"
    RESULT = "result"
    WITHDRAWAL = "withdrawal"
    JUDGEMENT = "judgement"
    MERGE_ATTEMPT = "merge_attempt"
    LINEAGE = "lineage"
    BASELINE = "baseline"
    REGRESSION = "regression"
    QUARANTINE = "quarantine"
    BRANCH_LOG = "branch_log"
    LEARNING = "learning"
    TOOL_RESULT = "tool_result"
    CLAIM = "claim"
    SESSION_PROFILE = "session_profile"
    MANIFEST = "manifest"
    LLM_USAGE = "llm_usage"


def require_token(value: str, name: str) -> str:
    """``value`` when it is a safe identifier (letters, digits, ``_.:@/-``); raises ``ValueError`` otherwise."""

    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ValueError(f"{name} must match {_TOKEN.pattern}, got {value!r}")
    return value


def clip_summary(text: str) -> str:
    """One line of at most :data:`SUMMARY_LIMIT` characters."""

    summary = " ".join(text.split())
    return summary if len(summary) <= SUMMARY_LIMIT else summary[: SUMMARY_LIMIT - 3] + "..."


def make_writer_id(role: Role, name: str) -> str:
    """A writer id unique to one handle: ``<role>.<name>.<host>.<pid>.<random>``."""

    host = _HOST_UNSAFE.sub("-", socket.gethostname())[:_HOST_LIMIT] or "host"
    return require_token(f"{role.value}.{name}.{host}.{os.getpid()}.{secrets.token_hex(2)}", "writer id")


@dataclass(frozen=True)
class Entry:
    """One immutable notebook row; ``entry_id`` is ``<writer>:<seq>`` and ``at`` is Unix seconds."""

    entry_id: str
    seq: int
    at: float
    writer: str
    role: Role
    actor: str
    kind: EntryKind
    branch: str
    parent: str
    summary: str
    detail: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "v": ENTRY_VERSION,
            "id": self.entry_id,
            "seq": self.seq,
            "at": self.at,
            "writer": self.writer,
            "role": self.role.value,
            "actor": self.actor,
            "kind": self.kind.value,
            "branch": self.branch,
            "parent": self.parent,
            "summary": self.summary,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> Entry:
        """Rebuild an entry; raises ``ValueError``, ``KeyError`` or ``TypeError`` on a row this version cannot read."""

        if row.get("v") != ENTRY_VERSION:
            raise ValueError(f"unsupported entry version {row.get('v')!r}")
        detail = row["detail"]
        if not isinstance(detail, dict):
            raise TypeError("entry detail must be an object")
        return cls(
            entry_id=str(row["id"]),
            seq=int(row["seq"]),
            at=float(row["at"]),
            writer=str(row["writer"]),
            role=Role(row["role"]),
            actor=str(row["actor"]),
            kind=EntryKind(row["kind"]),
            branch=str(row.get("branch") or ""),
            parent=str(row.get("parent") or ""),
            summary=str(row["summary"]),
            detail=detail,
        )


def _read_entries(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    for row in read_jsonl(path, require_dict=True, skip_malformed=True, skip_non_dict=True):
        try:
            entries.append(Entry.from_dict(row))
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("notebook: skipping unreadable entry in %s: %s", path, exc)
    return entries


class SegmentWriter:
    """Appends one writer's entries to its own segment, numbering them from 1; the file appears with its first row."""

    def __init__(self, segments_dir: Path, writer: str) -> None:
        self.writer = require_token(writer, "writer")
        self.path = Path(segments_dir) / f"{writer}{SEGMENT_SUFFIX}"
        self._last_seq = max((entry.seq for entry in _read_entries(self.path)), default=0) if self.path.exists() else 0

    def append(self, build: Callable[[int], Entry]) -> Entry:
        """Append the entry ``build`` makes for the next sequence number, and return it."""

        entry = build(self._last_seq + 1)
        if entry.writer != self.writer or entry.seq != self._last_seq + 1:
            raise ValueError(f"segment {self.writer} expects seq {self._last_seq + 1}, got {entry.entry_id}")
        append_jsonl(self.path, key_sorted(entry.to_dict()), make_parents=True)
        self._last_seq = entry.seq
        return entry


class SegmentReader:
    """Merges every segment of a notebook, re-parsing only the segments that changed."""

    def __init__(self, segments_dir: Path) -> None:
        self.segments_dir = Path(segments_dir)
        self._cache: dict[Path, tuple[tuple[int, int], list[Entry]]] = {}

    def entries(self) -> list[Entry]:
        """Every readable entry, ordered by time, then writer and sequence."""

        merged: list[Entry] = []
        for path in sorted(self.segments_dir.glob(f"*{SEGMENT_SUFFIX}")):
            merged.extend(self._segment(path))
        merged.sort(key=lambda entry: (entry.at, entry.writer, entry.seq))
        return merged

    def _segment(self, path: Path) -> list[Entry]:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return []
        version = (stat.st_size, stat.st_mtime_ns)
        cached = self._cache.get(path)
        if cached is not None and cached[0] == version:
            return cached[1]
        entries = _read_entries(path)
        self._cache[path] = (version, entries)
        return entries


__all__ = [
    "ENTRY_VERSION",
    "SEGMENT_SUFFIX",
    "SUMMARY_LIMIT",
    "Entry",
    "EntryKind",
    "SegmentReader",
    "SegmentWriter",
    "clip_summary",
    "make_writer_id",
    "require_token",
]
