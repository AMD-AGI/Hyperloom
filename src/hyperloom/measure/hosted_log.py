# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted backend's own journal of tests in flight: ``<session>/hosted/measurements.jsonl`` (design 4.4.4).

A line is appended, and flushed to disk, before the backend depends on what it says: ``submitted`` before anything is
written into the sandbox, ``sent`` once the sheet and the ``READY`` marker are in, ``finished`` once the outcome is
decided. After a restart the newest line of a request tells whether a sandbox may still be measuring it (``sent``:
re-attach and wait; ``submitted``: re-attach and send the sheet again, it may never have arrived) or the attempt is
over (``finished``: the next call is the next attempt). The sheet bytes of every attempt are kept beside the journal
so a re-attached attempt is verified against exactly the sheet that was sent.

A failed attempt's ``finished`` line also keeps what was said about the failure, since a runner sandbox's ``/out`` goes
with the sandbox: the runner's own ``status.json`` (only its ``state``, ``error_class`` and ``message``), or that it was
missing or could not be read, and the platform's message for a create that made no usable sandbox, each bounded to
:data:`FAILURE_DETAIL_CHARS`. No environment value is read.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.sandbox.socket import Sandbox, SandboxError

__all__ = [
    "FAILURE_DETAIL_CHARS",
    "JournalCorrupt",
    "MeasurementJournal",
    "Record",
    "STATE_FINISHED",
    "STATE_SENT",
    "STATE_SUBMITTED",
    "STATUS_JSON",
    "bounded",
    "runner_status",
    "status_evidence",
    "where",
]

STATE_SUBMITTED = "submitted"
STATE_SENT = "sent"
STATE_FINISHED = "finished"
_STATES = (STATE_SUBMITTED, STATE_SENT, STATE_FINISHED)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class JournalCorrupt(RuntimeError):
    """A line of the journal other than the last cannot be read: tests may have been lost, so nothing is guessed."""


@dataclass(frozen=True)
class Record:
    request_id: str
    attempt: int
    state: str
    sandbox_id: str | None
    output_dir: str
    task_id: str
    extra: Mapping[str, Any]

    def to_line(self) -> str:
        document = {
            "request_id": self.request_id,
            "attempt": self.attempt,
            "state": self.state,
            "sandbox_id": self.sandbox_id,
            "output_dir": self.output_dir,
            "task_id": self.task_id,
            **self.extra,
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_document(cls, document: Any) -> Record:
        if not isinstance(document, dict) or document.get("state") not in _STATES:
            raise ValueError("not a journal record")
        known = {"request_id", "attempt", "state", "sandbox_id", "output_dir", "task_id"}
        return cls(
            request_id=str(document["request_id"]),
            attempt=int(document["attempt"]),
            state=document["state"],
            sandbox_id=document.get("sandbox_id"),
            output_dir=str(document.get("output_dir", "")),
            task_id=str(document.get("task_id", "")),
            extra={k: v for k, v in document.items() if k not in known},
        )


class MeasurementJournal:
    """An append-only journal plus the sheets of every attempt, under one directory."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._path = self._dir / "measurements.jsonl"
        self._specs = self._dir / "specs"
        self._lock = threading.Lock()
        self._specs.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: Record) -> None:
        """Add a line and force it to disk before returning."""
        line = (record.to_line() + "\n").encode("utf-8")
        with self._lock:
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)

    def records(self) -> list[Record]:
        """Every readable line in order; an unreadable *last* line (a crash mid-write) is dropped."""
        try:
            raw = self._path.read_bytes()
        except FileNotFoundError:
            return []
        lines = raw.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        records: list[Record] = []
        for index, line in enumerate(lines):
            try:
                records.append(Record.from_document(json.loads(line)))
            except (ValueError, KeyError, TypeError):
                if index == len(lines) - 1 and not raw.endswith(b"\n"):
                    continue
                raise JournalCorrupt(f"{self._path}: line {index + 1} cannot be read") from None
        return records

    def latest(self, request_id: str) -> Record | None:
        """The newest line of a request, or ``None`` when it was never submitted."""
        found = [record for record in self.records() if record.request_id == request_id]
        return found[-1] if found else None

    def next_attempt(self, request_id: str) -> int:
        """The attempt number a fresh sandbox for this request gets: one past everything journalled."""
        attempts = [record.attempt for record in self.records() if record.request_id == request_id]
        return max(attempts, default=0) + 1

    def save_spec(self, request_id: str, attempt: int, raw: bytes) -> Path:
        path = self._spec_path(request_id, attempt)
        path.write_bytes(raw)
        return path

    def load_spec(self, request_id: str, attempt: int) -> bytes:
        return self._spec_path(request_id, attempt).read_bytes()

    def _spec_path(self, request_id: str, attempt: int) -> Path:
        if not _SAFE_ID.fullmatch(request_id):
            raise ValueError(f"request id {request_id!r} is not usable as a file name")
        return self._specs / f"{request_id}.{int(attempt)}.json"


# -- what a failed attempt's line keeps ------------------------------------------------------------------------------

#: How much of a message is kept: the start step's verify message whole, a runaway one cut.
FAILURE_DETAIL_CHARS = 2000
#: The key of the journal field that says what ``status.json`` held, or why nothing could be read.
STATUS_JSON = "status_json"
_STATUS_FIELDS = ("state", "error_class", "message")
_READ, _MISSING, _UNREADABLE = "read", "missing", "unreadable"


def bounded(text: object) -> str:
    """``text`` as a string of at most :data:`FAILURE_DETAIL_CHARS` characters, its beginning kept."""
    return "" if text is None else str(text)[:FAILURE_DETAIL_CHARS]


def runner_status(status: Mapping[str, Any]) -> dict[str, str]:
    """The runner's ``state``, ``error_class`` and ``message`` as it wrote them, bounded; nothing else of the file."""
    return {STATUS_JSON: _READ, **{name: bounded(status.get(name)) for name in _STATUS_FIELDS}}


def status_evidence(handle: Sandbox | None, path: str) -> dict[str, str]:
    """What ``path`` in the sandbox says, read once more, or why it could not be read (a sandbox already gone, say)."""
    if handle is None:
        return {STATUS_JSON: f"{_UNREADABLE}: the sandbox has no handle"}
    try:
        raw = handle.read_file(path)
    except FileNotFoundError:
        return {STATUS_JSON: _MISSING}
    except (SandboxError, OSError) as exc:
        return {STATUS_JSON: f"{_UNREADABLE}: {type(exc).__name__}"}
    try:
        document = json.loads(raw.decode("utf-8"))
    except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError both are
        return {STATUS_JSON: f"{_UNREADABLE}: {type(exc).__name__}"}
    if not isinstance(document, dict):
        return {STATUS_JSON: f"{_UNREADABLE}: not a JSON object"}
    return runner_status(document)


def where(sandbox_id: str | None, node_name: str | None) -> str:
    """``sandbox <id> on node <name>``: where a failure happened, for the reason a reader sees first."""
    return f"sandbox {sandbox_id or 'unknown'} on node {node_name or 'unknown'}"
