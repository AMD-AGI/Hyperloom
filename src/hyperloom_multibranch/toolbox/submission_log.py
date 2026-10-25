# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's submit log: what survives a restart (design 3.7, 4.17).

An append-only JSON-lines file, flushed and fsynced per entry. A request is logged before the notebook hears of it,
its outcome is logged before the notebook is written, and the entry is settled once the notebook holds it, so after a
crash every accepted request is in exactly one of three states: waiting to run, measured and waiting to be written, or
settled. Ids are UUID v5 of the campaign id and ``branch/sequence``; the sequence is per branch and is never reused
(a submit that failed after it was logged still used its number), so a restart hands out the numbers it would have.
A torn last line, the trace of a crash mid-write, is ignored.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ACCEPTED = "accepted"
OUTCOME = "outcome"
SETTLED = "settled"
ABORTED = "aborted"
RECIPE_DIGEST = "recipe_digest"


@dataclass
class LogEntry:
    """Everything the log knows of one request: its document, priority, outcome (once known) and settlement."""

    request_id: str
    branch: str
    sequence: int
    request: Mapping[str, object]
    priority: int
    queued_at: float
    outcome: Mapping[str, object] | None = None
    settled: bool = False
    aborted: bool = False
    #: The digest of the recipe the request measures; empty in an entry logged before it was recorded.
    recipe_digest: str = ""

    @property
    def waiting(self) -> bool:
        return self.outcome is None and not self.aborted

    @property
    def unwritten(self) -> bool:
        return self.outcome is not None and not self.settled and not self.aborted


@dataclass
class SubmissionLog:
    path: Path
    campaign_id: str
    _entries: dict[str, LogEntry] = field(default_factory=dict)
    _sequences: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._replay()

    def request_id(self, branch: str, sequence: int) -> str:
        try:
            namespace = uuid.UUID(self.campaign_id)
        except ValueError:
            namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"hyperloom-campaign:{self.campaign_id}")
        return str(uuid.uuid5(namespace, f"{branch}/{sequence}"))

    def next_sequence(self, branch: str) -> int:
        """Reserve the next number of ``branch``; a number is used the moment it is handed out."""

        self._sequences[branch] = self._sequences.get(branch, 0) + 1
        return self._sequences[branch]

    def get(self, request_id: str) -> LogEntry | None:
        return self._entries.get(request_id)

    def entries(self) -> list[LogEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.queued_at)

    def accept(self, entry: LogEntry) -> None:
        self._entries[entry.request_id] = entry
        self._append(
            {
                "type": ACCEPTED,
                "request_id": entry.request_id,
                "branch": entry.branch,
                "sequence": entry.sequence,
                "request": entry.request,
                "priority": entry.priority,
                "queued_at": entry.queued_at,
                **({RECIPE_DIGEST: entry.recipe_digest} if entry.recipe_digest else {}),
            }
        )

    def record_outcome(self, request_id: str, outcome: Mapping[str, object]) -> None:
        self._entries[request_id].outcome = outcome
        self._append({"type": OUTCOME, "request_id": request_id, "outcome": outcome})

    def settle(self, request_id: str) -> None:
        self._entries[request_id].settled = True
        self._append({"type": SETTLED, "request_id": request_id})

    def abort(self, request_id: str) -> None:
        """The submit failed after it was logged: the id is consumed and nothing will run."""

        self._entries[request_id].aborted = True
        self._append({"type": ABORTED, "request_id": request_id})

    def _append(self, record: Mapping[str, object]) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _replay(self) -> None:
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn write: only the last line can be one, and it was never acknowledged
            self._apply(record, number)

    def _apply(self, record: Mapping[str, object], number: int) -> None:
        kind, request_id = record["type"], str(record["request_id"])
        if kind == ACCEPTED:
            branch = str(record["branch"])
            sequence = int(record["sequence"])  # type: ignore[call-overload]
            self._sequences[branch] = max(self._sequences.get(branch, 0), sequence)
            self._entries[request_id] = LogEntry(
                request_id,
                branch,
                sequence,
                record["request"],  # type: ignore[arg-type]
                int(record["priority"]),  # type: ignore[call-overload]
                float(record["queued_at"]),  # type: ignore[arg-type]
                recipe_digest=str(record.get(RECIPE_DIGEST) or ""),
            )
            return
        entry = self._entries.get(request_id)
        if entry is None:
            raise ValueError(f"{self.path}:{number}: {kind} entry for the unlogged request {request_id}")
        if kind == OUTCOME:
            entry.outcome = record["outcome"]  # type: ignore[assignment]
        elif kind == SETTLED:
            entry.settled = True
        elif kind == ABORTED:
            entry.aborted = True
