# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which tool job answers a shared key, kept on disk so that a restarted toolbox answers as the one before it did.

A key names work whose answer is the same whoever asks: the analysis of one profile test by one tool (board #153
c3447). :class:`~.base.ToolJobs` records here the job it started for a key and, once the job ended, how it ended and
the pointer it pushed. A running job is joined and a succeeded one shared; a failed or cancelled one answers nothing,
so the next submission starts a new job and the key names that one. A job that was running when the toolbox stopped
ended with it: the ledger still says running, and the job is in no restarted runner, so it answers nothing either.

The ledger is one JSON file below the runner's results directory, rewritten whole on every change. One that cannot be
read is reported and set aside: the toolbox then starts new jobs for keys it would have shared, which costs a repeated
analysis and loses no result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json
from hyperloom_multibranch.contracts.blackboard.records import ToolResultPointer

__all__ = ["SHARED_JOBS_FILE", "SharedEntry", "SharedJobLedger"]

log = logging.getLogger(__name__)

SHARED_JOBS_FILE = "shared-jobs.json"
_SCHEMA = 1
_SUCCEEDED = "succeeded"


@dataclass(frozen=True)
class SharedEntry:
    """The job a key names: its id, its tool, its status as the runner last recorded it, and, once it succeeded, the
    pointer it pushed (what a branch that asks after a restart is given)."""

    job_id: str
    tool: str
    status: str
    pointer: ToolResultPointer | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == _SUCCEEDED and self.pointer is not None


def _entry(document: Any) -> SharedEntry:
    pointer = document.get("pointer")
    return SharedEntry(
        str(document["job_id"]),
        str(document["tool"]),
        str(document["status"]),
        ToolResultPointer(**pointer) if isinstance(pointer, dict) else None,
    )


class SharedJobLedger:
    """The keys of shared work and the job each names, in ``path``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, SharedEntry] = self._read()

    def _read(self) -> dict[str, SharedEntry]:
        if not self._path.exists():
            return {}
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
            return {str(key): _entry(value) for key, value in document["keys"].items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            log.error("the shared tool-job ledger %s cannot be read, so no earlier job is shared: %s", self._path, exc)
            return {}

    def get(self, key: str) -> SharedEntry | None:
        return self._entries.get(key)

    def record(self, key: str, entry: SharedEntry) -> None:
        """Name ``entry`` the job of ``key`` and write the ledger."""
        self._entries[key] = entry
        keys = {name: _document(found) for name, found in self._entries.items()}
        atomic_write_json(self._path, {"schema": _SCHEMA, "keys": keys})


def _document(entry: SharedEntry) -> dict[str, Any]:
    pointer = entry.pointer
    return {
        "job_id": entry.job_id,
        "tool": entry.tool,
        "status": entry.status,
        "pointer": None
        if pointer is None
        else {**asdict(replace(pointer, op_shares={})), "op_shares": dict(pointer.op_shares)},
    }
