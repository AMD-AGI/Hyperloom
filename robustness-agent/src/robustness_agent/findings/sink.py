# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Append-only JSONL sink for ladder findings.

File layout::

    {session_dir}/agents/robustness/findings/{session_id}.jsonl

Each line is one :class:`Finding`, serialised by :func:`finding_to_row`.
Writes go through :func:`asyncio.to_thread` so the reactor's tick
budget is not blocked on disk I/O.

The sink is best-effort: a write failure logs a single WARN per error
class but never raises into the reactor.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import asyncio

from ..decision.action_ladder import Finding


log = logging.getLogger(__name__)


@dataclass
class FindingSinkConfig:
    """Where the sink writes."""

    session_dir: Path
    session_id: str = "default"
    subdir: str = "agents/robustness/findings"

    @property
    def file_path(self) -> Path:
        safe = self.session_id or "default"
        return self.session_dir / self.subdir / f"{safe}.jsonl"


class FindingSink:
    """JSONL append sink with simple error suppression."""

    def __init__(self, config: FindingSinkConfig) -> None:
        self._config = config
        self._warned: set[str] = set()

    @property
    def file_path(self) -> Path:
        return self._config.file_path

    async def append_many(self, findings: Iterable[Finding]) -> int:
        rows = [finding_to_row(f) for f in findings]
        if not rows:
            return 0
        await asyncio.to_thread(self._write_rows, rows)
        return len(rows)

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        path = self._config.file_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False))
                    handle.write("\n")
        except OSError as exc:
            self._warn_once("io", f"finding sink io error: {exc}")

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        log.warning("findings sink: %s", message)
        self._warned.add(key)


def finding_to_row(finding: Finding) -> dict[str, Any]:
    """Serialise a :class:`Finding` for JSONL persistence."""
    row = asdict(finding)
    return row


__all__ = ["FindingSink", "FindingSinkConfig", "finding_to_row"]
