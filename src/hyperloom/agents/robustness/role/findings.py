# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Append-only JSONL sink for ladder findings."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import asyncio

from hyperloom.common.io import append_jsonl

from ..decision.action_ladder import Finding


log = logging.getLogger(__name__)


FINDINGS_SUBDIR: str = "agents/robustness/findings"


@dataclass
class FindingSinkConfig:
    """Where the sink writes."""

    session_dir: Path
    session_id: str = "default"
    subdir: str = FINDINGS_SUBDIR

    @property
    def file_path(self) -> Path:
        """Resolved JSONL file path for this session's findings."""
        safe = self.session_id or "default"
        return self.session_dir / self.subdir / f"{safe}.jsonl"


class FindingSink:
    """JSONL append sink with simple error suppression."""

    def __init__(self, config: FindingSinkConfig) -> None:
        """Initialise the sink."""
        self._config = config
        self._warned: set[str] = set()

    @property
    def file_path(self) -> Path:
        """Path of the JSONL file this sink appends to."""
        return self._config.file_path

    async def append_many(self, findings: Iterable[Finding]) -> int:
        """Append a batch of findings as JSONL rows off the event loop."""
        rows = [finding_to_row(f) for f in findings]
        if not rows:
            return 0
        await asyncio.to_thread(self._write_rows, rows)
        return len(rows)

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        """Append serialised rows to the JSONL file."""
        path = self._config.file_path
        try:
            for row in rows:
                append_jsonl(path, row, make_parents=True, ensure_ascii=False)
        except OSError as exc:
            self._warn_once("io", f"finding sink io error: {exc}")

    def _warn_once(self, key: str, message: str) -> None:
        """Log a warning at most once per error class."""
        if key in self._warned:
            return
        log.warning("findings sink: %s", message)
        self._warned.add(key)


def finding_to_row(finding: Finding) -> dict[str, Any]:
    """Serialise a :class:`Finding` for JSONL persistence."""
    row = asdict(finding)
    return row


__all__ = ["FINDINGS_SUBDIR", "FindingSink", "FindingSinkConfig", "finding_to_row"]
