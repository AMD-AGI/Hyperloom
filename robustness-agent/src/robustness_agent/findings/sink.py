# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Append-only JSONL sink for ladder findings.

Writes one :class:`Finding` per line to
``{session_dir}/agents/robustness/findings/{session_id}.jsonl`` via
:func:`asyncio.to_thread` (keeps the tick off the disk I/O path).
Best-effort: write failures log one WARN per error class, never raise.
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
        """Resolved JSONL file path for this session's findings.

        Returns:
            Path: ``session_dir/subdir/{session_id}.jsonl`` (falling back
            to ``default`` when ``session_id`` is empty).
        """
        safe = self.session_id or "default"
        return self.session_dir / self.subdir / f"{safe}.jsonl"


class FindingSink:
    """JSONL append sink with simple error suppression."""

    def __init__(self, config: FindingSinkConfig) -> None:
        """Initialise the sink.

        Args:
            config (FindingSinkConfig): Configuration describing the
                destination directory, session id, and subdirectory.
        """
        self._config = config
        self._warned: set[str] = set()

    @property
    def file_path(self) -> Path:
        """Path of the JSONL file this sink appends to.

        Returns:
            Path: The configured findings file path.
        """
        return self._config.file_path

    async def append_many(self, findings: Iterable[Finding]) -> int:
        """Append a batch of findings as JSONL rows off the event loop.

        Args:
            findings (Iterable[Finding]): Findings to serialise and
                append.

        Returns:
            int: The number of rows written (0 when ``findings`` is
            empty).
        """
        rows = [finding_to_row(f) for f in findings]
        if not rows:
            return 0
        await asyncio.to_thread(self._write_rows, rows)
        return len(rows)

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        """Append serialised rows to the JSONL file.

        Creates parent directories as needed. Write failures are
        suppressed and logged once per error class rather than raised.

        Args:
            rows (list[dict[str, Any]]): Pre-serialised finding rows.
        """
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
        """Log a warning at most once per error class.

        Args:
            key (str): Dedup key identifying the error class.
            message (str): The warning message to log.
        """
        if key in self._warned:
            return
        log.warning("findings sink: %s", message)
        self._warned.add(key)


def finding_to_row(finding: Finding) -> dict[str, Any]:
    """Serialise a :class:`Finding` for JSONL persistence.

    Args:
        finding (Finding): The finding to serialise.

    Returns:
        dict[str, Any]: A plain dict suitable for ``json.dumps``.
    """
    row = asdict(finding)
    return row


__all__ = ["FindingSink", "FindingSinkConfig", "finding_to_row"]
