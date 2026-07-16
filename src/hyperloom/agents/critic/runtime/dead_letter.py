# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Append-only KB dead-letter queue.

Failed KB writes go into ``KB_DEAD_LETTER_DIR/<endpoint>.jsonl`` (one JSON
record per line: ``endpoint``, ``payload``, ``attempts``, ``last_error``,
``ts``) so a cron/operator can replay them. The replay helper walks every
``*.jsonl`` and dispatches by ``endpoint``, dropping successes and
preserving failures.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hyperloom.common.io import append_jsonl
from hyperloom.common.timeutil import now_iso

from .errors import RuntimeAdapterError
from .metrics import CRITIC_KB_DEAD_LETTER_COUNT, get_registry


DEFAULT_DEAD_LETTER_DIR = "/var/lib/critic-kb-dlq"


@dataclass
class ReplaySummary:
    """Outcome counts from a :meth:`DeadLetter.replay` pass.

    Attributes:
        scanned (int): Total non-empty records examined.
        succeeded (int): Records that were dispatched successfully.
        failed (int): Records that failed to parse or dispatch.
        failed_details (list[dict[str, Any]]): Per-failure context entries.
    """

    scanned: int = 0
    succeeded: int = 0
    failed: int = 0
    failed_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a plain JSON-serialisable dict.

        Returns:
            dict[str, Any]: All summary counts and failure details.
        """
        return {
            "scanned": self.scanned,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "failed_details": list(self.failed_details),
        }


class DeadLetter:
    """Filesystem-backed dead-letter queue."""

    def __init__(self, root: str | Path | None = None):
        """Resolve the queue directory.

        Args:
            root (str | Path | None): Queue directory; falls back to the
                ``KB_DEAD_LETTER_DIR`` env var, then
                :data:`DEFAULT_DEAD_LETTER_DIR`.
        """
        if root is None:
            root = os.environ.get("KB_DEAD_LETTER_DIR", DEFAULT_DEAD_LETTER_DIR)
        self.root = Path(root)

    def _path_for(self, endpoint: str) -> Path:
        """Return the ``.jsonl`` path for an endpoint's dead-letter file.

        Args:
            endpoint (str): The KB endpoint name (no path separators).

        Returns:
            Path: ``<root>/<endpoint>.jsonl``.

        Raises:
            RuntimeAdapterError: If ``endpoint`` is empty or contains ``/``.
        """
        if not endpoint or "/" in endpoint:
            raise RuntimeAdapterError(f"invalid endpoint name: {endpoint!r}")
        return self.root / f"{endpoint}.jsonl"

    def append(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        attempts: int,
        last_error: str,
        context: dict[str, Any] | None = None,
    ) -> Path:
        """Append a failed write record to the endpoint's dead-letter file.

        Increments the dead-letter metric counter as a side effect.

        Args:
            endpoint (str): The KB endpoint the write targeted.
            payload (dict[str, Any]): The original request payload to retry.
            attempts (int): Number of attempts already made.
            last_error (str): String form of the last error encountered.
            context (dict[str, Any] | None): Optional extra audit context.

        Returns:
            Path: The file the record was appended to.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path_for(endpoint)
        record = {
            "ts": now_iso(timespec="microseconds"),
            "endpoint": endpoint,
            "payload": payload,
            "attempts": attempts,
            "last_error": last_error,
            "context": context or {},
        }
        append_jsonl(path, record, ensure_ascii=False)
        get_registry().counter(CRITIC_KB_DEAD_LETTER_COUNT).inc({"endpoint": endpoint})
        return path

    def files(self) -> list[Path]:
        """List the dead-letter files currently present.

        Returns:
            list[Path]: Sorted ``*.jsonl`` paths, or an empty list when the
            queue directory does not exist.
        """
        if not self.root.exists():
            return []
        return sorted(self.root.glob("*.jsonl"))

    def replay(
        self,
        dispatcher: Callable[[str, dict[str, Any]], None],
        *,
        delete_on_success: bool = True,
    ) -> ReplaySummary:
        """Walk every ``*.jsonl`` and call ``dispatcher(endpoint, payload)``.

        Successful rows are removed (file gets rewritten without them);
        failing rows stay so they can be retried later. The dispatcher
        should raise to signal failure.

        Args:
            dispatcher (Callable[[str, dict[str, Any]], None]): Callable that
                redelivers ``(endpoint, payload)``; raising signals failure.
            delete_on_success (bool): When True, rewrite files dropping
                succeeded rows (deleting files that fully drain).

        Returns:
            ReplaySummary: Counts of scanned / succeeded / failed records.
        """
        summary = ReplaySummary()
        for path in self.files():
            failed_lines: list[str] = []
            for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                summary.scanned += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    summary.failed += 1
                    summary.failed_details.append(
                        {
                            "file": str(path),
                            "line": lineno,
                            "reason": f"json: {exc}",
                        }
                    )
                    failed_lines.append(line)
                    continue
                endpoint = record.get("endpoint")
                payload = record.get("payload") or {}
                try:
                    dispatcher(endpoint, payload)
                    summary.succeeded += 1
                except Exception as exc:  # noqa: BLE001
                    summary.failed += 1
                    summary.failed_details.append(
                        {
                            "file": str(path),
                            "line": lineno,
                            "reason": str(exc),
                        }
                    )
                    failed_lines.append(line)
            if delete_on_success:
                if failed_lines:
                    path.write_text("\n".join(failed_lines) + "\n", encoding="utf-8")
                else:
                    path.unlink()
        return summary


__all__ = ["DEFAULT_DEAD_LETTER_DIR", "DeadLetter", "ReplaySummary"]
