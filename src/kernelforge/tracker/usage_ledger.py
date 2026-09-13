# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Always-current token ledger at one fixed path per experiments directory."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from kernelforge.durable_io import atomic_write_text
from kernelforge.tracker.usage import combine_usage_totals

log = logging.getLogger(__name__)

# Fixed for every producer: a caller that knows the experiments directory can read the run's spend without knowing
# which command produced it, whether a result file was requested, or what the experiment is called.
LEDGER_FILENAME = "llm_usage.json"

_COUNTER_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "calls",
)


def ledger_path(experiments_dir: str | Path) -> Path:
    """Where the shared token ledger for ``experiments_dir`` lives."""
    return Path(experiments_dir) / LEDGER_FILENAME


def read_usage_ledger(experiments_dir: str | Path) -> dict[str, Any]:
    """The directory's running token total, or ``{}`` when nothing was recorded."""
    try:
        payload = json.loads(ledger_path(experiments_dir).read_text())
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


class UsageLedgerFile:
    """Fold one process's spend into the ledger the whole run shares.

    A FlyDSL rewrite and the forge-loop it nests are separate processes writing one file, so a publish adds only what
    this process has spent since its own last publish instead of overwriting the file with its private view. The file
    is therefore the running total for the directory: reading it needs no knowledge of who contributed to it.
    """

    def __init__(self, experiments_dir: str | Path) -> None:
        self.path = ledger_path(experiments_dir)
        self._lock_path = self.path.with_name(f".{LEDGER_FILENAME}.lock")
        self._published: dict[str, Any] = {}

    def publish(self, totals: dict[str, Any]) -> None:
        """Record this process's totals so far. Never raises: accounting must not end a run."""
        try:
            self._publish(totals)
        except Exception as error:  # noqa: BLE001 - a ledger write must never break the caller
            log.debug("usage ledger publish skipped (%s: %s)", type(error).__name__, error)

    def _publish(self, totals: dict[str, Any]) -> None:
        delta = self._delta(totals)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if delta is None and self.path.is_file():
            return
        with open(self._lock_path, "a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                merged = combine_usage_totals(self._current(), delta)
                merged["updated_at"] = datetime.now().isoformat()
                atomic_write_text(self.path, json.dumps(merged, indent=2))
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        self._published = dict(totals)

    def _current(self) -> dict[str, Any]:
        """The ledger as it stands, read inside the lock."""
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError):
            # A ledger that was never written, or one a crash left unreadable, restarts from this process's own share
            # rather than costing the run its accounting.
            return {}
        return payload if isinstance(payload, dict) else {}

    def _delta(self, totals: dict[str, Any]) -> dict[str, Any] | None:
        """What this process has spent since its last publish, or None when nothing moved.

        The provenance of the whole record travels with the delta: a producer whose cost stops being complete can
        only ever turn the shared answer less certain, which is exactly how ``combine_usage_totals`` folds it.
        """
        delta: dict[str, Any] = {}
        moved = False
        for key in _COUNTER_KEYS:
            value = _counter(totals, key) - _counter(self._published, key)
            delta[key] = value
            moved = moved or value != 0
        cost = _cost(totals) - _cost(self._published)
        delta["total_cost_usd"] = round(cost, 6)
        delta["cost_available"] = totals.get("cost_available") is True
        delta["cost_source"] = totals.get("cost_source")
        return delta if moved or cost else None


def _counter(record: dict[str, Any], key: str) -> int:
    with contextlib.suppress(TypeError, ValueError):
        return int(record.get(key) or 0)
    return 0


def _cost(record: dict[str, Any]) -> float:
    with contextlib.suppress(TypeError, ValueError):
        return float(record.get("total_cost_usd") or 0.0)
    return 0.0


__all__ = [
    "LEDGER_FILENAME",
    "UsageLedgerFile",
    "ledger_path",
    "read_usage_ledger",
]
