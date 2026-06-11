# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Conductor integration — read events (and task/lease state) from the
Conductor SQLite DB. Intents are emitted via the subprocess transport in
:mod:`robustness_agent.runtime.cli`, not by writing into the DB.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from .models import ConductorEvent

log = logging.getLogger(__name__)


class ConductorReader:
    """Read events and task state from Conductor's SQLite DB."""

    def __init__(self, db_path: Path):
        """Initialise the reader against a Conductor SQLite database.

        Args:
            db_path (Path): Path to the Conductor ``conductor.db`` file.
                The connection is opened lazily by :meth:`connect`.
        """
        self._db_path = db_path
        self._last_event_id: int = 0
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> bool:
        """Open a read-only WAL connection to the Conductor DB.

        Returns:
            bool: True when connected; False when the file is missing or
            the connection failed.
        """
        if not self._db_path.exists():
            log.warning("Conductor DB not found at %s", self._db_path)
            return False
        try:
            self._conn = sqlite3.connect(
                str(self._db_path),
                timeout=5.0,
                isolation_level="DEFERRED",
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA query_only=ON")
            log.info("Connected to Conductor DB at %s", self._db_path)
            return True
        except sqlite3.Error as e:
            log.error("Failed to connect to Conductor DB: %s", e)
            return False

    def poll_events(self, limit: int = 100) -> list[ConductorEvent]:
        """Fetch new events since the last polled event id.

        Advances the internal cursor so each call only returns rows not
        seen before. JSON payload strings are decoded into dicts.

        Args:
            limit (int): Maximum number of events to return this call.

        Returns:
            list[ConductorEvent]: New events in ascending id order; empty
            when disconnected or on query error.
        """
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                "SELECT id, agent, intent_type, payload, timestamp, topic "
                "FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
                (self._last_event_id, limit),
            ).fetchall()
            events: list[ConductorEvent] = []
            for row in rows:
                payload = row["payload"]
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        payload = {"raw": payload}
                events.append(ConductorEvent(
                    event_id=row["id"],
                    agent=row["agent"] or "",
                    intent_type=row["intent_type"] or "",
                    payload=payload if isinstance(payload, dict) else {},
                    timestamp=row["timestamp"] or 0,
                    topic=row["topic"] or "",
                ))
                self._last_event_id = max(self._last_event_id, row["id"])
            return events
        except sqlite3.Error as e:
            log.warning("Failed to poll events: %s", e)
            return []

    def get_agent_last_activity(self) -> dict[str, float]:
        """Return {agent_name: last_event_timestamp} for stall detection.

        Returns:
            dict[str, float]: Mapping of agent name to its most recent
            event timestamp; empty when disconnected or on query error.
        """
        if not self._conn:
            return {}
        try:
            rows = self._conn.execute(
                "SELECT agent, MAX(timestamp) as last_ts FROM events "
                "GROUP BY agent",
            ).fetchall()
            return {row["agent"]: row["last_ts"] for row in rows if row["agent"]}
        except sqlite3.Error:
            return {}

    def get_task_state(self, task_id: str) -> Optional[dict[str, Any]]:
        """Look up a single task row by id.

        Args:
            task_id (str): The task id to query.

        Returns:
            Optional[dict[str, Any]]: The task row as a dict, or ``None``
            when not found, disconnected, or on query error.
        """
        if not self._conn:
            return None
        try:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            return dict(row) if row else None
        except sqlite3.Error:
            return None

    def get_active_leases(self) -> list[dict[str, Any]]:
        """Return all leases that have not yet been released.

        Returns:
            list[dict[str, Any]]: One dict per active lease row; empty when
            disconnected or on query error.
        """
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                "SELECT * FROM leases WHERE released_at IS NULL",
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    def close(self) -> None:
        """Close the underlying SQLite connection if one is open."""
        if self._conn:
            self._conn.close()
            self._conn = None
