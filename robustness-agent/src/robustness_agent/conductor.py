"""Conductor integration — read events from SQLite and emit intents.

This module bridges the Robustness agent with the Conductor's SQLite database.
It reads events to understand what other agents are doing, and writes intents
(alerts, kill_task, prune_branch, etc.) back.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from .models import Alert, ConductorEvent

log = logging.getLogger(__name__)


class ConductorReader:
    """Read events and task state from Conductor's SQLite DB."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._last_event_id: int = 0
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> bool:
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
        """Return {agent_name: last_event_timestamp} for stall detection."""
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
        if self._conn:
            self._conn.close()
            self._conn = None


class IntentEmitter:
    """Write robustness intents back to Conductor (legacy DB writer).

    Deprecated as of M1: the canonical integration runs the reactor
    behind the subprocess CLI in :mod:`robustness_agent.runtime.cli`
    and emits a validated ``intent_envelope`` for the host instead of
    writing into the SQLite DB. The class is retained for old DB-oriented
    tooling that has not migrated to the envelope transport.
    """

    def __init__(self, db_path: Path):
        import warnings

        warnings.warn(
            "IntentEmitter writes intents directly into conductor.db, which is "
            "deprecated. Migrate to `python -m robustness_agent.runtime.cli tick` "
            "(subprocess transport, mirrors critic-agent).",
            DeprecationWarning,
            stacklevel=2,
        )
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> bool:
        if not self._db_path.exists():
            return False
        try:
            self._conn = sqlite3.connect(str(self._db_path), timeout=5.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            return True
        except sqlite3.Error as e:
            log.error("IntentEmitter failed to connect: %s", e)
            return False

    def emit_alert(self, alert: Alert) -> None:
        self._write_event("alert", {
            "severity": alert.severity.value,
            "summary": alert.summary,
            "detail": alert.detail,
            "check_name": alert.check_name,
            "evidence": alert.evidence,
        })

    def emit_kill_task(self, task_id: str, reason: str) -> None:
        self._write_event("kill_task", {
            "task_id": task_id,
            "reason": reason,
            "scope": "task",
        })

    def emit_force_dispatch(self, task_id: str, reason: str) -> None:
        self._write_event("force_dispatch", {
            "task_id": task_id,
            "reason": reason,
        })

    def emit_prune_branch(self, family: str, reason: str) -> None:
        self._write_event("prune_branch", {
            "family": family,
            "reason": reason,
        })

    def emit_escalate(self, reason: str, hint: str = "") -> None:
        self._write_event("escalate_strategy_change", {
            "reason": reason,
            "next_action_hint": hint,
            "severity": "high",
        })

    def _write_event(self, intent_type: str, payload: dict[str, Any]) -> None:
        if not self._conn:
            log.warning("IntentEmitter not connected, dropping %s", intent_type)
            return
        try:
            self._conn.execute(
                "INSERT INTO events (agent, intent_type, payload, timestamp, topic) "
                "VALUES (?, ?, ?, ?, ?)",
                ("robustness", intent_type, json.dumps(payload), time.time(), intent_type),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            log.error("Failed to emit %s: %s", intent_type, e)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
