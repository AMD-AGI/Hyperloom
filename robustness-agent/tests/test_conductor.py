"""Tests for Conductor integration — event reading and intent emission."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


from robustness_agent.conductor import ConductorReader, IntentEmitter
from robustness_agent.models import Alert, Severity


class TestConductorReader:

    def test_poll_events(self, conductor_db: Path) -> None:
        conn = sqlite3.connect(str(conductor_db))
        for i in range(5):
            conn.execute(
                "INSERT INTO events (agent, intent_type, payload, timestamp, topic) "
                "VALUES (?, ?, ?, ?, ?)",
                ("kernel", "response", json.dumps({"status": "ok"}), time.time() + i, "kernel"),
            )
        conn.commit()
        conn.close()

        reader = ConductorReader(conductor_db)
        reader.connect()

        events = reader.poll_events()
        assert len(events) == 5
        assert events[0].agent == "kernel"

        events2 = reader.poll_events()
        assert len(events2) == 0
        reader.close()

    def test_agent_last_activity(self, conductor_db: Path) -> None:
        conn = sqlite3.connect(str(conductor_db))
        conn.execute(
            "INSERT INTO events (agent, intent_type, payload, timestamp, topic) "
            "VALUES (?, ?, ?, ?, ?)",
            ("orchestration", "propose_action", "{}", 1000.0, "action"),
        )
        conn.execute(
            "INSERT INTO events (agent, intent_type, payload, timestamp, topic) "
            "VALUES (?, ?, ?, ?, ?)",
            ("orchestration", "delegate", "{}", 2000.0, "task"),
        )
        conn.commit()
        conn.close()

        reader = ConductorReader(conductor_db)
        reader.connect()
        activity = reader.get_agent_last_activity()
        assert activity["orchestration"] == 2000.0
        reader.close()

    def test_missing_db(self, tmp_path: Path) -> None:
        reader = ConductorReader(tmp_path / "nonexistent.db")
        assert not reader.connect()
        assert reader.poll_events() == []


class TestIntentEmitter:

    def test_emit_alert(self, conductor_db: Path) -> None:
        emitter = IntentEmitter(conductor_db)
        emitter.connect()
        emitter.emit_alert(Alert(
            check_name="test_check",
            severity=Severity.WARNING,
            summary="test alert",
            timestamp=time.time(),
        ))
        emitter.close()

        conn = sqlite3.connect(str(conductor_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM events WHERE agent='robustness'").fetchall()
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["severity"] == "warning"
        assert payload["summary"] == "test alert"
        conn.close()

    def test_emit_kill_task(self, conductor_db: Path) -> None:
        emitter = IntentEmitter(conductor_db)
        emitter.connect()
        emitter.emit_kill_task("task-123", "stuck for 10min")
        emitter.close()

        conn = sqlite3.connect(str(conductor_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM events WHERE intent_type='kill_task'",
        ).fetchall()
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["task_id"] == "task-123"
        assert payload["scope"] == "task"
        conn.close()

    def test_emit_prune_branch(self, conductor_db: Path) -> None:
        emitter = IntentEmitter(conductor_db)
        emitter.connect()
        emitter.emit_prune_branch("deep_kernel", "3+ failures")
        emitter.close()

        conn = sqlite3.connect(str(conductor_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM events WHERE intent_type='prune_branch'",
        ).fetchall()
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["family"] == "deep_kernel"
        conn.close()
