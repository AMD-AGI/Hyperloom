"""Tests for Conductor integration — event reading from the SQLite DB."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


from robustness_agent.conductor import ConductorReader


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
