# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared fixtures for robustness-agent tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from robustness_agent.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "storage").mkdir()
    return Config(
        session_dir=session_dir,
        agent_stall_timeout_s=10.0,
    )


@pytest.fixture
def conductor_db(config: Config) -> Path:
    """Create a minimal Conductor SQLite DB for testing."""
    db_path = config.conductor_db_path
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT,
            intent_type TEXT,
            payload TEXT,
            timestamp REAL,
            topic TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            action_name TEXT,
            status TEXT,
            family TEXT,
            created_at REAL,
            updated_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS leases (
            lane TEXT,
            holder TEXT,
            acquired_at REAL,
            released_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cursors (
            agent TEXT PRIMARY KEY,
            last_event_id INTEGER
        )
    """)
    conn.commit()
    conn.close()
    return db_path
