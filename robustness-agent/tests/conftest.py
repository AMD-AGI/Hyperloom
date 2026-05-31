"""Shared fixtures for robustness-agent tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pytest

from robustness_agent.config import Config
from robustness_agent.models import DiskSnapshot, FaultEvent, GpuSnapshot, ProcessInfo
from robustness_agent.providers.base import MetricsProvider


class FakeProvider(MetricsProvider):
    """In-memory test provider."""

    def __init__(self) -> None:
        self.gpu_snapshots: list[GpuSnapshot] = []
        self.gpu_history_data: dict[int, list[GpuSnapshot]] = {}
        self.processes: list[ProcessInfo] = []
        self.disks: list[DiskSnapshot] = []
        self.faults: list[FaultEvent] = []

    async def get_gpu_metrics(self, gpu_id: Optional[int] = None) -> list[GpuSnapshot]:
        if gpu_id is not None:
            return [s for s in self.gpu_snapshots if s.gpu_id == gpu_id]
        return list(self.gpu_snapshots)

    async def get_gpu_history(self, gpu_id: int, window_seconds: int) -> list[GpuSnapshot]:
        return self.gpu_history_data.get(gpu_id, [])

    async def get_process_list(self) -> list[ProcessInfo]:
        return list(self.processes)

    async def get_disk_usage(self, path: str = "/") -> list[DiskSnapshot]:
        return list(self.disks)

    async def get_fault_events(self, since: float) -> list[FaultEvent]:
        return [f for f in self.faults if f.timestamp >= since]

    async def check_available(self) -> bool:
        return True


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "storage").mkdir()
    return Config(
        session_dir=session_dir,
        process_check_interval=1.0,
        gpu_check_interval=1.0,
        agent_stall_timeout_s=10.0,
        benchmark_timeout_s=5.0,
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
