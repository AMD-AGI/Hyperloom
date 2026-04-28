"""Tests for ``scripts/monitor`` — IMPL-CHECKLIST §9.12‒9.23."""
from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.scripts import monitor
from inference_optimizer.storage.connection import SqliteConnection


def _seed(db: SqliteConnection, *, lag: int = 0, zombie: bool = False) -> None:
    """Insert enough rows to make snapshot() output non-trivial."""
    cur = db.execute_sync(
        "INSERT INTO events (msg_id, from_agent, to_agent, topic, "
        "in_reply_to, payload, priority, ts) VALUES (?,?,?,?,?,?,?,?)",
        ("ev1", "x", "*", "event", None, "{}", 1, "2026-01-01"),
    )
    cur.close()
    cur = db.execute_sync(
        "INSERT INTO cursors (agent, last_processed_seq, "
        "last_processed_msg_id, processed_at) VALUES (?,?,?,?)",
        ("executor", 0 if lag else 1, "ev1", "2026-01-01"),
    )
    cur.close()
    if zombie:
        cur = db.execute_sync(
            "INSERT INTO tasks (task_id, kind, state, params, idempotency_key, "
            "requires_lanes, allowed_tools, side_effects, lease_ttl_sec, "
            "attempts, history, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "z1", "delegate", "running", "{}", "z-key",
                "[]", "[]", "[]", 0, 0, "[]",
                "2024-01-01T00:00:00.000000+00:00",
                "2024-01-01T00:00:00.000000+00:00",
            ),
        )
        cur.close()
    db.raw.commit()


def test_snapshot_healthy_returns_zero(tmp_path: Path, capsys):
    db_path = tmp_path / "conductor.db"
    db = SqliteConnection(db_path)
    _seed(db, lag=0)
    db.close()
    rc = monitor.snapshot(db_path, lag_threshold=10)
    out = capsys.readouterr().out
    assert rc == 0
    assert "events" in out
    assert "in-flight tasks" in out
    assert "cursors lag" in out


def test_snapshot_missing_db_returns_3(tmp_path: Path):
    rc = monitor.snapshot(tmp_path / "missing.db")
    assert rc == 3


def test_snapshot_zombie_task_flags_degraded(tmp_path: Path, capsys):
    db_path = tmp_path / "conductor.db"
    db = SqliteConnection(db_path)
    _seed(db, zombie=True)
    db.close()
    rc = monitor.snapshot(db_path, lag_threshold=100)
    err = capsys.readouterr().err
    assert rc == 1
    assert "STATUS=degraded" in err


def test_snapshot_per_agent_lists_cursors(tmp_path: Path, capsys):
    db_path = tmp_path / "conductor.db"
    db = SqliteConnection(db_path)
    _seed(db)
    db.close()
    rc = monitor.snapshot(db_path, per_agent=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "per-agent cursor lag" in out
    assert "executor" in out


def test_snapshot_top_events(tmp_path: Path, capsys):
    db_path = tmp_path / "conductor.db"
    db = SqliteConnection(db_path)
    _seed(db)
    db.close()
    rc = monitor.snapshot(db_path, top_events=5)
    out = capsys.readouterr().out
    assert rc == 0
    assert "last 5 events" in out


def test_snapshot_per_lane(tmp_path: Path, capsys):
    db_path = tmp_path / "conductor.db"
    db = SqliteConnection(db_path)
    _seed(db)
    # add a lease (schema requires pid + heartbeat_at)
    cur = db.execute_sync(
        "INSERT INTO leases (lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "benchmark_lane", "executor", "task-1", "bench_runner", 12345,
            "2026-01-01T00:00:00.000000+00:00",
            "2099-01-01T00:00:00.000000+00:00",
            "2026-01-01T00:00:00.000000+00:00",
        ),
    )
    cur.close()
    db.raw.commit()
    db.close()
    rc = monitor.snapshot(db_path, per_lane=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "active leases" in out
    assert "executor" in out


def test_resolve_db_uses_arg(tmp_path: Path):
    p = monitor._resolve_db(str(tmp_path / "x.db"))
    assert p == tmp_path / "x.db"


def test_resolve_db_uses_env_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DB_PATH", str(tmp_path / "y.db"))
    monkeypatch.delenv("SESSION_DIR", raising=False)
    assert monitor._resolve_db(None) == tmp_path / "y.db"


def test_resolve_db_uses_session_dir(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_DB_PATH", raising=False)
    monkeypatch.setenv("SESSION_DIR", str(tmp_path))
    assert monitor._resolve_db(None) == tmp_path / "storage" / "conductor.db"


def test_resolve_db_raises_when_unset(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_DB_PATH", raising=False)
    monkeypatch.delenv("SESSION_DIR", raising=False)
    with pytest.raises(SystemExit):
        monitor._resolve_db(None)
