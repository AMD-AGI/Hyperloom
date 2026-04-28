"""Tests for ``storage.backup`` — IMPL-CHECKLIST §6.20‒6.25."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from inference_optimizer.storage.backup import (
    DEFAULT_PERIOD_MIN,
    force_backup_after_keep,
    periodic_backup,
    restore_from_backup,
    vacuum_into,
)
from inference_optimizer.storage.connection import SqliteConnection


def _seed(db: SqliteConnection) -> None:
    """Insert a row each into events / cursors / tasks so the backup is
    nontrivial."""
    cur = db.execute_sync(
        "INSERT INTO events (msg_id, from_agent, to_agent, topic, "
        "in_reply_to, payload, priority, ts) VALUES (?,?,?,?,?,?,?,?)",
        ("ev1", "x", "y", "event", None, "{}", 1, "2026-01-01"),
    )
    cur.close()
    db.raw.commit()


# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_vacuum_into_writes_self_contained_db(tmp_path: Path):
    db = SqliteConnection(tmp_path / "live.db")
    _seed(db)
    backup = tmp_path / "out" / "snap.bak"
    out = await vacuum_into(db, backup)
    assert out == backup
    assert backup.is_file()
    assert backup.stat().st_size > 0
    # Open as a new connection to confirm self-containment.
    fresh = SqliteConnection(backup)
    rows = fresh.fetchall_sync("SELECT msg_id FROM events")
    assert any(r["msg_id"] == "ev1" for r in rows)
    fresh.close()
    db.close()


@pytest.mark.asyncio
async def test_vacuum_into_overwrites_existing(tmp_path: Path):
    db = SqliteConnection(tmp_path / "live.db")
    _seed(db)
    target = tmp_path / "snap.bak"
    target.write_bytes(b"old garbage")
    await vacuum_into(db, target)
    assert target.read_bytes()[:16] != b"old garbage"
    db.close()


@pytest.mark.asyncio
async def test_force_backup_after_keep_creates_timestamped_dir(tmp_path: Path):
    db = SqliteConnection(tmp_path / "live.db")
    _seed(db)
    cp_dir = tmp_path / "checkpoints"
    out = await force_backup_after_keep(db, cp_dir)
    assert out.parent.parent == cp_dir
    assert out.name == "conductor.db.bak"
    assert out.is_file()
    db.close()


@pytest.mark.asyncio
async def test_periodic_backup_loops_until_stop(tmp_path: Path):
    db = SqliteConnection(tmp_path / "live.db")
    _seed(db)
    cp_dir = tmp_path / "cps"
    stop = asyncio.Event()
    captured: list[Path] = []

    async def on_complete(path: Path) -> None:
        captured.append(path)
        if len(captured) >= 2:
            stop.set()

    task = asyncio.create_task(
        periodic_backup(
            db, cp_dir,
            period_min=0.005,  # ~ 0.3 s
            stop_event=stop,
            on_complete=on_complete,
        )
    )
    await asyncio.wait_for(task, timeout=5.0)
    assert len(captured) >= 2
    assert all(p.is_file() for p in captured)
    db.close()


@pytest.mark.asyncio
async def test_periodic_backup_continues_on_failure(tmp_path: Path, monkeypatch):
    db = SqliteConnection(tmp_path / "live.db")
    _seed(db)
    cp_dir = tmp_path / "cps"
    calls = {"n": 0}

    real_vacuum = vacuum_into

    async def flaky(db_arg, dest):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return await real_vacuum(db_arg, dest)

    import inference_optimizer.storage.backup as backup_mod
    monkeypatch.setattr(backup_mod, "vacuum_into", flaky)

    stop = asyncio.Event()
    captured: list[Path] = []

    async def on_complete(path: Path) -> None:
        captured.append(path)
        stop.set()

    task = asyncio.create_task(
        backup_mod.periodic_backup(
            db, cp_dir, period_min=0.005,
            stop_event=stop, on_complete=on_complete,
        )
    )
    await asyncio.wait_for(task, timeout=5.0)
    assert len(captured) == 1
    db.close()


def test_restore_from_backup_replaces_target(tmp_path: Path):
    src = tmp_path / "snap.bak"
    src.write_bytes(b"snapshot bytes")
    dest = tmp_path / "deep" / "live.db"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"old live")
    # also place a stray WAL file
    (dest.parent / "live.db-wal").write_bytes(b"wal")
    restore_from_backup(src, dest)
    assert dest.read_bytes() == b"snapshot bytes"
    assert not (dest.parent / "live.db-wal").exists()


def test_restore_from_backup_missing_source(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        restore_from_backup(tmp_path / "nope.bak", tmp_path / "live.db")


def test_default_period_min_constant():
    assert DEFAULT_PERIOD_MIN == 30
