# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Robustness tests for the orchestrator bus layer."""

from __future__ import annotations

import sqlite3

import pytest

from hyperloom.orchestrator.bus.storage.connection import SqliteConnection, open_connection
from hyperloom.orchestrator.bus.message_bus import MessageBus, Message
from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend


def test_open_connection_does_not_leak_on_bad_db_path(tmp_path):
    """open_connection must close the connection when pragma or schema setup fails."""
    import unittest.mock as mock

    opened: list[sqlite3.Connection] = []
    original_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    with mock.patch("hyperloom.orchestrator.bus.storage.connection.sqlite3.connect", side_effect=tracking_connect):
        with mock.patch(
            "hyperloom.orchestrator.bus.storage.connection.ensure_schema",
            side_effect=RuntimeError("schema boom"),
        ):
            with pytest.raises(RuntimeError, match="schema boom"):
                open_connection(tmp_path / "test.db")

    assert opened, "a connection must have been opened"
    for conn in opened:
        try:
            conn.execute("SELECT 1")
            assert False, "connection must be closed after a failed open_connection"
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            pass


@pytest.mark.asyncio
async def test_corrupt_expires_at_does_not_poison_acquire(tmp_path):
    """A single row with a corrupt expires_at must not prevent the whole acquire."""
    db = SqliteConnection(tmp_path / "leases.db")
    backend = SqliteLeaseBackend(db)

    now_iso = "2026-01-01T00:00:00+00:00"
    # Insert a row with a corrupt expires_at directly into the DB so the normal
    # path cannot produce it, then try to acquire — must not raise.
    db.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, acquired_at, expires_at, heartbeat_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("benchmark_lane", "dead_holder", "t0", "bench", 0, now_iso, "NOT_A_DATE", now_iso),
    )
    db.raw.commit()

    # acquire_many must treat the corrupt row as already-expired and delete it,
    # allowing the acquire to proceed.
    lease = await backend.acquire_many(
        lanes=["benchmark_lane"],
        holder_id="new_holder",
        task_id="t1",
        action="bench",
        ttl_sec=60,
    )
    assert lease is not None
    assert "benchmark_lane" in lease.lanes
    db.close()


@pytest.mark.asyncio
async def test_replay_for_respects_limit(tmp_path):
    """replay_for must honour the limit parameter and not load the entire table."""
    db = SqliteConnection(tmp_path / "bus.db")
    bus = MessageBus(db)

    # Append 20 messages; requesting limit=5 must return at most 5.
    for i in range(20):
        msg = Message.new(
            from_agent="coord",
            to_agent="agent1",
            topic="observation",
            payload={"n": i},
        )
        await bus.append_and_seq(msg)

    msgs = await bus.replay_for("agent1", after_seq=0, limit=5)
    assert len(msgs) == 5, f"expected 5 messages, got {len(msgs)}"

    msgs_all = await bus.replay_for("agent1", after_seq=0, limit=100)
    assert len(msgs_all) == 20, f"expected 20 messages, got {len(msgs_all)}"
    db.close()
