# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Failure-isolation contracts in the orchestrator bus layer.

A connection that fails midway through setup must not be left open, one lease
row with an unparseable ``expires_at`` must not abort an unrelated acquire, and
a resume replay must not load an unbounded number of events.
"""

from __future__ import annotations

import sqlite3
from unittest import mock

import pytest

from hyperloom.orchestrator.bus.message_bus import Message, MessageBus
from hyperloom.orchestrator.bus.resource_lock import SqliteLeaseBackend
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection, open_connection


def test_open_connection_closes_the_connection_when_setup_fails(tmp_path):
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def _tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    with (
        mock.patch(
            "hyperloom.orchestrator.bus.storage.connection.sqlite3.connect",
            side_effect=_tracking_connect,
        ),
        mock.patch(
            "hyperloom.orchestrator.bus.storage.connection.ensure_schema",
            side_effect=RuntimeError("schema boom"),
        ),
        pytest.raises(RuntimeError, match="schema boom"),
    ):
        open_connection(tmp_path / "test.db")

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")


@pytest.mark.asyncio
async def test_corrupt_expires_at_does_not_abort_acquire(tmp_path):
    db = SqliteConnection(tmp_path / "leases.db")
    backend = SqliteLeaseBackend(db)
    stamp = "2026-01-01T00:00:00+00:00"
    db.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, acquired_at, expires_at, heartbeat_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("benchmark_lane", "dead_holder", "t0", "bench", 0, stamp, "NOT_A_DATE", stamp),
    )
    db.raw.commit()

    lease = await backend.acquire_many(
        lanes=["benchmark_lane"],
        holder_id="new_holder",
        task_id="t1",
        action="bench",
        ttl_sec=60,
    )

    assert "benchmark_lane" in lease.lanes
    remaining = db.fetchall_sync("SELECT holder_id FROM leases WHERE lane='benchmark_lane'")
    assert [r["holder_id"] for r in remaining] == ["new_holder"], "the corrupt row must be reaped"
    db.close()


@pytest.mark.asyncio
async def test_replay_for_respects_limit(tmp_path):
    db = SqliteConnection(tmp_path / "bus.db")
    bus = MessageBus(db)
    for i in range(20):
        await bus.append_and_seq(
            Message.new(from_agent="coord", to_agent="agent1", topic="observation", payload={"n": i})
        )

    assert len(await bus.replay_for("agent1", after_seq=0, limit=5)) == 5
    assert len(await bus.replay_for("agent1", after_seq=0, limit=100)) == 20
    db.close()
