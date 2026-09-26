# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Failure-isolation contracts in the orchestrator bus layer."""

from __future__ import annotations

import sqlite3
from unittest import mock

import pytest

from hyperloom.orchestrator.bus.message_bus import Message, MessageBus
from hyperloom.orchestrator.bus.resource_lock import LaneBusy, SqliteLeaseBackend
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
    try:
        held = await backend.acquire_many(
            lanes=["benchmark_lane"],
            holder_id="live_holder",
            task_id="t0",
            action="bench",
            ttl_sec=60,
        )
        stamp = "2026-01-01T00:00:00+00:00"
        db.raw.execute(
            "UPDATE leases SET acquired_at=?, expires_at=?, heartbeat_at=? WHERE holder_id=?",
            (stamp, "NOT_A_DATE", stamp, held.holder_id),
        )
        db.raw.commit()
        retained = [dict(row) for row in db.fetchall_sync("SELECT * FROM leases ORDER BY lane")]
        assert len(retained) == len(held.lanes)
        assert all(row["expires_at"] == "NOT_A_DATE" for row in retained)

        with pytest.raises(LaneBusy) as exc:
            await backend.acquire_many(
                lanes=["benchmark_lane"],
                holder_id="new_holder",
                task_id="t1",
                action="bench",
                ttl_sec=60,
            )

        assert exc.value.busy_lanes == list(held.lanes)
        assert [dict(row) for row in db.fetchall_sync("SELECT * FROM leases ORDER BY lane")] == retained
        released = await backend.release(held)
        holders = await backend.lane_holders()
        assert released == len(held.lanes)
        assert holders == {}

        lease = await backend.acquire_many(
            lanes=["benchmark_lane"],
            holder_id="new_holder",
            task_id="t1",
            action="bench",
            ttl_sec=60,
        )

        assert lease.lanes == held.lanes
        remaining = db.fetchall_sync("SELECT lane, holder_id FROM leases ORDER BY lane")
        assert [(row["lane"], row["holder_id"]) for row in remaining] == [(lane, "new_holder") for lane in lease.lanes]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_replay_for_respects_limit(tmp_path):
    db = SqliteConnection(tmp_path / "bus.db")
    bus = MessageBus(db)
    for i in range(20):
        await bus.append_and_seq(
            Message.new(from_agent="coord", to_agent="critic", topic="observation", payload={"n": i})
        )

    assert len(await bus.replay_for("critic", after_seq=0, limit=5)) == 5
    assert len(await bus.replay_for("critic", after_seq=0, limit=100)) == 20
    db.close()
