# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :mod:`hyperloom.orchestrator.bus.storage.schema`."""

from __future__ import annotations

import sqlite3

import pytest

from hyperloom.orchestrator.bus.storage import schema


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def test_ensure_schema_creates_and_reports_version():
    conn = _conn()
    version = schema.ensure_schema(conn)
    assert version == schema.SCHEMA_VERSION
    # idempotent second call
    assert schema.ensure_schema(conn) == schema.SCHEMA_VERSION
    conn.close()


def test_ensure_schema_refuses_a_database_recorded_at_an_older_version(tmp_path):
    path = tmp_path / "coordinator.db"
    conn = sqlite3.connect(path)
    for stmt in schema._DDL:
        conn.execute(stmt)
    conn.execute("INSERT INTO schema_version(version, applied_at) VALUES (5, '2026-09-16 00:00:00')")
    conn.commit()
    with pytest.raises(RuntimeError) as excinfo:
        schema.ensure_schema(conn)
    assert str(excinfo.value) == (
        f"coordinator.db is at schema version 5, older than this build's {schema.SCHEMA_VERSION}: "
        "sessions created before this release cannot be resumed. Finish the session under the build "
        "that started it, or start a new session."
    )
    assert conn.execute("SELECT version FROM schema_version").fetchall() == [(5,)]
    assert conn.execute("SELECT COUNT(*) FROM lane_capacity").fetchone() == (0,)
    conn.close()


def test_set_lane_capacity_upserts():
    conn = _conn()
    schema.ensure_schema(conn)

    def _stored() -> int:
        return conn.execute("SELECT capacity FROM lane_capacity WHERE lane = 'research_lane'").fetchone()[0]

    schema.set_lane_capacity(conn, "research_lane", 4)
    assert _stored() == 4
    schema.set_lane_capacity(conn, "research_lane", 2)
    assert _stored() == 2
    conn.close()


class _SpyConn(sqlite3.Connection):
    """Connection subclass that records rollbacks and can force commit errors."""

    fail_commit = False
    rolled_back = False

    def commit(self):  # type: ignore[override]
        if self.fail_commit:
            raise sqlite3.OperationalError("boom")
        return super().commit()

    def rollback(self):  # type: ignore[override]
        self.rolled_back = True
        return super().rollback()


def test_set_lane_capacity_rolls_back_on_error():
    conn = sqlite3.connect(":memory:", factory=_SpyConn)
    schema.ensure_schema(conn)
    conn.fail_commit = True
    conn.rolled_back = False
    with pytest.raises(sqlite3.OperationalError):
        schema.set_lane_capacity(conn, "research_lane", 3)
    assert conn.rolled_back is True
    conn.close()


def test_ensure_schema_rolls_back_on_error():
    conn = sqlite3.connect(":memory:", factory=_SpyConn)
    conn.fail_commit = True
    conn.rolled_back = False
    with pytest.raises(sqlite3.OperationalError):
        schema.ensure_schema(conn)
    assert conn.rolled_back is True
    conn.close()
