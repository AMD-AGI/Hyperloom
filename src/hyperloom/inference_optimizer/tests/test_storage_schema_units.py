# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :mod:`hyperloom.orchestrator.bus.storage.schema`.

Covers fresh-DB creation, ``set_lane_capacity`` / ``get_lane_capacity``, and
the rollback-on-failure guards in ``set_lane_capacity`` / ``ensure_schema``.
"""

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


def test_get_lane_capacity_known_and_default():
    conn = _conn()
    schema.ensure_schema(conn)
    assert schema.get_lane_capacity(conn, "benchmark_lane") == 1
    # unknown lane with no row and no default -> defensive 1
    assert schema.get_lane_capacity(conn, "no_such_lane") == 1
    conn.close()


def test_set_lane_capacity_upserts():
    conn = _conn()
    schema.ensure_schema(conn)
    schema.set_lane_capacity(conn, "research_lane", 4)
    assert schema.get_lane_capacity(conn, "research_lane") == 4
    schema.set_lane_capacity(conn, "research_lane", 2)
    assert schema.get_lane_capacity(conn, "research_lane") == 2
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


def test_reset_schema_drops_and_recreates():
    conn = _conn()
    schema.ensure_schema(conn)
    conn.execute(
        "INSERT INTO tasks(task_id, kind, state, params, idempotency_key, created_at, updated_at) "
        "VALUES ('t1','baseline','queued','{}','idem1','a','b')"
    )
    conn.commit()
    schema.reset_schema(conn)
    (count,) = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
    assert count == 0
    conn.close()
