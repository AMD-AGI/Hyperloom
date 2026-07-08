# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for :mod:`hyperloom.orchestrator.bus.storage.schema`.

Focus on the migration / capacity / error-rollback branches that the
end-to-end resource-lane tests do not exercise: the legacy leases PK
migration variants, ``set_lane_capacity`` / ``get_lane_capacity`` error paths,
and the ``ensure_schema`` rollback-on-failure guard.
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
    # idempotent second call.
    assert schema.ensure_schema(conn) == schema.SCHEMA_VERSION
    conn.close()


def test_migrate_leases_no_table_returns_false():
    conn = _conn()
    cur = conn.cursor()
    # PRAGMA table_info on a missing table returns [] -> not migrated.
    assert schema._migrate_leases_v1_to_v2(cur) is False
    conn.close()


def test_migrate_leases_from_v1_pk():
    conn = _conn()
    cur = conn.cursor()
    # Legacy v1 leases table: PK is just (lane).
    cur.execute(
        """
        CREATE TABLE leases (
            lane TEXT PRIMARY KEY, holder_id TEXT NOT NULL, task_id TEXT NOT NULL,
            action TEXT NOT NULL, pid INTEGER NOT NULL, acquired_at TEXT NOT NULL,
            expires_at TEXT NOT NULL, heartbeat_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        "INSERT INTO leases VALUES ('benchmark_lane','h1','t1','baseline',1,'a','e','hb')"
    )
    assert schema._migrate_leases_v1_to_v2(cur) is True
    # Composite PK now allows two holders on the same lane.
    cur.execute("PRAGMA table_info(leases)")
    pk_cols = sorted(row[1] for row in cur.fetchall() if int(row[5] or 0) > 0)
    assert pk_cols == ["holder_id", "lane"]
    conn.close()


def test_migrate_leases_already_migrated_is_noop():
    conn = _conn()
    schema.ensure_schema(conn)  # creates composite-PK leases.
    cur = conn.cursor()
    assert schema._migrate_leases_v1_to_v2(cur) is False
    conn.close()


def test_migrate_leases_unknown_shape_left_alone():
    conn = _conn()
    cur = conn.cursor()
    # A leases table whose PK is neither (lane) nor (lane, holder_id).
    cur.execute("CREATE TABLE leases (task_id TEXT PRIMARY KEY, lane TEXT)")
    assert schema._migrate_leases_v1_to_v2(cur) is False
    conn.close()


def test_migrate_leases_pragma_operational_error():
    class _BoomCursor:
        def execute(self, *_a, **_k):
            raise sqlite3.OperationalError("pragma failed")

    assert schema._migrate_leases_v1_to_v2(_BoomCursor()) is False


def test_get_lane_capacity_known_and_default():
    conn = _conn()
    schema.ensure_schema(conn)
    assert schema.get_lane_capacity(conn, "benchmark_lane") == 1
    # An unknown lane with no row and no default -> defensive 1.
    assert schema.get_lane_capacity(conn, "no_such_lane") == 1
    conn.close()


def test_get_lane_capacity_missing_table_falls_back_to_default():
    conn = _conn()  # no ensure_schema -> lane_capacity table absent.
    assert schema.get_lane_capacity(conn, "research_lane") == 1
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
