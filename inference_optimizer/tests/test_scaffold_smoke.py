# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P0-0 scaffold smoke tests.

Verifies:

* Package importable + ``__version__`` set to v0.6.0
* ``paths.make_session_dir`` creates all standard subdirs under env override
* ``paths.db_path_for`` resolves to ``storage/coordinator.db`` under session
* ``storage.SqliteConnection`` opens DB with WAL pragmas + managed tables created
* Cross-table ``BEGIN IMMEDIATE`` transaction round-trips events + cursors
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import inference_optimizer
from inference_optimizer import paths
from inference_optimizer.storage import (
    SCHEMA_VERSION,
    SqliteConnection,
    ensure_schema,
    open_connection,
    reset_schema,
)


# ---------------------------------------------------------------------------
# package metadata
# ---------------------------------------------------------------------------
def test_package_version_is_v06():
    assert inference_optimizer.__version__ == "0.6.0"


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def test_make_session_dir_creates_all_subdirs(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    assert sd == tmp_path
    for sub in paths._SESSION_SKELETON:
        assert (sd / sub).is_dir(), f"missing subdir: {sub}"


def test_db_path_for_default_under_session(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    assert paths.db_path_for(sd) == sd / "storage" / "coordinator.db"


def test_asset_root_defaults_to_package(monkeypatch):
    monkeypatch.delenv(paths.ENV_OVERRIDE_ASSET_ROOT, raising=False)
    assert paths.asset_root() == paths.PACKAGE_ROOT


def test_asset_root_override_missing_raises(tmp_path, monkeypatch):
    bogus = tmp_path / "nope"
    monkeypatch.setenv(paths.ENV_OVERRIDE_ASSET_ROOT, str(bogus))
    with pytest.raises(paths.AssetRootNotFound):
        paths.asset_root()


def test_agent_session_dir_returns_path(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    ad = paths.agent_session_dir(sd, "orchestration")
    assert ad == sd / "agents" / "orchestration"
    # make_session_dir() pre-creates the lower-cased role subdirs.
    assert ad.is_dir()


# ---------------------------------------------------------------------------
# storage / schema
# ---------------------------------------------------------------------------
def _new_db(tmp_path) -> Path:
    return tmp_path / "test.db"


def test_open_connection_applies_wal_and_schema(tmp_path):
    db = _new_db(tmp_path)
    conn = open_connection(db)
    try:
        # WAL pragma actually applied
        cur = conn.execute("PRAGMA journal_mode")
        (mode,) = cur.fetchone()
        assert mode.lower() == "wal"
        # core tables + schema_version exist
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cur.fetchall()}
        for required in ("leases", "events", "cursors", "tasks", "schema_version"):
            assert required in tables, f"missing table: {required}"
        # schema_version row exists
        (version,) = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert version == SCHEMA_VERSION
    finally:
        conn.close()


def test_ensure_schema_is_idempotent(tmp_path):
    db = _new_db(tmp_path)
    conn = open_connection(db)
    try:
        v1 = ensure_schema(conn)
        v2 = ensure_schema(conn)
        assert v1 == v2 == SCHEMA_VERSION
    finally:
        conn.close()


def test_reset_schema_drops_and_recreates(tmp_path):
    db = _new_db(tmp_path)
    conn = open_connection(db)
    try:
        conn.execute(
            "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
            "acquired_at, expires_at, heartbeat_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("benchmark_lane", "h1", "t1", "bench_runner", 1, "t", "t", "t"),
        )
        conn.commit()
        (n,) = conn.execute("SELECT COUNT(*) FROM leases").fetchone()
        assert n == 1
        reset_schema(conn)
        (n,) = conn.execute("SELECT COUNT(*) FROM leases").fetchone()
        assert n == 0
    finally:
        conn.close()


def test_sqlite_connection_sync_round_trip(tmp_path):
    db = _new_db(tmp_path)
    sc = SqliteConnection(db)
    try:
        with sc.transaction_sync() as cur:
            cur.execute(
                "INSERT INTO events(msg_id, from_agent, to_agent, topic, "
                "payload, priority, ts) VALUES (?,?,?,?,?,?,?)",
                ("m1", "Orchestration", "Kernel", "request", "{}", 1, "t"),
            )
        rows = sc.fetchall_sync("SELECT msg_id, topic FROM events")
        assert len(rows) == 1
        assert rows[0]["msg_id"] == "m1"
        assert rows[0]["topic"] == "request"
    finally:
        sc.close()


@pytest.mark.asyncio
async def test_sqlite_connection_async_transaction_atomic(tmp_path):
    """ADR-42: one BEGIN IMMEDIATE txn covers events + cursors atomically."""
    db = _new_db(tmp_path)
    sc = SqliteConnection(db)
    try:
        async with sc.transaction() as cur:
            cur.execute(
                "INSERT INTO events(msg_id, from_agent, to_agent, topic, "
                "payload, priority, ts) VALUES (?,?,?,?,?,?,?)",
                ("m-tx", "Critic", "Orchestration", "review_verdict", "{}", 1, "t"),
            )
            cur.execute(
                "INSERT INTO cursors(agent, last_processed_seq, "
                "last_processed_msg_id, processed_at) VALUES (?,?,?,?)",
                ("Orchestration", 0, "", "t"),
            )
        ev = await sc.fetchall("SELECT msg_id FROM events")
        cu = await sc.fetchall("SELECT agent FROM cursors")
        assert len(ev) == 1 and ev[0]["msg_id"] == "m-tx"
        assert len(cu) == 1 and cu[0]["agent"] == "Orchestration"
    finally:
        sc.close()


@pytest.mark.asyncio
async def test_sqlite_connection_transaction_rollback_on_error(tmp_path):
    db = _new_db(tmp_path)
    sc = SqliteConnection(db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            async with sc.transaction() as cur:
                cur.execute(
                    "INSERT INTO events(msg_id, from_agent, to_agent, topic, "
                    "payload, priority, ts) VALUES (?,?,?,?,?,?,?)",
                    ("dup", "A", "B", "t", "{}", 0, "t"),
                )
                cur.execute(
                    "INSERT INTO events(msg_id, from_agent, to_agent, topic, "
                    "payload, priority, ts) VALUES (?,?,?,?,?,?,?)",
                    ("dup", "A", "B", "t", "{}", 0, "t"),
                )
        rows = await sc.fetchall("SELECT msg_id FROM events")
        assert rows == []
    finally:
        sc.close()
