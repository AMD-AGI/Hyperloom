# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SQLite schema for the unified Coordinator state DB (``$SESSION_DIR/storage/coordinator.db``)."""

from __future__ import annotations

import sqlite3

from hyperloom.inference_optimizer.session.coordinator_db import SCHEMA_VERSION, refuse_older_schema


# Default lane capacities; ``--research-lane-capacity`` overrides research_lane at boot.
DEFAULT_LANE_CAPACITIES: dict[str, int] = {
    "server_lifecycle": 1,
    "workspace_mutation": 1,
    "benchmark_lane": 1,
    "profile_lane": 1,
    "research_lane": 1,
    "gpu_research_lane": 1,
    "build_lane": 1,
}


_DDL = [
    # leases — lane ownership. Composite PK (lane, holder_id).
    """
    CREATE TABLE IF NOT EXISTS leases (
        lane          TEXT    NOT NULL,
        holder_id     TEXT    NOT NULL,
        task_id       TEXT    NOT NULL,
        action        TEXT    NOT NULL,
        pid           INTEGER NOT NULL,
        owner_scope   TEXT    NOT NULL DEFAULT '',
        acquired_at   TEXT    NOT NULL,
        expires_at    TEXT    NOT NULL,
        PRIMARY KEY (lane, holder_id)
    )
    """,
    # lane_capacity — per-lane concurrency cap
    """
    CREATE TABLE IF NOT EXISTS lane_capacity (
        lane     TEXT PRIMARY KEY,
        capacity INTEGER NOT NULL
    )
    """,
    # events — A2A message bus
    """
    CREATE TABLE IF NOT EXISTS events (
        seq           INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_id        TEXT    NOT NULL UNIQUE,
        from_agent    TEXT    NOT NULL,
        to_agent      TEXT    NOT NULL,
        topic         TEXT    NOT NULL,
        in_reply_to   TEXT,
        payload       TEXT    NOT NULL,
        ts            TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_to_agent ON events(to_agent, seq)",
    "CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic, seq)",
    # cursors — idempotent message processing
    """
    CREATE TABLE IF NOT EXISTS cursors (
        agent              TEXT PRIMARY KEY,
        last_processed_seq INTEGER NOT NULL
    )
    """,
    # tasks — DelegatedTask state machine
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id          TEXT PRIMARY KEY,
        kind             TEXT NOT NULL,
        state            TEXT NOT NULL CHECK (state IN
                           ('queued','running','succeeded','failed',
                            'cancelled')),
        params           TEXT NOT NULL,
        idempotency_key  TEXT NOT NULL UNIQUE,
        requires_lanes   TEXT NOT NULL DEFAULT '[]',
        lease_ttl_sec    INTEGER NOT NULL DEFAULT 0,
        history          TEXT NOT NULL DEFAULT '[]',
        created_at       TEXT NOT NULL,
        updated_at       TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state, updated_at)",
    # gpu_leases — specialist GPU pool (separate from serving lanes)
    """
    CREATE TABLE IF NOT EXISTS gpu_leases (
        gpu_id    INTEGER PRIMARY KEY,
        holder_id TEXT    NOT NULL,
        task_id   TEXT    NOT NULL
    )
    """,
    # bringup_rounds — the durable mutex deciding whether another round may
    # start. Only explicit settlement ends exclusion; timestamps describe budgets.
    """
    CREATE TABLE IF NOT EXISTS bringup_rounds (
        round_id             TEXT    PRIMARY KEY,
        state                TEXT    NOT NULL CHECK (state IN ('open','settled')),
        outcome              TEXT    NOT NULL DEFAULT '',
        holder_task_id       TEXT    NOT NULL,
        fence                INTEGER NOT NULL DEFAULT 1,
        opened_unix          REAL    NOT NULL,
        renewed_unix         REAL    NOT NULL,
        expires_unix         REAL    NOT NULL,
        settled_unix         REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_bringup_rounds_state ON bringup_rounds(state, opened_unix)",
    # round_events — append-only audit trail. Every attempt lands here with
    # its outcome, evidence and request id, applied or rejected.
    """
    CREATE TABLE IF NOT EXISTS round_events (
        event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        round_id      TEXT    NOT NULL,
        request_id    TEXT    NOT NULL,
        op            TEXT    NOT NULL,
        result        TEXT    NOT NULL CHECK (result IN ('applied','rejected','duplicate')),
        outcome       TEXT    NOT NULL DEFAULT '',
        fence         INTEGER NOT NULL DEFAULT 0,
        actor_task_id TEXT    NOT NULL DEFAULT '',
        reason        TEXT    NOT NULL DEFAULT '',
        evidence      TEXT    NOT NULL DEFAULT '{}',
        recorded_unix REAL    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_round_events_round ON round_events(round_id, event_id)",
    # schema_version — tracks future migrations
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version    INTEGER PRIMARY KEY,
        applied_at TEXT    NOT NULL
    )
    """,
]


def _seed_default_lane_capacity(cur: sqlite3.Cursor) -> None:
    """Idempotently insert default capacity rows; existing rows are left alone so a resume preserves the operator's choice."""
    for lane, capacity in DEFAULT_LANE_CAPACITIES.items():
        cur.execute(
            "INSERT OR IGNORE INTO lane_capacity(lane, capacity) VALUES (?, ?)",
            (lane, int(capacity)),
        )


def set_lane_capacity(
    conn: sqlite3.Connection,
    lane: str,
    capacity: int,
) -> None:
    """Upsert one ``lane_capacity`` row."""
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        cur.execute(
            "INSERT INTO lane_capacity(lane, capacity) VALUES (?, ?) "
            "ON CONFLICT(lane) DO UPDATE SET capacity = excluded.capacity",
            (str(lane), int(capacity)),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def ensure_schema(conn: sqlite3.Connection) -> int:
    """Idempotently create all tables, seed lane_capacity defaults, and record the schema version."""
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        refuse_older_schema(cur)
        for stmt in _DDL:
            cur.execute(stmt)
        _seed_default_lane_capacity(cur)
        cur.execute(
            "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (?, datetime('now'))",
            (SCHEMA_VERSION,),
        )
        cur.execute("SELECT MAX(version) FROM schema_version")
        (current,) = cur.fetchone()
        conn.commit()
        return int(current or 0)
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
