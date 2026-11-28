# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SQLite schema for the unified Coordinator state DB (``$SESSION_DIR/storage/coordinator.db``)."""

from __future__ import annotations

import sqlite3

from hyperloom.inference_optimizer.session.coordinator_db import SCHEMA_VERSION, refuse_older_schema


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
    # events — A2A message bus
    """
    CREATE TABLE IF NOT EXISTS events (
        seq           INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_id        TEXT    NOT NULL UNIQUE,
        from_agent    TEXT    NOT NULL,
        to_agent      TEXT    NOT NULL,
        topic         TEXT    NOT NULL,
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
    # schema_version — tracks future migrations
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version    INTEGER PRIMARY KEY,
        applied_at TEXT    NOT NULL
    )
    """,
]


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotently create all tables and record the schema version."""
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        refuse_older_schema(cur)
        for stmt in _DDL:
            cur.execute(stmt)
        cur.execute(
            "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (?, datetime('now'))",
            (SCHEMA_VERSION,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
