"""SQLite schema for the unified Conductor state DB (DESIGN §13.2 / ADR-33).

Four tables consolidated into a single WAL database
``$SESSION_DIR/storage/conductor.db``:

- ``leases``  — resource lock state (one row per lane, PK = lane name).
- ``events``  — A2A message bus source-of-truth, AUTOINCREMENT seq.
- ``cursors`` — per-agent ``last_processed_seq`` for idempotent replay.
- ``tasks``   — DelegatedTask lifecycle state machine.

We deliberately *don't* declare FK constraints between ``tasks`` and
``leases`` / ``events`` because lifetimes don't match: a task can complete
and its leases be reaped before the events that reference it are pruned.
The ``leases.task_id`` and ``events.in_reply_to`` columns are advisory.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

# NOTE: keep PRAGMAs in connection.py (per-connection); DDL only here.

_DDL = [
    # ------------------------------------------------------------------
    # leases — resource lock manager (DESIGN §3.5)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS leases (
        lane          TEXT PRIMARY KEY,
        holder_id     TEXT    NOT NULL,
        task_id       TEXT    NOT NULL,
        action        TEXT    NOT NULL,
        pid           INTEGER NOT NULL,
        acquired_at   TEXT    NOT NULL,
        expires_at    TEXT    NOT NULL,
        heartbeat_at  TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_leases_expires ON leases(expires_at)",
    # ------------------------------------------------------------------
    # events — A2A message bus (DESIGN §10.1)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS events (
        seq           INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_id        TEXT    NOT NULL UNIQUE,
        from_agent    TEXT    NOT NULL,
        to_agent      TEXT    NOT NULL,
        topic         TEXT    NOT NULL,
        in_reply_to   TEXT,
        payload       TEXT    NOT NULL,
        priority      INTEGER NOT NULL,
        ts            TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_to_agent ON events(to_agent, seq)",
    "CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic, seq)",
    # ------------------------------------------------------------------
    # cursors — idempotent message processing (DESIGN §13.3)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS cursors (
        agent                 TEXT PRIMARY KEY,
        last_processed_seq    INTEGER NOT NULL,
        last_processed_msg_id TEXT    NOT NULL,
        processed_at          TEXT    NOT NULL
    )
    """,
    # ------------------------------------------------------------------
    # tasks — DelegatedTask state machine (DESIGN §13.4)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id          TEXT PRIMARY KEY,
        kind             TEXT NOT NULL,
        state            TEXT NOT NULL CHECK (state IN
                           ('queued','running','succeeded','failed',
                            'cancelled','needs_manual_review')),
        params           TEXT NOT NULL,
        idempotency_key  TEXT NOT NULL UNIQUE,
        requires_lanes   TEXT NOT NULL DEFAULT '[]',
        allowed_tools    TEXT NOT NULL DEFAULT '[]',
        side_effects     TEXT NOT NULL DEFAULT '[]',
        lease_ttl_sec    INTEGER NOT NULL DEFAULT 0,
        attempts         INTEGER NOT NULL DEFAULT 0,
        history          TEXT NOT NULL DEFAULT '[]',
        created_at       TEXT NOT NULL,
        updated_at       TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state, updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_idem ON tasks(idempotency_key)",
    # ------------------------------------------------------------------
    # schema_version — for future migrations
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
]


def ensure_schema(conn: sqlite3.Connection) -> int:
    """Idempotently create all tables; record the schema version."""
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        for stmt in _DDL:
            cur.execute(stmt)
        cur.execute(
            "INSERT OR IGNORE INTO schema_version(version, applied_at) "
            "VALUES (?, datetime('now'))",
            (SCHEMA_VERSION,),
        )
        cur.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        (current,) = cur.fetchone()
        conn.commit()
        return int(current or 0)
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def reset_schema(conn: sqlite3.Connection) -> None:
    """Drop and recreate every managed table. Test-only convenience."""
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        for table in ("leases", "events", "cursors", "tasks", "schema_version"):
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
    finally:
        cur.close()
    ensure_schema(conn)
