# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SQLite schema for the unified Coordinator state DB
(``$SESSION_DIR/storage/coordinator.db``).

Tables: ``leases`` (composite PK ``(lane, holder_id)`` for multi-holder
lanes), ``lane_capacity``, ``events`` (A2A bus), ``cursors`` (idempotent
replay), ``tasks`` (lifecycle state machine), ``gpu_leases`` (specialist GPU
pool, separate from serving lanes).

No FK constraints between ``tasks`` and ``leases``/``events``: lifetimes
differ (a task's leases may be reaped before its events are pruned), so
``leases.task_id`` / ``events.in_reply_to`` are advisory only.
"""

from __future__ import annotations

import sqlite3

# v3 added gpu_leases; v2 widened the leases PK + added lane_capacity.
# ensure_schema migrates v1 DBs in place under BEGIN IMMEDIATE.
SCHEMA_VERSION = 3


# Default lane capacities; ``--research-lane-capacity`` overrides research_lane
# at boot. A fresh DB runs with research_lane=1 (single specialist).
DEFAULT_LANE_CAPACITIES: dict[str, int] = {
    "server_lifecycle":   1,
    "workspace_mutation": 1,
    "benchmark_lane":     1,
    "profile_lane":       1,
    "research_lane":      1,
}


_DDL = [
    # leases — Resource Lock Manager (DESIGN §3.5). Composite PK
    # (lane, holder_id); per-lane cap in lane_capacity (acquire_many
    # rolls back when count >= capacity).
    """
    CREATE TABLE IF NOT EXISTS leases (
        lane          TEXT    NOT NULL,
        holder_id     TEXT    NOT NULL,
        task_id       TEXT    NOT NULL,
        action        TEXT    NOT NULL,
        pid           INTEGER NOT NULL,
        acquired_at   TEXT    NOT NULL,
        expires_at    TEXT    NOT NULL,
        heartbeat_at  TEXT    NOT NULL,
        PRIMARY KEY (lane, holder_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_leases_expires ON leases(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_leases_lane ON leases(lane)",
    # lane_capacity — per-lane concurrency cap
    """
    CREATE TABLE IF NOT EXISTS lane_capacity (
        lane     TEXT PRIMARY KEY,
        capacity INTEGER NOT NULL
    )
    """,
    # events — A2A message bus (DESIGN §13.1)
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
    # cursors — idempotent message processing (DESIGN §17.3)
    """
    CREATE TABLE IF NOT EXISTS cursors (
        agent                 TEXT PRIMARY KEY,
        last_processed_seq    INTEGER NOT NULL,
        last_processed_msg_id TEXT    NOT NULL,
        processed_at          TEXT    NOT NULL
    )
    """,
    # tasks — DelegatedTask state machine (DESIGN §17.4)
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
    # gpu_leases — specialist GPU pool (separate from serving lanes)
    """
    CREATE TABLE IF NOT EXISTS gpu_leases (
        gpu_id       INTEGER PRIMARY KEY,
        holder_id    TEXT    NOT NULL,
        task_id      TEXT    NOT NULL,
        acquired_at  TEXT    NOT NULL,
        expires_at   TEXT    NOT NULL,
        heartbeat_at TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_gpu_leases_expires ON gpu_leases(expires_at)",
    # schema_version — tracks future migrations
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version    INTEGER PRIMARY KEY,
        applied_at TEXT    NOT NULL
    )
    """,
]


_MANAGED_TABLES = (
    "leases", "lane_capacity", "gpu_leases", "events", "cursors",
    "tasks", "schema_version",
)


def _migrate_leases_v1_to_v2(cur: sqlite3.Cursor) -> bool:
    """In-place widen of the legacy ``leases`` PK (``lane`` -> composite
    ``(lane, holder_id)``). Returns True when a migration ran, False when
    already migrated / unknown shape. Snapshots rows, recreates the table,
    re-inserts; runs inside the caller's BEGIN IMMEDIATE.
    """
    try:
        cur.execute("PRAGMA table_info(leases)")
    except sqlite3.OperationalError:
        return False
    info = cur.fetchall()
    if not info:
        return False
    pk_cols = sorted(
        (row[1] for row in info if int(row[5] or 0) > 0)
    )
    if pk_cols == ["holder_id", "lane"]:
        return False  # already migrated
    if pk_cols != ["lane"]:
        return False  # unknown shape — leave it alone
    cur.execute(
        "SELECT lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at FROM leases"
    )
    rows = cur.fetchall()
    cur.execute("DROP TABLE leases")
    cur.execute(
        """
        CREATE TABLE leases (
            lane          TEXT    NOT NULL,
            holder_id     TEXT    NOT NULL,
            task_id       TEXT    NOT NULL,
            action        TEXT    NOT NULL,
            pid           INTEGER NOT NULL,
            acquired_at   TEXT    NOT NULL,
            expires_at    TEXT    NOT NULL,
            heartbeat_at  TEXT    NOT NULL,
            PRIMARY KEY (lane, holder_id)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_leases_expires ON leases(expires_at)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_leases_lane ON leases(lane)"
    )
    for r in rows:
        cur.execute(
            "INSERT OR REPLACE INTO leases(lane, holder_id, task_id, "
            "action, pid, acquired_at, expires_at, heartbeat_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            tuple(r),
        )
    return True


def _seed_default_lane_capacity(cur: sqlite3.Cursor) -> None:
    """Idempotently insert default capacity rows; existing rows are left
    alone so a resume preserves the operator's choice."""
    for lane, capacity in DEFAULT_LANE_CAPACITIES.items():
        cur.execute(
            "INSERT OR IGNORE INTO lane_capacity(lane, capacity) "
            "VALUES (?, ?)",
            (lane, int(capacity)),
        )


def set_lane_capacity(
    conn: sqlite3.Connection, lane: str, capacity: int,
) -> None:
    """Upsert one ``lane_capacity`` row. Called by the CLI / Coordinator
    boot path once :data:`SharedState.research_lane_capacity` is known."""
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


def get_lane_capacity(conn: sqlite3.Connection, lane: str) -> int:
    """Return capacity for ``lane``, falling back to
    :data:`DEFAULT_LANE_CAPACITIES`. Returns ``1`` for unknown lanes
    (defensive; ``ensure_schema`` already seeds every KNOWN_LANES
    member)."""
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT capacity FROM lane_capacity WHERE lane = ?", (str(lane),),
        )
        row = cur.fetchone()
        if row is not None:
            return int(row[0])
    except sqlite3.OperationalError:
        pass
    finally:
        cur.close()
    return int(DEFAULT_LANE_CAPACITIES.get(lane, 1))


def ensure_schema(conn: sqlite3.Connection) -> int:
    """Idempotently create all tables, run the leases PK migration, seed
    lane_capacity defaults, and record the schema version. Single
    transaction so readers never see an intermediate schema.
    """
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        # schema_version first so migrations have somewhere to record.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version    INTEGER PRIMARY KEY,
                applied_at TEXT    NOT NULL
            )
            """
        )
        # Migrate before the _DDL CREATE IF NOT EXISTS, which would
        # otherwise keep the legacy PK on existing DBs.
        _migrate_leases_v1_to_v2(cur)
        for stmt in _DDL:
            cur.execute(stmt)
        _seed_default_lane_capacity(cur)
        cur.execute(
            "INSERT OR IGNORE INTO schema_version(version, applied_at) "
            "VALUES (?, datetime('now'))",
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


def reset_schema(conn: sqlite3.Connection) -> None:
    """Drop and recreate every managed table. Test-only convenience."""
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        for table in _MANAGED_TABLES:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
    finally:
        cur.close()
    ensure_schema(conn)
