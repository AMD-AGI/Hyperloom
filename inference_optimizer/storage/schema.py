# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SQLite schema for the unified Coordinator state DB.

Seven tables consolidated into a single WAL database
``$SESSION_DIR/storage/coordinator.db``:

* ``leases``         — resource lock state.  v0.6 had PK ``lane``
                        (one holder per lane); v0.8 M6 widens PK to
                        ``(lane, holder_id)`` so multi-holder lanes
                        (e.g. ``research_lane`` with capacity > 1)
                        can keep one row per concurrent specialist.
                        See ``ensure_schema()`` for the in-place
                        migration applied when an older DB is opened.
* ``lane_capacity``  — capacity table (v0.8 M6, KB_design §3.7).  Per-lane
                        ``capacity`` int; defaults inserted at boot.
* ``events``         — A2A message bus source-of-truth, AUTOINCREMENT seq
* ``cursors``        — per-agent ``last_processed_seq`` for idempotent replay
* ``tasks``          — DelegatedTask lifecycle state machine
* ``gpu_leases``     — specialist GPU pool leases. Separate from serving
                       lanes so short GPU specialist experiments can be
                       capacity-limited without blocking benchmark/profile.

We deliberately *don't* declare FK constraints between ``tasks`` and
``leases`` / ``events`` because lifetimes don't match: a task can complete
and its leases be reaped before the events that reference it are pruned.
``leases.task_id`` and ``events.in_reply_to`` are advisory only.
"""

from __future__ import annotations

import sqlite3

# bump from 2 → 3 to add the specialist GPU pool leases table.
# v2 marked the leases PK widening +
# lane_capacity table introduction. ``ensure_schema`` migrates v1 DBs
# in place by recreating ``leases`` with the new PK; the migration is
# defensive against active sessions because BEGIN IMMEDIATE serialises
# the rebuild.
SCHEMA_VERSION = 3


# default lane capacities. ``research_lane`` defaults to 1
# (single-specialist) so a fresh DB without operator config still runs;
# the CLI flag ``--research-lane-capacity`` upgrades this row at session
# boot. Serving lanes stay at 1.
DEFAULT_LANE_CAPACITIES: dict[str, int] = {
    "server_lifecycle":   1,
    "workspace_mutation": 1,
    "benchmark_lane":     1,
    "profile_lane":       1,
    "research_lane":      1,
}


_DDL = [
    # ------------------------------------------------------------------
    # leases — Resource Lock Manager (DESIGN §3.5)
    # ------------------------------------------------------------------
    # PK is the composite ``(lane, holder_id)``. The capacity
    # cap per lane lives in ``lane_capacity``; ``acquire_many`` selects
    # the current holder count and rolls back when ``count >= capacity``.
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
    # ------------------------------------------------------------------
    # lane_capacity — per-lane concurrency cap
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS lane_capacity (
        lane     TEXT PRIMARY KEY,
        capacity INTEGER NOT NULL
    )
    """,
    # ------------------------------------------------------------------
    # events — A2A message bus (DESIGN §13.1)
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
    # cursors — idempotent message processing (DESIGN §17.3)
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
    # tasks — DelegatedTask state machine (DESIGN §17.4)
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
    # gpu_leases — specialist GPU pool (separate from serving lanes)
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
    # schema_version — tracks future migrations
    # ------------------------------------------------------------------
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
    """In-place upgrade of the legacy ``leases`` table to the legacy M6 schema.

    Returns ``True`` when a migration actually ran, ``False`` when the
    existing table already has the composite PK. Safe to call on an
    empty DB (no rows to move).

    Strategy:

    * detect the v1 shape via ``PRAGMA table_info(leases)`` — v1 has a
      single PK column ``lane``; v2 has the composite PK
      ``(lane, holder_id)``.
    * snapshot the v1 rows into Python (small table — at most one row
      per lane), drop & recreate ``leases`` with the new shape, then
      re-insert. The whole transition runs inside the caller's
      ``BEGIN IMMEDIATE`` so a concurrent reader either sees the v1
      table or the v2 table, never a half-formed mix.

    Defensive: rows with ``expires_at`` already in the past are
    *dropped* during migration — they would just be reaped on the
    first ``reap_expired`` anyway, and dropping them avoids carrying
    stale state across a process restart.
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
        # Unknown shape — leave it alone (operator probably hand-edited).
        return False
    # Snapshot rows; the table is small.
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
    """Idempotently insert default capacity rows for known lanes.

    Existing rows are left alone so a session resume preserves the
    capacity the operator chose. Coordinator boot can override on
    top via :func:`set_lane_capacity`.
    """
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
    """Idempotently create all tables; record the current schema version.

    also runs the v1 → v2 ``leases`` PK widening when needed
    and seeds ``lane_capacity`` defaults. The whole sequence runs in
    one transaction so a concurrent reader never observes an
    intermediate state (a reader either sees the v1 schema or the v2
    schema, never half of each).
    """
    cur = conn.cursor()
    try:
        cur.execute("BEGIN IMMEDIATE")
        # Ensure schema_version exists first so subsequent migrations
        # have somewhere to record their pass.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version    INTEGER PRIMARY KEY,
                applied_at TEXT    NOT NULL
            )
            """
        )
        # Bring v1 ``leases`` up to the v2 composite PK before the
        # _DDL pass tries to ``CREATE TABLE IF NOT EXISTS leases`` —
        # the IF NOT EXISTS guard would otherwise prevent the new
        # constraint from landing on legacy DBs.
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
