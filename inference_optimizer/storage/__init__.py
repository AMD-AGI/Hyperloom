"""SQLite WAL atomic storage backend ().

Single ``$SESSION_DIR/storage/coordinator.db`` consolidates 4 tables:

* ``leases``  — resource lock state (one row per lane)
* ``events``  — A2A message bus source-of-truth (AUTOINCREMENT seq)
* ``cursors`` — per-agent ``last_processed_seq`` for idempotent replay
* ``tasks``   — DelegatedTask lifecycle state machine

WAL + ``BEGIN IMMEDIATE`` gives cross-table atomicity (ADR-33 promise).
"""

from .connection import SqliteConnection, open_connection
from .schema import (
    DEFAULT_LANE_CAPACITIES,
    SCHEMA_VERSION,
    ensure_schema,
    get_lane_capacity,
    reset_schema,
    set_lane_capacity,
)

__all__ = [
    "DEFAULT_LANE_CAPACITIES",
    "SCHEMA_VERSION",
    "SqliteConnection",
    "ensure_schema",
    "get_lane_capacity",
    "open_connection",
    "reset_schema",
    "set_lane_capacity",
]
