# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SQLite WAL atomic storage backend.

Single ``$SESSION_DIR/storage/coordinator.db`` consolidates these tables:

* ``leases``        — resource lock state (one row per lane/holder)
* ``lane_capacity`` — per-lane concurrency cap
* ``events``        — A2A message bus source-of-truth (AUTOINCREMENT seq)
* ``cursors``       — per-agent ``last_processed_seq`` for idempotent replay
* ``tasks``         — DelegatedTask lifecycle state machine
* ``gpu_leases``    — specialist GPU pool (separate from serving lanes)
* ``schema_version``— migration tracking

WAL + ``BEGIN IMMEDIATE`` gives cross-table atomicity.
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
