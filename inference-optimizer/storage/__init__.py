"""SQLite WAL atomic storage backend (DESIGN v0.6 §3.5 / §13.1 / ADR-42).

Single ``$SESSION_DIR/storage/coordinator.db`` consolidates 4 tables:

* ``leases``  — resource lock state (one row per lane)
* ``events``  — A2A message bus source-of-truth (AUTOINCREMENT seq)
* ``cursors`` — per-agent ``last_processed_seq`` for idempotent replay
* ``tasks``   — DelegatedTask lifecycle state machine

WAL + ``BEGIN IMMEDIATE`` gives cross-table atomicity (ADR-33 promise).
"""

from .connection import SqliteConnection, open_connection
from .schema import SCHEMA_VERSION, ensure_schema, reset_schema

__all__ = [
    "SCHEMA_VERSION",
    "SqliteConnection",
    "ensure_schema",
    "open_connection",
    "reset_schema",
]
