"""SQLite WAL atomic storage backend (DESIGN v0.5 §3.5 / §13)."""

from .connection import SqliteConnection, open_connection
from .schema import SCHEMA_VERSION, ensure_schema, reset_schema

__all__ = [
    "SqliteConnection",
    "open_connection",
    "SCHEMA_VERSION",
    "ensure_schema",
    "reset_schema",
]
