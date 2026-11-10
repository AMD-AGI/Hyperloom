# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Layout version of a session's ``storage/coordinator.db`` and the refusal of older databases."""

from __future__ import annotations

import sqlite3

# There are no in-place migrations: a database recorded at an older version is never opened.
SCHEMA_VERSION = 8


class OlderCoordinatorDB(RuntimeError):
    """The session's coordinator.db records a schema version older than this build's."""


def refuse_older_schema(db: sqlite3.Connection | sqlite3.Cursor) -> None:
    """Raise :class:`OlderCoordinatorDB` when ``db`` already records a version below :data:`SCHEMA_VERSION`."""
    if db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'").fetchone() is None:
        return
    (stored,) = db.execute("SELECT MAX(version) FROM schema_version").fetchone()
    if stored is not None and int(stored) < SCHEMA_VERSION:
        raise OlderCoordinatorDB(
            f"coordinator.db is at schema version {stored}, older than this build's {SCHEMA_VERSION}: "
            "sessions created before this release cannot be resumed. Finish the session under the build "
            "that started it, or start a new session."
        )


__all__ = ["SCHEMA_VERSION", "OlderCoordinatorDB", "refuse_older_schema"]
