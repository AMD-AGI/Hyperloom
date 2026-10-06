# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The PostgreSQL database behind Experience KBs: their identities, schemas, record index, state, sync, and writes.

Every table is keyed by the KB it belongs to, so one database can hold several KBs. Any number of service processes
may share a database: each write takes the next position of its KB under that KB's row lock, so positions are
committed in order and a reader paging by position never passes one still to commit.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SCHEMA_VERSION = 1
_MIGRATION_LOCK = "hyperloom-kb:migrate"
_POOL_MAX_SIZE = 16

_DDL_V1 = """
CREATE TABLE kbs (
    kb_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_sequence BIGINT NOT NULL DEFAULT 0,
    adopted_home TEXT NOT NULL DEFAULT ''
);
CREATE TABLE schemas (
    kb_id TEXT NOT NULL REFERENCES kbs,
    schema_ref TEXT NOT NULL,
    declaration JSONB NOT NULL,
    version BIGINT NOT NULL DEFAULT 0,
    registered_at TEXT NOT NULL,
    PRIMARY KEY (kb_id, schema_ref)
);
CREATE TABLE experiences (
    kb_id TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    sequence BIGINT NOT NULL,
    content_hash TEXT NOT NULL,
    bytes BIGINT NOT NULL,
    stored_at TEXT NOT NULL,
    PRIMARY KEY (kb_id, experience_id),
    UNIQUE (kb_id, sequence),
    FOREIGN KEY (kb_id, schema_ref) REFERENCES schemas
);
CREATE INDEX experiences_by_schema ON experiences (kb_id, schema_ref, sequence);
CREATE TABLE writes (
    kb_id TEXT NOT NULL,
    write_id BIGSERIAL,
    experience_id TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    result TEXT NOT NULL,
    source_kb_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    request_id TEXT NOT NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY (kb_id, write_id)
);
CREATE INDEX writes_by_source ON writes (kb_id, source_kb_id, received_at);
CREATE TABLE outside (
    kb_id TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    PRIMARY KEY (kb_id, schema_ref, experience_id)
);
CREATE TABLE exclusions (
    kb_id TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    excluded_at TEXT NOT NULL,
    PRIMARY KEY (kb_id, schema_ref, experience_id)
);
CREATE TABLE exclusion_history (
    kb_id TEXT NOT NULL,
    entry_id BIGSERIAL,
    schema_ref TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    at TEXT NOT NULL,
    PRIMARY KEY (kb_id, entry_id)
);
CREATE TABLE labels (
    kb_id TEXT NOT NULL,
    label_id TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    name TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY (kb_id, label_id)
);
CREATE TABLE label_members (
    kb_id TEXT NOT NULL,
    label_id TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    PRIMARY KEY (kb_id, label_id, experience_id)
);
CREATE TABLE label_exclusions (
    kb_id TEXT NOT NULL,
    label_id TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    excluded_at TEXT NOT NULL,
    PRIMARY KEY (kb_id, label_id, experience_id)
);
CREATE TABLE current_labels (
    kb_id TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    label_id TEXT NOT NULL,
    PRIMARY KEY (kb_id, schema_ref)
);
CREATE TABLE sync_cursors (
    kb_id TEXT NOT NULL,
    global_url TEXT NOT NULL,
    direction TEXT NOT NULL,
    position BIGINT NOT NULL,
    PRIMARY KEY (kb_id, global_url, direction)
);
CREATE TABLE sync_pulled (
    kb_id TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    PRIMARY KEY (kb_id, experience_id)
);
CREATE TABLE sync_held_back (
    kb_id TEXT NOT NULL,
    global_url TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    PRIMARY KEY (kb_id, global_url, experience_id)
);
CREATE TABLE sync_identities (
    kb_id TEXT NOT NULL,
    global_url TEXT NOT NULL,
    remote_kb_id TEXT NOT NULL,
    PRIMARY KEY (kb_id, global_url)
);
CREATE TABLE sync_on_global (
    kb_id TEXT NOT NULL,
    global_url TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    PRIMARY KEY (kb_id, global_url, schema_ref, experience_id)
);
CREATE TABLE sync_pulls_in_progress (
    kb_id TEXT NOT NULL,
    global_url TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    PRIMARY KEY (kb_id, global_url, schema_ref)
);
CREATE TABLE sync_pulled_states (
    kb_id TEXT NOT NULL,
    global_url TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    state TEXT NOT NULL,
    PRIMARY KEY (kb_id, global_url, schema_ref)
);
"""


class DatabaseError(RuntimeError):
    """Raised when the database cannot serve this service, such as one holding several KBs it cannot choose from."""


@dataclass(frozen=True)
class WriteSource:
    """Who a write came from: the KB that sent it, if it named itself, and the request that carried it."""

    kb_id: str = ""
    name: str = ""
    request_id: str = ""


class Database:
    """A connection pool to one database, migrated to this code's tables when it opens."""

    def __init__(self, conninfo: str, *, max_size: int = _POOL_MAX_SIZE) -> None:
        self.pool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        try:
            self.pool.open(wait=True, timeout=30)
            self._migrate()
        except (psycopg.Error, TimeoutError) as exc:
            self.pool.close()
            raise DatabaseError(f"cannot open the Experience KB database: {exc}") from exc

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        """One transaction, committed when the block finishes and rolled back when it raises."""

        with self.pool.connection() as connection:
            yield connection

    @contextmanager
    def exclusive(self, key: str) -> Iterator[None]:
        """Hold ``key`` against every other holder of it, in this process or any other sharing the database."""

        with self.pool.connection() as connection:
            connection.autocommit = True
            connection.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (key,))
            try:
                yield
            finally:
                connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,))
                connection.autocommit = False

    def close(self) -> None:
        self.pool.close()

    def _migrate(self) -> None:
        with self.transaction() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (_MIGRATION_LOCK,))
            connection.execute(
                "CREATE TABLE IF NOT EXISTS kb_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            row = connection.execute("SELECT COALESCE(MAX(version), 0) AS version FROM kb_migrations").fetchone()
            applied = int(row["version"]) if row else 0
            if applied > SCHEMA_VERSION:
                raise DatabaseError(
                    f"the database is at schema version {applied}, newer than this code's {SCHEMA_VERSION}"
                )
            if applied < 1:
                connection.execute(_DDL_V1)
                connection.execute("INSERT INTO kb_migrations(version, applied_at) VALUES (1, %s)", (utc_now(),))

    def resolve_kb(self, *, adopt_kb_id: str = "") -> str:
        """The KB this database serves: its one KB, or a new one, ``adopt_kb_id`` when an older home names it."""

        with self.transaction() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (_MIGRATION_LOCK,))
            rows = connection.execute("SELECT kb_id FROM kbs ORDER BY created_at, kb_id").fetchall()
            if len(rows) > 1:
                raise DatabaseError(
                    f"the database holds {len(rows)} Experience KBs; serving one of several is not supported yet"
                )
            if rows:
                return str(rows[0]["kb_id"])
            kb_id = adopt_kb_id or f"kb-{uuid.uuid4().hex}"
            connection.execute("INSERT INTO kbs(kb_id, created_at) VALUES (%s, %s)", (kb_id, utc_now()))
            return kb_id


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = ["SCHEMA_VERSION", "Database", "DatabaseError", "WriteSource", "utc_now"]
