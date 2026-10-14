# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The database behind Experience KBs: their identities, schemas, record index, state, sync, and writes.

A workspace's service keeps it in a SQLite file in its home. A global KB may keep it on a PostgreSQL server instead,
which any number of service processes then share. Every table is keyed by the KB it belongs to, so one database can
hold several KBs. Each write takes the next position of its KB under that KB's lock, so positions are committed in
order and a reader paging by position never passes one still to commit.

Callers write their SQL once, in what both databases speak, with psycopg's ``%s`` and ``%(name)s`` placeholders; a
SQLite connection translates the placeholders.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = 1
#: The SQLite file a service without a PostgreSQL server keeps under its home.
# Not main's kb.sqlite3, whose tables of the same names hold another shape: a home from main keeps that file unread.
SQLITE_FILE = "database.sqlite3"
_MIGRATION_LOCK = "hyperloom-kb:migrate"
_POOL_MAX_SIZE = 16
# Bounds one attempt to reach a server or a busy file, so one that does not answer fails instead of hanging.
_CONNECT_TIMEOUT_SECONDS = 10
_SQLITE_BUSY_SECONDS = 30

_DDL_V1 = """
CREATE TABLE kbs (
    kb_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_sequence BIGINT NOT NULL DEFAULT 0
);
CREATE TABLE schemas (
    kb_id TEXT NOT NULL REFERENCES kbs,
    schema_ref TEXT NOT NULL,
    declaration TEXT NOT NULL,
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
CREATE TABLE files (
    kb_id TEXT NOT NULL REFERENCES kbs,
    sha256 TEXT NOT NULL,
    bytes BIGINT NOT NULL,
    stored_at TEXT NOT NULL,
    PRIMARY KEY (kb_id, sha256)
);
CREATE TABLE writes (
    write_id {serial_key},
    kb_id TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    result TEXT NOT NULL,
    source_kb_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    request_id TEXT NOT NULL,
    received_at TEXT NOT NULL
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
    entry_id {serial_key},
    kb_id TEXT NOT NULL,
    schema_ref TEXT NOT NULL,
    experience_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE INDEX exclusion_history_by_schema ON exclusion_history (kb_id, schema_ref, experience_id);
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
    anchor TEXT NOT NULL DEFAULT '',
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
)
"""


class DatabaseError(RuntimeError):
    """Raised when the database cannot serve this service, such as one holding several KBs it cannot choose from."""


@dataclass(frozen=True)
class WriteSource:
    """Who a write came from: the KB that sent it, if it named itself, and the request that carried it."""

    kb_id: str = ""
    name: str = ""
    request_id: str = ""


class Cursor(Protocol):
    """A statement's result: its rows, each read by column name, and how many rows it changed."""

    rowcount: int

    def fetchone(self) -> Any: ...

    def fetchall(self) -> list[Any]: ...

    def __iter__(self) -> Iterator[Any]: ...


class Connection(Protocol):
    """One transaction's connection, in either database."""

    def execute(self, query: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> Cursor: ...

    def executemany(self, query: str, rows: Iterable[Sequence[Any]]) -> None: ...


class Database(ABC):
    """The database of one service, migrated to this code's tables when it opens."""

    #: The column type of a table's own increasing id.
    serial_key: str

    @abstractmethod
    def transaction(self, *, timeout: float | None = None) -> Any:
        """A context manager for one transaction, committed when its block finishes and rolled back when it raises.

        ``timeout`` bounds the wait for a connection; unset, the database's own applies.
        """

    @abstractmethod
    def exclusive(self, key: str) -> Any:
        """A context manager that holds ``key`` against every other holder of it sharing this database."""

    @abstractmethod
    def answers(self, timeout: float) -> bool:
        """Whether the database answers a query within ``timeout`` seconds."""

    @abstractmethod
    def size_bytes(self) -> int:
        """How large the database is."""

    def connection_stats(self) -> dict[str, int]:
        """What a connection pool reports of itself; empty for a database without one."""

        return {}

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def _hold_for_transaction(self, connection: Connection, key: str) -> None:
        """Hold ``key`` against other holders until ``connection``'s transaction ends."""

    def _migrate(self) -> None:
        with self.transaction() as connection:
            self._hold_for_transaction(connection, _MIGRATION_LOCK)
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
                for statement in _DDL_V1.format(serial_key=self.serial_key).split(";"):
                    if statement.strip():
                        connection.execute(statement)
                connection.execute("INSERT INTO kb_migrations(version, applied_at) VALUES (1, %s)", (utc_now(),))

    def resolve_kb(self) -> str:
        """The KB this database serves: its one KB, or a new one."""

        with self.transaction() as connection:
            self._hold_for_transaction(connection, _MIGRATION_LOCK)
            rows = connection.execute("SELECT kb_id FROM kbs ORDER BY created_at, kb_id").fetchall()
            if len(rows) > 1:
                raise DatabaseError(
                    f"the database holds {len(rows)} Experience KBs; serving one of several is not supported yet"
                )
            if rows:
                return str(rows[0]["kb_id"])
            kb_id = f"kb-{uuid.uuid4().hex}"
            connection.execute("INSERT INTO kbs(kb_id, created_at) VALUES (%s, %s)", (kb_id, utc_now()))
            return kb_id


_PLACEHOLDER_RE = re.compile(r"%\((\w+)\)s|%s")


def _sqlite_query(query: str) -> str:
    return _PLACEHOLDER_RE.sub(lambda match: f":{match.group(1)}" if match.group(1) else "?", query)


class _SqliteConnection:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def execute(self, query: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> Cursor:
        return self._connection.execute(_sqlite_query(query), params)

    def executemany(self, query: str, rows: Iterable[Sequence[Any]]) -> None:
        self._connection.executemany(_sqlite_query(query), rows)


# The locks of every SQLite database this process opened, by file and key: one service process serves a SQLite
# home, so holding a key within it holds it against every holder.
_SQLITE_LOCKS: dict[tuple[Path, str], threading.Lock] = {}
_SQLITE_LOCKS_GUARD = threading.Lock()


class SqliteDatabase(Database):
    """A database in one SQLite file, served by one service process.

    Each transaction takes the file's write lock when it begins, so transactions apply one after another; readers in
    other connections keep reading while one writes.
    """

    serial_key = "INTEGER PRIMARY KEY AUTOINCREMENT"

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect(_SQLITE_BUSY_SECONDS) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
            self._migrate()
        except (OSError, sqlite3.Error) as exc:
            raise DatabaseError(f"cannot open the Experience KB database {self.path}: {exc}") from exc

    @contextmanager
    def _connect(self, timeout: float) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=timeout, isolation_level=None, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, timeout: float | None = None) -> Iterator[Connection]:
        with self._connect(_SQLITE_BUSY_SECONDS if timeout is None else timeout) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield _SqliteConnection(connection)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    @contextmanager
    def exclusive(self, key: str) -> Iterator[None]:
        with _SQLITE_LOCKS_GUARD:
            lock = _SQLITE_LOCKS.setdefault((self.path, key), threading.Lock())
        with lock:
            yield

    def answers(self, timeout: float) -> bool:
        try:
            with self._connect(timeout) as connection:
                connection.execute("SELECT 1 FROM kbs LIMIT 1")
        except sqlite3.Error:
            return False
        return True

    def size_bytes(self) -> int:
        return sum(
            path.stat().st_size for path in (self.path, self.path.with_name(f"{self.path.name}-wal")) if path.exists()
        )

    def close(self) -> None:
        return None

    def _hold_for_transaction(self, connection: Connection, key: str) -> None:
        # Every SQLite transaction already holds the file's write lock.
        return None


class _PostgresConnection:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def execute(self, query: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> Cursor:
        return self._connection.execute(query, params)  # type: ignore[no-any-return]

    def executemany(self, query: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._connection.cursor() as cursor:
            cursor.executemany(query, rows)


class PostgresDatabase(Database):
    """A connection pool to one PostgreSQL database, which any number of service processes may share."""

    serial_key = "BIGSERIAL PRIMARY KEY"

    def __init__(self, conninfo: str, *, max_size: int = _POOL_MAX_SIZE) -> None:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.conninfo = conninfo
        # A connection is checked as it is handed out, so the first request after the server restarts or fails over
        # to a new primary gets a live one instead of failing on one the old server closed.
        self.pool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=max_size,
            kwargs={"row_factory": dict_row, "connect_timeout": _CONNECT_TIMEOUT_SECONDS},
            check=ConnectionPool.check_connection,
            open=False,
        )
        try:
            self.pool.open(wait=True, timeout=30)
            self._migrate()
        except (psycopg.Error, TimeoutError) as exc:
            self.pool.close()
            raise DatabaseError(f"cannot open the Experience KB database: {exc}") from exc

    @contextmanager
    def transaction(self, *, timeout: float | None = None) -> Iterator[Connection]:
        with self.pool.connection(timeout=timeout) as connection:
            yield _PostgresConnection(connection)

    def answers(self, timeout: float) -> bool:
        import psycopg
        from psycopg_pool import PoolTimeout

        try:
            with self.transaction(timeout=timeout) as connection:
                connection.execute("SELECT 1")
        except (psycopg.Error, PoolTimeout):
            return False
        return True

    @contextmanager
    def exclusive(self, key: str) -> Iterator[None]:
        with self.pool.connection() as connection:
            connection.autocommit = True
            connection.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (key,))
            try:
                yield
            finally:
                connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,))
                connection.autocommit = False

    def size_bytes(self) -> int:
        with self.transaction() as connection:
            row = connection.execute("SELECT pg_database_size(current_database()) AS size").fetchone()
        return int(row["size"]) if row else 0

    def connection_stats(self) -> dict[str, int]:
        return {stat: int(value) for stat, value in self.pool.get_stats().items()}

    def close(self) -> None:
        self.pool.close()

    def _hold_for_transaction(self, connection: Connection, key: str) -> None:
        connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (key,))


def open_database(home: Path, database_url: str = "") -> Database:
    """The database a service of ``home`` keeps: the PostgreSQL server ``database_url`` names, or a SQLite file in
    ``home``."""

    return PostgresDatabase(database_url) if database_url else SqliteDatabase(Path(home) / SQLITE_FILE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "SCHEMA_VERSION",
    "SQLITE_FILE",
    "Connection",
    "Cursor",
    "Database",
    "DatabaseError",
    "PostgresDatabase",
    "SqliteDatabase",
    "WriteSource",
    "open_database",
    "utc_now",
]
