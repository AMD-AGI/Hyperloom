"""SQLite connection wrapper ().

Design choices:

* Pure stdlib ``sqlite3`` — no extra dependency
* Mode = WAL with ``synchronous=FULL`` and ``busy_timeout=30s``
  (``FULL`` keeps writes crash-safe across WAL checkpoints; see issue #242)
* Async surface uses ``asyncio.to_thread`` so the rest of the codebase can
  ``await`` calls without rewiring; SQLite ops are short and IO-bound, so
  the thread-hop is negligible compared to actual I/O
* ``transaction()`` runs ``BEGIN IMMEDIATE`` so concurrent writers don't
  deadlock the way deferred transactions can. This is what gives us the
  cross-table atomicity that ADR-42 promises (one txn covering events +
  cursors + tasks + leases)
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

from .schema import ensure_schema


_PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = FULL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA busy_timeout = 30000",
    "PRAGMA temp_store = MEMORY",
)


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply the WAL / durability pragmas to a connection.

    Runs each statement in :data:`_PRAGMAS` (journal mode, synchronous
    level, foreign keys, busy timeout, temp store) on a throwaway
    cursor.

    Args:
        conn (sqlite3.Connection): Connection to configure.
    """
    cur = conn.cursor()
    try:
        for pragma in _PRAGMAS:
            cur.execute(pragma)
    finally:
        cur.close()


def open_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open one synchronous connection with WAL pragmas + schema applied.

    Creates the parent directory if needed, opens the connection with
    autocommit (``isolation_level=None``) and cross-thread access,
    sets a ``Row`` row factory, applies the pragmas, and ensures the
    schema exists.

    Args:
        db_path (str | Path): Path to the SQLite database file.

    Returns:
        sqlite3.Connection: A ready-to-use connection.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        timeout=30.0,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    ensure_schema(conn)
    return conn


class SqliteConnection:
    """Async-friendly wrapper over a single SQLite connection.

    The underlying connection is *single*; concurrent callers serialize
    through ``self._async_lock``. SQLite WAL gives multi-reader/single-writer
    at the file level anyway, so a writer-pool would only add lock
    contention without throughput gain for our workload.
    """

    def __init__(self, db_path: str | Path):
        """Open the wrapped connection and create its locks.

        Args:
            db_path (str | Path): Path to the SQLite database file;
                opened via :func:`open_connection`.
        """
        self.db_path = Path(db_path)
        self._conn = open_connection(self.db_path)
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.RLock()

    # ------------------------------------------------------------------
    # sync helpers (tests / boot / migrations)
    # ------------------------------------------------------------------
    @property
    def raw(self) -> sqlite3.Connection:
        """Return the underlying ``sqlite3.Connection``.

        Returns:
            sqlite3.Connection: The wrapped connection for callers that
                need direct access.
        """
        return self._conn

    def execute_sync(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Execute a statement synchronously under the sync lock.

        Args:
            sql (str): SQL statement to execute.
            params (Sequence[Any]): Bound parameters.

        Returns:
            sqlite3.Cursor: The cursor produced by ``execute``.
        """
        with self._sync_lock:
            return self._conn.execute(sql, params)

    def fetchall_sync(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a query synchronously and return all rows.

        Args:
            sql (str): SQL query to execute.
            params (Sequence[Any]): Bound parameters.

        Returns:
            list[sqlite3.Row]: All result rows.
        """
        with self._sync_lock:
            cur = self._conn.execute(sql, params)
            try:
                return cur.fetchall()
            finally:
                cur.close()

    def fetchone_sync(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        """Run a query synchronously and return the first row.

        Args:
            sql (str): SQL query to execute.
            params (Sequence[Any]): Bound parameters.

        Returns:
            sqlite3.Row | None: The first row, or ``None`` if empty.
        """
        with self._sync_lock:
            cur = self._conn.execute(sql, params)
            try:
                return cur.fetchone()
            finally:
                cur.close()

    @contextlib.contextmanager
    def transaction_sync(self) -> Iterator[sqlite3.Cursor]:
        """Synchronous BEGIN IMMEDIATE -> COMMIT/ROLLBACK."""
        with self._sync_lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE")
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    # ------------------------------------------------------------------
    # async surface
    # ------------------------------------------------------------------
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        """Execute a write statement asynchronously and commit.

        Runs on a worker thread so the event loop is not blocked; the
        async lock serialises against other async callers.

        Args:
            sql (str): SQL statement to execute.
            params (Sequence[Any]): Bound parameters.
        """
        async with self._async_lock:
            await asyncio.to_thread(self._exec_and_commit, sql, params)

    def _exec_and_commit(self, sql: str, params: Sequence[Any]) -> None:
        """Execute a statement and commit, under the sync lock.

        Args:
            sql (str): SQL statement to execute.
            params (Sequence[Any]): Bound parameters.
        """
        with self._sync_lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    async def fetchall(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[sqlite3.Row]:
        """Run a query asynchronously and return all rows.

        Args:
            sql (str): SQL query to execute.
            params (Sequence[Any]): Bound parameters.

        Returns:
            list[sqlite3.Row]: All result rows.
        """
        async with self._async_lock:
            return await asyncio.to_thread(self.fetchall_sync, sql, params)

    async def fetchone(
        self, sql: str, params: Sequence[Any] = ()
    ) -> sqlite3.Row | None:
        """Run a query asynchronously and return the first row.

        Args:
            sql (str): SQL query to execute.
            params (Sequence[Any]): Bound parameters.

        Returns:
            sqlite3.Row | None: The first row, or ``None`` if empty.
        """
        async with self._async_lock:
            return await asyncio.to_thread(self.fetchone_sync, sql, params)

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[sqlite3.Cursor]:
        """Async ``BEGIN IMMEDIATE`` -> COMMIT/ROLLBACK.

        Usage::

            async with conn.transaction() as cur:
                cur.execute("INSERT INTO events (...) VALUES (...)", row)
                cur.execute("UPDATE cursors SET ...", row2)
        """
        await self._async_lock.acquire()
        try:
            cur = await asyncio.to_thread(self._begin_immediate)
            try:
                yield cur
                await asyncio.to_thread(self._commit)
            except Exception:
                await asyncio.to_thread(self._rollback)
                raise
            finally:
                await asyncio.to_thread(cur.close)
        finally:
            self._async_lock.release()

    def _begin_immediate(self) -> sqlite3.Cursor:
        """Open a cursor and start a ``BEGIN IMMEDIATE`` transaction.

        Returns:
            sqlite3.Cursor: A cursor with an open immediate write
                transaction.
        """
        with self._sync_lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            return cur

    def _commit(self) -> None:
        """Commit the current transaction under the sync lock."""
        with self._sync_lock:
            self._conn.commit()

    def _rollback(self) -> None:
        """Roll back the current transaction under the sync lock."""
        with self._sync_lock:
            self._conn.rollback()

    def close(self) -> None:
        """Close the underlying connection under the sync lock."""
        with self._sync_lock:
            self._conn.close()
