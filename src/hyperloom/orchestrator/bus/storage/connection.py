# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SQLite connection wrapper.

Stdlib ``sqlite3``, WAL + ``synchronous=FULL``. Async surface wraps sync ops in
``asyncio.to_thread``. ``transaction()`` uses ``BEGIN IMMEDIATE`` for cross-table
atomicity (events + cursors + tasks + leases).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sqlite3
import threading
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from .schema import ensure_schema


log = logging.getLogger(__name__)


# Journal mode is env-overridable; WAL default. On networked filesystems
# (WekaFS / NFS) WAL's ``-shm`` mapping can corrupt the DB, so set
# ``INFERENCE_OPTIMIZER_SQLITE_JOURNAL_MODE=DELETE`` on such mounts.
_JOURNAL_MODE = os.environ.get("INFERENCE_OPTIMIZER_SQLITE_JOURNAL_MODE", "WAL").strip() or "WAL"


_PRAGMAS = (
    f"PRAGMA journal_mode = {_JOURNAL_MODE}",
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

    Single underlying connection; concurrent callers serialize through
    ``self._async_lock``.
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

    @property
    def raw(self) -> sqlite3.Connection:
        """Return the underlying ``sqlite3.Connection``.

        Returns:
            sqlite3.Connection: The wrapped connection for callers that
                need direct access.
        """
        return self._conn

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

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a query asynchronously and return all rows.

        Args:
            sql (str): SQL query to execute.
            params (Sequence[Any]): Bound parameters.

        Returns:
            list[sqlite3.Row]: All result rows.
        """
        async with self._async_lock:
            return await asyncio.to_thread(self.fetchall_sync, sql, params)

    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
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

        Yields:
            An open cursor inside the immediate write transaction; the
            transaction commits on clean exit and rolls back on any exception,
            cancellation included.
        """
        await self._async_lock.acquire()
        cur: sqlite3.Cursor | None = None
        try:
            try:
                cur = await asyncio.to_thread(self._begin_immediate)
                yield cur
                await asyncio.to_thread(self._commit)
            except BaseException:
                # ``BaseException``, not ``Exception``: ``CancelledError`` is
                # not an ``Exception``, and a cancel landing on any of the
                # ``to_thread`` hops here — including the one that returns the
                # cursor, after ``BEGIN IMMEDIATE`` already ran in the worker
                # thread — would otherwise skip the rollback. The shared
                # connection then stays inside a transaction for the rest of
                # the session and every later write fails with "cannot start a
                # transaction within a transaction".
                self._rollback_quietly()
                raise
            finally:
                if cur is not None:
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

    def _rollback_quietly(self) -> None:
        """Roll back an open transaction on the way out of a failed body.

        Runs inline rather than on a worker thread: the exception being handled
        is often this task's own cancellation, and an ``await`` there can be
        cancelled in turn — which is precisely how the connection would stay
        wedged. A rollback that itself fails is logged and swallowed so the
        caller's original exception is the one that propagates.
        """
        try:
            self._rollback()
        except sqlite3.Error as exc:
            log.warning("rollback after a failed transaction failed: %r", exc)

    def close(self) -> None:
        """Close the underlying connection under the sync lock."""
        with self._sync_lock:
            self._conn.close()
