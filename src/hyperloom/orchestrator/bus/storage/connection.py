# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SQLite connection wrapper.

Stdlib ``sqlite3``, ``synchronous=FULL``, journal mode resolved and verified at
open. Async surface wraps sync ops in ``asyncio.to_thread``. ``transaction()``
uses ``BEGIN IMMEDIATE`` for cross-table atomicity (events + cursors + tasks +
leases + bring-up rounds).
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


#: Journal mode used unless the environment names another one.
DEFAULT_JOURNAL_MODE = "WAL"

#: The journal modes SQLite accepts. An unrecognised mode leaves SQLite in its
#: current mode rather than failing, so it is checked before opening.
JOURNAL_MODES = frozenset({"DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"})

#: Environment variable naming the journal mode for every session database.
JOURNAL_MODE_ENV = "INFERENCE_OPTIMIZER_SQLITE_JOURNAL_MODE"


_PRAGMAS = (
    "PRAGMA synchronous = FULL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA busy_timeout = 30000",
    "PRAGMA temp_store = MEMORY",
)


class JournalModeError(RuntimeError):
    """The journal mode asked for is not the one the database ended up in."""


def resolve_journal_mode(requested: str | None = None) -> str:
    """Return the journal mode a session database must be opened in.

    WAL's ``-shm`` mapping can corrupt a database on a networked filesystem
    (WekaFS / NFS), so :data:`JOURNAL_MODE_ENV` overrides it per deployment.

    Args:
        requested: Explicit mode; the environment is consulted when omitted.

    Returns:
        str: The upper-cased mode to open with.

    Raises:
        JournalModeError: When the mode is not one SQLite defines.
    """
    raw = requested if requested is not None else os.environ.get(JOURNAL_MODE_ENV, "")
    mode = raw.strip().upper() or DEFAULT_JOURNAL_MODE
    if mode not in JOURNAL_MODES:
        raise JournalModeError(f"unknown SQLite journal mode {mode!r}; expected one of {sorted(JOURNAL_MODES)}")
    return mode


def _apply_pragmas(conn: sqlite3.Connection, journal_mode: str) -> None:
    """Apply the journal mode and durability pragmas, and verify the mode took.

    SQLite refuses a journal mode by leaving the old one in force rather than
    by raising, so ``PRAGMA journal_mode`` is read back and
    :class:`JournalModeError` raised when it differs.
    """
    cur = conn.cursor()
    try:
        cur.execute(f"PRAGMA journal_mode = {journal_mode}")  # nosec B608 - validated against JOURNAL_MODES.
        for pragma in _PRAGMAS:
            cur.execute(pragma)
        cur.execute("PRAGMA journal_mode")
        actual = str(cur.fetchone()[0]).strip().upper()
        if actual != journal_mode:
            raise JournalModeError(f"database opened in journal mode {actual!r}, not the requested {journal_mode!r}")
    finally:
        cur.close()


def open_connection(db_path: str | Path, *, journal_mode: str | None = None) -> sqlite3.Connection:
    """Open one synchronous connection with the durability pragmas + schema applied.

    Creates the parent directory if needed, opens with autocommit
    (``isolation_level=None``) and cross-thread access, sets a ``Row`` row
    factory, applies and verifies the pragmas, and ensures the schema exists.

    Args:
        db_path (str | Path): Path to the SQLite database file.
        journal_mode (str | None): Journal mode to enforce; resolved from the
            environment when omitted.

    Returns:
        sqlite3.Connection: A ready-to-use connection.

    Raises:
        JournalModeError: When the mode is unknown, or the database did not
            enter it.
    """
    mode = resolve_journal_mode(journal_mode)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        timeout=30.0,
        isolation_level=None,
        check_same_thread=False,
    )
    try:
        conn.row_factory = sqlite3.Row
        _apply_pragmas(conn, mode)
        ensure_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


class SqliteConnection:
    """Async-friendly wrapper over a single SQLite connection.

    Single underlying connection; concurrent callers serialize through
    ``self._async_lock``.
    """

    def __init__(self, db_path: str | Path, *, journal_mode: str | None = None):
        """Open the wrapped connection and create its locks.

        Args:
            db_path (str | Path): Path to the SQLite database file;
                opened via :func:`open_connection`.
            journal_mode (str | None): Journal mode to enforce; resolved from
                the environment when omitted.
        """
        self.db_path = Path(db_path)
        self.journal_mode = resolve_journal_mode(journal_mode)
        self._conn = open_connection(self.db_path, journal_mode=self.journal_mode)
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
                # not an ``Exception``, and a cancel landing on any ``to_thread``
                # hop here would leave the shared connection inside the
                # transaction for the rest of the session.
                await self._rollback_off_loop()
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

    async def _rollback_off_loop(self) -> None:
        """Roll back a failed transaction on a worker thread, uncancellably.

        Three constraints hold simultaneously: the rollback runs off the event
        loop, because ``_sync_lock`` may still be held by a worker parked in
        ``BEGIN IMMEDIATE`` for up to ``busy_timeout``; it must complete before
        ``_async_lock`` is released, or a later writer finds the transaction
        still open; and it must not itself be cancellable. :func:`asyncio.shield`
        protects the rollback but not the wait on it, so cancels landing on the
        wait are absorbed and the wait resumed, then the last one re-raised. A
        rollback that fails outright is logged and never masks the caller's
        original exception.
        """
        rolling_back = asyncio.ensure_future(asyncio.to_thread(self._rollback))
        cancel: asyncio.CancelledError | None = None
        while not rolling_back.done():
            try:
                await asyncio.shield(rolling_back)
            except asyncio.CancelledError as exc:
                cancel = exc
            except (sqlite3.Error, RuntimeError) as exc:
                # The rollback itself failed: a statement error, or the loop's
                # executor refusing new work during teardown. Anything else is
                # a bug worth surfacing rather than logging.
                log.warning("rollback after a failed transaction did not complete: %r", exc)
        if cancel is not None:
            raise cancel

    def close(self) -> None:
        """Close the underlying connection under the sync lock."""
        with self._sync_lock:
            self._conn.close()
