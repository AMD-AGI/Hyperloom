# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SQLite connection wrapper."""

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


# Journal mode is env-overridable; WAL default.
_JOURNAL_MODE = os.environ.get("INFERENCE_OPTIMIZER_SQLITE_JOURNAL_MODE", "WAL").strip() or "WAL"


_PRAGMAS = (
    f"PRAGMA journal_mode = {_JOURNAL_MODE}",
    "PRAGMA synchronous = FULL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA busy_timeout = 30000",
    "PRAGMA temp_store = MEMORY",
)


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply the WAL / durability pragmas to a connection."""
    cur = conn.cursor()
    try:
        for pragma in _PRAGMAS:
            cur.execute(pragma)
    finally:
        cur.close()


def open_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open one synchronous connection with WAL pragmas + schema applied."""
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
        _apply_pragmas(conn)
        ensure_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


class SqliteConnection:
    """Async-friendly wrapper over a single SQLite connection."""

    def __init__(self, db_path: str | Path):
        """Open the wrapped connection and create its locks."""
        self.db_path = Path(db_path)
        self._conn = open_connection(self.db_path)
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.RLock()

    @property
    def raw(self) -> sqlite3.Connection:
        """Return the underlying ``sqlite3.Connection``."""
        return self._conn

    def fetchall_sync(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a query synchronously and return all rows."""
        with self._sync_lock:
            cur = self._conn.execute(sql, params)
            try:
                return cur.fetchall()
            finally:
                cur.close()

    def fetchone_sync(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        """Run a query synchronously and return the first row."""
        with self._sync_lock:
            cur = self._conn.execute(sql, params)
            try:
                return cur.fetchone()
            finally:
                cur.close()

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        """Execute a write statement asynchronously and commit."""
        async with self._async_lock:
            await asyncio.to_thread(self._exec_and_commit, sql, params)

    def _exec_and_commit(self, sql: str, params: Sequence[Any]) -> None:
        """Execute a statement and commit, under the sync lock."""
        with self._sync_lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a query asynchronously and return all rows."""
        async with self._async_lock:
            return await asyncio.to_thread(self.fetchall_sync, sql, params)

    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        """Run a query asynchronously and return the first row."""
        async with self._async_lock:
            return await asyncio.to_thread(self.fetchone_sync, sql, params)

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[sqlite3.Cursor]:
        """Async ``BEGIN IMMEDIATE`` -> COMMIT/ROLLBACK."""
        await self._async_lock.acquire()
        cur: sqlite3.Cursor | None = None
        try:
            try:
                cur = await asyncio.to_thread(self._begin_immediate)
                yield cur
                await asyncio.to_thread(self._commit)
            except BaseException:
                # ``BaseException``, not ``Exception``: ``CancelledError`` is not an ``Exception``, and a cancel
                # landing on any of the ``to_thread`` hops here — including the one that returns the cursor, after
                # ``BEGIN IMMEDIATE`` already ran in the worker thread — would otherwise skip the rollback.
                await self._rollback_off_loop()
                raise
            finally:
                if cur is not None:
                    await asyncio.to_thread(cur.close)
        finally:
            self._async_lock.release()

    def _begin_immediate(self) -> sqlite3.Cursor:
        """Open a cursor and start a ``BEGIN IMMEDIATE`` transaction."""
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
        """Roll back a failed transaction on a worker thread, uncancellably."""
        rolling_back = asyncio.ensure_future(asyncio.to_thread(self._rollback))
        cancel: asyncio.CancelledError | None = None
        while not rolling_back.done():
            try:
                await asyncio.shield(rolling_back)
            except asyncio.CancelledError as exc:
                cancel = exc
            except (sqlite3.Error, RuntimeError) as exc:
                # The rollback itself failed: a statement error, or the loop's executor refusing new work during
                # teardown.
                log.warning("rollback after a failed transaction did not complete: %r", exc)
        if cancel is not None:
            raise cancel

    def close(self) -> None:
        """Close the underlying connection under the sync lock."""
        with self._sync_lock:
            self._conn.close()
