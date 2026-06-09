# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SQLite connection wrapper.

Stdlib ``sqlite3``, WAL + ``synchronous=FULL`` (crash-safe across checkpoints,
issue #242). Async surface wraps sync ops in ``asyncio.to_thread``.
``transaction()`` uses ``BEGIN IMMEDIATE`` for cross-table atomicity (ADR-42:
events + cursors + tasks + leases) without deferred-txn deadlocks.
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
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    ensure_schema(conn)
    return conn


class SqliteConnection:
    """Async-friendly wrapper over a single SQLite connection.

    Single underlying connection; concurrent callers serialize through
    ``self._async_lock`` (WAL is single-writer anyway, so a pool would only
    add contention).
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn = open_connection(self.db_path)
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.RLock()

    # sync helpers (tests / boot / migrations)
    @property
    def raw(self) -> sqlite3.Connection:
        return self._conn

    def execute_sync(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._sync_lock:
            return self._conn.execute(sql, params)

    def fetchall_sync(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._sync_lock:
            cur = self._conn.execute(sql, params)
            try:
                return cur.fetchall()
            finally:
                cur.close()

    def fetchone_sync(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
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

    # async surface
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        async with self._async_lock:
            await asyncio.to_thread(self._exec_and_commit, sql, params)

    def _exec_and_commit(self, sql: str, params: Sequence[Any]) -> None:
        with self._sync_lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    async def fetchall(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[sqlite3.Row]:
        async with self._async_lock:
            return await asyncio.to_thread(self.fetchall_sync, sql, params)

    async def fetchone(
        self, sql: str, params: Sequence[Any] = ()
    ) -> sqlite3.Row | None:
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
        with self._sync_lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            return cur

    def _commit(self) -> None:
        with self._sync_lock:
            self._conn.commit()

    def _rollback(self) -> None:
        with self._sync_lock:
            self._conn.rollback()

    def close(self) -> None:
        with self._sync_lock:
            self._conn.close()
