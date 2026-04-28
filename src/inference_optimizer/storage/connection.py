"""SQLite connection wrapper (DESIGN §3.5.4 / §13.2).

Design choices:

- Pure stdlib ``sqlite3`` — no extra dependency.
- Mode = WAL with ``synchronous=NORMAL`` and ``busy_timeout=30s``.
- Async surface uses ``asyncio.to_thread`` so the rest of the codebase can
  ``await`` calls without rewiring; SQLite ops are short and IO-bound, so
  the thread-hop is negligible compared to actual I/O.
- ``transaction()`` is an async context manager that runs ``BEGIN
  IMMEDIATE`` so concurrent writers don't deadlock the way deferred
  transactions can. This is what gives us the cross-table atomicity that
  ADR-33 promises (one txn covering events + cursors + tasks + leases).
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
    "PRAGMA synchronous = NORMAL",
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
    """Open one synchronous connection ready for use.

    Use ``SqliteConnection`` for the async wrapper used by the orchestrator.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        timeout=30.0,
        isolation_level=None,  # we manage txns manually
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    ensure_schema(conn)
    return conn


class SqliteConnection:
    """Async-friendly wrapper over a single SQLite connection.

    The underlying connection is *single*; concurrent callers serialize
    through ``self._lock``. SQLite's WAL mode gives us multi-reader,
    single-writer at the file level anyway, so we don't gain anything by
    running a pool of writer connections — we'd just create lock
    contention.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn = open_connection(self.db_path)
        # Single asyncio.Lock guards .execute / .transaction.
        # Use threading.RLock as well so synchronous helpers stay safe if
        # someone calls them outside the event loop (e.g. test setup).
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.RLock()

    # ------------------------------------------------------------------
    # sync helpers (for tests / boot / migrations)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # async surface
    # ------------------------------------------------------------------
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
        """Async ``BEGIN IMMEDIATE``. Use for any multi-statement write
        that must be atomic (e.g. cursor advance + emit event)."""
        await self._async_lock.acquire()
        try:
            cur = await asyncio.to_thread(self._begin_immediate)
            try:
                yield cur
                await asyncio.to_thread(self._conn.commit)
            except Exception:
                await asyncio.to_thread(self._conn.rollback)
                raise
            finally:
                await asyncio.to_thread(cur.close)
        finally:
            self._async_lock.release()

    def _begin_immediate(self) -> sqlite3.Cursor:
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        return cur

    def close(self) -> None:
        with self._sync_lock:
            self._conn.close()
