# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SQLite connection wrapper."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sqlite3
import stat
import threading
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from hyperloom_common.shared_storage import SHARED_FILE_MODE, make_shared_dirs, open_shared, ownership

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


#: What SQLite keeps beside the database. It gives the ones it creates the database's mode, but it reuses ones a killed
#: process left as they are, so they are narrowed with the database.
_SIDE_FILES = ("-wal", "-shm", "-journal")
#: The permission bits of the group and of others.
_OTHERS = 0o077


class DatabaseNotPrivate(PermissionError):
    """The session holds a credential, so its database and the files beside it must be owner-only, and one of them cannot
    be made so: the credential would be written to a file the group can read."""


def _refusal(path: Path, file_mode: int, why: str) -> str:
    message = f"{path}{ownership(path)} {why}"
    if file_mode & _OTHERS:
        return message
    return (
        f"{message}; the session holds a credential and its database must be {file_mode:04o}, or the credential would be "
        "written to a file the group can read. Resume it as the account that owns the file, or start a new session"
    )


def _cannot_narrow(path: Path, file_mode: int, why: str) -> None:
    """Say that ``path`` cannot be narrowed to ``file_mode``: an error when that mode is owner-only (nothing may be
    written to a wider file), else a warning, as a session without a credential still resumes."""
    if file_mode & _OTHERS:
        log.warning("%s", _refusal(path, file_mode, why))
    else:
        raise DatabaseNotPrivate(_refusal(path, file_mode, why))


def _narrow(descriptor: int, path: Path, file_mode: int) -> None:
    """Narrow the open file to what it has in common with ``file_mode``, or say that it cannot be."""
    current = stat.S_IMODE(os.fstat(descriptor).st_mode)
    if current & ~file_mode:
        try:
            os.fchmod(descriptor, current & file_mode)
        except OSError as exc:
            _cannot_narrow(
                path, file_mode, f"is {current:04o}, wider than {file_mode:04o}, and not narrowed ({exc.strerror})"
            )


def _open_database_file(db_path: Path, file_mode: int) -> int:
    """The database file, created ``file_mode`` when it is not there; a refusal to open it (another account's, a link)
    is a :class:`DatabaseNotPrivate` when the mode is owner-only, else it is raised as it is."""
    try:
        return open_shared(db_path, os.O_RDWR, file_mode)
    except OSError as exc:
        if file_mode & _OTHERS:
            raise
        raise DatabaseNotPrivate(_refusal(db_path, file_mode, f"cannot be opened ({exc.strerror})")) from exc


def _make_database_file(db_path: Path, file_mode: int) -> None:
    """Create the database ``file_mode`` before SQLite does, and narrow it and the side files a killed process left when
    they are wider (a credential has come into the session since). Each is opened without following a link. When
    ``file_mode`` is owner-only, a file that cannot be opened or narrowed (another account's, a link, a failed
    ``fchmod``) is a :class:`DatabaseNotPrivate`, raised before SQLite opens anything: otherwise a warning."""
    descriptor = _open_database_file(db_path, file_mode)
    try:
        _narrow(descriptor, db_path, file_mode)
    finally:
        os.close(descriptor)
    for suffix in _SIDE_FILES:
        side = db_path.with_name(db_path.name + suffix)
        try:
            descriptor = os.open(side, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        except FileNotFoundError:
            continue
        except OSError as exc:
            _cannot_narrow(side, file_mode, f"cannot be opened to narrow it ({exc.strerror})")
            continue
        try:
            _narrow(descriptor, side, file_mode)
        finally:
            os.close(descriptor)


def open_connection(
    db_path: str | Path, *, journal_mode: str | None = None, file_mode: int = SHARED_FILE_MODE
) -> sqlite3.Connection:
    """Open one synchronous connection with the durability pragmas + schema applied.

    Creates the parent directory if needed, opens with autocommit
    (``isolation_level=None``) and cross-thread access, sets a ``Row`` row
    factory, applies and verifies the pragmas, and ensures the schema exists.

    Args:
        db_path (str | Path): Path to the SQLite database file.
        journal_mode (str | None): Journal mode to enforce; resolved from the
            environment when omitted.
        file_mode (int): The most the database file's permission bits may be:
            ``664`` so the next account of the group can open it, ``600`` when
            the session holds a credential the bus then stores.

    Returns:
        sqlite3.Connection: A ready-to-use connection.

    Raises:
        JournalModeError: When the mode is unknown, or the database did not
            enter it.
        DatabaseNotPrivate: When ``file_mode`` is owner-only and the database
            or a file beside it cannot be made so; nothing has been written.
    """
    mode = resolve_journal_mode(journal_mode)
    db_path = Path(db_path)
    make_shared_dirs(db_path.parent)
    _make_database_file(db_path, file_mode)
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
    """Async-friendly wrapper over a single SQLite connection."""

    def __init__(self, db_path: str | Path, *, journal_mode: str | None = None, file_mode: int = SHARED_FILE_MODE):
        """Open the wrapped connection and create its locks.

        Args:
            db_path (str | Path): Path to the SQLite database file;
                opened via :func:`open_connection`.
            journal_mode (str | None): Journal mode to enforce; resolved from
                the environment when omitted.
            file_mode (int): See :func:`open_connection`.
        """
        self.db_path = Path(db_path)
        self._conn = open_connection(self.db_path, journal_mode=journal_mode, file_mode=file_mode)
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
