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
#: The permission bits of the group and of others, and the write bits among them.
_OTHERS = 0o077
_WRITE_BY_OTHERS = stat.S_IWGRP | stat.S_IWOTH
#: The mode of the directories made for a private database and of the files in them.
_OWNER_DIRECTORY = stat.S_IRWXU
_OWNER_FILE = stat.S_IRUSR | stat.S_IWUSR
#: `os` open flags missing on Windows: 0 there, so the same expressions build on both platforms.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class DatabaseNotPrivate(PermissionError):
    """The session holds a credential, so its database must be kept where no other account can read it or replace a
    directory on its path, and that cannot be had: the credential would be written where another account can reach it."""


def _this_account() -> int:
    return os.geteuid()


def _refusal(path: Path, why: str) -> str:
    return (
        f"{path}{ownership(path)} {why}; the session holds a credential, so its database must be kept in a directory "
        "of this account's that no other account can enter, below directories no other account can rename entries in. "
        "Point TMPDIR at one, or resume the session as the account that made it"
    )


def _narrow(descriptor: int, path: Path, file_mode: int) -> None:
    """Narrow the open file to what it has in common with ``file_mode``, or warn that it cannot be."""
    current = stat.S_IMODE(os.fstat(descriptor).st_mode)
    if current & ~file_mode:
        try:
            os.fchmod(descriptor, current & file_mode)
        except OSError as exc:
            log.warning("%s is %04o, wider than %04o, and not narrowed (%s)", path, current, file_mode, exc.strerror)


def _make_database_file(db_path: Path, file_mode: int) -> None:
    """Create the database ``file_mode`` before SQLite does, and narrow it and the side files a killed process left when
    they are wider. Each is opened without following a link; one that cannot be opened or narrowed is a warning."""
    descriptor = open_shared(db_path, os.O_RDWR, file_mode)
    try:
        _narrow(descriptor, db_path, file_mode)
    finally:
        os.close(descriptor)
    for suffix in _SIDE_FILES:
        side = db_path.with_name(db_path.name + suffix)
        try:
            descriptor = os.open(side, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC)
        except FileNotFoundError:
            continue
        except OSError as exc:
            log.warning("%s cannot be opened to narrow it to %04o (%s)", side, file_mode, exc.strerror)
            continue
        try:
            _narrow(descriptor, side, file_mode)
        finally:
            os.close(descriptor)


def _require_safe_chain(directory: Path) -> None:
    """Refuse unless ``directory`` and every directory above it are real directories (never a link: one that was there
    when the path was resolved is gone from it, so one found now was put there since) that belong to root or this
    account and cannot be written by the group or others (a sticky one can: it lets nobody rename an entry that is not
    theirs, and the next directory down is checked to be root's or this account's). SQLite reaches the database, and
    makes its journal files, by path: a directory on that path in which another account could rename entries would let it
    put a database of its own where SQLite opens one, between any two lookups."""
    for each in (directory, *directory.parents):
        try:
            info = os.lstat(each)
        except OSError as exc:
            raise DatabaseNotPrivate(_refusal(each, f"cannot be examined ({exc.strerror})")) from exc
        if stat.S_ISLNK(info.st_mode):
            raise DatabaseNotPrivate(_refusal(each, "is a link, which another account could point elsewhere"))
        if not stat.S_ISDIR(info.st_mode):
            raise DatabaseNotPrivate(_refusal(each, "is not a directory"))
        if info.st_uid not in (0, _this_account()):
            raise DatabaseNotPrivate(_refusal(each, "is another account's, which could replace what is below it"))
        if info.st_mode & _WRITE_BY_OTHERS and not info.st_mode & stat.S_ISVTX:
            raise DatabaseNotPrivate(
                _refusal(each, "can be written by its group or others, who could replace what is below it")
            )


def _require_owner_only(directory: Path) -> None:
    """Refuse unless ``directory`` (not a link) is this account's with no access for anyone else."""
    try:
        descriptor = os.open(directory, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise DatabaseNotPrivate(_refusal(directory, f"cannot be opened as a directory ({exc.strerror})")) from exc
    try:
        info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if info.st_uid != _this_account() or info.st_mode & _OTHERS:
        raise DatabaseNotPrivate(
            _refusal(directory, "is not a directory of this account's that no other account can enter")
        )


def _remove(made: list[Path]) -> None:
    """Remove what was made, last first (a directory only when it is empty)."""
    for path in reversed(made):
        with contextlib.suppress(OSError):
            path.rmdir() if path.is_dir() and not path.is_symlink() else path.unlink()


def _make_private_directories(directory: Path) -> list[Path]:
    """Make ``directory`` and the missing directories above it ``0700``, after checking the existing ones: nothing is
    made when the chain is refused, and what was made is removed when it is refused after. Returns what it made."""
    missing, existing = [], directory
    while not os.path.lexists(existing):
        missing.append(existing)
        existing = existing.parent
    _require_safe_chain(existing)
    made: list[Path] = []
    try:
        for path in reversed(missing):
            try:
                os.mkdir(path, _OWNER_DIRECTORY)
            except FileExistsError:
                _require_safe_chain(path)  # made by someone since it was looked for: a real directory, not a link
                continue
            except OSError as exc:
                raise DatabaseNotPrivate(_refusal(path, f"cannot be made ({exc.strerror})")) from exc
            made.append(path)
        _require_safe_chain(directory)
        _require_owner_only(directory)
    except BaseException:
        _remove(made)
        raise
    return made


def _open_private_file(path: Path, *, create: bool) -> int | None:
    """The file in the private directory, opened without following a link and made when ``create`` and it is not there;
    ``None`` for a file that is not there and is not made. One that cannot be opened is refused."""
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        if create:
            with contextlib.suppress(FileExistsError):
                return os.open(path, flags | os.O_CREAT | os.O_EXCL, _OWNER_FILE)
        return os.open(path, flags)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError) and not create:
            return None
        raise DatabaseNotPrivate(_refusal(path, f"cannot be opened ({exc.strerror})")) from exc


def _prepare_private(directory: Path, name: str, made: list[Path]) -> None:
    """Create the database in the private directory when it is not there, and narrow it and the side files in it to
    owner-only: each opened without following a link, and each must be a regular file of this account's."""
    for each in (name, *(name + suffix for suffix in _SIDE_FILES)):
        path = directory / each
        fresh = each == name and not os.path.lexists(path)
        descriptor = _open_private_file(path, create=each == name)
        if descriptor is None:
            continue
        if fresh:
            made.append(path)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != _this_account():
                raise DatabaseNotPrivate(_refusal(path, "is not a regular file of this account's"))
            try:
                os.fchmod(descriptor, stat.S_IMODE(info.st_mode) & _OWNER_FILE)
            except OSError as exc:
                raise DatabaseNotPrivate(_refusal(path, f"cannot be made owner-only ({exc.strerror})")) from exc
        finally:
            os.close(descriptor)


def _private_database(db_path: Path) -> Path:
    """The path of a database that holds a credential, made ready before SQLite opens anything: its directory is made
    ``0700`` (or found, and refused unless it is this account's and closed to everyone else) below directories only root
    or this account can write in, and the database and the side files in it are narrowed to ``0600``. Returns the path
    with every link resolved. Nothing is left behind when it is refused."""
    directory = Path(os.path.realpath(db_path.parent))
    made = _make_private_directories(directory)
    try:
        _prepare_private(directory, db_path.name, made)
    except BaseException:
        _remove(made)
        raise
    return directory / db_path.name


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
            the session holds a credential the bus then stores. At ``600`` the
            database must lie in a directory of this account's that no other
            account can enter, below directories no other account can rename
            entries in (``private_db_path_for`` names one); it is refused
            otherwise.

    Returns:
        sqlite3.Connection: A ready-to-use connection.

    Raises:
        JournalModeError: When the mode is unknown, or the database did not
            enter it.
        DatabaseNotPrivate: When the database must be private and such a
            location, or a file in it, cannot be had; nothing has been written
            and nothing is left made.
    """
    mode = resolve_journal_mode(journal_mode)
    db_path = Path(db_path)
    if file_mode & _OTHERS == 0:
        db_path = _private_database(db_path)
    else:
        make_shared_dirs(db_path.parent)
        _make_database_file(db_path, file_mode)
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None, check_same_thread=False)
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
