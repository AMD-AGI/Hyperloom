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

from hyperloom.inference_optimizer.session.paths import PRIVATE_STORAGE_DIRNAME
from hyperloom_common.shared_storage import SHARED_FILE_MODE, make_shared_dirs, open_directory, open_shared, ownership

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
#: The mode of the owner-only directory and of the files in it.
_OWNER_DIRECTORY = stat.S_IRWXU
_OWNER_FILE = stat.S_IRUSR | stat.S_IWUSR
_DIRECTORY = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


class DatabaseNotPrivate(PermissionError):
    """The session holds a credential, so its database must live in a directory only this account can enter, and that
    cannot be had: the credential would be written where another account of the group can read it."""


def _this_account() -> int:
    return os.geteuid()


def _refusal(path: Path, why: str) -> str:
    return (
        f"{path}{ownership(path)} {why}; the session holds a credential, so its database must be kept in a directory "
        "only this account can enter. Resume it as the account that owns that directory, or start a new session"
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
            descriptor = os.open(side, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        except FileNotFoundError:
            continue
        except OSError as exc:
            log.warning("%s cannot be opened to narrow it to %04o (%s)", side, file_mode, exc.strerror)
            continue
        try:
            _narrow(descriptor, side, file_mode)
        finally:
            os.close(descriptor)


def _private_directory(storage: int, shown: Path) -> int:
    """The owner-only directory below ``storage`` (a descriptor of it), made with its mode when it is not there, reached
    without following a link. One that is there already must be this account's with no access for anyone else: another
    account's, or a wider one, is refused."""
    with contextlib.suppress(FileExistsError):
        os.mkdir(shown.name, _OWNER_DIRECTORY, dir_fd=storage)
    try:
        descriptor = os.open(shown.name, _DIRECTORY, dir_fd=storage)
    except OSError as exc:
        raise DatabaseNotPrivate(
            _refusal(shown, f"cannot be opened as a directory without following a link ({exc.strerror})")
        ) from exc
    info = os.fstat(descriptor)
    if info.st_uid != _this_account() or info.st_mode & _OTHERS:
        os.close(descriptor)
        raise DatabaseNotPrivate(
            _refusal(shown, "is not a directory of this account's that no other account can enter")
        )
    return descriptor


def _files_of(name: str) -> list[str]:
    return [name, *(name + suffix for suffix in _SIDE_FILES)]


def _regular_file_of_ours(directory: int, name: str) -> bool | None:
    """Whether ``name`` in the directory held open as ``directory`` is a regular file of this account's, without following
    a link; ``None`` when it is not there."""
    try:
        descriptor = os.open(name, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return stat.S_ISREG(info.st_mode) and info.st_uid == _this_account()


def _move_in(storage: int, private: int, name: str, shown: Path) -> None:
    """Move the database a session made before it held a credential, with the side files a killed process left, from
    ``storage`` into the owner-only directory: all of them must be this account's regular files, or it is refused (a
    journal another account left beside the database would be rolled into it). Where the owner-only directory holds a
    database already, that one is the session's, and where ``storage`` holds none there is nothing to move: files
    beside a database that is not there are never read, since SQLite looks only in the owner-only directory."""
    if _regular_file_of_ours(private, name) is not None or _regular_file_of_ours(storage, name) is None:
        return
    present = []
    for each in _files_of(name):
        ours = _regular_file_of_ours(storage, each)
        if ours is False:
            raise DatabaseNotPrivate(
                _refusal(
                    shown.parent / each,
                    "was made before the session held a credential and is not a regular file of this account's",
                )
            )
        if ours:
            present.append(each)
    for each in present:
        try:
            os.rename(each, each, src_dir_fd=storage, dst_dir_fd=private)
        except OSError as exc:
            raise DatabaseNotPrivate(
                _refusal(shown.parent / each, f"cannot be moved into the owner-only directory ({exc.strerror})")
            ) from exc


def _open_in_private(private: int, each: str, *, create: bool, shown: Path) -> int | None:
    """The file ``each`` of the owner-only directory, opened without following a link and made when ``create`` and it is
    not there; ``None`` for a file that is not there and is not made. One that cannot be opened is refused."""
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        if create:
            with contextlib.suppress(FileExistsError):
                return os.open(each, flags | os.O_CREAT | os.O_EXCL, _OWNER_FILE, dir_fd=private)
        return os.open(each, flags, dir_fd=private)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError) and not create:
            return None
        raise DatabaseNotPrivate(_refusal(shown / each, f"cannot be opened ({exc.strerror})")) from exc


def _prepare_private(private: int, name: str, shown: Path) -> None:
    """Create the database in the owner-only directory when it is not there, and narrow it and the side files in it to
    owner-only: each opened without following a link, and each must be a regular file of this account's."""
    for each in _files_of(name):
        descriptor = _open_in_private(private, each, create=each == name, shown=shown)
        if descriptor is None:
            continue
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != _this_account():
                raise DatabaseNotPrivate(_refusal(shown / each, "is not a regular file of this account's"))
            try:
                os.fchmod(descriptor, stat.S_IMODE(info.st_mode) & _OWNER_FILE)
            except OSError as exc:
                raise DatabaseNotPrivate(_refusal(shown / each, f"cannot be made owner-only ({exc.strerror})")) from exc
        finally:
            os.close(descriptor)


def _private_database(db_path: Path) -> tuple[int, Path]:
    """The database of a session that holds a credential, made ready in the owner-only directory below ``storage/``
    (a descriptor of it, and the path of the database in it), before SQLite opens anything: the directory is made (or
    found, and refused unless it is this account's and closed to everyone else), a database a session made before it
    held a credential is moved into it, and the files are narrowed. The session directory is one the group can write,
    and SQLite makes its journal files by name: a file another account put there first would be written to."""
    inside = db_path.parent.name == PRIVATE_STORAGE_DIRNAME
    shared = db_path.parent.parent if inside else db_path.parent
    shown = shared / PRIVATE_STORAGE_DIRNAME
    make_shared_dirs(shared)
    storage = open_directory(shared)
    try:
        private = _private_directory(storage, shown)
        try:
            _move_in(storage, private, db_path.name, shown)
            _prepare_private(private, db_path.name, shown)
        except BaseException:
            os.close(private)
            raise
    finally:
        os.close(storage)
    return private, shown / db_path.name


def _is_same_file(path: Path, directory: int) -> bool:
    """Whether ``path`` names the file of that name in the directory held open as ``directory``."""
    try:
        reached = os.stat(path)
        held = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
    except OSError:
        return False
    return (reached.st_dev, reached.st_ino) == (held.st_dev, held.st_ino)


def _connect(path: Path, directory: int | None) -> sqlite3.Connection:
    """The connection to the database at ``path``; for one in an owner-only directory (held open as ``directory``),
    refused when SQLite reached another file by that name (the directory was replaced since it was checked)."""
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None, check_same_thread=False)
    if directory is not None and not _is_same_file(path, directory):
        conn.close()
        raise DatabaseNotPrivate(_refusal(path.parent, "was replaced while the database was opened"))
    return conn


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
            the session holds a credential the bus then stores. At ``600``, and
            for a database already in the owner-only directory
            (:data:`~hyperloom.inference_optimizer.session.paths.PRIVATE_STORAGE_DIRNAME`
            below its directory), the database lives in that directory.

    Returns:
        sqlite3.Connection: A ready-to-use connection.

    Raises:
        JournalModeError: When the mode is unknown, or the database did not
            enter it.
        DatabaseNotPrivate: When the database must be private and the
            owner-only directory, or a file in it, cannot be had; nothing has
            been written.
    """
    mode = resolve_journal_mode(journal_mode)
    db_path = Path(db_path)
    private = file_mode & _OTHERS == 0 or db_path.parent.name == PRIVATE_STORAGE_DIRNAME
    directory = None
    if private:
        directory, db_path = _private_database(db_path)
    else:
        make_shared_dirs(db_path.parent)
        _make_database_file(db_path, file_mode)
    try:
        conn = _connect(db_path, directory)
    finally:
        if directory is not None:
            os.close(directory)
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
