# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Directories and files that several accounts of one group write in turn.

On shared storage each container of a run may be another host account. The accounts share one group, and the key
directories are made setgid in it once (``2775``), so what is created below them takes that group. What this module
creates has an explicit mode, set by ``fchmod`` after the create: directories ``2775`` and files ``664``, whatever the
umask. Nothing is chowned, and an existing entry is never chmodded (another account's would fail ``EPERM``): a file is
replaced by rename instead, and never opened in place, since a member of the group may have put a link there. Nor is
a link followed on the way to it once the way has passed a directory another account can write
(:func:`open_directory`): a member of the group could have put one there for any directory of the path. The umask of
this process is never changed, since its threads share it and it would also open up the files only this account may
read.

Nothing the application executes or sources may live in a tree the group can write.
"""

from __future__ import annotations

import errno
import logging
import os
import secrets
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path

log = logging.getLogger(__name__)

SHARED_DIR_MODE = 0o2775
SHARED_FILE_MODE = 0o664

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
#: How each directory of a path is opened on the way (:func:`open_directory`): only to name what is below it, which
#: needs no read permission (a home directory may be ``0711``), and never through a link.
_WAY = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
#: How a link on the way is opened to check whose it is and read it (:meth:`_Walk.follow`).
_LINK = os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
_NEW_FILE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_PROBE_PREFIX = ".hl-writable-"
#: What a new file is made with, before ``fchmod`` gives it its mode: the umask never decides it.
_OWNER_ONLY = 0o600
#: The suffix of the new file a writer makes beside the one it replaces.
_TEMPORARY_SUFFIX = ".tmp"
#: How many names a new temporary file is tried at before the writer gives up.
_TEMPORARY_TRIES = 100
#: How many links one path may go through, as the kernel allows (``MAXSYMLINKS``).
_MAX_LINKS = 40
#: A directory with one of these bits lets accounts other than its owner put a link in it.
_OPEN_TO_OTHERS = stat.S_IWGRP | stat.S_IWOTH
#: When a link cannot be made, a new file is made at its final name instead (:func:`create_new_shared_file`).
_NO_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS})


def open_directory(path: Path, *, make: int | None = None) -> int:
    """``path`` opened as a directory: an ``O_PATH`` descriptor (the caller closes it) that what is written below it
    goes through, as ``dir_fd``; :func:`read_directory` opens it for listing or a flush.

    Each directory of the path is opened below the one before it, never following another account's link once the way
    has passed a directory that accounts other than its owner can write (:data:`_OPEN_TO_OTHERS`, the shared tree
    itself, ``/tmp``): any member of the group could have put one there, to any directory the account that follows it
    can write. Links above that point (``/var/run``, a mount reached through a link of the host's), and this account's
    and root's own anywhere, are followed. With ``make``, each missing directory is made with that mode
    (:func:`make_shared_dirs`).

    Raises:
        OSError: ``ELOOP`` for a link not followed or too many links, ``ENOENT`` for a missing directory without
            ``make``, ``FileExistsError`` (with ``make``) or ``ENOTDIR`` for something else where a directory belongs;
            each names the part of the path it is about.
    """
    parts = list(Path(os.path.abspath(path)).parts[1:])
    walk = _Walk(os.open("/", _WAY))
    try:
        while parts:
            name = parts.pop(0)
            if name in ("", "."):
                continue
            info = walk.look_up(name, make)
            if info is None:
                parts.insert(0, name)  # made meanwhile by another process: looked at again
            elif stat.S_ISLNK(info.st_mode):
                parts[:0] = walk.follow(name)
            else:
                walk.enter(name, info, make)
    except BaseException:
        os.close(walk.descriptor)
        raise
    return walk.descriptor


class _Walk:
    """Where :func:`open_directory` is on its way: the directory open as :attr:`descriptor` (``O_PATH``) is
    :attr:`here`, and :attr:`shared` once a directory passed can be written by accounts other than its owner."""

    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor
        self.here = Path("/")
        self.shared = False
        self.links = 0

    def look_up(self, name: str, make: int | None) -> os.stat_result | None:
        """``lstat`` of ``name`` here, made first with ``make`` when it is missing; ``None`` when another process
        made it meanwhile."""
        self.shared = self.shared or bool(os.fstat(self.descriptor).st_mode & _OPEN_TO_OTHERS)
        try:
            return os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            if make is None:
                raise FileNotFoundError(errno.ENOENT, "no such directory", str(self.here / name)) from None
        if not _make_directory(self.descriptor, name, make, self.here / name):
            return None
        return os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)

    def follow(self, name: str) -> tuple[str, ...]:
        """The parts of the link ``name`` here to walk instead of it; an absolute one starts again from ``/``.

        Below a directory others can write, only a link of this account's or root's is followed, which no other account
        can make: it is opened itself (``O_PATH``), so the link whose owner is checked is the one read.

        Raises:
            OSError: ``ELOOP`` for another account's link below a directory others can write, or after too many links.
        """
        link = os.open(name, _LINK, dir_fd=self.descriptor)
        try:
            owner = os.fstat(link).st_uid
            if self.shared and owner not in (0, os.geteuid()):
                raise OSError(
                    errno.ELOOP,
                    f"a link of uid {owner} in a directory other accounts can write is not followed",
                    str(self.here / name),
                )
            target = Path(os.readlink("", dir_fd=link))
        finally:
            os.close(link)
        self.links += 1
        if self.links > _MAX_LINKS:
            raise OSError(errno.ELOOP, os.strerror(errno.ELOOP), str(self.here / name))
        if not target.is_absolute():
            return target.parts
        self._move(os.open("/", _WAY), Path("/"))
        self.shared = False  # the target's own way decides from here
        return target.parts[1:]

    def enter(self, name: str, info: os.stat_result, make: int | None) -> None:
        """Go into the directory ``name`` here, whose ``lstat`` is ``info``."""
        if not stat.S_ISDIR(info.st_mode):
            if make is not None:
                raise FileExistsError(errno.EEXIST, "not a directory", str(self.here / name))
            raise NotADirectoryError(errno.ENOTDIR, "not a directory", str(self.here / name))
        child = os.open(name, _WAY, dir_fd=self.descriptor)  # ELOOP or ENOTDIR if it changed since
        self._move(child, self.here.parent if name == ".." else self.here / name)

    def _move(self, descriptor: int, here: Path) -> None:
        os.close(self.descriptor)
        self.descriptor, self.here = descriptor, here


def read_directory(descriptor: int) -> int:
    """The directory open as ``descriptor`` (:func:`open_directory`) opened again for reading: to list it or flush it.
    The caller closes it."""
    return os.open(".", _DIRECTORY, dir_fd=descriptor)


def _make_directory(parent: int, name: str, mode: int, made: Path) -> bool:
    """Make the directory ``name`` (``made``) of ``parent`` with ``mode`` set by ``fchmod``; ``False`` when it was there
    already."""
    try:
        os.mkdir(name, mode, dir_fd=parent)
    except FileExistsError:
        return False
    _share(os.open(name, _DIRECTORY, dir_fd=parent), mode, made)
    return True


def _share(descriptor: int, mode: int, directory: Path) -> None:
    """Give the directory open as ``descriptor`` its ``mode`` and close it; logged when it does not keep setgid."""
    try:
        os.fchmod(descriptor, mode)
        if mode & stat.S_ISGID and not os.fstat(descriptor).st_mode & stat.S_ISGID:
            log.warning(
                "%s did not keep the setgid bit: this account is not in its group, so what is made below it takes this "
                "account's own group",
                directory,
            )
    finally:
        os.close(descriptor)


def make_shared_dirs(path: Path, *, mode: int = SHARED_DIR_MODE) -> Path:
    """Create each missing directory of ``path`` with ``mode`` (default ``2775``) and return ``path``.

    The way is walked as :func:`open_directory` walks it, so a link a member of the group put there is never followed.
    The mode is set by ``fchmod`` on the new directory, so the umask never decides it; the group comes from a setgid
    parent. An existing directory is left as it is. A new directory that did not keep the setgid bit (this account is
    not in its group, which a setgid parent gave it) is logged: what is made below it takes this account's group,
    which the next account may not share.
    """
    os.close(open_directory(path, make=mode))
    return path


def share_new_directory(directory: Path, *, mode: int = SHARED_DIR_MODE) -> None:
    """Give a directory this process has just made (``mkdir``, ``mkdtemp``) its shared ``mode``, by ``fchmod`` on the
    directory reached as :func:`open_directory` reaches it; logged as in :func:`make_shared_dirs` when it does not keep
    setgid."""
    way = open_directory(directory)
    try:
        _share(read_directory(way), mode, directory)
    finally:
        os.close(way)


@contextmanager
def _directory_of(path: Path) -> Iterator[tuple[int, str]]:
    """The directory of ``path`` opened by :func:`open_directory`, and the name of ``path`` in it."""
    descriptor = open_directory(Path(path).parent)
    try:
        yield descriptor, Path(path).name
    finally:
        os.close(descriptor)


def _new_temporary(directory: int, name: str, mode: int, flags: int = 0) -> tuple[int, str]:
    """A new file beside ``name`` in ``directory`` under a name of its own (``O_EXCL``, never through a link), with
    ``mode`` set by ``fchmod``: its descriptor, open for writing, and that name."""
    for _ in range(_TEMPORARY_TRIES):
        temporary = f".{name}.{secrets.token_hex(6)}{_TEMPORARY_SUFFIX}"
        try:
            descriptor = os.open(
                temporary, _NEW_FILE | flags, _OWNER_ONLY, dir_fd=directory
            )  # the umask never decides it
        except FileExistsError:
            continue
        try:
            os.fchmod(descriptor, mode)
        except BaseException:
            os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise
        return descriptor, temporary
    raise FileExistsError(errno.EEXIST, "no free name for a new file beside it", name)


def create_shared_file(path: Path) -> None:
    """Create ``path`` empty with mode ``664``; an existing file is replaced by rename, never truncated or chmodded."""
    with _directory_of(path) as (directory, name):
        try:
            descriptor = os.open(
                name, _NEW_FILE, _OWNER_ONLY, dir_fd=directory
            )  # the umask never decides it: fchmod below
        except FileExistsError:
            descriptor = None
        if descriptor is not None:
            try:
                os.fchmod(descriptor, SHARED_FILE_MODE)
            finally:
                os.close(descriptor)
            return
    write_shared_bytes(path, b"")


def write_shared_bytes(
    path: Path, data: bytes, *, mode: int = SHARED_FILE_MODE, fsync: bool = False, fsync_dir: bool = False
) -> None:
    """Write ``data`` to a new file beside ``path`` with ``mode`` and rename it over ``path``: the file is then this
    account's, whoever owned the one it replaced. ``fsync`` flushes the file before the rename, ``fsync_dir`` the
    directory after it."""
    with _directory_of(path) as (directory, name):
        descriptor, temporary = _new_temporary(directory, name, mode)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                if fsync:
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise
        if fsync_dir:
            flushed = read_directory(directory)
            try:
                os.fsync(flushed)
            finally:
                os.close(flushed)


def open_shared_replacement(path: Path, *, mode: int = SHARED_FILE_MODE) -> int:
    """A new, empty file with ``mode``, already renamed over ``path``: a descriptor open for writing it.

    For a file a process keeps writing while it runs (a log): the file is made beside ``path`` under a new name of its
    own (``O_EXCL``, so never through a link there) and put in place before the first byte, so ``path`` names this
    process's file from the start. What was at ``path`` (another leg's file, a link a member of the group put there)
    is replaced, never opened. The caller closes the descriptor.
    """
    with _directory_of(path) as (directory, name):
        descriptor, temporary = _new_temporary(directory, name, mode)
        try:
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise
    return descriptor


def create_new_shared_file(path: Path, *, mode: int = SHARED_FILE_MODE) -> int:
    """Create ``path`` new with ``mode``: a descriptor open for appending to it.

    The file is made under a name of its own and given its mode first, then linked to ``path`` (which fails for any
    name there already, a link included), so ``path`` never names it with another mode, even when the process is
    killed meanwhile. Where the file system makes no links it is made at ``path`` itself (``O_EXCL``, never through a
    link).

    Raises:
        FileExistsError: When anything, a link included, has that name already; it is left as it is.
    """
    with _directory_of(path) as (directory, name):
        descriptor, temporary = _new_temporary(directory, name, mode, os.O_APPEND)
        try:
            os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        except OSError as exc:
            os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            if exc.errno not in _NO_LINKS:
                raise
            return _create_in_place(directory, name, mode)
        except BaseException:
            os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise
        with suppress(OSError):
            os.unlink(temporary, dir_fd=directory)
    return descriptor


def _create_in_place(directory: int, name: str, mode: int) -> int:
    descriptor = os.open(name, _NEW_FILE | os.O_APPEND, _OWNER_ONLY, dir_fd=directory)
    try:
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def copy_shared_file(source: Path, path: Path, *, mode: int = SHARED_FILE_MODE) -> None:
    """Copy ``source`` to a new file beside ``path`` with ``mode`` and rename it over ``path``, as
    :func:`write_shared_bytes` does, without holding the content in memory."""
    with _directory_of(path) as (directory, name):
        descriptor, temporary = _new_temporary(directory, name, mode)
        try:
            with os.fdopen(descriptor, "wb") as handle, open(source, "rb") as content:
                shutil.copyfileobj(content, handle)
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise


def write_shared_text(path: Path, text: str, *, mode: int = SHARED_FILE_MODE) -> None:
    """:func:`write_shared_bytes` of ``text`` as UTF-8."""
    write_shared_bytes(path, text.encode("utf-8"), mode=mode)


def current_umask() -> int:
    """This process's umask, read from ``/proc/self/status``: setting it to learn it would race the process's threads.

    Raises:
        OSError: When the kernel does not report it (``ENOTSUP``).
    """
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("Umask:"):
                return int(line.split()[1], 8)
    raise OSError(errno.ENOTSUP, "the kernel reports no umask in /proc/self/status")


def writable_problems(dirs: Mapping[str, Path]) -> list[str]:
    """One line for each named directory this account cannot create and remove a file in:
    ``"<name> <path>: <errno name> (owner <uid>:<gid> <mode>)"``.

    A real write, not ``os.access``: on NFS and in a user namespace neither ``access()`` nor a comparison of ids says
    what a write will do.
    """
    problems = []
    for name, directory in dirs.items():
        try:
            descriptor, probe = tempfile.mkstemp(prefix=_PROBE_PREFIX, dir=directory)
            os.close(descriptor)
            os.unlink(probe)
        except OSError as exc:
            problems.append(f"{name} {directory}: {_errno_name(exc)}{ownership(directory)}")
    return problems


def setgid_warnings(dirs: Mapping[str, Path]) -> list[str]:
    """One line for each named directory without the setgid bit: what an account creates there takes that account's own
    group, so the next account of the run may not be able to write it."""
    warnings = []
    for name, directory in dirs.items():
        try:
            mode = os.stat(directory).st_mode
        except OSError:
            continue  # :func:`writable_problems` names a directory that cannot be read
        if not mode & stat.S_ISGID:
            warnings.append(
                f"{name} {directory} is not setgid: what is made in it takes the group of the account that makes it, "
                f"which the next account of the run may not share (chmod g+s {directory})"
            )
    return warnings


def _errno_name(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno, str(exc.errno)) if exc.errno is not None else type(exc).__name__


def ownership(path: Path) -> str:
    """``" (owner <uid>:<gid> <mode>)"`` of ``path`` for a message, or ``""`` when it cannot be read."""
    try:
        info = os.stat(path)
    except OSError:
        return ""
    return f" (owner {info.st_uid}:{info.st_gid} {stat.S_IMODE(info.st_mode):04o})"


__all__ = [
    "SHARED_DIR_MODE",
    "SHARED_FILE_MODE",
    "copy_shared_file",
    "create_new_shared_file",
    "create_shared_file",
    "current_umask",
    "make_shared_dirs",
    "open_directory",
    "read_directory",
    "open_shared_replacement",
    "ownership",
    "share_new_directory",
    "setgid_warnings",
    "writable_problems",
    "write_shared_bytes",
    "write_shared_text",
]
