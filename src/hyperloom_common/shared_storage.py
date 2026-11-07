# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Directories and files that several accounts of one group write in turn.

On shared storage each container of a run may be another host account. The accounts share one group, and the key
directories are made setgid in it once (``2775``), so what is created below them takes that group. What this module
creates has an explicit mode, set by ``fchmod`` after the create: directories ``2775`` and files ``664``, whatever the
umask. Nothing is chowned, and an existing entry is never chmodded (another account's would fail ``EPERM``): a file is
replaced by rename instead. A child process that writes such a tree gets :data:`SHARED_CHILD_UMASK` through its own
launch (``Popen(umask=...)``); the umask of this process is never changed, since its threads share it and it would
also open up the files only this account may read.

Nothing the application executes or sources may live in a tree the group can write.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

log = logging.getLogger(__name__)

SHARED_DIR_MODE = 0o2775
SHARED_FILE_MODE = 0o664
#: The umask of a child process that creates entries in a shared tree (git, shells, installers).
SHARED_CHILD_UMASK = 0o002

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_NEW_FILE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_APPEND = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC
_PROBE_PREFIX = ".hl-writable-"


def make_shared_dirs(path: Path, *, mode: int = SHARED_DIR_MODE) -> Path:
    """Create each missing directory of ``path`` with ``mode`` (default ``2775``) and return ``path``.

    The mode is set by ``fchmod`` on the new directory, opened without following a link, so the umask never decides it;
    the group comes from a setgid parent. An existing directory is left as it is. A new directory that did not keep the
    setgid bit (this account is not in its group, which a setgid parent gave it) is logged, naming it: what is made
    below it takes this account's group, which the next account may not share.
    """
    missing: list[Path] = []
    probe = Path(os.path.abspath(path))
    while not probe.is_dir():
        missing.append(probe)
        probe = probe.parent
    for directory in reversed(missing):
        try:
            os.mkdir(directory, mode)
        except FileExistsError:
            if directory.is_dir():
                continue  # made meanwhile by another process: theirs, left as it is
            raise
        share_new_directory(directory, mode=mode)
    return path


def share_new_directory(directory: Path, *, mode: int = SHARED_DIR_MODE) -> None:
    """Give a directory this process has just made (``mkdir``, ``mkdtemp``) its shared ``mode``, by ``fchmod`` on the
    directory opened without following a link; logged as in :func:`make_shared_dirs` when it does not keep setgid."""
    descriptor = os.open(directory, _DIRECTORY)
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


def create_shared_file(path: Path) -> None:
    """Create ``path`` empty with mode ``664``; an existing file is replaced by rename, never truncated or chmodded."""
    try:
        descriptor = os.open(path, _NEW_FILE, 0o600)  # the umask never decides it: fchmod below
    except FileExistsError:
        write_shared_bytes(path, b"")
        return
    try:
        os.fchmod(descriptor, SHARED_FILE_MODE)
    finally:
        os.close(descriptor)


def open_shared_append(path: Path) -> int:
    """``path`` opened for appending, never through a link: made ``664`` by ``fchmod`` when this call creates it, so the
    next account of the group appends to it too, and opened as it is when it exists. One another account made without
    group write fails ``EACCES``: a file that is appended to cannot be replaced by rename."""
    try:
        descriptor = os.open(path, _APPEND | os.O_CREAT | os.O_EXCL, 0o600)  # the umask never decides it: fchmod below
    except FileExistsError:
        return os.open(path, _APPEND)
    try:
        os.fchmod(descriptor, SHARED_FILE_MODE)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def open_shared(path: Path, access: int, mode: int = SHARED_FILE_MODE) -> int:
    """Open ``path`` for ``access`` (``os.O_RDWR``, ``os.O_WRONLY | os.O_APPEND``) without following a link, creating it
    with ``mode`` (default ``664``) when it is not there, and return the descriptor.

    A file that exists is opened as it is, never truncated, chmodded or replaced by rename: a lock or a log that
    another process holds open would be left on an unlinked inode. Another account's ``0600`` file therefore raises
    ``PermissionError``.
    """
    flags = access | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags, mode)  # the umask never decides it: fchmod below
    except FileExistsError:
        return os.open(path, access | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def write_shared_bytes(path: Path, data: bytes, *, mode: int = SHARED_FILE_MODE) -> None:
    """Write ``data`` to a new file beside ``path`` with ``mode`` and rename it over ``path``: the file is then this
    account's, whoever owned the one it replaced."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            os.fchmod(handle.fileno(), mode)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
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
    "SHARED_CHILD_UMASK",
    "SHARED_DIR_MODE",
    "SHARED_FILE_MODE",
    "create_shared_file",
    "current_umask",
    "make_shared_dirs",
    "open_shared",
    "open_shared_append",
    "ownership",
    "share_new_directory",
    "setgid_warnings",
    "writable_problems",
    "write_shared_bytes",
    "write_shared_text",
]
