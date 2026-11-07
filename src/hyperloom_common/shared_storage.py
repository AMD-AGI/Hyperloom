# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Directories and files that several accounts of one group write in turn.

On shared storage each container of a run may be another host account. The accounts share one group, and the key
directories are made setgid in it once (``2775``), so what is created below them takes that group.

Permissions are widened only where the storage is set up to be shared: when the nearest existing directory above what
is created is setgid, group-writable and not writable by others (:func:`parent_is_shared`). There what this module
creates has an explicit mode, set by ``fchmod`` after the create: directories ``2775`` (each one of a chain it makes)
and files ``664``, whatever the umask. Anywhere else (a ``0755`` home, a ``1777`` ``/tmp``, a setgid directory the group
cannot write) the umask alone decides, exactly as for a plain ``mkdir`` or ``open``, so what is made there stays as
private as the account keeps its files; a mode asked for is masked by the umask, never widened. Nothing is chowned,
and an existing entry is never chmodded (another account's would fail ``EPERM``): a file is replaced by rename instead.
A mode narrower than the shared one (``0600`` for a file that holds a credential) is kept as asked, shared or not.
A child process that writes such a tree gets :data:`SHARED_CHILD_UMASK` through its own
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
_PROBE_PREFIX = ".hl-writable-"


def parent_is_shared(path: Path) -> bool:
    """Whether what is created at ``path`` lands in storage set up to be shared: the nearest existing directory above
    ``path`` is setgid, group-writable and not writable by others.

    That directory is read with ``lstat``, so a link standing there is not followed and counts as not shared; the
    components above it are resolved as for any path. A directory that cannot be read counts as not shared: the umask
    then decides, which can only keep what is made more private.
    """
    probe = Path(os.path.abspath(path)).parent
    while True:
        try:
            mode = os.lstat(probe).st_mode
        except FileNotFoundError:
            if probe.parent == probe:
                return False
            probe = probe.parent
            continue
        except OSError:
            return False
        return (
            stat.S_ISDIR(mode) and bool(mode & stat.S_ISGID) and bool(mode & stat.S_IWGRP) and not mode & stat.S_IWOTH
        )


def _private_mode(mode: int) -> int:
    """``mode`` as a plain create under this process's umask would leave it."""
    return mode & ~current_umask()


def make_shared_dirs(path: Path, *, mode: int = SHARED_DIR_MODE) -> Path:
    """Create each missing directory of ``path`` with ``mode`` (default ``2775``) and return ``path``.

    Below shared storage (:func:`parent_is_shared`, judged once, at the nearest directory that exists) the mode is set
    by ``fchmod`` on each new directory, opened without following a link, so the umask never decides it; the group
    comes from a setgid parent. Anywhere else ``mode`` is only asked of ``mkdir``, which masks it with the umask. An
    existing directory is left as it is. A new shared directory that did not keep the setgid bit (this account is not in
    its group, which a setgid parent gave it) is logged, naming it: what is made below it takes this account's group,
    which the next account may not share.
    """
    missing: list[Path] = []
    probe = Path(os.path.abspath(path))
    while not probe.is_dir():
        missing.append(probe)
        probe = probe.parent
    shared = bool(missing) and parent_is_shared(missing[-1])
    for directory in reversed(missing):
        try:
            os.mkdir(directory, mode)
        except FileExistsError:
            if directory.is_dir():
                continue  # made meanwhile by another process: theirs, left as it is
            raise
        _widen_new_directory(directory, mode, shared=shared)
    return path


def share_new_directory(directory: Path, *, mode: int = SHARED_DIR_MODE) -> None:
    """Give a directory this process has just made (``mkdir``, ``mkdtemp``) its shared ``mode``, by ``fchmod`` on the
    directory opened without following a link, when it is made below shared storage (:func:`parent_is_shared`); logged
    as in :func:`make_shared_dirs` when it does not keep setgid. Anywhere else it is left as it was made; a link there
    is refused either way."""
    _widen_new_directory(directory, mode, shared=parent_is_shared(directory))


def _widen_new_directory(directory: Path, mode: int, *, shared: bool) -> None:
    descriptor = os.open(directory, _DIRECTORY)  # a link is refused, shared or not
    try:
        if not shared:
            return
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
    """Create ``path`` empty with mode ``664`` below shared storage (:func:`parent_is_shared`), else as the umask leaves
    ``664``; an existing file is replaced by rename, never truncated or chmodded."""
    if not parent_is_shared(path):
        try:
            os.close(os.open(path, _NEW_FILE, SHARED_FILE_MODE))  # the umask masks it, as for any create
        except FileExistsError:
            write_shared_bytes(path, b"")
        return
    try:
        descriptor = os.open(path, _NEW_FILE, 0o600)  # the umask never decides it: fchmod below
    except FileExistsError:
        write_shared_bytes(path, b"")
        return
    try:
        os.fchmod(descriptor, SHARED_FILE_MODE)
    finally:
        os.close(descriptor)


def write_shared_bytes(path: Path, data: bytes, *, mode: int = SHARED_FILE_MODE) -> None:
    """Write ``data`` to a new file beside ``path`` with ``mode`` and rename it over ``path``: the file is then this
    account's, whoever owned the one it replaced. Below shared storage (:func:`parent_is_shared`) the file has ``mode``
    whatever the umask; anywhere else ``mode`` masked by the umask."""
    final_mode = mode if parent_is_shared(path) else _private_mode(mode)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            os.fchmod(handle.fileno(), final_mode)
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
    group, and the helpers of this module keep it as private as the umask leaves it, so the next account of the run may
    not be able to write it."""
    warnings = []
    for name, directory in dirs.items():
        try:
            mode = os.stat(directory).st_mode
        except OSError:
            continue  # :func:`writable_problems` names a directory that cannot be read
        if not mode & stat.S_ISGID:
            warnings.append(
                f"{name} {directory} is not setgid: what is made in it takes the group of the account that makes it and is "
                f"not opened to the group, so the next account of the run may not be able to write it "
                f"(chmod g+s {directory})"
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
    "ownership",
    "parent_is_shared",
    "share_new_directory",
    "setgid_warnings",
    "writable_problems",
    "write_shared_bytes",
    "write_shared_text",
]
