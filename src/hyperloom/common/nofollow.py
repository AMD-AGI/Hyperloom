# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read and write paths a less-privileged process can rewrite, without following its symlinks.

A specialist that runs as another user owns the group-write bit on its worktree
and workspace, so it can replace any name in them with a symlink. The app (root,
or a member of the account's group) must not follow one: a followed link makes
the app truncate or read a file the specialist could never touch itself.

Every function here refuses a symlink in the last component, and for writes in
every directory below the ``base`` the caller names, by opening each component
with ``O_NOFOLLOW`` relative to the directory just opened. A refusal is an
``OSError`` (``ELOOP``/``ENOTDIR``/``EISDIR``), so callers that already handle
a failed read or write need no new branch.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from pathlib import Path
from typing import IO, Callable, Iterator, Sequence

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_below(base: Path, parts: Sequence[str], *, create: bool) -> int:
    """Open ``base`` then each of ``parts`` as a directory, never through a symlink; return the last fd."""
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for name in parts:
            if create:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(name, 0o777, dir_fd=fd)
            nxt = os.open(name, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
    except BaseException:
        os.close(fd)
        raise
    return fd


def read_bytes_nofollow(path: Path | str, *, limit: int | None = None) -> bytes:
    """Return the bytes of the regular file at ``path``; refuse a symlink, a FIFO or a device.

    ``limit``: refuse (``EFBIG``) a file longer than this many bytes, without reading past it.
    """
    # O_NONBLOCK keeps a planted FIFO from hanging the app; regular files ignore it.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        data = handle.read(None if limit is None else limit + 1)
    if limit is not None and len(data) > limit:
        raise OSError(errno.EFBIG, "file too large", str(path))
    return data


def read_text_nofollow(path: Path | str) -> str:
    """Return the UTF-8 text of the regular file at ``path``, refusing a symlink as :func:`read_bytes_nofollow` does."""
    return read_bytes_nofollow(path).decode("utf-8")


def create_nofollow(base: Path, rel: Sequence[str], *, mode: int = 0o600) -> IO[str]:
    """Create ``base/<rel...>`` fresh and return it open for text writing.

    Directories on the way are opened without following links and created when
    missing. An existing entry at the final name (a symlink, a hard link, a
    FIFO) is unlinked, never written through, and the file is created with
    ``O_EXCL``. ``mode`` (owner-only by default) is subject to the process umask.
    """
    *dirs, name = rel
    dir_fd = _open_below(base, dirs, create=True)
    try:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name, dir_fd=dir_fd)
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    return os.fdopen(fd, "w", encoding="utf-8")


def write_text_nofollow(base: Path, rel: Sequence[str], text: str, *, mode: int = 0o600) -> Path:
    """Write ``text`` to ``base/<rel...>`` as :func:`create_nofollow` creates it, and return the path."""
    with create_nofollow(base, rel, mode=mode) as handle:
        handle.write(text)
    return Path(base, *rel)


def rewrite_lines_nofollow(path: Path | str, transform: Callable[[str], str], *, mode: int = 0o600) -> None:
    """Rewrite the regular file ``path`` line by line through ``transform``, in place, without following links.

    The result is staged in ``<name>.redacting`` beside the file, created fresh
    (an entry already there is unlinked, never written through) and moved over
    the file only when ``transform`` changed something. A symlink, a FIFO or a
    multiply linked file at ``path`` is refused. The file ends with ``mode``.
    Raises ``OSError`` when it cannot be done; a staged file is removed first.
    """
    target = Path(path)
    name, staged = target.name, target.name + ".redacting"
    dir_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        src_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dir_fd)
        with os.fdopen(src_fd, "r", encoding="utf-8", errors="replace") as source:
            info = os.fstat(src_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError(errno.EINVAL, "not a plain regular file", str(target))
            with contextlib.suppress(FileNotFoundError):
                os.unlink(staged, dir_fd=dir_fd)
            out_fd = os.open(
                staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=dir_fd
            )
            try:
                changed = False
                with os.fdopen(out_fd, "w", encoding="utf-8") as destination:
                    for line in source:
                        new = transform(line)
                        changed = changed or new != line
                        destination.write(new)
                    destination.flush()
                    os.fchmod(out_fd, mode)
                if changed:
                    os.replace(staged, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
                else:
                    os.unlink(staged, dir_fd=dir_fd)
                    os.fchmod(src_fd, mode)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(staged, dir_fd=dir_fd)
                raise
    finally:
        os.close(dir_fd)


def regular_files_nofollow(base: Path, subdir: str, suffixes: Sequence[str]) -> Iterator[Path]:
    """Sorted regular files ``base/subdir/*<suffix>``; nothing when ``subdir`` or an entry is a symlink."""
    try:
        dir_fd = _open_below(base, (subdir,), create=False)
    except OSError:
        return
    try:
        names = sorted(n for n in os.listdir(dir_fd) if n.endswith(tuple(suffixes)) and _is_regular(n, dir_fd))
    finally:
        os.close(dir_fd)
    for name in names:
        yield base / subdir / name


def _is_regular(name: str, dir_fd: int) -> bool:
    try:
        return stat.S_ISREG(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
    except OSError:
        return False
