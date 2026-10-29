# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared filesystem helpers: file classification, path helpers, and reads and writes that never follow a link a less-privileged process planted."""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from pathlib import Path
from typing import IO, Callable, Iterator, Sequence


# Mounts that can be revoked mid-run: a process whose cwd lives on one sees relative-path writes fail with ENOENT
# after a flap, so callers place anything they must keep writing to on local disk instead.
_NETWORK_FS_TYPES: frozenset[str] = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "lustre",
        "glusterfs",
        "ceph",
        "fuse.weka",
        "wekafs",
        "wekafsgw",
        "fuse.juicefs",
        "fuse.s3fs",
        "fuse.sshfs",
        "9p",
    }
)


def _path_fstype(path: str) -> str:
    """Return the filesystem type backing ``path`` per ``/proc/mounts``."""
    try:
        rp = os.path.realpath(path)
    except OSError:
        return ""
    best_mp = ""
    best_type = ""
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                # /proc/mounts octal-escapes spaces in the mountpoint.
                try:
                    mp = parts[1].encode("latin-1").decode("unicode_escape")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    mp = parts[1]
                fstype = parts[2]
                norm = mp.rstrip("/") or "/"
                is_under = norm == "/" or rp == norm or rp.startswith(norm + "/")
                if is_under and len(norm) >= len(best_mp):
                    best_mp = norm
                    best_type = fstype
    except OSError:
        return ""
    return best_type


def is_network_fs(path: str) -> bool:
    """True when ``path`` is backed by a revocable network filesystem."""
    return _path_fstype(path).lower() in _NETWORK_FS_TYPES


_PRIVATE_MODE = 0o600
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


def create_nofollow(base: Path, rel: Sequence[str], *, mode: int = _PRIVATE_MODE) -> IO[str]:
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


def write_text_nofollow(base: Path, rel: Sequence[str], text: str, *, mode: int = _PRIVATE_MODE) -> Path:
    """Write ``text`` to ``base/<rel...>`` as :func:`create_nofollow` creates it, and return the path."""
    with create_nofollow(base, rel, mode=mode) as handle:
        handle.write(text)
    return Path(base, *rel)


def rewrite_lines_nofollow(path: Path | str, transform: Callable[[str], str], *, mode: int = _PRIVATE_MODE) -> None:
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
            try:
                if _stage_transformed(source, dir_fd, staged, transform, mode):
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


def _stage_transformed(source: IO[str], dir_fd: int, staged: str, transform: Callable[[str], str], mode: int) -> bool:
    """Write ``source`` through ``transform`` into a fresh ``staged``; whether any line changed."""
    with contextlib.suppress(FileNotFoundError):
        os.unlink(staged, dir_fd=dir_fd)
    out_fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=dir_fd)
    changed = False
    with os.fdopen(out_fd, "w", encoding="utf-8") as destination:
        for line in source:
            new = transform(line)
            changed = changed or new != line
            destination.write(new)
        destination.flush()
        os.fchmod(out_fd, mode)
    return changed


def append_text_nofollow(base: Path, rel: Sequence[str], text: str, *, mode: int = _PRIVATE_MODE) -> None:
    """Append ``text`` to the regular file ``base/<rel...>``, creating it; refuse a symlink or a multiply linked file."""
    *dirs, name = rel
    dir_fd = _open_below(base, dirs, create=True)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError(errno.EINVAL, "not a plain regular file", str(Path(base, *rel)))
        handle.write(text)


def read_text_below_nofollow(base: Path, rel: Sequence[str]) -> str:
    """Return the UTF-8 text of the regular file ``base/<rel...>``, opening every component without following a link."""
    *dirs, name = rel
    dir_fd = _open_below(base, dirs, create=False)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(Path(base, *rel)))
        return handle.read().decode("utf-8", errors="replace")


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


def relative_posix(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` as a POSIX string, both resolved; ``str(path)`` when it is not under ``root``."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)
