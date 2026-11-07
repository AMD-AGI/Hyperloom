# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomic filesystem writes (canonical ``atomic_write*``) and the fsync primitives that make them durable."""

from __future__ import annotations

import json as _json
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.shared_storage import make_shared_dirs, write_shared_bytes

_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)


def _fsync_handle(fh: Any) -> None:
    fh.flush()
    os.fsync(fh.fileno())


def _fsync_file(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: str | Path) -> None:
    """Flush one directory's metadata so a rename into it survives a crash; a no-op where directories have no fds."""
    if not hasattr(os, "O_DIRECTORY"):  # Windows has no directory fds
        return
    descriptor = os.open(str(path), _DIRECTORY_FLAGS)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _walk_depth_first(root: Path) -> Iterator[tuple[Path, list[str]]]:
    """Yield ``(directory, filenames)`` under ``root``, children before parents.

    A directory that cannot be enumerated raises rather than being skipped: the
    caller is about to rename this tree into place and its durability claim only
    holds if the whole tree was visited.
    """

    def _reraise(error: OSError) -> None:
        raise error

    for directory, _subdirectories, filenames in os.walk(root, topdown=False, onerror=_reraise):
        yield Path(directory), filenames


def fsync_tree(root: Path) -> None:
    """Flush every file and directory under ``root`` before it is renamed."""
    for directory, filenames in _walk_depth_first(root):
        for filename in filenames:
            _fsync_file(directory / filename)
        fsync_directory(directory)


def fsync_tree_directories(root: Path) -> None:
    """Flush every directory under ``root`` for a tree whose files were fsynced as they were written."""
    for directory, _filenames in _walk_depth_first(root):
        fsync_directory(directory)


@dataclass(frozen=True)
class WriteOptions:
    """How an atomic replace treats parent directories, durability and permission bits.

    A new file is owner-only (``0o600``). ``mode`` is applied with group/other bits
    stripped; ``preserve_mode`` instead keeps an existing file's permission bits and
    is ignored when ``mode`` is given. ``group_read`` lets the directory's group read a
    new file when the directory is shared with that group (setgid and group-readable);
    never other-readable, and ignored when ``mode`` or ``preserve_mode`` applies.
    ``shared`` is the one exception to owner-only, and only where the storage is set up to
    be shared (the nearest existing directory is setgid, group-writable and not writable by
    others): there the new file is ``0o664``
    (:data:`~hyperloom_common.shared_storage.SHARED_FILE_MODE`), for a tree the accounts of
    one group write in turn, and the rename replaces the file whoever owned it; the three
    options above are then ignored. The parents ``make_parents`` creates follow
    :func:`~hyperloom_common.shared_storage.make_shared_dirs` (``2775`` there, the umask
    elsewhere). Anywhere else ``shared`` widens nothing and the file has the mode the other
    options ask for. Shared or not, no link is followed on the way below a directory other
    accounts can write (:func:`~hyperloom_common.shared_storage.write_shared_bytes`). Never
    use it for a file that holds a credential or that anything executes or sources.
    ``fsync`` and ``fsync_dir`` raise the
    ``OSError`` of a flush that fails: a caller that asks for durability learns
    when it did not get it.
    """

    make_parents: bool = False
    fsync: bool = False
    fsync_dir: bool = False
    mode: int | None = None
    preserve_mode: bool = False
    group_read: bool = False
    shared: bool = False


#: A crash-safe replace: parents created, the data and the rename flushed to disk, an existing file's mode kept.
DURABLE = WriteOptions(make_parents=True, fsync=True, fsync_dir=True, preserve_mode=True)


@dataclass(frozen=True)
class JsonFormat:
    """How :func:`atomic_write_json` serializes a document."""

    indent: int | None = 2
    sort_keys: bool = True
    ensure_ascii: bool = True
    trailing_newline: bool = False


@dataclass(frozen=True)
class JsonLineFormat:
    """How :func:`append_jsonl` serializes one row onto its line."""

    ensure_ascii: bool = True
    sort_keys: bool = False


_JSON_WRITE = WriteOptions(make_parents=True)


def _owner_mode(tmp: Path, path: Path, options: WriteOptions) -> None:
    os.chmod(tmp, _private_mode(path, options))


def _private_mode(path: Path, options: WriteOptions) -> int:
    """The mode of a new file at ``path`` (a ``shared`` write's outside shared storage): owner-only, ``options.mode``
    with group and other bits stripped, an existing file's with ``preserve_mode``, or with ``group_read`` the group's
    read of a setgid directory."""
    if options.mode is not None:
        # Strip group/other bits: written files may hold sensitive payloads, so never expose them beyond the owner
        # regardless of caller intent.
        return options.mode & 0o700
    if options.preserve_mode and path.is_file():
        return stat.S_IMODE(path.stat().st_mode)
    if options.group_read:
        directory = path.parent.stat().st_mode
        if directory & stat.S_ISGID:
            return stat.S_IRUSR | stat.S_IWUSR | (directory & stat.S_IRGRP)
    return stat.S_IRUSR | stat.S_IWUSR


def atomic_write_bytes(path: str | Path, data: bytes, *, options: WriteOptions = WriteOptions()) -> None:
    """Atomically write ``data`` to ``path`` (temp file in same dir + ``os.replace``)."""
    path = Path(path)
    if options.shared:
        if options.make_parents:
            make_shared_dirs(path.parent)
        write_shared_bytes(
            path,
            data,
            private_mode=_private_mode(path, options),
            fsync=options.fsync,
            fsync_dir=options.fsync_dir,
        )
        return
    if options.make_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            if options.fsync:
                _fsync_handle(fh)
        _owner_mode(tmp, path, options)
        os.replace(tmp, path)
        if options.fsync_dir:
            fsync_directory(path.parent)
    except Exception:
        with suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_text(path: str | Path, text: str, *, options: WriteOptions = WriteOptions()) -> None:
    """Atomically write ``text`` to ``path`` as UTF-8; see :func:`atomic_write_bytes`."""
    atomic_write_bytes(path, text.encode("utf-8"), options=options)


def atomic_write_json(
    path: str | Path,
    data: Any,
    *,
    fmt: JsonFormat = JsonFormat(),
    options: WriteOptions = _JSON_WRITE,
) -> None:
    """Atomically write ``data`` as JSON to ``path``; parent directories are created by default."""
    text = _json.dumps(data, indent=fmt.indent, sort_keys=fmt.sort_keys, ensure_ascii=fmt.ensure_ascii)
    if fmt.trailing_newline:
        text += "\n"
    atomic_write_text(path, text, options=options)


def append_jsonl(
    path: Path,
    row: Any,
    *,
    make_parents: bool = False,
    fsync: bool = False,
    fmt: JsonLineFormat = JsonLineFormat(),
) -> None:
    """Append one JSON object as a line to a JSONL file."""
    path = Path(path)
    if make_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    line = _json.dumps(row, ensure_ascii=fmt.ensure_ascii, sort_keys=fmt.sort_keys)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        if fsync:
            _fsync_handle(fh)


def safe_mtime(path: Path) -> float:
    """Return ``path``'s modification time, or ``0.0`` when ``stat()`` fails."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


__all__ = [
    "DURABLE",
    "JsonFormat",
    "JsonLineFormat",
    "WriteOptions",
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "append_jsonl",
    "fsync_directory",
    "fsync_tree",
    "fsync_tree_directories",
    "safe_mtime",
]
