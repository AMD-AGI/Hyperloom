# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomic filesystem writes (canonical ``atomic_write*``)."""

from __future__ import annotations

import json as _json
import os
import shutil
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _best_effort_fsync(fh: Any) -> None:
    """``os.fsync`` the file handle, swallowing OSError (tmpfs/path reject it)."""
    with suppress(OSError):
        fh.flush()
        os.fsync(fh.fileno())


def _best_effort_fsync_dir(directory: Path) -> None:
    """``os.fsync`` a directory so a rename survives a crash."""
    if not hasattr(os, "O_DIRECTORY"):  # Windows has no directory fds
        return
    with suppress(OSError):
        fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@dataclass(frozen=True)
class WriteOptions:
    """How an atomic replace treats parent directories, durability and permission bits.

    A new file is owner-only (``0o600``). ``mode`` is applied with group/other bits
    stripped; ``preserve_mode`` instead keeps an existing file's permission bits and
    is ignored when ``mode`` is given.
    """

    make_parents: bool = False
    fsync: bool = False
    fsync_dir: bool = False
    mode: int | None = None
    preserve_mode: bool = False


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


def atomic_write_bytes(path: Path, data: bytes, *, options: WriteOptions = WriteOptions()) -> None:
    """Atomically write ``data`` to ``path`` (temp file in same dir + ``os.replace``)."""
    path = Path(path)
    if options.make_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            if options.fsync:
                _best_effort_fsync(fh)
        if options.mode is not None:
            # Strip group/other bits: written files may hold sensitive payloads, so never expose them beyond the owner
            # regardless of caller intent.
            os.chmod(tmp, options.mode & 0o700)
        elif options.preserve_mode and path.is_file():
            shutil.copymode(path, tmp)
        os.replace(tmp, path)
        if options.fsync_dir:
            _best_effort_fsync_dir(path.parent)
    except Exception:
        with suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_text(path: Path, text: str, *, options: WriteOptions = WriteOptions()) -> None:
    """Atomically write ``text`` to ``path`` as UTF-8; see :func:`atomic_write_bytes`."""
    atomic_write_bytes(path, text.encode("utf-8"), options=options)


def atomic_write_json(
    path: Path,
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
            _best_effort_fsync(fh)


def safe_mtime(path: Path) -> float:
    """Return ``path``'s modification time, or ``0.0`` when ``stat()`` fails."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


__all__ = [
    "JsonFormat",
    "JsonLineFormat",
    "WriteOptions",
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "append_jsonl",
    "safe_mtime",
]
