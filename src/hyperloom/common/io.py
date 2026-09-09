# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomic filesystem writes (canonical ``atomic_write*``)."""

from __future__ import annotations

import json as _json
import os
import tempfile
from contextlib import suppress
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


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    make_parents: bool = False,
    fsync: bool = False,
    mode: int | None = None,
) -> None:
    """Atomically write ``data`` to ``path`` (temp file in same dir + ``os.replace``)."""
    path = Path(path)
    if make_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            if fsync:
                _best_effort_fsync(fh)
        if mode is not None:
            # Strip group/other bits: written files may hold sensitive payloads, so never expose them beyond the owner
            # regardless of caller intent.
            os.chmod(tmp, mode & 0o700)
        os.replace(tmp, path)
    except Exception:
        with suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    make_parents: bool = False,
    fsync: bool = False,
    fsync_dir: bool = False,
    mode: int | None = None,
) -> None:
    """Atomically write ``text`` to ``path`` (temp file in same dir + ``os.replace``)."""
    path = Path(path)
    if make_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            if fsync:
                _best_effort_fsync(fh)
        if mode is not None:
            # Strip group/other bits: never expose written payloads beyond owner.
            os.chmod(tmp, mode & 0o700)
        os.replace(tmp, path)
        if fsync_dir:
            _best_effort_fsync_dir(path.parent)
    except Exception:
        with suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
    trailing_newline: bool = False,
    make_parents: bool = True,
    fsync: bool = False,
    fsync_dir: bool = False,
    mode: int | None = None,
) -> None:
    """Atomically write ``data`` as JSON to ``path``."""
    text = _json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    if trailing_newline:
        text += "\n"
    atomic_write_text(
        path,
        text,
        make_parents=make_parents,
        fsync=fsync,
        fsync_dir=fsync_dir,
        mode=mode,
    )


def append_jsonl(
    path: Path,
    row: Any,
    *,
    make_parents: bool = False,
    fsync: bool = False,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> None:
    """Append one JSON object as a line to a JSONL file."""
    path = Path(path)
    if make_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    line = _json.dumps(row, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
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
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "append_jsonl",
    "safe_mtime",
]
