# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomic filesystem writes (canonical ``atomic_write*``).

Write to a sibling temp file in the same directory, then ``os.replace`` into
place, so a reader never observes a half-written file. Stdlib-only so any
package may depend on it without creating an import cycle.

Behaviour-preserving flags let each call site delegate here without any
observable change:

* ``make_parents`` — create ``path.parent`` first.
* ``atomic_write_json``: ``indent`` / ``sort_keys`` / ``ensure_ascii`` /
  ``trailing_newline`` mirror the exact ``json.dump`` shape each site used.

Sites intentionally NOT delegated here (kept local by design):

* ``action_executors/_magpie_patcher.atomic_write_text`` — returns ``bool``,
  takes keyword args, ``chmod``-mirrors the target, and relies on
  module-global ``os``/``tempfile`` being monkeypatched by its tests.
* ``src/hyperloom/agents/kernel/tools/geak_prompt_patcher._atomic_write`` —
  ``shutil.copystat`` preserves the target's mode.
* ``multi_node/scripts/*._atomic_write_bytes`` — shipped to remote nodes and run
  standalone, so they must not gain a ``hyperloom`` import dependency.
"""

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


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    make_parents: bool = False,
    fsync: bool = False,
    mode: int | None = None,
) -> None:
    """Atomically write ``data`` to ``path`` (temp file in same dir + ``os.replace``).

    Args:
        path: Destination file path.
        data: Bytes to write.
        make_parents: When ``True``, create ``path.parent`` (``parents=True,
            exist_ok=True``) before writing.
        fsync: When ``True``, best-effort ``os.fsync`` the temp file before the
            rename (OSError swallowed on mounts that reject the syscall).
        mode: Optional file mode applied to the temp file before rename.

    Raises:
        Exception: Re-raised after a best-effort unlink of the temp file when
            writing or replacing fails.
    """
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
            # Strip group/other bits: written files may hold sensitive payloads,
            # so never expose them beyond the owner regardless of caller intent.
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
    mode: int | None = None,
) -> None:
    """Atomically write ``text`` to ``path`` (temp file in same dir + ``os.replace``).

    Args:
        path: Destination file path.
        text: Full file contents to write.
        encoding: Text encoding for the temp file (default ``utf-8``).
        make_parents: When ``True``, create ``path.parent`` before writing.
        fsync: When ``True``, best-effort ``os.fsync`` the temp file before the
            rename (OSError swallowed on mounts that reject the syscall).
        mode: Optional file mode applied to the temp file before rename.

    Raises:
        Exception: Re-raised after a best-effort unlink of the temp file when
            writing or replacing fails.
    """
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
    mode: int | None = None,
) -> None:
    """Atomically write ``data`` as JSON to ``path``.

    Args:
        path: Destination file path.
        data: JSON-serialisable object.
        indent: ``json.dumps`` indent (default ``2``).
        sort_keys: ``json.dumps`` ``sort_keys`` (default ``True``).
        ensure_ascii: ``json.dumps`` ``ensure_ascii`` (default ``True``).
        trailing_newline: Append a final ``"\\n"`` after the JSON body.
        make_parents: When ``True`` (default), create ``path.parent`` first.
        fsync: When ``True``, best-effort ``os.fsync`` before the rename.
        mode: Optional file mode applied to the temp file before rename.
    """
    text = _json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    if trailing_newline:
        text += "\n"
    atomic_write_text(path, text, make_parents=make_parents, fsync=fsync, mode=mode)


def append_jsonl(
    path: Path,
    row: Any,
    *,
    make_parents: bool = False,
    fsync: bool = False,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> None:
    """Append one JSON object as a line to a JSONL file.

    Serialises *row* with ``json.dumps`` and writes it plus a trailing newline
    in ``"a"`` mode. Not atomic across processes, but a single ``write`` of a
    compact single-line record is the standard append-log idiom.

    Args:
        path: Destination JSONL file.
        row: JSON-serialisable value to append.
        make_parents: When ``True``, create ``path.parent`` first.
        fsync: When ``True``, best-effort ``os.fsync`` after the write.
        ensure_ascii: ``json.dumps`` ``ensure_ascii`` (default ``True``).
        sort_keys: ``json.dumps`` ``sort_keys`` (default ``False``).
    """
    path = Path(path)
    if make_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    line = _json.dumps(row, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        if fsync:
            _best_effort_fsync(fh)


def safe_mtime(path: Path) -> float:
    """Return ``path``'s modification time, or ``0.0`` when ``stat()`` fails.

    Never raises: a missing entry (concurrent cleanup mid-scan) or an
    ``OSError`` from ``stat`` (e.g. an NFS stale handle) degrades to ``0.0``,
    which sorts oldest for the ``key=`` / mtime-cutoff comparisons that use it.

    Args:
        path: Filesystem path to stat.

    Returns:
        The ``st_mtime`` of ``path``, or ``0.0`` on any ``stat()`` failure.
    """
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
