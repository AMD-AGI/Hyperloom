# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Atomic filesystem writes (tree-reform.MD §7 — canonical ``atomic_write*``).

Single home for the "write to a sibling temp file in the same directory, then
``os.replace`` into place" idiom that was independently re-implemented across
the codebase. A reader never observes a half-written file: it sees either the
old contents or the complete new contents, never a truncated one.

Zero first-party imports (stdlib only) so any package may depend on it without
creating an import cycle (see ``tree-reform.MD`` §7 "防环规则").

Behaviour-preserving flags let each legacy call site delegate here without any
observable change:

* ``make_parents`` — create ``path.parent`` first (some sites did, some did not).
* ``atomic_write_json``: ``indent`` / ``sort_keys`` / ``trailing_newline`` mirror
  the exact ``json.dump`` shape each site used.

Sites intentionally NOT delegated here (kept local by design):

* ``action_executors/_magpie_patcher.atomic_write_text`` — returns ``bool``,
  takes keyword args, ``chmod``-mirrors the target, and relies on
  module-global ``os``/``tempfile`` being monkeypatched by its tests.
* ``src/hyperloom/agents/kernel/tools/geak_prompt_patcher._atomic_write`` —
  ``shutil.copystat`` preserves the target's mode.
* ``recipe_kb/local_store._atomic_write_json`` — best-effort ``fsync`` + DEBUG
  logging for durability on journaling mounts.
* ``multi_node/scripts/*._atomic_write_bytes`` — shipped to remote nodes and run
  standalone, so they must not gain a ``hyperloom`` import dependency.
"""

from __future__ import annotations

import json as _json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path, data: bytes, *, make_parents: bool = False) -> None:
    """Atomically write ``data`` to ``path`` (temp file in same dir + ``os.replace``).

    Args:
        path: Destination file path.
        data: Bytes to write.
        make_parents: When ``True``, create ``path.parent`` (``parents=True,
            exist_ok=True``) before writing.

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
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    make_parents: bool = False,
) -> None:
    """Atomically write ``text`` to ``path`` (temp file in same dir + ``os.replace``).

    Args:
        path: Destination file path.
        text: Full file contents to write.
        encoding: Text encoding for the temp file (default ``utf-8``).
        make_parents: When ``True``, create ``path.parent`` before writing.

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
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    trailing_newline: bool = False,
    make_parents: bool = True,
) -> None:
    """Atomically write ``data`` as JSON to ``path``.

    Args:
        path: Destination file path.
        data: JSON-serialisable object.
        indent: ``json.dumps`` indent (default ``2``).
        sort_keys: ``json.dumps`` ``sort_keys`` (default ``True``).
        trailing_newline: Append a final ``"\\n"`` after the JSON body.
        make_parents: When ``True`` (default), create ``path.parent`` first.
    """
    text = _json.dumps(data, indent=indent, sort_keys=sort_keys)
    if trailing_newline:
        text += "\n"
    atomic_write_text(path, text, make_parents=make_parents)


__all__ = ["atomic_write_bytes", "atomic_write_text", "atomic_write_json"]
