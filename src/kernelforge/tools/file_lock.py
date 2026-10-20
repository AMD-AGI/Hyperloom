# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Advisory ``flock`` helpers that coordinate separate processes through one file."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from typing import IO, Any


@contextmanager
def locked_json_state(path: Path) -> Iterator[dict[str, Any]]:
    """Read one small JSON object under an exclusive lock and write it back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", errors="replace") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read().strip()
            try:
                state = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                state = {}
            if not isinstance(state, dict):
                state = {}
            yield state
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(state, sort_keys=True))
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def try_lock_exclusive(path: Path) -> IO[str] | None:
    """Take an exclusive lock on an existing file without waiting, or return None."""
    try:
        handle = path.open("r+", encoding="utf-8")
    except OSError:
        return None
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def release_lock(handle: IO[str]) -> None:
    """Drop a lock taken by ``try_lock_exclusive`` and close its handle."""
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
