# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Live extensions to a running specialist's kill instant, and how far each may reach."""

from __future__ import annotations

import math

__all__ = [
    "clear_wall_budget_extension",
    "grant_wall_budget_extension",
    "limit_wall_budget_extension",
    "wall_budget_extension",
]

# Extensions ``extend_lease`` granted while a specialist is already spawned, keyed by task_id. The reap loop re-reads
# this every poll, so an extension moves the kill instant of a run in flight; the run clears its entry when it ends.
_WALL_BUDGET_EXTENSIONS: dict[str, float] = {}

# The most a task's extensions may add up to, set when it is dispatched; a task without an entry is not limited.
_WALL_BUDGET_EXTENSION_ROOM: dict[str, float] = {}


def _key(task_id: str) -> str:
    """The registry key for ``task_id``."""
    return str(task_id or "").strip()


def limit_wall_budget_extension(task_id: str, room_sec: float | None) -> None:
    """Bound how far ``extend_lease`` may move a specialist's kill instant for its coming run.

    Args:
        task_id: The specialist task being dispatched.
        room_sec: The most its extensions may add up to; ``None`` leaves them unlimited.
    """
    if room_sec is None:
        _WALL_BUDGET_EXTENSION_ROOM.pop(_key(task_id), None)
    else:
        _WALL_BUDGET_EXTENSION_ROOM[_key(task_id)] = max(0.0, float(room_sec))


def grant_wall_budget_extension(task_id: str, extra_sec: float) -> float:
    """Add ``extra_sec`` to a live specialist's hard wall-clock deadline, within the room its dispatch allowed.

    Safe to call for a task that is not running a subprocess (the entry is
    simply never read and is cleared on the next dispatch of that task_id).

    Args:
        task_id: The specialist task whose deadline should move.
        extra_sec: Seconds to add; non-positive values are a no-op.

    Returns:
        The task's cumulative granted extension in seconds.
    """
    key = _key(task_id)
    if not key or extra_sec <= 0:
        return _WALL_BUDGET_EXTENSIONS.get(key, 0.0)
    total = min(
        _WALL_BUDGET_EXTENSIONS.get(key, 0.0) + float(extra_sec),
        _WALL_BUDGET_EXTENSION_ROOM.get(key, math.inf),
    )
    _WALL_BUDGET_EXTENSIONS[key] = total
    return total


def wall_budget_extension(task_id: str) -> float:
    """Return the cumulative live extension granted to ``task_id`` (0 if none).

    Args:
        task_id: The specialist task to look up.

    Returns:
        Seconds of extension granted so far.
    """
    return _WALL_BUDGET_EXTENSIONS.get(_key(task_id), 0.0)


def clear_wall_budget_extension(task_id: str) -> None:
    """Drop any recorded extension for ``task_id``.

    Args:
        task_id: The specialist task whose entry should be removed.
    """
    _WALL_BUDGET_EXTENSIONS.pop(_key(task_id), None)
