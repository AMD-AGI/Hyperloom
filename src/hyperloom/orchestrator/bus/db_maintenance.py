# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DB retention / pruning for multi-day single-session runs."""

from __future__ import annotations

from dataclasses import dataclass

from .storage.connection import SqliteConnection


# Keep at least this many most-recent events: the most one inbox replay reads.
DEFAULT_EVENTS_KEEP_RECENT: int = 5000
# Keep at least this many most-recent done tasks (succeeded/cancelled).
DEFAULT_TASKS_KEEP_DONE: int = 2000

# Only these task states are safe to prune: truly done, never retried, and not pending manual attention.
_PRUNABLE_TASK_STATES: tuple[str, ...] = ("succeeded", "cancelled")


@dataclass
class RetentionResult:
    events_deleted: int = 0
    tasks_deleted: int = 0


async def prune_events(
    db: SqliteConnection,
    *,
    keep_recent: int = DEFAULT_EVENTS_KEEP_RECENT,
) -> int:
    """Delete old events outside the recent window."""
    row = await db.fetchone("SELECT MAX(seq) AS m FROM events")
    max_seq = int(row["m"]) if row and row["m"] is not None else 0
    if max_seq <= 0:
        return 0
    delete_below = max_seq - max(0, int(keep_recent))
    if delete_below <= 0:
        return 0
    async with db.transaction() as cur:
        cur.execute("DELETE FROM events WHERE seq <= ?", (delete_below,))
        return int(cur.rowcount or 0)


async def prune_tasks(
    db: SqliteConnection,
    *,
    keep_done: int = DEFAULT_TASKS_KEEP_DONE,
) -> int:
    """Delete old done (succeeded/cancelled) tasks beyond ``keep_done``."""
    keep_done = max(0, int(keep_done))
    placeholders = ",".join("?" * len(_PRUNABLE_TASK_STATES))
    async with db.transaction() as cur:
        cur.execute(
            f"DELETE FROM tasks WHERE state IN ({placeholders}) "  # nosec B608 - generated placeholders only.
            f"AND task_id NOT IN ("
            f"  SELECT task_id FROM tasks WHERE state IN ({placeholders}) "
            f"  ORDER BY updated_at DESC LIMIT ?"
            f")",
            (*_PRUNABLE_TASK_STATES, *_PRUNABLE_TASK_STATES, keep_done),
        )
        return int(cur.rowcount or 0)


async def run_db_retention(db: SqliteConnection) -> RetentionResult:
    """Run all DB retention passes; safe to call periodically from the reaper."""
    return RetentionResult(events_deleted=await prune_events(db), tasks_deleted=await prune_tasks(db))


__all__ = [
    "DEFAULT_EVENTS_KEEP_RECENT",
    "DEFAULT_TASKS_KEEP_DONE",
    "RetentionResult",
    "prune_events",
    "prune_tasks",
    "run_db_retention",
]
