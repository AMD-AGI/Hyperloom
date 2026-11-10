# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DB retention / pruning for multi-day single-session runs."""

from __future__ import annotations

from dataclasses import dataclass

from .storage.connection import SqliteConnection


# Keep at least this many most-recent events so a recently-emitted reply is still resolvable by ``lookup_by_id``.
DEFAULT_EVENTS_KEEP_RECENT: int = 5000
# Keep at least this many most-recent done tasks (succeeded/cancelled).
DEFAULT_TASKS_KEEP_DONE: int = 2000

# Ids of proposals that already carry a verdict. It must exclude NULL and empty targets: ``msg_id NOT IN (.., NULL)``
# is NULL for every row, so one malformed verdict would make every proposal look decided. ``replay_for_resume`` applies
# the same rule in Python.
DECIDED_PROPOSAL_IDS_SQL = """
    SELECT json_extract(payload, '$.target_proposal_msg_id')
    FROM events
    WHERE topic = 'review_verdict'
      AND json_extract(payload, '$.target_proposal_msg_id') IS NOT NULL
      AND json_extract(payload, '$.target_proposal_msg_id') != ''
"""

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
    """Delete old events outside the recent window, protecting pending proposals."""
    row = await db.fetchone("SELECT MAX(seq) AS m FROM events")
    max_seq = int(row["m"]) if row and row["m"] is not None else 0
    if max_seq <= 0:
        return 0
    delete_below = max_seq - max(0, int(keep_recent))
    if delete_below <= 0:
        return 0
    # Never prune a ``proposal`` row that has no verdict yet.
    async with db.transaction() as cur:
        cur.execute(
            f"""
            DELETE FROM events
            WHERE seq <= ?
              AND seq NOT IN (
                SELECT seq FROM events
                WHERE topic = 'proposal'
                  AND msg_id NOT IN ({DECIDED_PROPOSAL_IDS_SQL})
              )
            """,  # nosec B608 - interpolates a constant SQL fragment only.
            (delete_below,),
        )
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
        cur.execute(  # nosec B608 - placeholders string is generated solely from fixed state count.
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
    "DECIDED_PROPOSAL_IDS_SQL",
    "DEFAULT_EVENTS_KEEP_RECENT",
    "DEFAULT_TASKS_KEEP_DONE",
    "RetentionResult",
    "prune_events",
    "prune_tasks",
    "run_db_retention",
]
