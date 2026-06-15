# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""DB retention / pruning for multi-day single-session runs (R4).

The ``events`` and ``tasks`` tables are otherwise append-only and would grow
without bound over a multi-day run (no process restart clears them). These
helpers bound them while preserving the two correctness invariants:

* **Resume safety** — ``events`` are replayed at resume via
  :meth:`MessageBus.replay_for(after_seq=cursor)`. An event is only deletable
  once *every* agent cursor has advanced past it (``seq <= min(cursor)``) AND it
  falls outside a recent-window margin (so ``lookup_by_id`` of a fresh
  ``in_reply_to`` still resolves). The prune watermark is therefore
  ``min(min_processed_seq, max_seq - keep_recent)`` — strictly below the resume
  anchor.
* **In-flight safety** — ``tasks`` pruning never touches ``queued`` / ``running``
  / ``failed`` / ``needs_manual_review`` rows; only truly-done
  (``succeeded`` / ``cancelled``) rows beyond a keep-recent count are removed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cursor_store import CursorStore
from ..storage.connection import SqliteConnection


# Keep at least this many most-recent events regardless of cursor watermark, so
# a recently-emitted reply is still resolvable by ``lookup_by_id``.
DEFAULT_EVENTS_KEEP_RECENT: int = 5000
# Keep at least this many most-recent done tasks (succeeded/cancelled).
DEFAULT_TASKS_KEEP_DONE: int = 2000

# Only these task states are safe to prune: truly done, never retried, and not
# pending manual attention. (``failed`` can transition back to ``running``.)
_PRUNABLE_TASK_STATES: tuple[str, ...] = ("succeeded", "cancelled")


@dataclass
class RetentionResult:
    events_deleted: int = 0
    tasks_deleted: int = 0

    @property
    def total(self) -> int:
        return self.events_deleted + self.tasks_deleted


async def _min_processed_seq(cursors: CursorStore) -> int | None:
    """Lowest ``last_processed_seq`` across all agent cursors.

    ``None`` when no cursor exists yet (nothing is safe to prune — every event
    may still need replay)."""
    states = await cursors.all()
    if not states:
        return None
    return min(int(s.last_processed_seq) for s in states.values())


async def prune_events(
    db: SqliteConnection,
    cursors: CursorStore,
    *,
    keep_recent: int = DEFAULT_EVENTS_KEEP_RECENT,
) -> int:
    """Delete fully-processed events below the resume anchor + recent margin.

    Returns the number of rows deleted.
    """
    min_cursor = await _min_processed_seq(cursors)
    if min_cursor is None or min_cursor <= 0:
        return 0
    row = await db.fetchone("SELECT MAX(seq) AS m FROM events")
    max_seq = int(row["m"]) if row and row["m"] is not None else 0
    if max_seq <= 0:
        return 0
    # Deletable iff processed by all agents AND outside the recent window.
    delete_below = min(min_cursor, max_seq - max(0, int(keep_recent)))
    if delete_below <= 0:
        return 0
    async with db.transaction() as cur:
        cur.execute("DELETE FROM events WHERE seq <= ?", (delete_below,))
        deleted = int(cur.rowcount or 0)
    return deleted


async def prune_tasks(
    db: SqliteConnection,
    *,
    keep_done: int = DEFAULT_TASKS_KEEP_DONE,
) -> int:
    """Delete old done (succeeded/cancelled) tasks beyond ``keep_done``.

    Never touches queued/running/failed/needs_manual_review rows. Returns the
    number of rows deleted.
    """
    keep_done = max(0, int(keep_done))
    placeholders = ",".join("?" * len(_PRUNABLE_TASK_STATES))
    async with db.transaction() as cur:
        cur.execute(
            f"DELETE FROM tasks WHERE state IN ({placeholders}) "
            f"AND task_id NOT IN ("
            f"  SELECT task_id FROM tasks WHERE state IN ({placeholders}) "
            f"  ORDER BY updated_at DESC LIMIT ?"
            f")",
            (*_PRUNABLE_TASK_STATES, *_PRUNABLE_TASK_STATES, keep_done),
        )
        deleted = int(cur.rowcount or 0)
    return deleted


async def run_db_retention(
    db: SqliteConnection,
    cursors: CursorStore,
    *,
    events_keep_recent: int = DEFAULT_EVENTS_KEEP_RECENT,
    tasks_keep_done: int = DEFAULT_TASKS_KEEP_DONE,
) -> RetentionResult:
    """Run all DB retention passes; safe to call periodically from the reaper."""
    events_deleted = await prune_events(
        db, cursors, keep_recent=events_keep_recent,
    )
    tasks_deleted = await prune_tasks(db, keep_done=tasks_keep_done)
    return RetentionResult(
        events_deleted=events_deleted, tasks_deleted=tasks_deleted,
    )


__all__ = [
    "DEFAULT_EVENTS_KEEP_RECENT",
    "DEFAULT_TASKS_KEEP_DONE",
    "RetentionResult",
    "prune_events",
    "prune_tasks",
    "run_db_retention",
]
