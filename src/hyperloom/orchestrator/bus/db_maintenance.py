# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DB retention / pruning for multi-day single-session runs.

The ``events`` and ``tasks`` tables are otherwise append-only and would grow
without bound over a multi-day run (no process restart clears them). These
helpers bound them while preserving the two correctness invariants:

* **Resume safety** — at resume, pending proposals are reconstructed from the
  event log via a pending-proposal anti-join (``replay_for_resume``). An event
  is only deletable when it falls outside the ``keep_recent`` window AND is not
  a semantically pending proposal (one without a ``review_verdict`` targeting
  its ``msg_id``). The pending-proposal anti-join is the sole resume-safety
  mechanism; the pruner never needs to know about agent cursors.
* **In-flight safety** — ``tasks`` pruning never touches ``queued`` / ``running``
  / ``failed`` rows; only truly-done (``succeeded`` / ``cancelled``) rows beyond
  a keep-recent count are removed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .storage.connection import SqliteConnection


# Keep at least this many most-recent events so a recently-emitted reply is
# still resolvable by ``lookup_by_id``.
DEFAULT_EVENTS_KEEP_RECENT: int = 5000
# Keep at least this many most-recent done tasks (succeeded/cancelled).
DEFAULT_TASKS_KEEP_DONE: int = 2000

# Only these task states are safe to prune: truly done, never retried, and not
# pending manual attention.
_PRUNABLE_TASK_STATES: tuple[str, ...] = ("succeeded", "cancelled")


@dataclass
class RetentionResult:
    events_deleted: int = 0
    tasks_deleted: int = 0

    @property
    def total(self) -> int:
        """Total rows deleted across the events and tasks retention passes.

        Returns:
            The sum of ``events_deleted`` and ``tasks_deleted``.
        """
        return self.events_deleted + self.tasks_deleted


# A ``proposal`` event is semantically pending until a ``review_verdict`` targets
# its ``msg_id``; pruning a pending proposal's row would lose the only durable
# record a post-resume late verdict can attach to. Canonical form of that set;
# ``prune_events`` inlines a copy — keep both in lockstep with ``replay_for_resume``.
# The inner SELECT must exclude NULL/empty targets, else ``msg_id NOT IN (..,
# NULL)`` evaluates to NULL and every proposal would wrongly look decided.
_PENDING_PROPOSAL_SEQS_SQL = """
    SELECT seq FROM events
    WHERE topic = 'proposal'
      AND msg_id NOT IN (
        SELECT json_extract(payload, '$.target_proposal_msg_id')
        FROM events
        WHERE topic = 'review_verdict'
          AND json_extract(payload, '$.target_proposal_msg_id') IS NOT NULL
          AND json_extract(payload, '$.target_proposal_msg_id') != ''
      )
"""


async def prune_events(
    db: SqliteConnection,
    *,
    keep_recent: int = DEFAULT_EVENTS_KEEP_RECENT,
) -> int:
    """Delete old events outside the recent window, protecting pending proposals.

    Returns the number of rows deleted.

    Args:
        db: Open SQLite connection to prune the ``events`` table on.
        keep_recent: Minimum number of most-recent events to retain.

    Returns:
        The number of event rows deleted.
    """
    row = await db.fetchone("SELECT MAX(seq) AS m FROM events")
    max_seq = int(row["m"]) if row and row["m"] is not None else 0
    if max_seq <= 0:
        return 0
    delete_below = max_seq - max(0, int(keep_recent))
    if delete_below <= 0:
        return 0
    # Never prune a ``proposal`` row still semantically pending; the anti-join
    # mirrors ``replay_for_resume``.
    async with db.transaction() as cur:
        cur.execute(
            """
            DELETE FROM events
            WHERE seq <= ?
              AND seq NOT IN (
                SELECT seq FROM events
                WHERE topic = 'proposal'
                  AND msg_id NOT IN (
                    SELECT json_extract(payload, '$.target_proposal_msg_id')
                    FROM events
                    WHERE topic = 'review_verdict'
                      AND json_extract(payload, '$.target_proposal_msg_id') IS NOT NULL
                      AND json_extract(payload, '$.target_proposal_msg_id') != ''
                  )
              )
            """,
            (delete_below,),
        )
        deleted = int(cur.rowcount or 0)
    return deleted


async def prune_tasks(
    db: SqliteConnection,
    *,
    keep_done: int = DEFAULT_TASKS_KEEP_DONE,
) -> int:
    """Delete old done (succeeded/cancelled) tasks beyond ``keep_done``.

    Never touches queued/running/failed rows. Returns the number of rows
    deleted.

    Args:
        db: Open SQLite connection to prune the ``tasks`` table on.
        keep_done: Minimum number of most-recent done tasks to retain.

    Returns:
        The number of task rows deleted.
    """
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
        deleted = int(cur.rowcount or 0)
    return deleted


async def run_db_retention(
    db: SqliteConnection,
    *,
    events_keep_recent: int = DEFAULT_EVENTS_KEEP_RECENT,
    tasks_keep_done: int = DEFAULT_TASKS_KEEP_DONE,
) -> RetentionResult:
    """Run all DB retention passes; safe to call periodically from the reaper.

    Args:
        db: Open SQLite connection to run retention on.
        events_keep_recent: Minimum number of most-recent events to retain.
        tasks_keep_done: Minimum number of most-recent done tasks to retain.

    Returns:
        A :class:`RetentionResult` with the per-table deletion counts.
    """
    events_deleted = await prune_events(
        db,
        keep_recent=events_keep_recent,
    )
    tasks_deleted = await prune_tasks(db, keep_done=tasks_keep_done)
    return RetentionResult(
        events_deleted=events_deleted,
        tasks_deleted=tasks_deleted,
    )


__all__ = [
    "DEFAULT_EVENTS_KEEP_RECENT",
    "DEFAULT_TASKS_KEEP_DONE",
    "RetentionResult",
    "prune_events",
    "prune_tasks",
    "run_db_retention",
]
