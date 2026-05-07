"""Session ↔ pod assignment repository.

Persists the time-window mapping consumed by the query API. Two
operating modes share this surface:

* **Open** an assignment when the KV watcher / event consumer / SaFE
  reconciler observes a fresh (session, pod) pair. Open implies
  ``t_end IS NULL``; a duplicate open touches ``last_seen_at`` so the
  reconciler can detect stale assignments.
* **Close** an open assignment when the source signals termination
  (KV expire, sandbox delete, terminal event). Close stamps ``t_end``
  but keeps the historical row so range queries against the past
  still resolve.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg

from ..models import (
    PodAssignment,
    PodAssignmentSource,
    PodRef,
    PodRole,
)


class AssignmentsRepository:
    """CRUD on ``session_pod_assignment``."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def open_assignment(
        self,
        *,
        session_id: str,
        pod: PodRef,
        role: PodRole,
        source: PodAssignmentSource,
        observed_at: datetime,
    ) -> int:
        """Open or refresh an assignment.

        Returns the ``assignment_id`` of the open row. If a row with
        ``(session_id, namespace, name, role)`` is already open, we
        keep its ``t_start`` and just bump ``last_seen_at`` — this
        prevents the reconciler from creating duplicate open rows.
        """

        async with self._pool.acquire() as conn:
            existing = await conn.fetchrow(
                """
                SELECT assignment_id
                  FROM session_pod_assignment
                 WHERE session_id    = $1
                   AND pod_namespace = $2
                   AND pod_name      = $3
                   AND role          = $4
                   AND t_end IS NULL
              ORDER BY t_start DESC
                 LIMIT 1
                """,
                session_id,
                pod.namespace,
                pod.name,
                role.value,
            )
            if existing is not None:
                await conn.execute(
                    """
                    UPDATE session_pod_assignment
                       SET last_seen_at = GREATEST(last_seen_at, $2),
                           pod_uid      = COALESCE(pod_uid, $3)
                     WHERE assignment_id = $1
                    """,
                    existing["assignment_id"],
                    observed_at,
                    pod.pod_uid,
                )
                return int(existing["assignment_id"])
            row = await conn.fetchrow(
                """
                INSERT INTO session_pod_assignment (
                    session_id, pod_namespace, pod_name, pod_uid,
                    role, source, t_start, last_seen_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
                RETURNING assignment_id
                """,
                session_id,
                pod.namespace,
                pod.name,
                pod.pod_uid,
                role.value,
                source.value,
                observed_at,
            )
            return int(row["assignment_id"])

    async def close_assignment(
        self,
        *,
        session_id: str,
        pod: PodRef,
        role: PodRole,
        closed_at: datetime,
    ) -> int:
        """Close any open assignment for this (session, pod, role).

        Returns the number of rows touched. Idempotent: closing an
        already-closed assignment is a no-op (the predicate
        ``t_end IS NULL`` filters it out).
        """

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE session_pod_assignment
                   SET t_end        = $5,
                       last_seen_at = GREATEST(last_seen_at, $5)
                 WHERE session_id    = $1
                   AND pod_namespace = $2
                   AND pod_name      = $3
                   AND role          = $4
                   AND t_end IS NULL
                """,
                session_id,
                pod.namespace,
                pod.name,
                role.value,
                closed_at,
            )
        return _rows_affected(result)

    async def close_all_for_session(
        self,
        *,
        session_id: str,
        closed_at: datetime,
    ) -> int:
        """Close every open assignment for a session.

        Used when a session reaches a terminal state — any pod still
        open inherits the session's terminal timestamp.
        """

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE session_pod_assignment
                   SET t_end        = $2,
                       last_seen_at = GREATEST(last_seen_at, $2)
                 WHERE session_id = $1
                   AND t_end IS NULL
                """,
                session_id,
                closed_at,
            )
        return _rows_affected(result)

    async def expire_stale_open(
        self,
        *,
        source: PodAssignmentSource,
        last_seen_before: datetime,
        closed_at: datetime,
    ) -> int:
        """Close open assignments the reconciler hasn't seen lately.

        Used by the SaFE reconciler to garbage-collect rows for hands
        pods that vanished without a sandbox_delete event. ``source``
        is filtered so a slow reconciler does not race with the KV
        watcher's brain assignments.
        """

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE session_pod_assignment
                   SET t_end        = $3,
                       last_seen_at = GREATEST(last_seen_at, $3)
                 WHERE source       = $1
                   AND t_end IS NULL
                   AND last_seen_at < $2
                """,
                source.value,
                last_seen_before,
                closed_at,
            )
        return _rows_affected(result)

    async def list_for_session(
        self,
        *,
        session_id: str,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> list[PodAssignment]:
        """All assignments overlapping ``[window_start, window_end]``.

        ``None`` bounds collapse to "all-time". The overlap predicate
        is symmetric: a row whose ``t_end`` is NULL is treated as
        still open and therefore overlaps any future window.
        """

        clauses = ["session_id = $1"]
        params: list[Any] = [session_id]
        if window_start is not None:
            clauses.append(
                "(t_end IS NULL OR t_end >= ${i})".replace(
                    "${i}", f"${len(params) + 1}"
                )
            )
            params.append(window_start)
        if window_end is not None:
            clauses.append(f"t_start <= ${len(params) + 1}")
            params.append(window_end)
        sql = f"""
            SELECT assignment_id, session_id, pod_namespace, pod_name, pod_uid,
                   role, source, t_start, t_end, last_seen_at
              FROM session_pod_assignment
             WHERE {' AND '.join(clauses)}
          ORDER BY t_start ASC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [_row_to_assignment(r) for r in rows]


def _row_to_assignment(row: asyncpg.Record | dict[str, Any]) -> PodAssignment:
    return PodAssignment(
        assignment_id=row["assignment_id"],
        session_id=row["session_id"],
        pod=PodRef(
            namespace=row["pod_namespace"],
            name=row["pod_name"],
            pod_uid=row["pod_uid"],
        ),
        role=PodRole(row["role"]),
        source=PodAssignmentSource(row["source"]),
        t_start=row["t_start"],
        t_end=row["t_end"],
        last_seen_at=row["last_seen_at"],
    )


def _rows_affected(execute_result: str) -> int:
    """asyncpg returns ``UPDATE n`` / ``INSERT 0 n`` / ``DELETE n``.

    Pulling out the trailing integer makes it usable as an idempotency
    counter without the caller having to parse the tag.
    """

    parts = execute_result.split()
    if not parts:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0
