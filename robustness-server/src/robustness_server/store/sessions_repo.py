"""Sessions repository.

Owns INSERT/UPDATE on ``hyperloom_robustness.sessions``. The reconciler
drives the row's lifecycle by upserting on every observed event:

* first observation creates the row with ``t_start = occurred_at`` and
  ``last_event_at = occurred_at``;
* subsequent observations refresh ``last_event_at``;
* terminal events stamp ``t_end`` and ``final_state`` once.

We deliberately leave session enrichment (user id, plugin id) to be
filled in opportunistically — the consumer hands whatever fields the
event happens to carry; missing fields stay NULL and may be filled by a
later event for the same session.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg

from ..models import Session, SessionState


class SessionsRepository:
    """Plain CRUD for ``sessions`` keyed by ``session_id``."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_observation(
        self,
        *,
        session_id: str,
        occurred_at: datetime,
        user_id: str | None = None,
        plugin_id: str | None = None,
    ) -> None:
        """Record that an event was observed for ``session_id``.

        Idempotent: a fresh insert sets ``t_start`` and
        ``last_event_at`` to ``occurred_at``; a subsequent call only
        nudges ``last_event_at`` forward and backfills NULL columns.
        """

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (
                    session_id, user_id, plugin_id,
                    t_start, last_event_at
                )
                VALUES ($1, $2, $3, $4, $4)
                ON CONFLICT (session_id) DO UPDATE SET
                    user_id       = COALESCE(sessions.user_id,   EXCLUDED.user_id),
                    plugin_id     = COALESCE(sessions.plugin_id, EXCLUDED.plugin_id),
                    last_event_at = GREATEST(sessions.last_event_at, EXCLUDED.last_event_at)
                """,
                session_id,
                user_id,
                plugin_id,
                occurred_at,
            )

    async def mark_terminal(
        self,
        *,
        session_id: str,
        terminal_at: datetime,
        final_state: SessionState,
    ) -> None:
        """Stamp ``t_end`` / ``final_state`` once a terminal arrives.

        Latches: if a later event tries to overwrite the terminal we
        keep the earliest ``t_end`` and the first non-NULL state.
        """

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE sessions
                   SET t_end       = COALESCE(sessions.t_end, $2),
                       final_state = COALESCE(sessions.final_state, $3),
                       last_event_at = GREATEST(sessions.last_event_at, $2)
                 WHERE session_id = $1
                """,
                session_id,
                terminal_at,
                final_state.value,
            )

    async def get(self, session_id: str) -> Session | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT session_id, user_id, plugin_id,
                       t_start, t_end, final_state, last_event_at
                  FROM sessions
                 WHERE session_id = $1
                """,
                session_id,
            )
        return _row_to_session(row) if row else None

    async def list_recent(self, *, limit: int = 50) -> list[Session]:
        """Return the most recently active sessions."""

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, user_id, plugin_id,
                       t_start, t_end, final_state, last_event_at
                  FROM sessions
              ORDER BY last_event_at DESC
                 LIMIT $1
                """,
                limit,
            )
        return [_row_to_session(r) for r in rows]


def _row_to_session(row: asyncpg.Record | dict[str, Any]) -> Session:
    final = row["final_state"]
    return Session(
        session_id=row["session_id"],
        user_id=row["user_id"],
        plugin_id=row["plugin_id"],
        t_start=row["t_start"],
        t_end=row["t_end"],
        final_state=SessionState(final) if final else None,
        last_event_at=row["last_event_at"],
    )
