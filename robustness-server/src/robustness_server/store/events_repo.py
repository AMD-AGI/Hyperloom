"""Session events repository.

Append-only audit log for every NATS message we accept. Reads happen
from the ``/sessions/{id}/events`` endpoint and from analytics scripts
mining the JSONB body. The table is intentionally not normalised
beyond a few canonical columns — callers that want to filter on a
deeply-nested field should use ``body @> '{"key": "value"}'``.
"""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any

import asyncpg

from ..models import EventEnvelope


class EventsRepository:
    """Append + range-query on ``session_events``."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append(self, event: EventEnvelope) -> int:
        """Persist one event; returns the assigned ``event_id``."""

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO session_events (
                    session_id, event_type, subject,
                    occurred_at, received_at,
                    pod_name, plugin_id, body
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                RETURNING event_id
                """,
                event.session_id,
                event.raw_type,
                event.subject,
                event.occurred_at,
                event.received_at,
                event.pod_name,
                event.plugin_id,
                json.dumps(event.body),
            )
        return int(row["event_id"])

    async def list_for_session(
        self,
        *,
        session_id: str,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return events for a session within an optional window.

        Returns plain dicts (not pydantic) because the body is opaque
        JSONB and callers serialise it directly to HTTP. ``limit`` caps
        unbounded queries; the API layer surfaces this as a hard cap
        rather than paginating because the typical use case is "tail
        the last N events" not "scroll back forever".
        """

        clauses = ["session_id = $1"]
        params: list[Any] = [session_id]
        if window_start is not None:
            clauses.append(f"occurred_at >= ${len(params) + 1}")
            params.append(window_start)
        if window_end is not None:
            clauses.append(f"occurred_at <= ${len(params) + 1}")
            params.append(window_end)
        params.append(limit)
        sql = f"""
            SELECT event_id, session_id, event_type, subject,
                   occurred_at, received_at, pod_name, plugin_id, body
              FROM session_events
             WHERE {' AND '.join(clauses)}
          ORDER BY occurred_at DESC
             LIMIT ${len(params)}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [
            {
                "event_id": r["event_id"],
                "session_id": r["session_id"],
                "event_type": r["event_type"],
                "subject": r["subject"],
                "occurred_at": r["occurred_at"],
                "received_at": r["received_at"],
                "pod_name": r["pod_name"],
                "plugin_id": r["plugin_id"],
                "body": _decode_body(r["body"]),
            }
            for r in rows
        ]


def _decode_body(value: Any) -> Any:
    """asyncpg returns JSONB as a string by default; decode lazily."""

    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value
