# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CursorStore

Per-agent ``last_processed_seq`` cursor. Single SQL UPSERT replaces the
per-file tmp+rename pattern from v0.4. Combined with the tasks/events
tables, this gives the cross-table atomicity that ADR-42 promises.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..storage.connection import SqliteConnection


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        str: Timestamp with microsecond precision in UTC.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class CursorState:
    """A single agent's bus-processing cursor.

    Attributes:
        agent (str): Identifier of the agent owning the cursor.
        last_processed_seq (int): Highest bus sequence processed so far.
        last_processed_msg_id (str): Message id at that sequence.
        processed_at (str): ISO-8601 timestamp of the last advance.
    """

    agent: str
    last_processed_seq: int
    last_processed_msg_id: str
    processed_at: str

    @classmethod
    def empty(cls, agent: str) -> "CursorState":
        """Build a zeroed cursor for an agent with no prior state.

        Args:
            agent (str): Identifier of the agent.

        Returns:
            CursorState: A cursor at sequence 0 with an empty msg id.
        """
        return cls(
            agent=agent,
            last_processed_seq=0,
            last_processed_msg_id="",
            processed_at=_now_iso(),
        )

    @classmethod
    def from_row(cls, row) -> "CursorState":
        """Build a cursor from a ``cursors`` table row.

        Args:
            row: Mapping-like DB row with ``agent``,
                ``last_processed_seq``, ``last_processed_msg_id``, and
                ``processed_at`` keys.

        Returns:
            CursorState: The cursor populated from the row.
        """
        return cls(
            agent=row["agent"],
            last_processed_seq=row["last_processed_seq"],
            last_processed_msg_id=row["last_processed_msg_id"],
            processed_at=row["processed_at"],
        )


class CursorStore:
    """SQLite-backed store of per-agent bus-processing cursors.

    Attributes:
        db (SqliteConnection): Connection used for cursor reads/writes.
    """

    def __init__(self, db: SqliteConnection):
        """Bind the store to a SQLite connection.

        Args:
            db (SqliteConnection): Connection backing the ``cursors``
                table.
        """
        self.db = db

    async def load(self, agent: str) -> CursorState:
        """Load one agent's cursor, defaulting to empty when absent.

        Args:
            agent (str): Identifier of the agent to load.

        Returns:
            CursorState: The stored cursor, or an empty cursor when the
            agent has no row.
        """
        row = await self.db.fetchone(
            "SELECT * FROM cursors WHERE agent=?", (agent,)
        )
        if row is None:
            return CursorState.empty(agent)
        return CursorState.from_row(row)

    async def all(self) -> dict[str, CursorState]:
        """Load every agent's cursor.

        Returns:
            dict[str, CursorState]: Cursors keyed by agent identifier.
        """
        rows = await self.db.fetchall("SELECT * FROM cursors")
        return {r["agent"]: CursorState.from_row(r) for r in rows}

    async def advance(
        self,
        agent: str,
        *,
        seq: int,
        msg_id: str,
    ) -> CursorState:
        """Advance an agent's cursor via UPSERT, never moving backwards.

        Args:
            agent (str): Identifier of the agent to advance.
            seq (int): Sequence to advance to; ignored when not greater
                than the current sequence.
            msg_id (str): Message id at the new sequence.

        Returns:
            CursorState: The resulting cursor state (unchanged when
            ``seq`` would move the cursor backwards).
        """
        async with self.db.transaction() as cur:
            cur.execute(
                "SELECT last_processed_seq, last_processed_msg_id, processed_at "
                "FROM cursors WHERE agent=?",
                (agent,),
            )
            row = cur.fetchone()
            current = int(row["last_processed_seq"]) if row else 0
            if seq <= current:
                return CursorState(
                    agent=agent,
                    last_processed_seq=current,
                    last_processed_msg_id=row["last_processed_msg_id"] if row else "",
                    processed_at=row["processed_at"] if row else _now_iso(),
                )
            now = _now_iso()
            cur.execute(
                "INSERT INTO cursors(agent, last_processed_seq, "
                "                   last_processed_msg_id, processed_at) "
                "VALUES (?,?,?,?) "
                "ON CONFLICT(agent) DO UPDATE SET "
                "  last_processed_seq=excluded.last_processed_seq, "
                "  last_processed_msg_id=excluded.last_processed_msg_id, "
                "  processed_at=excluded.processed_at",
                (agent, seq, msg_id, now),
            )
            return CursorState(
                agent=agent,
                last_processed_seq=seq,
                last_processed_msg_id=msg_id,
                processed_at=now,
            )

    async def is_already_processed(self, agent: str, seq: int) -> bool:
        """Return whether a sequence has already been processed.

        Args:
            agent (str): Identifier of the agent.
            seq (int): Bus sequence to test.

        Returns:
            bool: ``True`` when ``seq`` is at or below the agent's last
            processed sequence.
        """
        cur = await self.load(agent)
        return seq <= cur.last_processed_seq


__all__ = ["CursorState", "CursorStore"]
