# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CursorStore."""

from __future__ import annotations

from dataclasses import dataclass

from hyperloom.common.timeutil import now_iso

from .storage.connection import SqliteConnection

_now_iso = now_iso


@dataclass
class CursorState:
    """A single agent's bus-processing cursor."""

    agent: str
    last_processed_seq: int
    last_processed_msg_id: str
    processed_at: str

    @classmethod
    def empty(cls, agent: str) -> "CursorState":
        """Build a zeroed cursor for an agent with no prior state."""
        return cls(
            agent=agent,
            last_processed_seq=0,
            last_processed_msg_id="",
            processed_at=_now_iso(),
        )

    @classmethod
    def from_row(cls, row) -> "CursorState":
        """Build a cursor from a ``cursors`` table row."""
        return cls(
            agent=row["agent"],
            last_processed_seq=row["last_processed_seq"],
            last_processed_msg_id=row["last_processed_msg_id"],
            processed_at=row["processed_at"],
        )


class CursorStore:
    """SQLite-backed store of per-agent bus-processing cursors."""

    def __init__(self, db: SqliteConnection):
        """Bind the store to a SQLite connection."""
        self.db = db

    async def load(self, agent: str) -> CursorState:
        """Load one agent's cursor, defaulting to empty when absent."""
        row = await self.db.fetchone("SELECT * FROM cursors WHERE agent=?", (agent,))
        if row is None:
            return CursorState.empty(agent)
        return CursorState.from_row(row)

    async def advance(
        self,
        agent: str,
        *,
        seq: int,
        msg_id: str,
    ) -> CursorState:
        """Advance an agent's cursor via UPSERT, never moving backwards."""
        async with self.db.transaction() as cur:
            cur.execute(
                "SELECT last_processed_seq, last_processed_msg_id, processed_at FROM cursors WHERE agent=?",
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


__all__ = ["CursorState", "CursorStore"]
