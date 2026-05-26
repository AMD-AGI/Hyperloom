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
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class CursorState:
    agent: str
    last_processed_seq: int
    last_processed_msg_id: str
    processed_at: str

    @classmethod
    def empty(cls, agent: str) -> "CursorState":
        return cls(
            agent=agent,
            last_processed_seq=0,
            last_processed_msg_id="",
            processed_at=_now_iso(),
        )

    @classmethod
    def from_row(cls, row) -> "CursorState":
        return cls(
            agent=row["agent"],
            last_processed_seq=row["last_processed_seq"],
            last_processed_msg_id=row["last_processed_msg_id"],
            processed_at=row["processed_at"],
        )


class CursorStore:
    def __init__(self, db: SqliteConnection):
        self.db = db

    async def load(self, agent: str) -> CursorState:
        row = await self.db.fetchone(
            "SELECT * FROM cursors WHERE agent=?", (agent,)
        )
        if row is None:
            return CursorState.empty(agent)
        return CursorState.from_row(row)

    async def all(self) -> dict[str, CursorState]:
        rows = await self.db.fetchall("SELECT * FROM cursors")
        return {r["agent"]: CursorState.from_row(r) for r in rows}

    async def advance(
        self,
        agent: str,
        *,
        seq: int,
        msg_id: str,
    ) -> CursorState:
        """UPSERT — refuses to move backwards. Returns the resulting state."""
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
        cur = await self.load(agent)
        return seq <= cur.last_processed_seq


__all__ = ["CursorState", "CursorStore"]
