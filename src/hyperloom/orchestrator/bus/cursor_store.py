# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CursorStore."""

from __future__ import annotations

from dataclasses import dataclass

from .storage.connection import SqliteConnection


@dataclass
class CursorState:
    """A single agent's bus-processing cursor."""

    agent: str
    last_processed_seq: int


class CursorStore:
    """SQLite-backed store of per-agent bus-processing cursors."""

    def __init__(self, db: SqliteConnection):
        """Bind the store to a SQLite connection."""
        self.db = db

    async def load(self, agent: str) -> CursorState:
        """Load one agent's cursor, defaulting to zero when absent."""
        row = await self.db.fetchone("SELECT last_processed_seq FROM cursors WHERE agent=?", (agent,))
        return CursorState(agent=agent, last_processed_seq=int(row["last_processed_seq"]) if row else 0)

    async def advance(self, agent: str, *, seq: int) -> None:
        """Advance an agent's cursor via UPSERT, never moving backwards."""
        async with self.db.transaction() as cur:
            cur.execute("SELECT last_processed_seq FROM cursors WHERE agent=?", (agent,))
            row = cur.fetchone()
            if seq <= (int(row["last_processed_seq"]) if row else 0):
                return
            cur.execute(
                "INSERT INTO cursors(agent, last_processed_seq) VALUES (?,?) "
                "ON CONFLICT(agent) DO UPDATE SET last_processed_seq=excluded.last_processed_seq",
                (agent, seq),
            )


__all__ = ["CursorState", "CursorStore"]
