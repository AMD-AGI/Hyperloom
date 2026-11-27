# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MessageBus — the ``events`` table is the source of truth; ``seq`` (AUTOINCREMENT) gives a monotonic id. Topics validated against an allowlist."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from hyperloom_common.timeutil import now_iso

from .db_maintenance import DEFAULT_EVENTS_KEEP_RECENT
from .storage.connection import SqliteConnection


TOPIC_ALLOWLIST = frozenset(
    {
        "observation",
        "event",
        "decision",
        "alert",
        "delegated_result",
        "request",
        "response",
        "strategy_change",
    }
)

# The ``source`` of the reply to a request deferred because the bench lanes are busy. The reply asks nothing of its
# reader: the same request is deferred the same way until the lanes free.
LANES_BUSY_SOURCE = "lanes_busy"
# An agent's inbox past a sequence id. Its own rows stay out: they would repeat in its next prompt what it just said,
# and a turn would wake the next one.
_INBOX_WHERE = "seq > ? AND from_agent != ?"


@dataclass
class Message:
    """One bus message persisted in the ``events`` table."""

    msg_id: str
    from_agent: str
    to_agent: str
    topic: str
    payload: dict[str, Any]
    in_reply_to: str | None = None
    ts: str = field(default_factory=now_iso)
    seq: int | None = None  # DB-assigned on insert

    @classmethod
    def new(
        cls,
        from_agent: str,
        to_agent: str,
        topic: str,
        payload: dict[str, Any],
        *,
        in_reply_to: str | None = None,
    ) -> "Message":
        """Construct a new message with a fresh ``msg_id``."""
        return cls(
            msg_id=uuid.uuid4().hex,
            from_agent=from_agent,
            to_agent=to_agent,
            topic=topic,
            payload=payload,
            in_reply_to=in_reply_to,
        )

    @classmethod
    def from_row(cls, row) -> "Message":
        """Build a :class:`Message` from an ``events`` table row."""
        return cls(
            msg_id=row["msg_id"],
            from_agent=row["from_agent"],
            to_agent=row["to_agent"],
            topic=row["topic"],
            payload=json.loads(row["payload"]),
            in_reply_to=row["in_reply_to"],
            ts=row["ts"],
            seq=row["seq"],
        )

    def to_db_row(self) -> tuple:
        """Serialise the message into an ``events`` INSERT tuple."""
        return (
            self.msg_id,
            self.from_agent,
            self.to_agent,
            self.topic,
            self.in_reply_to,
            json.dumps(self.payload),
            self.ts,
        )


class MessageBus:
    """Append-only message log backed by the ``events`` table."""

    def __init__(self, db: SqliteConnection):
        """Initialise the bus."""
        self.db = db

    async def append_and_seq(self, msg: Message) -> int:
        """Append one message and return its assigned sequence id."""
        if msg.topic not in TOPIC_ALLOWLIST:
            raise ValueError(f"unknown topic: {msg.topic!r}")
        if not isinstance(msg.to_agent, str):
            raise ValueError(f"to_agent must be a string, got {type(msg.to_agent).__name__}")
        async with self.db.transaction() as cur:
            cur.execute(
                "INSERT INTO events (msg_id, from_agent, to_agent, topic, "
                "in_reply_to, payload, ts) VALUES (?,?,?,?,?,?,?)",
                msg.to_db_row(),
            )
            msg.seq = int(cur.lastrowid)
        return msg.seq

    async def record_observation(self, source: str, topic: str, payload: dict) -> None:
        """Broadcast one ``source`` message under ``topic`` to every agent."""
        await self.append_and_seq(Message.new(source, "*", topic, payload))

    async def tail(
        self,
        n: int = 200,
        *,
        to_agent: str | None = None,
        topic: str | None = None,
    ) -> list[Message]:
        """Return the most recent messages matching the given filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if to_agent is not None:
            clauses.append("(to_agent = ? OR to_agent = '*')")
            params.append(to_agent)
        if topic is not None:
            clauses.append("topic = ?")
            params.append(topic)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM events{where} ORDER BY seq DESC LIMIT ?"  # nosec B608 - clauses are selected from fixed templates.
        params.append(n)
        rows = await self.db.fetchall(sql, params)
        return [Message.from_row(r) for r in rows]

    def count_sync(self) -> int:
        """Return how many events the log holds."""
        return int(self.db.fetchone_sync("SELECT COUNT(*) FROM events")[0])

    def inbox_context_sync(self, to_agent: str, *, after_seq: int = 0) -> list[Message]:
        """Read an uncapped recipient inbox, including all topics and self-sent events."""
        rows = self.db.fetchall_sync(
            "SELECT * FROM events WHERE seq > ? AND (to_agent = ? OR to_agent = '*') ORDER BY seq ASC",
            (after_seq, to_agent),
        )
        return [Message.from_row(row) for row in rows]

    def recent_outcomes_context_sync(self, *, limit: int) -> list[Message]:
        """Read the latest outcomes, newest first."""
        rows = self.db.fetchall_sync(
            "SELECT * FROM events WHERE topic = 'delegated_result' ORDER BY seq DESC LIMIT ?",
            (limit,),
        )
        return [Message.from_row(row) for row in rows]

    async def replay_for(
        self,
        agent: str,
        *,
        after_seq: int,
        limit: int = DEFAULT_EVENTS_KEEP_RECENT,
    ) -> list[Message]:
        """Inbox for ``agent``: every message past ``after_seq`` it did not send itself.

        Raw-DB readers (``tail``, ``inbox_context_sync``, ``recent_outcomes_context_sync``) show its own rows too.
        """
        rows = await self.db.fetchall(
            f"SELECT * FROM events WHERE {_INBOX_WHERE} ORDER BY seq ASC LIMIT ?",  # nosec B608 - fixed template.
            (after_seq, agent, limit),
        )
        return [Message.from_row(r) for r in rows]

    async def has_actionable_inbox(self, agent: str, *, after_seq: int) -> bool:
        """Whether ``agent``'s inbox past ``after_seq`` holds a message that asks something of it."""
        row = await self.db.fetchone(
            f"SELECT EXISTS(SELECT 1 FROM events WHERE {_INBOX_WHERE}"  # nosec B608 - fixed template.
            " AND NOT (topic = 'response' AND json_extract(payload, '$.source') IS ?)) AS found",
            (after_seq, agent, LANES_BUSY_SOURCE),
        )
        return bool(row and row["found"])


__all__ = ["LANES_BUSY_SOURCE", "Message", "MessageBus", "TOPIC_ALLOWLIST"]
