# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MessageBus — the ``events`` table is the source of truth; ``seq`` (AUTOINCREMENT) gives a monotonic id. Topics + priorities validated against an allowlist."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.timeutil import now_iso

from .db_maintenance import DEFAULT_EVENTS_KEEP_RECENT
from .storage.connection import SqliteConnection


TOPIC_ALLOWLIST = frozenset(
    {
        # Optimization-loop topics
        "proposal",
        "observation",
        "event",
        "decision",
        "alert",
        "historical_warning",
        "reflection_tick",
        "do_postmortem",
        "do_strategic_review",
        "do_emergency_rca",
        "synthesize_for_kb",
        "graceful_stop",
        "heartbeat",
        "delegated_result",
        "intent_emitted",
        "rca_done",
        # Storage-layer events
        "lease_expired",
        "lease_acquire_failed",
        # Agent-to-agent RPC (REQUEST / RESPONSE intents).
        "request",
        "response",
        # Critic Review Protocol verdict broadcast.
        "review_verdict",
        "advice",
        "strategy_change",
        # Reserved for a dynamic-specialist dispatch audit trail; no emitter in-tree yet.
        "dynamic_specialist_dispatched",
        "dynamic_specialist_status",
        "dynamic_specialist_results",
        "dynamic_specialist_error",
    }
)


_now_iso = now_iso


@dataclass
class Message:
    """One bus message persisted in the ``events`` table."""

    msg_id: str
    from_agent: str
    to_agent: str
    topic: str
    payload: dict[str, Any]
    priority: int = 1
    in_reply_to: str | None = None
    ts: str = field(default_factory=_now_iso)
    seq: int | None = None  # DB-assigned on insert

    @classmethod
    def new(
        cls,
        from_agent: str,
        to_agent: str,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: int = 1,
        in_reply_to: str | None = None,
    ) -> "Message":
        """Construct a new message with a fresh ``msg_id``."""
        return cls(
            msg_id=uuid.uuid4().hex,
            from_agent=from_agent,
            to_agent=to_agent,
            topic=topic,
            payload=payload,
            priority=priority,
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
            priority=row["priority"],
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
            self.priority,
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
        if not (0 <= msg.priority <= 3):
            raise ValueError(f"priority must be 0..3, got {msg.priority}")
        async with self.db.transaction() as cur:
            cur.execute(
                "INSERT INTO events (msg_id, from_agent, to_agent, topic, "
                "in_reply_to, payload, priority, ts) VALUES (?,?,?,?,?,?,?,?)",
                msg.to_db_row(),
            )
            msg.seq = int(cur.lastrowid)
        return msg.seq

    async def tail(
        self,
        n: int = 200,
        *,
        after_seq: int = 0,
        to_agent: str | None = None,
        topic: str | None = None,
    ) -> list[Message]:
        """Return the most recent messages matching the given filters."""
        clauses = ["seq > ?"]
        params: list[Any] = [after_seq]
        if to_agent is not None:
            clauses.append("(to_agent = ? OR to_agent = '*')")
            params.append(to_agent)
        if topic is not None:
            clauses.append("topic = ?")
            params.append(topic)
        sql = f"SELECT * FROM events WHERE {' AND '.join(clauses)} ORDER BY seq DESC LIMIT ?"  # nosec B608 - clauses are selected from fixed templates.
        params.append(n)
        rows = await self.db.fetchall(sql, params)
        return [Message.from_row(r) for r in rows]

    async def replay_for(
        self,
        to_agent: str,
        *,
        after_seq: int,
        limit: int = DEFAULT_EVENTS_KEEP_RECENT,
    ) -> list[Message]:
        """Used at resume — returns events in monotonic seq order."""
        rows = await self.db.fetchall(
            "SELECT * FROM events WHERE seq > ? AND (to_agent = ? OR to_agent = '*') ORDER BY seq ASC LIMIT ?",
            (after_seq, to_agent, limit),
        )
        return [Message.from_row(r) for r in rows]

    async def lookup_by_id(self, msg_id: str) -> Message | None:
        """Look up a single message by its ``msg_id``."""
        row = await self.db.fetchone("SELECT * FROM events WHERE msg_id = ?", (msg_id,))
        return Message.from_row(row) if row else None


__all__ = ["Message", "MessageBus", "TOPIC_ALLOWLIST"]
