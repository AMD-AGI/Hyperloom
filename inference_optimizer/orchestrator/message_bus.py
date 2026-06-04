"""MessageBus ().

The ``events`` table is the source of truth. ``seq`` is the AUTOINCREMENT
primary key, so we never have to coordinate sequence allocation in
application code; SQLite gives us a globally monotonic id for free.

Topics + priorities are validated against an allowlist (DESIGN §13.2)
before insert. v0.6 changes:

* Removed ``objection`` / ``vote`` / ``vote_request`` / ``parliament_open``
  (parliament gone — ADR-38).
* Added ``review_verdict`` (Critic Review Protocol).
* ``kill`` mirror topic stays; emitted by Robustness via Coordinator.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..storage.connection import SqliteConnection


TOPIC_ALLOWLIST = frozenset({
    # Optimization-loop topics
    "proposal", "question", "answer",
    "observation", "event", "decision",
    "alert",
    "historical_warning", "reflection_tick",
    "do_postmortem", "do_strategic_review", "do_emergency_rca",
    "synthesize_for_kb", "graceful_stop", "heartbeat",
    "delegated_result", "intent_emitted", "rca_done",
    # Storage-layer events
    "lease_expired", "lease_acquire_failed",
    # Agent-to-agent RPC topics carrying REQUEST / RESPONSE intents.
    # Coordinator mirrors request/response payloads onto these topics so the
    # target agent's inbox picks them up (kernel agent contract).
    "request", "response",
    # Critic Review Protocol — verdict broadcast topic.
    "review_verdict", "advice", "strategy_change",
    # Robustness handle / scheduling-police mirror topics for audit trail.
    "kill",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class Message:
    msg_id: str
    from_agent: str
    to_agent: str
    topic: str
    payload: dict[str, Any]
    priority: int = 1
    in_reply_to: str | None = None
    ts: str = field(default_factory=_now_iso)
    seq: int | None = None  # filled in by the bus on insert

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


# ---------------------------------------------------------------------------
class MessageBus:
    def __init__(self, db: SqliteConnection):
        self.db = db

    async def append_and_seq(self, msg: Message) -> int:
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

    async def append_batch(self, messages: Iterable[Message]) -> list[int]:
        """Bulk-insert as a single transaction."""
        seqs: list[int] = []
        async with self.db.transaction() as cur:
            for msg in messages:
                if msg.topic not in TOPIC_ALLOWLIST:
                    raise ValueError(f"unknown topic: {msg.topic!r}")
                cur.execute(
                    "INSERT INTO events (msg_id, from_agent, to_agent, topic, "
                    "in_reply_to, payload, priority, ts) VALUES (?,?,?,?,?,?,?,?)",
                    msg.to_db_row(),
                )
                msg.seq = int(cur.lastrowid)
                seqs.append(msg.seq)
        return seqs

    async def tail(
        self,
        n: int = 200,
        *,
        after_seq: int = 0,
        to_agent: str | None = None,
        topic: str | None = None,
    ) -> list[Message]:
        clauses = ["seq > ?"]
        params: list[Any] = [after_seq]
        if to_agent is not None:
            clauses.append("(to_agent = ? OR to_agent = '*')")
            params.append(to_agent)
        if topic is not None:
            clauses.append("topic = ?")
            params.append(topic)
        sql = (
            f"SELECT * FROM events WHERE {' AND '.join(clauses)} "
            f"ORDER BY seq DESC LIMIT ?"
        )
        params.append(n)
        rows = await self.db.fetchall(sql, params)
        return [Message.from_row(r) for r in rows]

    async def replay_for(self, to_agent: str, *, after_seq: int) -> list[Message]:
        """Used at resume — returns events in monotonic seq order."""
        rows = await self.db.fetchall(
            "SELECT * FROM events WHERE seq > ? AND (to_agent = ? OR to_agent = '*') "
            "ORDER BY seq ASC",
            (after_seq, to_agent),
        )
        return [Message.from_row(r) for r in rows]

    async def lookup_by_id(self, msg_id: str) -> Message | None:
        row = await self.db.fetchone(
            "SELECT * FROM events WHERE msg_id = ?", (msg_id,)
        )
        return Message.from_row(row) if row else None

    async def count(self) -> int:
        row = await self.db.fetchone("SELECT COUNT(*) AS c FROM events")
        return int(row["c"]) if row else 0

    @staticmethod
    def message_to_dict(msg: Message) -> dict:
        return asdict(msg)


__all__ = ["Message", "MessageBus", "TOPIC_ALLOWLIST"]
