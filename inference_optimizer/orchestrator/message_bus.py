# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""MessageBus — the ``events`` table is the source of truth; ``seq`` (AUTOINCREMENT) gives a monotonic id. Topics + priorities validated against an allowlist (DESIGN §13.2)."""

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
    # Agent-to-agent RPC (REQUEST / RESPONSE intents).
    "request", "response",
    # Critic Review Protocol verdict broadcast.
    "review_verdict", "advice", "strategy_change",
    # Dynamic-specialist dispatch audit trail (free-form CPU-only
    # specialist dispatch via dynamic_dispatch_tools). These are
    # write-only observation-style records the Coordinator emits so the
    # dispatch / poll / collect lifecycle is visible in the bus; no
    # consumer keys off them, but they must be allow-listed or
    # ``append_and_seq`` rejects them with ``unknown topic``.
    "dynamic_specialist_dispatched", "dynamic_specialist_status",
    "dynamic_specialist_results", "dynamic_specialist_error",
})


def _now_iso() -> str:
    """Return the current UTC time as a microsecond-precision ISO 8601 string.

    Returns:
        str: The current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class Message:
    """One bus message persisted in the ``events`` table.

    Attributes:
        msg_id (str): Unique message identifier.
        from_agent (str): Sending agent id.
        to_agent (str): Recipient agent id (``"*"`` for broadcast).
        topic (str): Topic (validated against :data:`TOPIC_ALLOWLIST`).
        payload (dict[str, Any]): The message payload.
        priority (int): Priority in ``0..3``. Defaults to ``1``.
        in_reply_to (str | None): The ``msg_id`` this replies to, if any.
        ts (str): ISO timestamp the message was created.
        seq (int | None): Monotonic sequence id assigned by the bus on insert.
    """

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
        """Construct a new message with a fresh ``msg_id``.

        Args:
            from_agent (str): Sending agent id.
            to_agent (str): Recipient agent id (``"*"`` for broadcast).
            topic (str): Topic.
            payload (dict[str, Any]): The message payload.
            priority (int): Priority in ``0..3``. Defaults to ``1``.
            in_reply_to (str | None): The ``msg_id`` this replies to, if any.

        Returns:
            Message: The constructed message (``seq`` unset until inserted).
        """
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
        """Build a :class:`Message` from an ``events`` table row.

        Args:
            row: A mapping-like DB row with the ``events`` columns.

        Returns:
            Message: The reconstructed message.
        """
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
        """Serialise the message into an ``events`` INSERT tuple.

        Returns:
            tuple: The column values in INSERT order (``payload`` JSON-encoded);
                ``seq`` is intentionally omitted as it is DB-assigned.
        """
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
    """Append-only message log backed by the ``events`` table.

    Attributes:
        db (SqliteConnection): The backing SQLite connection.
    """

    def __init__(self, db: SqliteConnection):
        """Initialise the bus.

        Args:
            db (SqliteConnection): The backing SQLite connection.
        """
        self.db = db

    async def append_and_seq(self, msg: Message) -> int:
        """Append one message and return its assigned sequence id.

        Args:
            msg (Message): The message to append; its ``seq`` is set in place.

        Returns:
            int: The monotonic sequence id assigned by SQLite.

        Raises:
            ValueError: If the topic is unknown or priority is out of ``0..3``.
        """
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
        """Bulk-insert as a single transaction.

        Args:
            messages (Iterable[Message]): Messages to insert; each ``seq`` is
                set in place.

        Returns:
            list[int]: The assigned sequence ids, in insertion order.

        Raises:
            ValueError: If any message carries an unknown topic.
        """
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
        """Return the most recent messages matching the given filters.

        Args:
            n (int): Maximum number of messages to return. Defaults to ``200``.
            after_seq (int): Only return messages with ``seq`` greater than
                this. Defaults to ``0``.
            to_agent (str | None): Filter to this recipient (plus broadcasts).
            topic (str | None): Filter to this topic.

        Returns:
            list[Message]: Matching messages ordered by descending ``seq``.
        """
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
        """Used at resume — returns events in monotonic seq order.

        Args:
            to_agent (str): Recipient agent id (broadcasts are included).
            after_seq (int): Only return messages with ``seq`` greater than
                this.

        Returns:
            list[Message]: Matching messages ordered by ascending ``seq``.
        """
        rows = await self.db.fetchall(
            "SELECT * FROM events WHERE seq > ? AND (to_agent = ? OR to_agent = '*') "
            "ORDER BY seq ASC",
            (after_seq, to_agent),
        )
        return [Message.from_row(r) for r in rows]

    async def lookup_by_id(self, msg_id: str) -> Message | None:
        """Look up a single message by its ``msg_id``.

        Args:
            msg_id (str): The message identifier.

        Returns:
            Message | None: The matching message, or ``None`` if absent.
        """
        row = await self.db.fetchone(
            "SELECT * FROM events WHERE msg_id = ?", (msg_id,)
        )
        return Message.from_row(row) if row else None

    async def count(self) -> int:
        """Return the total number of messages in the bus.

        Returns:
            int: The row count of the ``events`` table.
        """
        row = await self.db.fetchone("SELECT COUNT(*) AS c FROM events")
        return int(row["c"]) if row else 0

    @staticmethod
    def message_to_dict(msg: Message) -> dict:
        """Convert a message to a plain dict.

        Args:
            msg (Message): The message to convert.

        Returns:
            dict: The dataclass fields as a dict.
        """
        return asdict(msg)


__all__ = ["Message", "MessageBus", "TOPIC_ALLOWLIST"]
