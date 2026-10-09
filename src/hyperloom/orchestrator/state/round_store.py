# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The durable bring-up round store: who may start a round, and when."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from hyperloom.orchestrator.bus.resource_lock import drop_round_lane, hold_round_lane
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

#: Round states.
OPEN = "open"
SETTLED = "settled"

#: Round outcomes.
BOOTED = "booted"
FAILED = "failed"
ABANDONED = "abandoned"

#: The round cleared a new boot failure without reaching a clean boot.
ADVANCED = "advanced"

#: The lease ran out and the holder's processes were confirmed killed.
EXPIRED_REAPED = "expired_reaped"

#: The lease ran out and nothing ever confirmed the holder dead.
EXPIRED_UNREAPED = "expired_unreaped"

#: The two ways a lease can run out, kept apart because the reap either
#: confirmed the holder dead or did not.
EXPIRY_OUTCOMES = frozenset({EXPIRED_REAPED, EXPIRED_UNREAPED})

OUTCOMES = frozenset({BOOTED, FAILED, ABANDONED, ADVANCED, *EXPIRY_OUTCOMES})

#: Rejection reasons, recorded on the outbox row.
UNKNOWN_ROUND = "unknown_round"
NOT_OWNER = "not_owner"
STALE_FENCE = "stale_fence"
NOT_OPEN = "not_open"
EXCLUDED = "excluded"
ALREADY_EXISTS = "already_exists"
ALREADY_SETTLED = "already_settled"

#: Default for an attempt that records nothing beyond the operation itself.
_NO_EVIDENCE: Mapping[str, Any] = MappingProxyType({})

#: Ownership ends at explicit settlement, never at a budget timestamp.
_LIVE_EXCLUSION = f"state = '{OPEN}'"

__all__ = [
    "ABANDONED",
    "ALREADY_SETTLED",
    "BOOTED",
    "EXCLUDED",
    "EXPIRED_REAPED",
    "EXPIRED_UNREAPED",
    "FAILED",
    "NOT_OWNER",
    "OPEN",
    "SETTLED",
    "STALE_FENCE",
    "Round",
    "RoundEvent",
    "RoundResult",
    "RoundStore",
]


@dataclass(frozen=True)
class Round:
    """One ``bringup_rounds`` row.

    Attributes:
        round_id: Identity of the round.
        state: :data:`OPEN` or :data:`SETTLED`.
        outcome: One of :data:`OUTCOMES`, empty while the round is open.
        holder_task_id: The task that currently holds the round.
        fence: Monotone token; only :meth:`RoundStore.handoff` advances it.
        opened_unix: When the round was acquired.
        renewed_unix: When its lease was last extended.
        expires_unix: When its lease runs out; the same instant the round's
            lane row carries.
        settled_unix: When it was settled, or ``None``.
    """

    round_id: str
    state: str
    outcome: str
    holder_task_id: str
    fence: int
    opened_unix: float
    renewed_unix: float
    expires_unix: float
    settled_unix: float | None

    @classmethod
    def from_row(cls, row: Any) -> "Round":
        """Build a :class:`Round` from a ``bringup_rounds`` row.

        Args:
            row: A mapping-like database row (``sqlite3.Row`` in production).

        Returns:
            Round: The decoded row.
        """
        settled = row["settled_unix"]
        return cls(
            round_id=str(row["round_id"]),
            state=str(row["state"]),
            outcome=str(row["outcome"]),
            holder_task_id=str(row["holder_task_id"]),
            fence=int(row["fence"]),
            opened_unix=float(row["opened_unix"]),
            renewed_unix=float(row["renewed_unix"]),
            expires_unix=float(row["expires_unix"]),
            settled_unix=None if settled is None else float(settled),
        )

    def excludes_at(self, now_unix: float) -> bool:
        """Report whether this round denies an acquire at ``now_unix``.

        Args:
            now_unix: The instant to test.

        Returns:
            bool: ``True`` when an acquire must be denied.
        """
        return self.state == OPEN


@dataclass(frozen=True)
class RoundEvent:
    """One ``round_events`` row: an attempt, and what became of it.

    Attributes:
        event_id: Monotone id; also the outbox's order.
        round_id: The round the attempt addressed.
        request_id: The caller's id for this attempt, carried into a re-drive.
        op: ``open`` / ``renew`` / ``handoff`` / ``settle``.
        result: ``applied`` / ``rejected`` / ``duplicate``.
        outcome: The outcome a settle asked for.
        fence: The fence the caller presented.
        actor_task_id: The task that made the attempt.
        reason: Why a rejected attempt was rejected.
        evidence: Whatever the caller recorded alongside the attempt.
        recorded_unix: When it was recorded.
    """

    event_id: int
    round_id: str
    request_id: str
    op: str
    result: str
    outcome: str
    fence: int
    actor_task_id: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    recorded_unix: float = 0.0

    @classmethod
    def from_row(cls, row: Any) -> "RoundEvent":
        """Build a :class:`RoundEvent` from a ``round_events`` row.

        Args:
            row: A mapping-like database row (``sqlite3.Row`` in production).

        Returns:
            RoundEvent: The decoded row, with ``evidence`` JSON-decoded.
        """
        return cls(
            event_id=int(row["event_id"]),
            round_id=str(row["round_id"]),
            request_id=str(row["request_id"]),
            op=str(row["op"]),
            result=str(row["result"]),
            outcome=str(row["outcome"]),
            fence=int(row["fence"]),
            actor_task_id=str(row["actor_task_id"]),
            reason=str(row["reason"]),
            evidence=json.loads(row["evidence"]),
            recorded_unix=float(row["recorded_unix"]),
        )


@dataclass(frozen=True)
class RoundResult:
    """What an operation did.

    Attributes:
        ok: Whether the store's state now reflects what was asked.
        round_id: The round addressed.
        fence: The fence in force after the operation.
        state: The round's state after the operation.
        outcome: The round's outcome after the operation.
        reason: Why a failed operation failed; empty when ``ok``.
        duplicate: Whether this was a replay of an attempt already applied.
        event_id: The outbox row this attempt wrote.
    """

    ok: bool
    round_id: str
    fence: int = 0
    state: str = ""
    outcome: str = ""
    reason: str = ""
    duplicate: bool = False
    event_id: int = 0


class RoundStore:
    """Durable acquire / renew / handoff / settle for bring-up rounds.

    Attributes:
        db (SqliteConnection): The session database.
    """

    def __init__(self, db: SqliteConnection):
        """Initialise the store.

        Args:
            db: The session database.
        """
        self.db = db

    async def open(
        self,
        round_id: str,
        *,
        holder_task_id: str,
        lease_sec: float,
        now_unix: float,
        request_id: str,
        join: Callable[[sqlite3.Cursor], None] | None = None,
        evidence: Mapping[str, Any] = _NO_EVIDENCE,
    ) -> RoundResult:
        """Acquire the round, if and only if no live exclusion denies it.

        Args:
            round_id: Identity of the round to acquire.
            holder_task_id: The task that will hold it.
            lease_sec: How long the acquire is good for without a renewal.
            now_unix: Current wall time.
            request_id: The caller's id for this attempt.
            join: Ran with the acquiring cursor once the insert succeeds; its
                writes commit with the acquire or not at all.
            evidence: Recorded on the outbox row.

        Returns:
            RoundResult: ``ok`` when the round was acquired; otherwise
            ``reason`` is :data:`EXCLUDED` or :data:`ALREADY_EXISTS`.
        """
        attempt = _Attempt(
            round_id=round_id,
            request_id=request_id,
            op="open",
            actor_task_id=holder_task_id,
            fence=0,
            outcome="",
            evidence=evidence,
            now_unix=now_unix,
        )
        opened = Round(
            round_id=round_id,
            state=OPEN,
            outcome="",
            holder_task_id=holder_task_id,
            fence=1,
            opened_unix=now_unix,
            renewed_unix=now_unix,
            expires_unix=now_unix + lease_sec,
            settled_unix=None,
        )
        async with self.db.transaction() as cur:
            cur.execute(
                "INSERT INTO bringup_rounds ("
                "  round_id, state, outcome, holder_task_id, fence,"
                "  opened_unix, renewed_unix, expires_unix, settled_unix"
                ") SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?"
                f" WHERE NOT EXISTS (SELECT 1 FROM bringup_rounds WHERE {_LIVE_EXCLUSION})"  # nosec B608 - a fixed predicate constant, no caller input.
                "   AND NOT EXISTS (SELECT 1 FROM bringup_rounds WHERE round_id = ?)",
                (
                    opened.round_id,
                    opened.state,
                    opened.outcome,
                    opened.holder_task_id,
                    opened.fence,
                    opened.opened_unix,
                    opened.renewed_unix,
                    opened.expires_unix,
                    opened.settled_unix,
                    round_id,
                ),
            )
            if cur.rowcount != 1:
                cur.execute("SELECT 1 FROM bringup_rounds WHERE round_id = ?", (round_id,))
                return attempt.reject(cur, None, ALREADY_EXISTS if cur.fetchone() is not None else EXCLUDED)
            _place_lane(cur, opened)
            if join is not None:
                join(cur)
            return attempt.applied(cur, opened)

    async def renew(
        self,
        held: Round,
        *,
        lease_sec: float,
        now_unix: float,
        request_id: str,
        evidence: Mapping[str, Any] = _NO_EVIDENCE,
    ) -> RoundResult:
        """Extend an open round's lease without changing who holds it.

        Args:
            held: The round as its holder last read it.
            lease_sec: How much longer the lease is good for, from ``now_unix``.
            now_unix: Current wall time.
            request_id: The caller's id for this attempt.
            evidence: Recorded on the outbox row.

        Returns:
            RoundResult: ``ok`` when the lease moved; otherwise ``reason``.
        """
        return await self._swap(
            held,
            "renew",
            lambda current: replace(current, renewed_unix=now_unix, expires_unix=now_unix + lease_sec),
            request_id=request_id,
            now_unix=now_unix,
            evidence=evidence,
        )

    async def handoff(
        self,
        held: Round,
        *,
        new_holder_task_id: str,
        lease_sec: float,
        now_unix: float,
        request_id: str,
        evidence: Mapping[str, Any] = _NO_EVIDENCE,
    ) -> RoundResult:
        """Move an open round to a new holder, advancing the fence.

        Args:
            held: The round as its current holder last read it.
            new_holder_task_id: The task taking it over.
            lease_sec: The new holder's lease, from ``now_unix``.
            now_unix: Current wall time.
            request_id: The caller's id for this attempt.
            evidence: Recorded on the outbox row.

        Returns:
            RoundResult: ``ok`` with the advanced ``fence``; otherwise
            ``reason``.
        """
        return await self._swap(
            held,
            "handoff",
            lambda current: replace(
                current,
                holder_task_id=new_holder_task_id,
                fence=current.fence + 1,
                renewed_unix=now_unix,
                expires_unix=now_unix + lease_sec,
            ),
            request_id=request_id,
            now_unix=now_unix,
            evidence=evidence,
        )

    async def settle(
        self,
        held: Round,
        *,
        outcome: str,
        now_unix: float,
        request_id: str,
        evidence: Mapping[str, Any] = _NO_EVIDENCE,
    ) -> RoundResult:
        """End the round, releasing the machine.

        Args:
            held: The round as its holder last read it.
            outcome: One of :data:`OUTCOMES`.
            now_unix: Current wall time.
            request_id: The caller's id for this attempt.
            evidence: Recorded on the outbox row.

        Returns:
            RoundResult: ``ok`` when the round settled, ``duplicate`` when this
            settle had already been applied, otherwise ``reason``.

        Raises:
            ValueError: When ``outcome`` is not one of :data:`OUTCOMES`.
        """
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown round outcome {outcome!r}; expected one of {sorted(OUTCOMES)}")
        return await self._swap(
            held,
            "settle",
            lambda current: replace(current, state=SETTLED, outcome=outcome, settled_unix=now_unix),
            request_id=request_id,
            now_unix=now_unix,
            evidence=evidence,
            outcome=outcome,
        )

    async def _swap(
        self,
        held: Round,
        op: str,
        advance: Callable[[Round], Round],
        *,
        request_id: str,
        now_unix: float,
        evidence: Mapping[str, Any],
        outcome: str = "",
    ) -> RoundResult:
        """Replace the round with ``advance(current)`` while ``held`` still names its holder and fence.

        Only a settle can replay: any other attempt on a settled round is :data:`NOT_OPEN`.
        """
        attempt = _Attempt(
            round_id=held.round_id,
            request_id=request_id,
            op=op,
            actor_task_id=held.holder_task_id,
            fence=held.fence,
            outcome=outcome,
            evidence=evidence,
            now_unix=now_unix,
        )
        async with self.db.transaction() as cur:
            current = _load(cur, held.round_id)
            if current is None:
                return attempt.reject(cur, None, UNKNOWN_ROUND)
            if op == "settle" and current.state == SETTLED:
                if (
                    current.outcome == outcome
                    and current.holder_task_id == held.holder_task_id
                    and current.fence == held.fence
                ):
                    return attempt.applied(cur, current, duplicate=True)
                return attempt.reject(cur, current, ALREADY_SETTLED)
            reason = _cas_reason(current, held)
            if reason:
                return attempt.reject(cur, current, reason)
            after = advance(current)
            # The transaction is BEGIN IMMEDIATE, so the check above and this write are one step.
            cur.execute(
                "UPDATE bringup_rounds SET state = ?, outcome = ?, holder_task_id = ?, fence = ?,"
                "  renewed_unix = ?, expires_unix = ?, settled_unix = ?"
                " WHERE round_id = ?",
                (
                    after.state,
                    after.outcome,
                    after.holder_task_id,
                    after.fence,
                    after.renewed_unix,
                    after.expires_unix,
                    after.settled_unix,
                    after.round_id,
                ),
            )
            _place_lane(cur, after)
            return attempt.applied(cur, after)

    async def held(self) -> Round | None:
        """Return the newest open round, or ``None``.

        Returns:
            Round | None: The open round.
        """
        row = await self.db.fetchone(
            "SELECT * FROM bringup_rounds WHERE state = ? ORDER BY opened_unix DESC LIMIT 1",
            (OPEN,),
        )
        return None if row is None else Round.from_row(row)

    async def open_rounds(self) -> list[Round]:
        """Return every round still open, oldest first.

        Returns:
            list[Round]: The open rounds.
        """
        rows = await self.db.fetchall(
            "SELECT * FROM bringup_rounds WHERE state = ? ORDER BY opened_unix ASC",
            (OPEN,),
        )
        return [Round.from_row(r) for r in rows]

    async def get(self, round_id: str) -> Round | None:
        """Return one round, or ``None`` when it was never opened.

        Args:
            round_id: The round to read.

        Returns:
            Round | None: The round row.
        """
        row = await self.db.fetchone("SELECT * FROM bringup_rounds WHERE round_id = ?", (round_id,))
        return None if row is None else Round.from_row(row)

    async def excluding(self, now_unix: float) -> list[Round]:
        """Return every round that denies an acquire at ``now_unix``.

        Args:
            now_unix: The instant to test.

        Returns:
            list[Round]: The rounds currently excluding, oldest first.
        """
        rows = await self.db.fetchall(
            f"SELECT * FROM bringup_rounds WHERE {_LIVE_EXCLUSION} ORDER BY opened_unix ASC",  # nosec B608 - a fixed predicate constant, no caller input.
        )
        return [Round.from_row(r) for r in rows]

    async def consecutive_stalled(self) -> int:
        """Count the newest settled rounds that bought no ground.

        Returns:
            int: Consecutive stalled rounds, newest first.
        """
        rows = await self.db.fetchall(
            "SELECT outcome FROM bringup_rounds WHERE state = ? ORDER BY settled_unix DESC",
            (SETTLED,),
        )
        neutral = {ABANDONED, *EXPIRY_OUTCOMES}
        count = 0
        for row in rows:
            outcome = row["outcome"]
            if outcome in (BOOTED, ADVANCED):
                break
            if outcome not in neutral:
                count += 1
        return count


def _load(cur: sqlite3.Cursor, round_id: str) -> Round | None:
    """Read one round inside the caller's transaction."""
    cur.execute("SELECT * FROM bringup_rounds WHERE round_id = ?", (round_id,))
    row = cur.fetchone()
    return None if row is None else Round.from_row(row)


def _cas_reason(current: Round, held: Round) -> str:
    """Return why a compare-and-swap on ``current`` must be refused, or ``""`` when ``held`` still names it."""
    if current.state != OPEN:
        return NOT_OPEN
    if current.holder_task_id != held.holder_task_id:
        return NOT_OWNER
    if current.fence != held.fence:
        return STALE_FENCE
    return ""


def _place_lane(cur: sqlite3.Cursor, after: Round) -> None:
    """Make the round's lane row match ``after``: held to its lease while open, gone once settled."""
    if after.state == OPEN:
        hold_round_lane(
            cur,
            round_id=after.round_id,
            holder_task_id=after.holder_task_id,
            expires_unix=after.expires_unix,
            now_unix=after.renewed_unix,
        )
    else:
        drop_round_lane(cur, round_id=after.round_id)


@dataclass(frozen=True, kw_only=True)
class _Attempt:
    """One caller request against a round, as the outbox records it."""

    round_id: str
    request_id: str
    op: str
    actor_task_id: str
    fence: int
    outcome: str
    evidence: Mapping[str, Any]
    now_unix: float

    def applied(self, cur: sqlite3.Cursor, after: Round, *, duplicate: bool = False) -> RoundResult:
        """Record this attempt as having left the round at ``after``, and describe ``after`` to the caller.

        The outbox row carries ``after``'s fence and holder, so an open or a handoff records the ones it put in place.
        """
        event_id = self._append(
            cur,
            result="duplicate" if duplicate else "applied",
            fence=after.fence,
            actor_task_id=after.holder_task_id,
            reason="",
        )
        return RoundResult(
            ok=True,
            round_id=after.round_id,
            fence=after.fence,
            state=after.state,
            outcome=after.outcome,
            duplicate=duplicate,
            event_id=event_id,
        )

    def reject(self, cur: sqlite3.Cursor, current: Round | None, reason: str) -> RoundResult:
        """Record this attempt as refused with what it presented, and describe the round as it stands to the caller."""
        event_id = self._append(
            cur,
            result="rejected",
            fence=self.fence,
            actor_task_id=self.actor_task_id,
            reason=reason,
        )
        return RoundResult(
            ok=False,
            round_id=self.round_id,
            fence=0 if current is None else current.fence,
            state="" if current is None else current.state,
            outcome="" if current is None else current.outcome,
            reason=reason,
            event_id=event_id,
        )

    def _append(self, cur: sqlite3.Cursor, *, result: str, fence: int, actor_task_id: str, reason: str) -> int:
        """Append this attempt to the outbox and return its ``event_id``."""
        cur.execute(
            "INSERT INTO round_events ("
            "  round_id, request_id, op, result, outcome, fence,"
            "  actor_task_id, reason, evidence, recorded_unix"
            ") VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                self.round_id,
                self.request_id,
                self.op,
                result,
                self.outcome,
                fence,
                actor_task_id,
                reason,
                json.dumps(dict(self.evidence), sort_keys=True),
                self.now_unix,
            ),
        )
        event_id = cur.lastrowid
        if event_id is None:
            raise sqlite3.DatabaseError("the round_events insert reported no row id")
        return event_id
