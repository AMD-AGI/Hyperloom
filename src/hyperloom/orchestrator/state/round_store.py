# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The durable bring-up round store: who may start a round, and when.

Mutual exclusion between server bring-ups lives in the session database, not in
Coordinator memory. Five operations over ``bringup_rounds`` -- ``open``,
``renew``, ``handoff``, ``observe``, ``settle`` -- each append to the
``round_events`` outbox whether they applied or not.

An open round also holds a row in ``leases`` on
:data:`~hyperloom.orchestrator.bus.resource_lock.BRINGUP_ROUND_LANE`, written by
the same transaction that writes the round row. That row is the round's clock
and the only place the loop looks to decide whether a round has run out, so the
lease reaper is the one sweep over both. What the lease layer cannot express --
the fence, the outcome, the exclusion a settled round leaves behind, the outbox
and its re-drive -- stays here.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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

#: The lease ran out and the holder's processes were confirmed killed.
EXPIRED_REAPED = "expired_reaped"

#: The lease ran out and nothing ever confirmed the holder dead.
EXPIRED_UNREAPED = "expired_unreaped"

#: The two ways a lease can run out, kept apart because the reap either
#: confirmed the holder dead or did not.
EXPIRY_OUTCOMES = frozenset({EXPIRED_REAPED, EXPIRED_UNREAPED})

OUTCOMES = frozenset({BOOTED, FAILED, ABANDONED, *EXPIRY_OUTCOMES})

#: Rejection reasons, recorded on the outbox row.
UNKNOWN_ROUND = "unknown_round"
NOT_OWNER = "not_owner"
STALE_FENCE = "stale_fence"
NOT_OPEN = "not_open"
EXCLUDED = "excluded"
ALREADY_EXISTS = "already_exists"
ALREADY_SETTLED = "already_settled"

#: The op an observation is recorded under.
OBSERVE = "observe"

#: Evidence keys an observation event carries.
EVIDENCE_STAGE = "stage"
EVIDENCE_DIGEST = "failure_digest"

#: Default for an attempt that records nothing beyond the operation itself.
_NO_EVIDENCE: Mapping[str, Any] = MappingProxyType({})

#: The admission predicate: a round holds the machine while it is open and its
#: lease is live. Time-bounded on purpose -- an exclusion that outlived every
#: reader is what trapped a session before, and the repair pass settles an open
#: round at the top of a tick, before anything asks to acquire.
_LIVE_EXCLUSION = f"state = '{OPEN}' AND expires_unix > ?"

__all__ = [
    "ABANDONED",
    "ALREADY_SETTLED",
    "BOOTED",
    "EVIDENCE_DIGEST",
    "EVIDENCE_STAGE",
    "EXCLUDED",
    "EXPIRED_REAPED",
    "EXPIRED_UNREAPED",
    "EXPIRY_OUTCOMES",
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
            or ``None`` when nothing ever confirmed it.
        reap_backend: What performed (or would have performed) the reap.
        probe_origin: What caused the round to be opened.
        provisional: Whether the round's result is not yet trustworthy.
        correctness_verified: Whether the round's server passed correctness.
        stage_high_water: Highest ladder stage the round ever reached.
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
    reap_backend: str
    probe_origin: str
    provisional: bool
    correctness_verified: bool
    stage_high_water: int

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
            reap_backend=str(row["reap_backend"]),
            probe_origin=str(row["probe_origin"]),
            provisional=bool(row["provisional"]),
            correctness_verified=bool(row["correctness_verified"]),
            stage_high_water=int(row["stage_high_water"]),
        )

    def excludes_at(self, now_unix: float) -> bool:
        """Report whether this round denies an acquire at ``now_unix``.

        The Python mirror of :data:`_LIVE_EXCLUSION`.

        Args:
            now_unix: The instant to test.

        Returns:
            bool: ``True`` when an acquire must be denied.
        """
        return self.state == OPEN and self.expires_unix > float(now_unix)


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

    Each operation runs inside one ``BEGIN IMMEDIATE``.

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
        probe_origin: str = "",
        reap_backend: str = "",
        provisional: bool = False,
        join: Callable[[sqlite3.Cursor], None] | None = None,
        evidence: Mapping[str, Any] = _NO_EVIDENCE,
    ) -> RoundResult:
        """Acquire the round, if and only if no live exclusion denies it.

        The round's lane row is written by the same transaction, so a round row
        without its lease cannot exist.

        Args:
            round_id: Identity of the round to acquire.
            holder_task_id: The task that will hold it.
            lease_sec: How long the acquire is good for without a renewal.
            now_unix: Current wall time.
            request_id: The caller's id for this attempt.
            probe_origin: What caused this round to be opened.
            reap_backend: What would reap this round's processes.
            provisional: Whether the round's result starts untrustworthy.
            join: Ran with the acquiring cursor once the insert succeeds; its
                writes commit with the acquire or not at all.
            evidence: Recorded on the outbox row.

        Returns:
            RoundResult: ``ok`` when the round was acquired; otherwise
            ``reason`` is :data:`EXCLUDED` or :data:`ALREADY_EXISTS`.
        """
        now = float(now_unix)
        expires = now + max(0.0, float(lease_sec))
        async with self.db.transaction() as cur:
            cur.execute(
                "INSERT INTO bringup_rounds ("
                "  round_id, state, outcome, holder_task_id, fence,"
                "  opened_unix, renewed_unix, expires_unix, settled_unix,"
                "  reap_backend, probe_origin, provisional,"
                "  correctness_verified, stage_high_water"
                ") SELECT ?, ?, '', ?, 1, ?, ?, ?, NULL, ?, ?, ?, 0, 0"
                f" WHERE NOT EXISTS (SELECT 1 FROM bringup_rounds WHERE {_LIVE_EXCLUSION})"  # nosec B608 - a fixed predicate constant, no caller input.
                "   AND NOT EXISTS (SELECT 1 FROM bringup_rounds WHERE round_id = ?)",
                (
                    round_id,
                    OPEN,
                    holder_task_id,
                    now,
                    now,
                    expires,
                    reap_backend,
                    probe_origin,
                    1 if provisional else 0,
                    now,
                    round_id,
                ),
            )
            if cur.rowcount != 1:
                cur.execute("SELECT 1 FROM bringup_rounds WHERE round_id = ?", (round_id,))
                reason = ALREADY_EXISTS if cur.fetchone() is not None else EXCLUDED
                event_id = _record(
                    cur,
                    round_id=round_id,
                    request_id=request_id,
                    op="open",
                    result="rejected",
                    outcome="",
                    fence=0,
                    actor_task_id=holder_task_id,
                    reason=reason,
                    evidence=evidence,
                    now_unix=now,
                )
                return RoundResult(ok=False, round_id=round_id, reason=reason, event_id=event_id)
            hold_round_lane(
                cur,
                round_id=round_id,
                holder_task_id=holder_task_id,
                expires_unix=expires,
                now_unix=now,
            )
            if join is not None:
                join(cur)
            event_id = _record(
                cur,
                round_id=round_id,
                request_id=request_id,
                op="open",
                result="applied",
                outcome="",
                fence=1,
                actor_task_id=holder_task_id,
                reason="",
                evidence=evidence,
                now_unix=now,
            )
        return RoundResult(ok=True, round_id=round_id, fence=1, state=OPEN, event_id=event_id)

    async def renew(
        self,
        round_id: str,
        *,
        holder_task_id: str,
        fence: int,
        lease_sec: float,
        now_unix: float,
        request_id: str,
        evidence: Mapping[str, Any] = _NO_EVIDENCE,
    ) -> RoundResult:
        """Extend an open round's lease without changing who holds it.

        The fence names the holder, not the lease, so a renewal leaves it alone.
        The round's lane row is stamped with the same new expiry.

        Args:
            round_id: The round to renew.
            holder_task_id: The task claiming to hold it.
            fence: The fence the holder acquired under.
            lease_sec: How much longer the lease is good for, from ``now_unix``.
            now_unix: Current wall time.
            request_id: The caller's id for this attempt.
            evidence: Recorded on the outbox row.

        Returns:
            RoundResult: ``ok`` when the lease moved; otherwise ``reason``.
        """
        now = float(now_unix)
        expires = now + max(0.0, float(lease_sec))
        async with self.db.transaction() as cur:
            round_row = _load(cur, round_id)
            reason = _cas_reason(round_row, holder_task_id=holder_task_id, fence=fence)
            if reason:
                return _reject(
                    cur, round_row, round_id, request_id, "renew", "", fence, holder_task_id, reason, evidence, now
                )
            cur.execute(
                "UPDATE bringup_rounds SET renewed_unix = ?, expires_unix = ?"
                " WHERE round_id = ? AND holder_task_id = ? AND fence = ? AND state = ?",
                (now, expires, round_id, holder_task_id, int(fence), OPEN),
            )
            hold_round_lane(
                cur,
                round_id=round_id,
                holder_task_id=holder_task_id,
                expires_unix=expires,
                now_unix=now,
            )
            event_id = _record(
                cur,
                round_id=round_id,
                request_id=request_id,
                op="renew",
                result="applied",
                outcome="",
                fence=int(fence),
                actor_task_id=holder_task_id,
                reason="",
                evidence=evidence,
                now_unix=now,
            )
        return RoundResult(ok=True, round_id=round_id, fence=int(fence), state=OPEN, event_id=event_id)

    async def handoff(
        self,
        round_id: str,
        *,
        holder_task_id: str,
        fence: int,
        new_holder_task_id: str,
        lease_sec: float,
        now_unix: float,
        request_id: str,
        evidence: Mapping[str, Any] = _NO_EVIDENCE,
    ) -> RoundResult:
        """Move an open round to a new holder, advancing the fence.

        The only fence increment in the store, so a fence value identifies
        exactly one holder for exactly one span. The round stays open, so its
        lane row moves to the new holder rather than being released.

        Args:
            round_id: The round to hand off.
            holder_task_id: The task currently holding it.
            fence: The fence the current holder acquired under.
            new_holder_task_id: The task taking it over.
            lease_sec: The new holder's lease, from ``now_unix``.
            now_unix: Current wall time.
            request_id: The caller's id for this attempt.
            evidence: Recorded on the outbox row.

        Returns:
            RoundResult: ``ok`` with the advanced ``fence``; otherwise
            ``reason``.
        """
        now = float(now_unix)
        expires = now + max(0.0, float(lease_sec))
        next_fence = int(fence) + 1
        async with self.db.transaction() as cur:
            round_row = _load(cur, round_id)
            reason = _cas_reason(round_row, holder_task_id=holder_task_id, fence=fence)
            if reason:
                return _reject(
                    cur, round_row, round_id, request_id, "handoff", "", fence, holder_task_id, reason, evidence, now
                )
            cur.execute(
                "UPDATE bringup_rounds SET holder_task_id = ?, fence = ?, renewed_unix = ?,"
                "  expires_unix = ?"
                " WHERE round_id = ? AND holder_task_id = ? AND fence = ? AND state = ?",
                (
                    new_holder_task_id,
                    next_fence,
                    now,
                    expires,
                    round_id,
                    holder_task_id,
                    int(fence),
                    OPEN,
                ),
            )
            hold_round_lane(
                cur,
                round_id=round_id,
                holder_task_id=new_holder_task_id,
                expires_unix=expires,
                now_unix=now,
            )
            event_id = _record(
                cur,
                round_id=round_id,
                request_id=request_id,
                op="handoff",
                result="applied",
                outcome="",
                fence=next_fence,
                actor_task_id=new_holder_task_id,
                reason="",
                evidence=evidence,
                now_unix=now,
            )
        return RoundResult(ok=True, round_id=round_id, fence=next_fence, state=OPEN, event_id=event_id)

    async def observe(
        self,
        round_id: str,
        *,
        actor_task_id: str,
        stage: int,
        failure_digest: str,
        now_unix: float,
        request_id: str,
        evidence: Mapping[str, Any] = _NO_EVIDENCE,
    ) -> RoundResult:
        """Record what a boot was seen to do, and raise the high-water mark.

        The only writer of ``stage_high_water``, and it only moves it up. The
        append takes no compare-and-swap, so no observation is ever dropped.

        Args:
            round_id: The round the boot ran under; ``""`` when none did.
            actor_task_id: The task that watched the boot.
            stage: Ladder stage value the boot reached.
            failure_digest: Stable identity of the wall it stopped at, or ``""``
                for a boot that did not stop at one.
            now_unix: Current wall time.
            request_id: The caller's id for this attempt.
            evidence: Extra fields recorded alongside the stage and digest.

        Returns:
            RoundResult: ``ok``, carrying the event id of the ledger row.
        """
        now = float(now_unix)
        recorded = dict(evidence)
        recorded[EVIDENCE_STAGE] = max(0, int(stage))
        recorded[EVIDENCE_DIGEST] = failure_digest
        async with self.db.transaction() as cur:
            cur.execute(
                "UPDATE bringup_rounds SET stage_high_water = MAX(stage_high_water, ?) WHERE round_id = ?",
                (max(0, int(stage)), round_id),
            )
            event_id = _record(
                cur,
                round_id=round_id,
                request_id=request_id,
                op=OBSERVE,
                result="applied",
                outcome="",
                fence=0,
                actor_task_id=actor_task_id,
                reason="",
                evidence=recorded,
                now_unix=now,
            )
        return RoundResult(ok=True, round_id=round_id, event_id=event_id)

    async def settle(
        self,
        round_id: str,
        *,
        holder_task_id: str,
        fence: int,
        outcome: str,
        now_unix: float,
        request_id: str,
        reap_backend: str = "",
        correctness_verified: bool = False,
        provisional: bool = False,
        evidence: Mapping[str, Any] = _NO_EVIDENCE,
    ) -> RoundResult:
        """End the round, releasing the machine.

        A settled round denies nothing. The lane row goes with it, so what the
        next acquire sees is the absence of an open round rather than a record
        of this one.

        Args:
            round_id: The round to settle.
            holder_task_id: The task claiming to hold it.
            fence: The fence the holder acquired under.
            outcome: One of :data:`OUTCOMES`.
            now_unix: Current wall time.
            request_id: The caller's id for this attempt.
            reap_backend: What performed the reap.
            correctness_verified: Whether the round's server passed correctness.
            provisional: Whether the round's result is untrustworthy.
            evidence: Recorded on the outbox row; a rejected settle is
                re-driven from it.

        Returns:
            RoundResult: ``ok`` when the round settled, ``duplicate`` when this
            settle had already been applied, otherwise ``reason``.

        Raises:
            ValueError: When ``outcome`` is not one of :data:`OUTCOMES`.
        """
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown round outcome {outcome!r}; expected one of {sorted(OUTCOMES)}")
        now = float(now_unix)
        async with self.db.transaction() as cur:
            round_row = _load(cur, round_id)
            if round_row is not None and round_row.state == SETTLED:
                if (
                    round_row.outcome == outcome
                    and round_row.holder_task_id == holder_task_id
                    and round_row.fence == int(fence)
                ):
                    event_id = _record(
                        cur,
                        round_id=round_id,
                        request_id=request_id,
                        op="settle",
                        result="duplicate",
                        outcome=outcome,
                        fence=int(fence),
                        actor_task_id=holder_task_id,
                        reason="",
                        evidence=evidence,
                        now_unix=now,
                    )
                    return RoundResult(
                        ok=True,
                        round_id=round_id,
                        fence=round_row.fence,
                        state=SETTLED,
                        outcome=round_row.outcome,
                        duplicate=True,
                        event_id=event_id,
                    )
                return _reject(
                    cur,
                    round_row,
                    round_id,
                    request_id,
                    "settle",
                    outcome,
                    fence,
                    holder_task_id,
                    ALREADY_SETTLED,
                    evidence,
                    now,
                )
            reason = _cas_reason(round_row, holder_task_id=holder_task_id, fence=fence)
            if reason:
                return _reject(
                    cur,
                    round_row,
                    round_id,
                    request_id,
                    "settle",
                    outcome,
                    fence,
                    holder_task_id,
                    reason,
                    evidence,
                    now,
                )
            cur.execute(
                "UPDATE bringup_rounds SET"
                "  state = ?, outcome = ?, settled_unix = ?,"
                "  reap_backend = CASE WHEN ? <> '' THEN ? ELSE reap_backend END,"
                "  correctness_verified = ?, provisional = ?"
                " WHERE round_id = ? AND holder_task_id = ? AND fence = ? AND state = ?",
                (
                    SETTLED,
                    outcome,
                    now,
                    reap_backend,
                    reap_backend,
                    1 if correctness_verified else 0,
                    1 if provisional else 0,
                    round_id,
                    holder_task_id,
                    int(fence),
                    OPEN,
                ),
            )
            drop_round_lane(cur, round_id=round_id)
            event_id = _record(
                cur,
                round_id=round_id,
                request_id=request_id,
                op="settle",
                result="applied",
                outcome=outcome,
                fence=int(fence),
                actor_task_id=holder_task_id,
                reason="",
                evidence=evidence,
                now_unix=now,
            )
        return RoundResult(
            ok=True,
            round_id=round_id,
            fence=int(fence),
            state=SETTLED,
            outcome=outcome,
            event_id=event_id,
        )

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
            (float(now_unix),),
        )
        return [Round.from_row(r) for r in rows]

    async def observations(self) -> list[RoundEvent]:
        """Return every observation this session recorded, oldest first.

        Order is load-bearing: a failure digest is new exactly once.

        Returns:
            list[RoundEvent]: The applied observation rows.
        """
        rows = await self.db.fetchall(
            "SELECT * FROM round_events WHERE op = ? AND result = 'applied' ORDER BY event_id ASC",
            (OBSERVE,),
        )
        return [RoundEvent.from_row(r) for r in rows]

    async def redrivable_settles(self) -> list[RoundEvent]:
        """Return rejected settles whose round is still not settled.

        Returns:
            list[RoundEvent]: The rejected settles still worth re-driving,
            oldest first.
        """
        rows = await self.db.fetchall(
            "SELECT e.* FROM round_events e"
            " LEFT JOIN bringup_rounds r ON r.round_id = e.round_id"
            " WHERE e.op = 'settle' AND e.result = 'rejected'"
            "   AND (r.round_id IS NULL OR r.state <> ?)"
            " ORDER BY e.event_id ASC",
            (SETTLED,),
        )
        return [RoundEvent.from_row(r) for r in rows]


def _load(cur: sqlite3.Cursor, round_id: str) -> Round | None:
    """Read one round inside the caller's transaction."""
    cur.execute("SELECT * FROM bringup_rounds WHERE round_id = ?", (round_id,))
    row = cur.fetchone()
    return None if row is None else Round.from_row(row)


def _cas_reason(round_row: Round | None, *, holder_task_id: str, fence: int) -> str:
    """Return why a compare-and-swap must be refused, or ``""`` when it may proceed."""
    if round_row is None:
        return UNKNOWN_ROUND
    if round_row.state != OPEN:
        return NOT_OPEN
    if round_row.holder_task_id != holder_task_id:
        return NOT_OWNER
    if round_row.fence != int(fence):
        return STALE_FENCE
    return ""


def _reject(
    cur: sqlite3.Cursor,
    round_row: Round | None,
    round_id: str,
    request_id: str,
    op: str,
    outcome: str,
    fence: int,
    actor_task_id: str,
    reason: str,
    evidence: Mapping[str, Any],
    now_unix: float,
) -> RoundResult:
    """Record a refused attempt on the outbox and describe it to the caller."""
    event_id = _record(
        cur,
        round_id=round_id,
        request_id=request_id,
        op=op,
        result="rejected",
        outcome=outcome,
        fence=int(fence),
        actor_task_id=actor_task_id,
        reason=reason,
        evidence=evidence,
        now_unix=now_unix,
    )
    return RoundResult(
        ok=False,
        round_id=round_id,
        fence=0 if round_row is None else round_row.fence,
        state="" if round_row is None else round_row.state,
        outcome="" if round_row is None else round_row.outcome,
        reason=reason,
        event_id=event_id,
    )


def _record(
    cur: sqlite3.Cursor,
    *,
    round_id: str,
    request_id: str,
    op: str,
    result: str,
    outcome: str,
    fence: int,
    actor_task_id: str,
    reason: str,
    evidence: Mapping[str, Any],
    now_unix: float,
) -> int:
    """Append one attempt to the outbox and return its ``event_id``."""
    cur.execute(
        "INSERT INTO round_events ("
        "  round_id, request_id, op, result, outcome, fence,"
        "  actor_task_id, reason, evidence, recorded_unix"
        ") VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            round_id,
            request_id,
            op,
            result,
            outcome,
            int(fence),
            actor_task_id,
            reason,
            json.dumps(dict(evidence), sort_keys=True),
            float(now_unix),
        ),
    )
    event_id = cur.lastrowid
    if event_id is None:
        raise sqlite3.DatabaseError("the round_events insert reported no row id")
    return int(event_id)
