# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The durable bring-up round store: who may start a round, and when."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, replace
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

#: The lease ran out and the holder's processes were killed.
EXPIRED_REAPED = "expired_reaped"

OUTCOMES = frozenset({BOOTED, FAILED, ABANDONED, ADVANCED, EXPIRED_REAPED})

#: Rejection reasons, returned as :attr:`RoundResult.reason`.
UNKNOWN_ROUND = "unknown_round"
NOT_OWNER = "not_owner"
STALE_FENCE = "stale_fence"
NOT_OPEN = "not_open"
EXCLUDED = "excluded"
ALREADY_EXISTS = "already_exists"
ALREADY_SETTLED = "already_settled"

#: Ownership ends at explicit settlement, never at a budget timestamp.
_LIVE_EXCLUSION = f"state = '{OPEN}'"

__all__ = [
    "ABANDONED",
    "ALREADY_SETTLED",
    "BOOTED",
    "EXCLUDED",
    "EXPIRED_REAPED",
    "FAILED",
    "NOT_OWNER",
    "OPEN",
    "SETTLED",
    "STALE_FENCE",
    "Round",
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


@dataclass(frozen=True)
class RoundResult:
    """What an operation did.

    Attributes:
        ok: Whether the store's state now reflects what was asked.
        reason: Why a failed operation failed; empty when ``ok``.
    """

    ok: bool
    reason: str = ""


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
        now_unix: float,
        lease_sec: float,
        join: Callable[[sqlite3.Cursor], None] | None = None,
    ) -> RoundResult:
        """Acquire the round, if and only if no live exclusion denies it.

        Args:
            round_id: Identity of the round to acquire.
            holder_task_id: The task that will hold it.
            now_unix: Current wall time.
            lease_sec: How long the acquire is good for without a renewal.
            join: Ran with the acquiring cursor once the insert succeeds; its
                writes commit with the acquire or not at all.

        Returns:
            RoundResult: ``ok`` when the round was acquired; otherwise
            ``reason`` is :data:`EXCLUDED` or :data:`ALREADY_EXISTS`.
        """
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
                return RoundResult(ok=False, reason=ALREADY_EXISTS if cur.fetchone() is not None else EXCLUDED)
            _place_lane(cur, opened)
            if join is not None:
                join(cur)
            return RoundResult(ok=True)

    async def renew(
        self,
        held: Round,
        *,
        now_unix: float,
        lease_sec: float,
    ) -> RoundResult:
        """Extend an open round's lease without changing who holds it.

        Args:
            held: The round as its holder last read it.
            now_unix: Current wall time.
            lease_sec: How much longer the lease is good for, from ``now_unix``.

        Returns:
            RoundResult: ``ok`` when the lease moved; otherwise ``reason``.
        """
        return await self._swap(
            held,
            lambda current: replace(current, renewed_unix=now_unix, expires_unix=now_unix + lease_sec),
        )

    async def handoff(
        self,
        held: Round,
        *,
        new_holder_task_id: str,
        now_unix: float,
        lease_sec: float,
    ) -> RoundResult:
        """Move an open round to a new holder, advancing the fence.

        Args:
            held: The round as its current holder last read it.
            new_holder_task_id: The task taking it over.
            now_unix: Current wall time.
            lease_sec: The new holder's lease, from ``now_unix``.

        Returns:
            RoundResult: ``ok`` when the round moved; otherwise ``reason``.
        """
        return await self._swap(
            held,
            lambda current: replace(
                current,
                holder_task_id=new_holder_task_id,
                fence=current.fence + 1,
                renewed_unix=now_unix,
                expires_unix=now_unix + lease_sec,
            ),
        )

    async def settle(
        self,
        held: Round,
        *,
        outcome: str,
        now_unix: float,
    ) -> RoundResult:
        """End the round, releasing the machine.

        Args:
            held: The round as its holder last read it.
            outcome: One of :data:`OUTCOMES`.
            now_unix: Current wall time.

        Returns:
            RoundResult: ``ok`` when the round settled or this settle had
            already been applied; otherwise ``reason``.

        Raises:
            ValueError: When ``outcome`` is not one of :data:`OUTCOMES`.
        """
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown round outcome {outcome!r}; expected one of {sorted(OUTCOMES)}")
        return await self._swap(
            held,
            lambda current: replace(current, state=SETTLED, outcome=outcome, settled_unix=now_unix),
            settles_as=outcome,
        )

    async def _swap(
        self,
        held: Round,
        advance: Callable[[Round], Round],
        *,
        settles_as: str = "",
    ) -> RoundResult:
        """Replace the round with ``advance(current)`` while ``held`` still names its holder and fence.

        Only a settle (``settles_as`` names its outcome) can replay: any other attempt on a settled round is
        :data:`NOT_OPEN`.
        """
        async with self.db.transaction() as cur:
            current = _load(cur, held.round_id)
            if current is None:
                return RoundResult(ok=False, reason=UNKNOWN_ROUND)
            if settles_as and current.state == SETTLED:
                replayed = (
                    current.outcome == settles_as
                    and current.holder_task_id == held.holder_task_id
                    and current.fence == held.fence
                )
                return RoundResult(ok=True) if replayed else RoundResult(ok=False, reason=ALREADY_SETTLED)
            reason = _cas_reason(current, held)
            if reason:
                return RoundResult(ok=False, reason=reason)
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
            return RoundResult(ok=True)

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
        """Return every round that denies an acquire, oldest first.

        Returns:
            list[Round]: The open rounds.
        """
        rows = await self.db.fetchall(
            f"SELECT * FROM bringup_rounds WHERE {_LIVE_EXCLUSION} ORDER BY opened_unix ASC",  # nosec B608 - a fixed predicate constant, no caller input.
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

    async def consecutive_stalled(self) -> int:
        """Count the newest settled rounds that bought no ground.

        Returns:
            int: Consecutive stalled rounds, newest first.
        """
        rows = await self.db.fetchall(
            "SELECT outcome FROM bringup_rounds WHERE state = ? ORDER BY settled_unix DESC",
            (SETTLED,),
        )
        neutral = {ABANDONED, EXPIRED_REAPED}
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
