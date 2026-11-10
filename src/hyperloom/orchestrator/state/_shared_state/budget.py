# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_BudgetStateMixin`` — the wall-clock session budget of :class:`..shared_state.SharedState`: run legs, elapsed time and deadlines."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from hyperloom.common.deadline import Deadline

from .elapsed import elapsed_sec_of


class _BudgetStateMixin:
    # Session budget: elapsed is summed forward across legs.
    def begin_leg(self, *, now_unix: float | None = None) -> None:
        """Open a run leg: start charging elapsed time from this instant.

        A leg is one process's turn at the session. Time between legs is not
        charged; every second inside one is, folded into
        :attr:`elapsed_charged_sec` by :meth:`charge_elapsed`. How the previous
        leg ended is deliberately not consulted, so no leg can begin by handing
        itself a fresh budget.

        The baseline failures a previous leg booked stay in
        :attr:`baseline_failure_streak`, which is what ends a stuck run; what
        does not carry over is their claim on *this* leg's budget, so
        :attr:`baseline_failures_this_leg` starts at zero. A leg resumed past
        the optimization reserve is owed the one baseline it was resumed for
        before a retry is priced as a retry.

        Args:
            now_unix: Wall instant the leg starts; defaults to ``time.time()``.
        """
        self.leg_anchor_unix = float(time.time() if now_unix is None else now_unix)
        self.baseline_failures_this_leg = 0

    def charge_elapsed(self, *, now_unix: float | None = None) -> float:
        """Fold the time since the last charge into the session's elapsed total.

        Called on every state save. A no-op until :meth:`begin_leg` opens a
        leg, so reading and re-saving a state bills the session nothing.

        Args:
            now_unix: Wall instant to charge up to; defaults to ``time.time()``.

        Returns:
            float: The new :attr:`elapsed_charged_sec`.
        """
        anchor = self.leg_anchor_unix
        if anchor <= 0.0:
            return self.elapsed_charged_sec
        now = float(time.time() if now_unix is None else now_unix)
        self.elapsed_charged_sec += max(0.0, now - anchor)
        self.leg_anchor_unix = now
        return self.elapsed_charged_sec

    def elapsed_minutes(self, *, now: datetime | None = None) -> float:
        """Minutes of budget this session has consumed, summed over every leg.

        The charged total plus whatever the live leg has run since the last
        charge.

        Args:
            now (datetime | None): Reference time; defaults to the current UTC
                time.

        Returns:
            float: Minutes consumed; never negative.
        """
        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        return self.elapsed_sec_of(vars(self), now_unix=now_dt.timestamp()) / 60.0

    elapsed_sec_of = staticmethod(elapsed_sec_of)

    def extend_budget_minutes(self, minutes: float) -> float:
        """Grant more wall-clock budget to this session.

        Args:
            minutes: Minutes to add; non-positive is a no-op.

        Returns:
            float: The session's budget in minutes after the grant; ``0.0``
                when the session is unbounded and nothing was granted.
        """
        added = float(minutes)
        if added <= 0.0:
            return float(self.max_minutes)
        if not self.max_minutes:
            # Granting an unbounded session a budget would bound it.
            return 0.0
        self.max_minutes = int(float(self.max_minutes) + added)
        return float(self.max_minutes)

    def remaining_minutes(self, *, now: datetime | None = None) -> float | None:
        """Minutes left in the wall-clock budget; ``None`` when unbounded, else clamped at 0.

        :meth:`session_deadline` keeps the sign, for a caller that must tell
        "just expired" from "expired an hour ago".

        Args:
            now (datetime | None): Reference time; defaults to the current UTC
                time.

        Returns:
            float | None: Minutes remaining in the budget (clamped at 0.0), or
                ``None`` when the session is unbounded.
        """
        if not self.max_minutes:
            return None
        return max(0.0, float(self.max_minutes) - self.elapsed_minutes(now=now))

    def session_deadline(self, *, now: datetime | None = None) -> Deadline | None:
        """The absolute instant this session's budget runs out.

        ``None`` means unbounded, and only that: an exhausted session returns a
        :class:`Deadline` already in the past.

        Args:
            now (datetime | None): Reference time for the elapsed sum; defaults
                to the current UTC time.

        Returns:
            Deadline | None: The stop instant, or ``None`` when unbounded.
        """
        if not self.max_minutes:
            return None
        spent_min = self.elapsed_minutes(now=now)
        return Deadline.after((float(self.max_minutes) - spent_min) * 60.0)

    def session_budget_usable_sec(
        self,
        *,
        reserve_sec: float | None = None,
    ) -> float | None:
        """Seconds of wall-clock budget left once the closing reserve is held back."""
        remaining = self.remaining_minutes()
        if remaining is None:
            return None
        reserve = self.closing_reserve_sec() if reserve_sec is None else float(reserve_sec)
        return max(0.0, remaining * 60.0 - reserve)

    def grid_session_deadline_sec(
        self,
        *,
        reserve_sec: float | None = None,
    ) -> float | None:
        """``time.monotonic()`` deadline for grid variant loops, or ``None`` when the budget is unbounded."""
        usable = self.session_budget_usable_sec(reserve_sec=reserve_sec)
        if usable is None:
            return None
        return time.monotonic() + usable
