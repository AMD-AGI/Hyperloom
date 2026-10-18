# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduler's answer to the rule gate's branch requests: a fork for a correctness fix, a repair of a blocked gain.

:class:`ForkDesk` is the scheduler's :class:`~hyperloom_multibranch.contracts.fork.ForkAdmissionService`. A request of
a branch that is on the roster and not stopped is queued, first in first out, under a reserved branch id; it enters the
roster when a slot is free and nothing queued before it waits, which for a request that finds the queue empty and a
slot free is at once, answered ``ACCEPTED`` with its branch. One that has to wait is answered ``PENDING``; its reserved
branch stays the scheduler's own, in the admission decision, and it enters the roster when the explore loop fills the
slots freed after it, before the judge's new branches. The queue holds at most ``max_branches`` requests. A request is
refused when its branch is unknown or stopped, a repair of the same intent on the same commit is open or queued, the
session is stopping, or the queue is full. A queued request is voided when its branch stops or the session stops; a
voided request takes no slot and frees its intent for another repair.

A request that enters the roster takes a slot of the branch cap, with the K of the branch that asked, admitted but not
started, and waits on the desk until the explore loop starts it: a repair at the commit the blocked candidate was
measured against, a fork once the rule gate has landed its fix. The desk needs no peer, so it is built before them and
handed to the specialist runtime, which builds its rule gates on it.

An admitted repair carries its request, which the branch control hands the new specialist as its branch's task; the
diagnosis is also kept in the ``REPAIR`` decision. The fix a fork lands reaches its specialist through the notebook.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from hyperloom_multibranch.contracts.fork import AdmissionStatus, ForkAdmission, ForkRequest, RepairRequest
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_settle_voided
from hyperloom_multibranch.scheduling.roster import BranchOrigin, Roster

DECISION_KINDS = {BranchOrigin.FORK: DecisionKind.FORK, BranchOrigin.REPAIR: DecisionKind.REPAIR}
QUEUE_FULL = "pending queue full"
WAITING = "waiting for a branch slot"


@dataclass(frozen=True)
class AdmittedBranch:
    """A fork or repair the desk admitted, cut from ``parent_commit``; ``repair`` is the request a repair was admitted
    for, ``None`` for a fork."""

    branch: str
    parent_commit: str
    repair: RepairRequest | None = None


@dataclass(frozen=True)
class PendingBranch:
    """A fork or repair waiting for a branch slot under its reserved ``branch`` id, asked by ``from_branch`` and queued
    at clock time ``queued_at``."""

    branch: str
    origin: BranchOrigin
    from_branch: str
    parent_commit: str
    queued_at: float
    repair: RepairRequest | None = None


def _parent_refusal(roster: Roster, from_branch: str) -> str:
    parent = next((entry for entry in roster.entries() if entry.branch == from_branch), None)
    if parent is None:
        return f"unknown parent branch {from_branch!r}"
    return "parent branch is stopped" if parent.stop is not None else ""


def _repair_key(repair: RepairRequest) -> tuple[str, str]:
    return (repair.intent.parent_commit, repair.intent.fingerprint)


class ForkDesk:
    """Admits the rule gate's fork and repair requests onto the roster, queueing those it has no slot for."""

    def __init__(self, roster: Roster, decisions: DecisionLog, clock: Clock) -> None:
        self._roster = roster
        self._decisions = decisions
        self._clock = clock
        self._queue: deque[PendingBranch] = deque()
        self._waiting: dict[str, AdmittedBranch] = {}
        self._repairs: set[tuple[str, str]] = set()

    async def admit(self, request: ForkRequest) -> ForkAdmission:
        pending = PendingBranch("", BranchOrigin.FORK, request.from_branch, request.parent_commit, self._clock.now())
        return self._admit(pending, request.reason, _parent_refusal(self._roster, request.from_branch))

    async def admit_repair(self, request: RepairRequest) -> ForkAdmission:
        intent = request.intent
        refusal = _parent_refusal(self._roster, intent.branch)
        if not refusal and _repair_key(request) in self._repairs:
            refusal = f"a repair of this intent on {intent.parent_commit} is already open or queued"
        pending = PendingBranch(
            "", BranchOrigin.REPAIR, intent.branch, intent.parent_commit, self._clock.now(), request
        )
        answer = self._admit(pending, f"repair of {intent.mechanism} on {intent.target}", refusal)
        if answer.status is not AdmissionStatus.REJECTED:
            self._repairs.add(_repair_key(request))
        return answer

    def _admit(self, pending: PendingBranch, reason: str, refusal: str) -> ForkAdmission:
        """Refuse, queue or admit ``pending``; nothing jumps the queue, even onto a slot freed while it waits."""

        refusal = refusal or self._queue_refusal()
        if refusal:
            self._record(pending, refusal, AdmissionStatus.REJECTED.value)
            return ForkAdmission(AdmissionStatus.REJECTED, refusal)
        waits = bool(self._queue) or bool(self._roster.open_refusal())
        pending = replace(pending, branch=self._roster.next_branch_id())
        if waits:
            self._queue.append(pending)
            answer = ForkAdmission(AdmissionStatus.PENDING, WAITING)
        else:
            self._enter(pending)
            answer = ForkAdmission(AdmissionStatus.ACCEPTED, reason, pending.branch)
        self._record(pending, reason, answer.status.value)
        return answer

    def _queue_refusal(self) -> str:
        if self._roster.closed:
            return self._roster.open_refusal()
        return QUEUE_FULL if len(self._queue) >= self._roster.caps.max_branches else ""

    def _enter(self, pending: PendingBranch) -> None:
        k = self._roster.entry(pending.from_branch).k
        self._roster.open(pending.branch, origin=pending.origin, start_commit="", k=k, from_branch=pending.from_branch)
        self._waiting[pending.branch] = AdmittedBranch(pending.branch, pending.parent_commit, pending.repair)

    def _record(self, pending: PendingBranch, reason: str, status: str) -> None:
        detail: dict[str, object] = {
            "status": status,
            "from_branch": pending.from_branch,
            "parent_commit": pending.parent_commit,
        }
        if pending.repair is not None:
            detail["diagnosis"] = pending.repair.diagnosis
        self._decisions.record(Decision(DECISION_KINDS[pending.origin], reason, pending.branch, detail))

    def fill_free_slots(self) -> None:
        """Move the queued requests, oldest first, onto the roster's free slots; each waits on the desk to start."""

        while self._queue and not self._roster.open_refusal():
            pending = self._queue.popleft()
            self._enter(pending)
            self._record(pending, f"dequeued after {self._clock.now() - pending.queued_at:.0f}s", "dequeued")

    async def void_from(self, parent: str, reason: str) -> None:
        """Void every queued request ``parent`` asked for, as ``parent`` has stopped for ``reason``."""

        await self._void(tuple(pending for pending in self._queue if pending.from_branch == parent), reason)

    async def void_all(self, reason: str) -> None:
        """Void the whole queue, as the session stops for ``reason``."""

        await self._void(tuple(self._queue), reason)

    async def _void(self, voided: tuple[PendingBranch, ...], reason: str) -> None:
        """Take ``voided`` off the queue before the first await, so no fill moves one onto the roster meanwhile."""

        self._queue = deque(pending for pending in self._queue if pending not in voided)
        for pending in voided:
            if pending.repair is not None:
                self._repairs.discard(_repair_key(pending.repair))
            self._record(pending, f"voided: {reason}", "voided")
        for pending in voided:
            await mb_fake_settle_voided(pending.branch, reason)

    def pending(self) -> tuple[PendingBranch, ...]:
        """The requests waiting for a slot, oldest first."""

        return tuple(self._queue)

    def waiting(self) -> tuple[AdmittedBranch, ...]:
        """The admitted branches not started yet, oldest first."""

        return tuple(self._waiting.values())

    def release(self, branch: str) -> None:
        """Take ``branch`` off the desk: it started, or it never will."""

        del self._waiting[branch]


__all__ = ["DECISION_KINDS", "QUEUE_FULL", "WAITING", "AdmittedBranch", "ForkDesk", "PendingBranch"]
