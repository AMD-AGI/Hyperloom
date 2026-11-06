# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduler's answer to the requests for a new branch: the rule gate's fork for a correctness fix and repair of a
blocked gain, and a mapping branch's share branch, which runs an Arbor agent on its share of the map.

:class:`ForkDesk` is the scheduler's :class:`~hyperloom_multibranch.contracts.fork.ForkAdmissionService`. A request of a
branch that is on the roster and not stopped is queued, first in first out, under a reserved branch id; it enters the
roster when a slot is free and nothing queued before it waits, which for a request that finds the queue empty and a slot
free is at once, answered ``ACCEPTED`` with its branch. One that has to wait is answered ``PENDING`` with its reserved
branch, which stays off the roster until it enters when the explore loop fills the slots freed after it, before the
judge's new branches. The queue holds at most ``max_branches`` requests. A request is refused when its branch is unknown
or stopped, the session is stopping, or the queue is full; a repair is also refused while a repair of the same change,
on any head, or of the same intent on the same commit is open or queued, and the refusal names that repair's branch,
whose result is the one the change waits for: one blocked change replayed onto two heads is repaired once. A queued
request is voided when its branch stops or the session stops; a voided request takes no slot and frees its change and
intent for another repair.

A request that enters the roster takes a slot of the branch cap, with the K of the branch that asked, admitted but not
started, and waits on the desk until the explore loop starts it: a repair at the commit the blocked candidate was
measured against, a fork once the rule gate has landed its fix. The desk needs no peer, so it is built before them and
handed to the specialist runtime, which builds its rule gates on it.

An admitted repair carries its request, which the branch control hands the new specialist as its branch's task; the
diagnosis is also kept in the ``REPAIR`` decision. An admitted share carries its agent and plan the same way, and is cut
at the commit its mapping branch opened on. The fix a fork lands reaches its specialist through the blackboard.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from hyperloom_multibranch.contracts.fork import (
    AdmissionStatus,
    ForkAdmission,
    ForkRequest,
    RepairRequest,
    ShareRequest,
)
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.roster import BranchOrigin, Roster

DECISION_KINDS = {
    BranchOrigin.FORK: DecisionKind.FORK,
    BranchOrigin.REPAIR: DecisionKind.REPAIR,
    BranchOrigin.SHARE: DecisionKind.SHARE,
}
QUEUE_FULL = "pending queue full"
WAITING = "waiting for a branch slot"


@dataclass(frozen=True)
class AdmittedBranch:
    """A fork, repair or share the desk admitted, cut from ``parent_commit``; ``repair`` and ``share`` are the request a
    repair or a share was admitted for, ``None`` otherwise."""

    branch: str
    parent_commit: str
    repair: RepairRequest | None = None
    share: ShareRequest | None = None


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
    share: ShareRequest | None = None


def _parent_refusal(roster: Roster, from_branch: str) -> str:
    parent = next((entry for entry in roster.entries() if entry.branch == from_branch), None)
    if parent is None:
        return f"unknown parent branch {from_branch!r}"
    return "parent branch is stopped" if parent.stop is not None else ""


def _share_refusal(roster: Roster, request: ShareRequest) -> str:
    """A share is asked by its mapping branch as that branch's run ends, so only an unknown branch refuses it."""

    if any(entry.branch == request.from_branch for entry in roster.entries()):
        return ""
    return f"unknown parent branch {request.from_branch!r}"


def _repair_keys(repair: RepairRequest) -> tuple[tuple[str, ...], ...]:
    """What makes two repairs the same: the blocked change, whose action fingerprint is the same on whatever head it
    was measured, and its intent on the commit it was measured against."""

    return ((repair.candidate_delta.fingerprint(),), (repair.intent.parent_commit, repair.intent.fingerprint))


class ForkDesk:
    """Admits the rule gate's fork and repair requests onto the roster, queueing those it has no slot for."""

    def __init__(self, roster: Roster, decisions: DecisionLog, clock: Clock) -> None:
        self._roster = roster
        self._decisions = decisions
        self._clock = clock
        self._queue: deque[PendingBranch] = deque()
        self._waiting: dict[str, AdmittedBranch] = {}
        self._repairs: dict[tuple[str, ...], str] = {}

    async def admit(self, request: ForkRequest) -> ForkAdmission:
        pending = PendingBranch("", BranchOrigin.FORK, request.from_branch, request.parent_commit, self._clock.now())
        return self._admit(pending, request.reason, _parent_refusal(self._roster, request.from_branch))

    async def admit_repair(self, request: RepairRequest) -> ForkAdmission:
        intent = request.intent
        keys = _repair_keys(request)
        refusal = _parent_refusal(self._roster, intent.branch)
        repairing = next((self._repairs[key] for key in keys if key in self._repairs), "")
        if not refusal and repairing:
            refusal = (
                f"{repairing} already repairs this change, or this intent on {intent.parent_commit}, and is open or "
                "queued; replay what it lands instead"
            )
        pending = PendingBranch(
            "", BranchOrigin.REPAIR, intent.branch, intent.parent_commit, self._clock.now(), request
        )
        answer = self._admit(pending, f"repair of {intent.mechanism} on {intent.target}", refusal)
        if answer.status is not AdmissionStatus.REJECTED:
            self._repairs.update(dict.fromkeys(keys, answer.branch))
        return answer

    async def admit_share(self, request: ShareRequest) -> ForkAdmission:
        pending = PendingBranch(
            "", BranchOrigin.SHARE, request.from_branch, request.start_commit, self._clock.now(), share=request
        )
        self._roster.unhold(request.from_branch)
        return self._admit(pending, f"share {request.plan} of {request.agent}", _share_refusal(self._roster, request))

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
            answer = ForkAdmission(AdmissionStatus.PENDING, WAITING, pending.branch)
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
        self._waiting[pending.branch] = AdmittedBranch(
            pending.branch, pending.parent_commit, pending.repair, pending.share
        )

    def _record(self, pending: PendingBranch, reason: str, status: str) -> None:
        detail: dict[str, object] = {
            "status": status,
            "from_branch": pending.from_branch,
            "parent_commit": pending.parent_commit,
        }
        if pending.repair is not None:
            detail["diagnosis"] = pending.repair.diagnosis
        if pending.share is not None:
            detail.update(agent=pending.share.agent, plan=pending.share.plan)
        self._decisions.record(Decision(DECISION_KINDS[pending.origin], reason, pending.branch, detail))

    def fill_free_slots(self) -> None:
        """Move the queued requests, oldest first, onto the roster's free slots; each waits on the desk to start."""

        while self._queue and not self._roster.open_refusal():
            pending = self._queue.popleft()
            self._enter(pending)
            self._record(pending, f"dequeued after {self._clock.now() - pending.queued_at:.0f}s", "dequeued")

    def void_from(self, parent: str, reason: str) -> tuple[str, ...]:
        """Void every queued fork and repair ``parent`` asked for, as ``parent`` has stopped for ``reason``; their
        reserved branch ids, oldest first. A share outlives its mapping branch, whose run ends once it has asked."""

        voided = (p for p in self._queue if p.from_branch == parent and p.origin is not BranchOrigin.SHARE)
        return self._void(tuple(voided), reason)

    def void_all(self, reason: str) -> tuple[str, ...]:
        """Void the whole queue, as the session stops for ``reason``; the reserved branch ids, oldest first."""

        return self._void(tuple(self._queue), reason)

    def _void(self, voided: tuple[PendingBranch, ...], reason: str) -> tuple[str, ...]:
        self._queue = deque(pending for pending in self._queue if pending not in voided)
        for pending in voided:
            if pending.repair is not None:
                for key in _repair_keys(pending.repair):
                    del self._repairs[key]
            self._record(pending, f"voided: {reason}", "voided")
        return tuple(pending.branch for pending in voided)

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
