# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduler's answer to the rule gate's branch requests: a fork for a correctness fix, a repair of a blocked gain.

:class:`ForkDesk` is the scheduler's :class:`~hyperloom_multibranch.contracts.fork.ForkAdmissionService`. A request is
admitted when the roster may open a branch and the branch that asks is on the roster and not stopped, as stopping a
branch withdraws everything of it; what the roster cannot take is answered by :func:`mb_fake_refused_fork`. An intent
is repaired at most once per commit. An admitted branch is put on the roster at once, admitted but not started, with
the K of the branch that asked, so it takes a slot of the branch cap, and it waits on the desk until the branch control
starts it: a repair at the commit the blocked candidate was measured against, a fork once the gate has cut it and
landed the fix on it. The desk needs no peer, so it is built before them and handed to the specialist runtime, which
builds its rule gates on it.

The specialist runtime reads a new branch's brief from the notebook, so neither the fix a fork lands nor a repair's
diagnosis reaches the new specialist from here; the diagnosis is kept in the ``REPAIR`` decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from hyperloom_multibranch.contracts.fork import AdmissionStatus, ForkAdmission, ForkRequest, RepairRequest
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_refused_fork
from hyperloom_multibranch.scheduling.roster import BranchOrigin, Roster

DECISION_KINDS = {BranchOrigin.FORK: DecisionKind.FORK, BranchOrigin.REPAIR: DecisionKind.REPAIR}


@dataclass(frozen=True)
class AdmittedBranch:
    """A fork or repair the desk admitted, cut from ``parent_commit``."""

    branch: str
    parent_commit: str


@dataclass(frozen=True)
class _Ask:
    origin: BranchOrigin
    parent_commit: str
    from_branch: str
    reason: str
    detail: dict[str, object]


def _parent_refusal(roster: Roster, from_branch: str) -> str:
    parent = next((entry for entry in roster.entries() if entry.branch == from_branch), None)
    if parent is None:
        return f"unknown parent branch {from_branch!r}"
    return "parent branch is stopped" if parent.stop is not None else ""


class ForkDesk:
    """Admits the rule gate's fork and repair requests onto the roster."""

    def __init__(self, roster: Roster, decisions: DecisionLog) -> None:
        self._roster = roster
        self._decisions = decisions
        self._waiting: dict[str, AdmittedBranch] = {}
        self._repairs: set[tuple[str, str]] = set()

    async def admit(self, request: ForkRequest) -> ForkAdmission:
        ask = _Ask(BranchOrigin.FORK, request.parent_commit, request.from_branch, request.reason, {})
        return self._admit(ask, _parent_refusal(self._roster, request.from_branch))

    async def admit_repair(self, request: RepairRequest) -> ForkAdmission:
        intent = request.intent
        key = (intent.parent_commit, intent.fingerprint)
        refusal = _parent_refusal(self._roster, intent.branch)
        if not refusal and key in self._repairs:
            refusal = f"a repair of this intent on {intent.parent_commit} is already open"
        reason = f"repair of {intent.mechanism} on {intent.target}"
        ask = _Ask(BranchOrigin.REPAIR, intent.parent_commit, intent.branch, reason, {"diagnosis": request.diagnosis})
        answer = self._admit(ask, refusal)
        if answer.status is AdmissionStatus.ACCEPTED:
            self._repairs.add(key)
        return answer

    def _admit(self, ask: _Ask, refusal: str) -> ForkAdmission:
        roster_refusal = self._roster.open_refusal()
        if refusal:
            answer = ForkAdmission(AdmissionStatus.REJECTED, refusal)
        elif roster_refusal:
            answer = mb_fake_refused_fork(roster_refusal)
        else:
            branch = self._roster.next_branch_id()
            k = self._roster.entry(ask.from_branch).k
            self._roster.open(branch, origin=ask.origin, start_commit="", k=k, from_branch=ask.from_branch)
            self._waiting[branch] = AdmittedBranch(branch, ask.parent_commit)
            answer = ForkAdmission(AdmissionStatus.ACCEPTED, ask.reason, branch)
        detail = {
            "status": answer.status.value,
            "from_branch": ask.from_branch,
            "parent_commit": ask.parent_commit,
            **ask.detail,
        }
        self._decisions.record(Decision(DECISION_KINDS[ask.origin], answer.reason, answer.branch, detail))
        return answer

    def waiting(self) -> tuple[AdmittedBranch, ...]:
        """The admitted branches not started yet, oldest first."""

        return tuple(self._waiting.values())

    def release(self, branch: str) -> None:
        """Take ``branch`` off the desk: it started, or it never will."""

        del self._waiting[branch]


__all__ = ["DECISION_KINDS", "AdmittedBranch", "ForkDesk"]
