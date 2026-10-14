# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduler's answer to the rule gate's branch requests: a fork for a correctness fix, a repair of a blocked gain.

:class:`ForkDesk` is the scheduler's :class:`~hyperloom_multibranch.contracts.fork.BranchRequests`. A request is
admitted when the roster may open a branch; what one the roster cannot take gets is :func:`mb_fake_refused_fork`. An
admitted branch is put on the roster at once, admitted but not started, so it takes a slot of the branch cap, and
:meth:`ForkDesk.drain` hands it to the branch control, which opens it at the requested commit and starts its specialist
with a brief that carries the change to build on: the fix a fork lands first, or the blocked candidate and its
diagnosis a repair starts from. The desk needs no peer, so it is built before them and handed to the specialist
runtime, which builds its rule gates on it.

A repair names the branch that asks, through its intent, and inherits its K; one of a branch the roster does not hold,
or of one already stopped, is refused, as stopping a branch withdraws everything of it, and an intent is repaired at
most once per commit. A fork names no branch, so it starts at K 1 and is opened from no branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.fork import Admission, AdmissionStatus, ForkRequest, RepairRequest
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_refused_fork
from hyperloom_multibranch.scheduling.roster import BranchOrigin, Roster

#: The K a fork starts with: the request names no branch to inherit one from.
FORK_K = 1
DECISION_KINDS = {BranchOrigin.FORK: DecisionKind.FORK, BranchOrigin.REPAIR: DecisionKind.REPAIR}


@dataclass(frozen=True)
class AdmittedBranch:
    """A fork or repair the desk admitted: its specialist starts at ``start_commit`` with ``brief``."""

    branch: str
    start_commit: str
    brief: str


@dataclass(frozen=True)
class _Ask:
    origin: BranchOrigin
    start_commit: str
    from_branch: str
    brief: str
    reason: str


def _change(delta: RecipeDelta) -> str:
    return json.dumps(delta.to_document(), indent=2, sort_keys=True)


def fork_brief(request: ForkRequest) -> str:
    return "\n".join(
        (
            "This branch carries a correctness fix that cost the branch it was found on performance.",
            f"Why: {request.reason}",
            f"Reproduction case, which fails without the fix: {request.repro_case}",
            "Land the fix first, then optimize on top of it. The fix, as a recipe change:",
            _change(request.fix_delta),
        )
    )


def repair_brief(request: RepairRequest) -> str:
    intent, judgement = request.intent, request.judgement
    return "\n".join(
        (
            f"This branch repairs a gain the rule gate blocked on {intent.branch}: {intent.mechanism} on "
            f"{intent.target} ({intent.layer.value}), {judgement.gain_pct:+.2f}% held back by "
            f"{', '.join(judgement.refused_by) or 'the accuracy gate'}.",
            f"Diagnosis and repair plan: {request.diagnosis}",
            "The blocked candidate, as a recipe change:",
            _change(request.candidate_delta),
        )
    )


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
        self._admitted: list[AdmittedBranch] = []
        self._repairs: set[tuple[str, str]] = set()

    async def request_fork(self, request: ForkRequest) -> Admission:
        ask = _Ask(BranchOrigin.FORK, request.parent_commit, "", fork_brief(request), request.reason)
        return self._admit(ask, "")

    async def request_repair(self, request: RepairRequest) -> Admission:
        intent = request.intent
        key = (intent.parent_commit, intent.fingerprint)
        refusal = _parent_refusal(self._roster, intent.branch)
        if not refusal and key in self._repairs:
            refusal = f"a repair of this intent on {intent.parent_commit} is already open"
        reason = f"repair of {intent.mechanism} on {intent.target}"
        answer = self._admit(
            _Ask(BranchOrigin.REPAIR, intent.parent_commit, intent.branch, repair_brief(request), reason), refusal
        )
        if answer.status is AdmissionStatus.ACCEPTED:
            self._repairs.add(key)
        return answer

    def _admit(self, ask: _Ask, refusal: str) -> Admission:
        branch = ""
        roster_refusal = self._roster.open_refusal()
        if refusal:
            answer = Admission(AdmissionStatus.REJECTED, refusal)
        elif roster_refusal:
            answer = mb_fake_refused_fork(roster_refusal)
        else:
            k = self._roster.entry(ask.from_branch).k if ask.from_branch else FORK_K
            branch = self._roster.next_branch_id()
            self._roster.open(branch, origin=ask.origin, start_commit="", k=k, from_branch=ask.from_branch)
            self._admitted.append(AdmittedBranch(branch, ask.start_commit, ask.brief))
            answer = Admission(AdmissionStatus.ACCEPTED, ask.reason)
        detail = {"status": answer.status.value, "from_branch": ask.from_branch, "start_commit": ask.start_commit}
        self._decisions.record(Decision(DECISION_KINDS[ask.origin], answer.reason, branch, detail))
        return answer

    def drain(self) -> tuple[AdmittedBranch, ...]:
        """The branches admitted since the last call, oldest first; never waits."""

        drained, self._admitted = tuple(self._admitted), []
        return drained


__all__ = ["DECISION_KINDS", "FORK_K", "AdmittedBranch", "ForkDesk", "fork_brief", "repair_brief"]
