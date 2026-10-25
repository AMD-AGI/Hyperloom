# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Judging a branch's measured candidates: the half of the specialist runtime that is not a model.

The specialist proposes and the toolbox measures; this code compares each measurement through the rule gate, retests a
gain on another node, checks a declared correctness fix (``fix_check``) and lands it or asks for its fork, and tells the
specialist in its branch log what landed and what the gate could not land. A retest that fails as a candidate settles
the candidate as regressed; one whose measurement failed leaves it unjudged. Every step is read back from the notebook,
so a restarted runtime takes each step once; a fix check interrupted by a restart runs its repro tests again.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from hyperloom.contracts.measurement import Purpose, Validity
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.fork import AdmissionStatus, ForkAdmission
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook.interfaces import (
    BranchLogStore,
    CommitQuery,
    Inbox,
    NotRecorded,
    SnapshotQuery,
)
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.contracts.notebook.vocab import InboxKind
from hyperloom_multibranch.contracts.toolbox import Toolbox
from hyperloom_multibranch.rulegate.decision import Action, decide
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.readings import UnreadableResult
from hyperloom_multibranch.rulegate.rules import Rule, repeats, retest_request
from hyperloom_multibranch.specialist.fix_check import FixCheck, FixChecking, check_fix, declares_fix

#: Opens every branch log entry the runtime writes, so the specialist tells them from its own.
NOTICE = "rule gate:"
#: The endings of the notices after which the judge never judges the candidate: a retest that ended without a
#: measurement, a result it cannot read, a result that did not re-verify. Every other notice precedes a judgement.
_NOTHING_JUDGED = "nothing was judged"
_CANNOT_BE_JUDGED = "cannot be judged:"
_NOT_COMMITTED = "was not committed:"


class JudgeNotebook(Inbox, SnapshotQuery, BranchLogStore, CommitQuery, Protocol):
    """What judging a branch reads, and the branch log it writes as the branch's runtime."""


@dataclass(frozen=True)
class Judging:
    """What every branch's judge settles candidates with: the workload's rule, the gate that lands what clears it, the
    toolbox a retest goes to, and how a declared correctness fix is checked (``None``: a declared fix is judged on its
    performance alone, and its specialist told so)."""

    gate: RuleGate
    toolbox: Toolbox
    rule: Rule
    fixes: FixChecking | None = None


class BranchJudge:
    """Judges one branch's measured candidates; one task per branch, so its steps never race each other."""

    def __init__(self, branch: str, notebook: JudgeNotebook, judging: Judging) -> None:
        self._branch = branch
        self._notebook = notebook
        self._gate, self._toolbox, self._rule = judging.gate, judging.toolbox, judging.rule
        self._fixes = judging.fixes

    async def run(self, poll_seconds: float, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        """Judge what waits, then again whenever the inbox brings a result, until cancelled; ``sleep`` waits between
        two polls."""
        cursor = (await self._notebook.poll(self._branch)).cursor
        await self.reconcile()
        while True:
            await sleep(poll_seconds)
            page = await self._notebook.poll(self._branch, cursor)
            cursor = page.cursor
            if any(item.kind is InboxKind.RESULT for item in page.items):
                await self.reconcile()

    async def reconcile(self) -> None:
        """Take the runtime's next step for every candidate of the branch still waiting for one."""
        snapshot = await self._notebook.snapshot()
        records = [record for record in snapshot.records if record.request.branch_id == self._branch]
        logged = {record.entry for record in snapshot.branch_logs.get(self._branch, ())}
        retests: dict[str, list[TestRecord]] = {}
        for record in records:
            if record.request.retest_of is not None:
                retests.setdefault(record.request.retest_of, []).append(record)
        for record in records:
            if _awaits_judgement(record):
                await self._judge(record, retests.get(record.request_id, []), logged)

    async def _judge(self, first: TestRecord, retests: Sequence[TestRecord], logged: set[BranchLogEntry]) -> None:
        retests = [retest for retest in retests if repeats(retest, first, self._rule)]
        try:
            parent = (await self._notebook.recorded_value(first.request.parent_commit)).values
            judgement = await self._rule.first(first, parent)
            if judgement.verdict is Verdict.NEEDS_CONFIRM and retests:
                measured = [retest for retest in retests if _succeeded(retest)]
                if not measured:
                    if any(retest.in_flight for retest in retests):
                        return
                    last = max(retests, key=lambda retest: retest.requested_at)
                    if not _failed_as_candidate(last):
                        content = f"the retest of {first.request_id} {_ended(last)}; {_NOTHING_JUDGED}"
                        await self._notice(logged, content, first.request_id, last.request_id)
                        return
                    content = f"the retest of {first.request_id} {_ended(last)}; recorded as regressed"
                    await self._notice(logged, content, first.request_id, last.request_id)
                    await self._gate.settle(first.request, _failed_retest(judgement, last))
                    return
                latest = max(measured, key=lambda retest: retest.requested_at)
                judgement = await self._rule.retest(judgement, latest, parent)
        except (UnreadableResult, NotRecorded) as error:
            await self._notice(logged, f"{first.request_id} {_CANNOT_BE_JUDGED} {error}", first.request_id)
            return
        await self._settle(first, judgement, logged)

    async def _settle(self, first: TestRecord, judgement: Judgement, logged: set[BranchLogEntry]) -> None:
        fix = None
        if declares_fix(first.request, judgement):
            checked = await self._check_fix(first, judgement, logged)
            judgement, fix = checked.judgement, checked.proof
        if decide(judgement) is Action.DIAGNOSE:
            # Before the gate holds the candidate: once held, it is never judged again, so a notice lost after that
            # would never be written.
            held_back_by = ", ".join(judgement.refused_by) or "the accuracy gate"
            content = f"{first.request_id} held its gain on the retest and is held back by {held_back_by}"
            await self._notice(
                logged, f"{content}; diagnose it, then request_repair or decline_repair", *judgement.request_ids
            )
        try:
            action = await self._gate.settle(first.request, judgement, fix=fix)
            if action is Action.REQUEST_FORK:
                reason = f"correctness fix of {first.request_id}"
                admission = await self._gate.fork(first.request, judgement, reason, fix=fix)
                await self._notice(logged, _forked(first, admission), *judgement.request_ids)
        except ResultUnverified as error:
            unverified = "a result behind it did not re-verify against its signed bundle"
            content = f"{first.request_id} {_NOT_COMMITTED} {unverified} ({error.request_id}: {error.reason})"
            await self._notice(logged, content, *judgement.request_ids)
            return
        if action is Action.COMMIT:
            landed = f"{first.request_id} landed on your branch at {judgement.gain_pct:+.2f}% over {first.request.parent_commit}"
            if fix is not None:
                landed = f"{landed} with its repro case, which now guards every later commit"
            await self._notice(logged, f"{landed}; your next candidates go on the new head", *judgement.request_ids)
        elif action is Action.RETEST:
            await self._toolbox.submit_test(retest_request(first, self._rule))
        elif action is Action.RECORD_ONLY and decide(judgement) is Action.COMMIT:
            moved = f"{first.request_id} improved on {first.request.parent_commit}, but the branch has moved on"
            await self._notice(logged, f"{moved}; submit it again on the head as a new test", *judgement.request_ids)

    async def _check_fix(self, first: TestRecord, judgement: Judgement, logged: set[BranchLogEntry]) -> FixCheck:
        if self._fixes is None:
            checked = FixCheck(judgement, None, "this session does not check correctness fixes")
        else:
            checked = await check_fix(self._fixes, self._notebook, first.request, judgement)
        if checked.alone:
            content = f"{first.request_id} is judged on its performance alone: {checked.alone}"
            await self._notice(logged, content, first.request_id)
        return checked

    async def _notice(self, logged: set[BranchLogEntry], content: str, *request_ids: str) -> None:
        entry = BranchLogEntry(self._branch, BranchLogType.RESULT, f"{NOTICE} {content}", request_ids)
        if entry not in logged:
            await self._notebook.append_branch_log(entry)
            logged.add(entry)


def left_unjudged(notice: str) -> bool:
    """Whether the judge's ``notice`` closes its candidate without a judgement; a notice such as a fix judged on its
    performance alone or a diagnosis precedes the gate's judgement and commit."""
    return _NOTHING_JUDGED in notice or _CANNOT_BE_JUDGED in notice or _NOT_COMMITTED in notice


def _awaits_judgement(record: TestRecord) -> bool:
    """A candidate's first test, measured and not judged yet; a failed measurement takes no judgement."""
    return record.request.purpose is Purpose.EXPLORE and record.judgement is None and _succeeded(record)


def _succeeded(record: TestRecord) -> bool:
    return record.result is not None and record.result.validity is Validity.SUCCEEDED


def _failed_as_candidate(record: TestRecord) -> bool:
    return record.result is not None and record.result.validity is Validity.CANDIDATE_FAILED


def _failed_retest(first: Judgement, retest: TestRecord) -> Judgement:
    """The candidate's judgement once its retest failed as a candidate: the change itself failed, so its gain stands
    refused and no fix is checked on it, and the record says so where an unjudged candidate would leave no trace."""
    return replace(
        first,
        verdict=Verdict.REGRESSED,
        request_ids=(*first.request_ids, retest.request_id),
        refused_by=(*first.refused_by, Validity.CANDIDATE_FAILED.value),
        fixes_correctness=False,
        foundation_fix=False,
    )


def _forked(first: TestRecord, admission: ForkAdmission) -> str:
    fix = f"{first.request_id} fixes a defect of {first.request.parent_commit}"
    if admission.status is AdmissionStatus.ACCEPTED:
        return f"{fix}: it goes to a new branch, {admission.branch}, cut at that commit; you go on on yours"
    if admission.status is AdmissionStatus.PENDING:
        return f"{fix}: it waits for a branch slot as {admission.branch}; if the fork is dropped it is recorded only"
    return f"{fix}, but its fork was refused ({admission.reason}); it is recorded only"


def _ended(record: TestRecord) -> str:
    if record.result is None:
        return f"was withdrawn ({record.withdrawal_reason})"
    return f"ended {record.result.validity.value} ({record.result.failure_reason or 'no reason given'})"


__all__ = ["NOTICE", "BranchJudge", "JudgeNotebook", "Judging", "left_unjudged"]
