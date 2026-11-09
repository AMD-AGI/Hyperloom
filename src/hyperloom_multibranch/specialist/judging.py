# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Judging a branch's measured candidates: the half of the specialist runtime that is not a model.

The specialist proposes and the toolbox measures; this code compares each measurement through the rule gate, retests a
gain on another node, checks a declared correctness fix (``fix_check``) and lands it or asks for its fork, and tells the
specialist in its branch log what landed and what the gate could not land. A retest that fails as a candidate settles
the candidate as regressed; one whose measurement failed leaves it unjudged, and a notice that leaves a candidate
unjudged closes it for good. Every step is read back from the blackboard, so a restarted runtime takes each step once
and announces what landed from the recorded judgement; a fix check interrupted by a restart runs its repro tests again.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from hyperloom.contracts.measurement import Purpose, TestRequest, Validity
from hyperloom.measure.compile import BlobStore
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.fork import AdmissionStatus
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.blackboard.interfaces import (
    BranchLogStore,
    CommitQuery,
    Inbox,
    NotRecorded,
    SnapshotQuery,
)
from hyperloom_multibranch.contracts.blackboard.records import RetractionRef, TestRecord
from hyperloom_multibranch.contracts.blackboard.vocab import BranchDriver, Conclusion, InboxKind
from hyperloom_multibranch.contracts.toolbox import RequestRejected, Toolbox
from hyperloom_multibranch.rulegate.decision import Action, decide
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.readings import UnreadableResult
from hyperloom_multibranch.rulegate.rules import (
    RETRACTION_ID,
    Rule,
    lands_as_retraction,
    repeats,
    retest_request,
    retracts,
    stopped_at_checkpoint,
)
from hyperloom_multibranch.specialist.fix_check import FixCheck, FixChecking, check_fix, declares_fix

#: Opens every branch log entry the runtime writes, so the specialist tells them from its own.
NOTICE = "rule gate:"
#: The endings of the notices after which the judge never judges the candidate: a retest that ended without a
#: measurement, a result it cannot read, a result that did not re-verify. Every other notice precedes a judgement.
_NOTHING_JUDGED = "nothing was judged"
_CANNOT_BE_JUDGED = "cannot be judged:"
_NOT_COMMITTED = "was not committed:"


class JudgeBlackboard(Inbox, SnapshotQuery, BranchLogStore, CommitQuery, Protocol):
    """What judging a branch reads, and the branch log it writes as the branch's runtime."""


@dataclass(frozen=True)
class Judging:
    """What every branch's judge settles candidates with: the workload's rule, the gate that lands what clears it, the
    toolbox a retest goes to, the blob store a retraction's attachment is read from, and how a declared correctness fix is checked (``None``: a declared fix is judged on its
    performance alone, and its specialist told so)."""

    gate: RuleGate
    toolbox: Toolbox
    rule: Rule
    blobs: BlobStore
    fixes: FixChecking | None = None


class BranchJudge:
    """Judges one branch's measured candidates; one task per branch, so its steps never race each other."""

    def __init__(self, branch: str, blackboard: JudgeBlackboard, judging: Judging) -> None:
        self._branch = branch
        self._blackboard = blackboard
        self._gate, self._toolbox, self._rule, self._blobs = judging.gate, judging.toolbox, judging.rule, judging.blobs
        self._fixes = judging.fixes
        self._finishing = False
        self._finished = False
        self._submitted = False
        self._autonomous = False

    async def run(self, poll_seconds: float, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        """Judge what waits, then again whenever the inbox brings a result, until cancelled or, once :meth:`finish`
        was called, until no test of the branch is left in flight; ``sleep`` waits between two polls."""
        cursor = (await self._blackboard.poll(self._branch)).cursor
        await self.reconcile()
        while not self._finished:
            await sleep(poll_seconds)
            page = await self._blackboard.poll(self._branch, cursor)
            cursor = page.cursor
            if self._finishing or any(item.kind is InboxKind.RESULT for item in page.items):
                await self.reconcile()

    def finish(self) -> None:
        """The branch's specialist is gone: judge only the tests it left in flight, and the retests they bring."""
        self._finishing = True

    async def reconcile(self) -> None:
        """Take the runtime's next step for every candidate of the branch still waiting for one."""
        # Only a snapshot read after finish() can show that nothing of the branch is left to arrive.
        finishing, self._submitted = self._finishing, False
        snapshot = await self._blackboard.snapshot()
        self._autonomous = any(
            node.branch == self._branch and node.driver is BranchDriver.AUTONOMOUS for node in snapshot.lineage.branches
        )
        records = [record for record in snapshot.records if record.request.branch_id == self._branch]
        logged = {record.entry for record in snapshot.branch_logs.get(self._branch, ())}
        closed = {
            request_id
            for entry in logged
            if entry.content.startswith(NOTICE) and left_unjudged(entry.content)
            for request_id in entry.request_ids
        }
        guarded = {case.fix_commit for case in snapshot.regressions}
        retests: dict[str, list[TestRecord]] = {}
        for record in records:
            if record.request.retest_of is not None:
                retests.setdefault(record.request.retest_of, []).append(record)
        for record in records:
            if _awaits_judgement(record) and record.request_id not in closed:
                await self._judge(record, retests.get(record.request_id, []), logged)
            elif record.judgement is not None:
                # A stop or a failure between the gate's step and its notice must not leave a landing untold.
                announced = _announced(record, record.judgement, record.commit_id in guarded)
                if announced:
                    await self._notice(logged, announced, *record.judgement.request_ids)
        if finishing and not self._submitted and not any(record.in_flight for record in records):
            self._finished = True

    async def _judge(self, first: TestRecord, retests: Sequence[TestRecord], logged: set[BranchLogEntry]) -> None:
        retests = [retest for retest in retests if repeats(retest, first, self._rule)]
        try:
            parent = (await self._blackboard.recorded_value(first.request.parent_commit)).values
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
        if self._autonomous and retracts(first, self._rule) and lands_as_retraction(first, judgement, self._rule):
            await self._retract(first, judgement, logged)
            return
        fix = None
        if declares_fix(first.request, judgement):
            checked = await self._check_fix(first, judgement, logged)
            judgement, fix = checked.judgement, checked.proof
        # Before the gate's step: a held or refused candidate is never judged again, and its record cannot tell a
        # refused fork from a voided one, so a notice lost after the step would never be written.
        if decide(judgement) is Action.DIAGNOSE:
            held_back_by = ", ".join(judgement.refused_by) or "the accuracy gate"
            content = f"{first.request_id} held its gain on the retest and is held back by {held_back_by}"
            await self._notice(
                logged, f"{content}; diagnose it, then request_repair or decline_repair", *judgement.request_ids
            )
        elif decide(judgement) is Action.REQUEST_FORK:
            content = f"{_fixes(first.request)}; its fork is asked for, and if it is refused it is recorded only"
            await self._notice(logged, content, *judgement.request_ids)
        try:
            action = await self._gate.settle(first.request, judgement, fix=fix)
            if action is Action.REQUEST_FORK:
                reason = f"correctness fix of {first.request_id}"
                admission = await self._gate.fork(first.request, judgement, reason, fix=fix)
                if admission.status is AdmissionStatus.REJECTED:
                    forked = f"{_fixes(first.request)}: its fork was refused ({admission.reason}); it is recorded only"
                else:
                    forked = _forked_to(first.request, pending=admission.status is AdmissionStatus.PENDING)
                await self._notice(logged, forked, *judgement.request_ids)
        except ResultUnverified as error:
            await self._unverified(logged, first, judgement, error)
            return
        if action is Action.COMMIT:
            await self._notice(logged, _landed(first.request, judgement, fix is not None), *judgement.request_ids)
        elif action is Action.RETEST:
            try:
                await self._toolbox.submit_test(retest_request(first, self._rule))
            except RequestRejected as error:
                content = f"the retest of {first.request_id} was refused ({error}); {_NOTHING_JUDGED}"
                await self._notice(logged, content, first.request_id)
                return
            self._submitted = True
        elif action is Action.RECORD_ONLY and decide(judgement) is Action.COMMIT:
            await self._notice(logged, _moved_on(first.request), *judgement.request_ids)

    async def _retract(self, first: TestRecord, judgement: Judgement, logged: set[BranchLogEntry]) -> None:
        try:
            landed = await self._gate.retract(first.request, judgement, self._retraction_ref(first))
        except ResultUnverified as error:
            await self._unverified(logged, first, judgement, error)
            return
        if landed:
            content = _retracted(first.request, judgement)
        else:
            content = f"{first.request_id} retracts a change, but the branch has moved on; it is recorded only"
        await self._notice(logged, content, *judgement.request_ids)

    def _retraction_ref(self, first: TestRecord) -> RetractionRef:
        attachment = next(item for item in first.request.attachments if item.id == RETRACTION_ID)
        return RetractionRef(**json.loads(self._blobs.get(attachment.sha256)))

    async def _unverified(
        self, logged: set[BranchLogEntry], first: TestRecord, judgement: Judgement, error: ResultUnverified
    ) -> None:
        unverified = "a result behind it did not re-verify against its signed bundle"
        content = f"{first.request_id} {_NOT_COMMITTED} {unverified} ({error.request_id}: {error.reason})"
        await self._notice(logged, content, *judgement.request_ids)

    async def _check_fix(self, first: TestRecord, judgement: Judgement, logged: set[BranchLogEntry]) -> FixCheck:
        if self._fixes is None:
            checked = FixCheck(judgement, None, "this session does not check correctness fixes")
        else:
            checked = await check_fix(self._fixes, self._blackboard, first.request, judgement)
        if checked.alone:
            content = f"{first.request_id} is judged on its performance alone: {checked.alone}"
            await self._notice(logged, content, first.request_id)
        return checked

    async def _notice(self, logged: set[BranchLogEntry], content: str, *request_ids: str) -> None:
        entry = BranchLogEntry(self._branch, BranchLogType.RESULT, f"{NOTICE} {content}", request_ids)
        if entry not in logged:
            await self._blackboard.append_branch_log(entry)
            logged.add(entry)


def left_unjudged(notice: str) -> bool:
    """Whether the judge's ``notice`` closes its candidate without a judgement; a notice such as a fix judged on its
    performance alone or a diagnosis precedes the gate's judgement and commit."""
    return _NOTHING_JUDGED in notice or _CANNOT_BE_JUDGED in notice or _NOT_COMMITTED in notice


def _awaits_judgement(record: TestRecord) -> bool:
    """A candidate's first test, measured, or stopped by the AgentX checkpoint, and not judged yet; any other failed
    measurement takes no judgement."""
    measured = _succeeded(record) or stopped_at_checkpoint(record)
    return record.request.purpose is Purpose.EXPLORE and record.judgement is None and measured


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


def _announced(record: TestRecord, judgement: Judgement, guarded: bool) -> str:
    """The notice of a commit, a fork or a gain the moved-on branch kept from landing that the gate recorded for a
    judged candidate, "" for any other conclusion; the same text :meth:`BranchJudge._settle` writes after the step,
    so the one written there is not written twice."""
    if record.landed_retraction:
        return _retracted(record.request, judgement)
    if record.conclusion in (Conclusion.COMMITTED, Conclusion.COMMITTED_WITH_REPRO):
        return _landed(record.request, judgement, guarded)
    if record.conclusion is Conclusion.RECORDED_ONLY and decide(judgement) is Action.COMMIT:
        return _moved_on(record.request)
    if record.conclusion in (Conclusion.SPLIT_REQUESTED, Conclusion.SPLIT_COMMITTED):
        return _forked_to(record.request, pending=False)
    if record.conclusion is Conclusion.SPLIT_PENDING:
        return _forked_to(record.request, pending=True)
    return ""


def _landed(request: TestRequest, judgement: Judgement, guarded: bool) -> str:
    landed = f"{request.request_id} landed on your branch at {judgement.gain_pct:+.2f}% over {request.parent_commit}"
    if guarded:
        landed = f"{landed} with its repro case, which now guards every later commit"
    return f"{landed}; your next candidates go on the new head"


def _retracted(request: TestRequest, judgement: Judgement) -> str:
    retraction = f"{request.request_id} landed on your branch as a retraction at {judgement.gain_pct:+.2f}%"
    return (
        f"{retraction} over {request.parent_commit}, held to the rule's errors, guards, accuracy and, where its tests "
        "run them, consistency checks, not its speed"
    )


def _moved_on(request: TestRequest) -> str:
    moved = f"{request.request_id} improved on {request.parent_commit}, but the branch has moved on"
    return f"{moved}; submit it again on the head as a new test"


def _fixes(request: TestRequest) -> str:
    return f"{request.request_id} fixes a defect of {request.parent_commit}"


def _forked_to(request: TestRequest, *, pending: bool) -> str:
    fix = _fixes(request)
    if pending:
        return f"{fix}: it waits for a branch slot; if the fork is dropped it is recorded only"
    return f"{fix}: it goes to a new branch cut at that commit; you go on on yours"


def _ended(record: TestRecord) -> str:
    if record.result is None:
        return f"was withdrawn ({record.withdrawal_reason})"
    return f"ended {record.result.validity.value} ({record.result.failure_reason or 'no reason given'})"


__all__ = ["NOTICE", "BranchJudge", "JudgeBlackboard", "Judging", "left_unjudged"]
