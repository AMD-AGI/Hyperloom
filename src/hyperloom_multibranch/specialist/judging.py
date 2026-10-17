# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Judging a branch's measured candidates: the half of the specialist runtime that is not a model.

The specialist proposes and the toolbox measures; this code compares each measurement through the rule gate, retests a
gain on another node, and tells the specialist in its branch log what landed and what the gate could not land. Every step is read back
from the notebook, so a restarted runtime takes each step once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from hyperloom.contracts.measurement import Purpose, Validity
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
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

#: Opens every branch log entry the runtime writes, so the specialist tells them from its own.
NOTICE = "rule gate:"


class JudgeNotebook(Inbox, SnapshotQuery, BranchLogStore, CommitQuery, Protocol):
    """What judging a branch reads, and the branch log it writes as the branch's runtime."""


@dataclass(frozen=True)
class Judging:
    """What every branch's judge settles candidates with: the workload's rule, the gate that lands what clears it, and
    the toolbox a retest goes to."""

    gate: RuleGate
    toolbox: Toolbox
    rule: Rule


class BranchJudge:
    """Judges one branch's measured candidates; one task per branch, so its steps never race each other."""

    def __init__(self, branch: str, notebook: JudgeNotebook, judging: Judging) -> None:
        self._branch = branch
        self._notebook = notebook
        self._gate, self._toolbox, self._rule = judging.gate, judging.toolbox, judging.rule

    async def run(self, poll_seconds: float) -> None:
        """Judge what waits, then again whenever the inbox brings a result, until cancelled."""
        cursor = (await self._notebook.poll(self._branch)).cursor
        await self.reconcile()
        while True:
            await asyncio.sleep(poll_seconds)
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
                    if not any(retest.in_flight for retest in retests):
                        last = max(retests, key=lambda retest: retest.requested_at)
                        content = f"the retest of {first.request_id} {_ended(last)}; nothing was judged"
                        await self._notice(logged, content, first.request_id, last.request_id)
                    return
                latest = max(measured, key=lambda retest: retest.requested_at)
                judgement = await self._rule.retest(judgement, latest, parent)
        except (UnreadableResult, NotRecorded) as error:
            await self._notice(logged, f"{first.request_id} cannot be judged: {error}", first.request_id)
            return
        await self._settle(first, judgement, logged)

    async def _settle(self, first: TestRecord, judgement: Judgement, logged: set[BranchLogEntry]) -> None:
        if decide(judgement) is Action.DIAGNOSE:
            # Before the gate holds the candidate: once held, it is never judged again, so a notice lost after that
            # would never be written.
            held_back_by = ", ".join(judgement.refused_by) or "the accuracy gate"
            content = f"{first.request_id} cleared the bar twice and is held back by {held_back_by}"
            await self._notice(
                logged, f"{content}; diagnose it, then request_repair or decline_repair", *judgement.request_ids
            )
        try:
            action = await self._gate.settle(first.request, judgement)
        except ResultUnverified as error:
            unverified = "a result behind it did not re-verify against its signed bundle"
            content = f"{first.request_id} was not committed: {unverified} ({error.request_id}: {error.reason})"
            await self._notice(logged, content, *judgement.request_ids)
            return
        if action is Action.COMMIT:
            landed = f"{first.request_id} landed on your branch at {judgement.gain_pct:+.2f}% over {first.request.parent_commit}"
            await self._notice(logged, f"{landed}; your next candidates go on the new head", *judgement.request_ids)
        elif action is Action.RETEST:
            await self._toolbox.submit_test(retest_request(first, self._rule))
        elif action is Action.REQUEST_FORK:
            raise RuntimeError(f"{first.request_id} calls for a fork, which the runtime does not request yet")
        elif action is Action.RECORD_ONLY and decide(judgement) is Action.COMMIT:
            moved = f"{first.request_id} improved on {first.request.parent_commit}, but the branch has moved on"
            await self._notice(logged, f"{moved}; submit it again on the head as a new test", *judgement.request_ids)

    async def _notice(self, logged: set[BranchLogEntry], content: str, *request_ids: str) -> None:
        entry = BranchLogEntry(self._branch, BranchLogType.RESULT, f"{NOTICE} {content}", request_ids)
        if entry not in logged:
            await self._notebook.append_branch_log(entry)
            logged.add(entry)


def _awaits_judgement(record: TestRecord) -> bool:
    """A candidate's first test, measured and not judged yet; a failed measurement takes no judgement."""
    return record.request.purpose is Purpose.EXPLORE and record.judgement is None and _succeeded(record)


def _succeeded(record: TestRecord) -> bool:
    return record.result is not None and record.result.validity is Validity.SUCCEEDED


def _ended(record: TestRecord) -> str:
    if record.result is None:
        return f"was withdrawn ({record.withdrawal_reason})"
    return f"ended {record.result.validity.value} ({record.result.failure_reason or 'no reason given'})"


__all__ = ["NOTICE", "BranchJudge", "JudgeNotebook", "Judging"]
