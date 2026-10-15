# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Judging a branch's measured candidates: the half of the specialist runtime that is not a model.

The specialist proposes and the toolbox measures; this code compares each measurement through the rule gate, retests a
gain on another node, and tells the specialist in its branch log what the gate could not land. Every step is read back
from the notebook, so a restarted runtime takes each step once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest, TestResult, Validity
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
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
from hyperloom_multibranch.rulegate import agentx, inferencex
from hyperloom_multibranch.rulegate.baseline import read_inferencex_baseline
from hyperloom_multibranch.rulegate.decision import Action, decide
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.readings import UnreadableResult, agentx_reading, inferencex_reading

#: Opens every branch log entry the runtime writes, so the specialist tells them from its own.
NOTICE = "rule gate:"


class Rule(Protocol):
    """How the session's workload is judged; ``accuracy_round`` says whether its tests run the eval it reads."""

    accuracy_round: bool

    async def first(self, record: TestRecord, parent: Mapping[str, float]) -> Judgement: ...

    async def retest(self, first: Judgement, record: TestRecord, parent: Mapping[str, float]) -> Judgement: ...


@dataclass(frozen=True)
class InferenceXRule:
    """Throughput at the graded concurrency against the PRELUDE bar, the eval score against the baseline's."""

    notebook: CommitQuery
    concurrency: int
    accuracy_round = True

    async def first(self, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        reading = inferencex_reading(_result(record), self.concurrency)
        return inferencex.compare(reading, await read_inferencex_baseline(self.notebook), parent, record.request_id)

    async def retest(self, first: Judgement, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        reading = inferencex_reading(_result(record), self.concurrency)
        baseline = await read_inferencex_baseline(self.notebook)
        return inferencex.confirm(first, reading, baseline, parent, record.request_id)


class AgentXRule:
    """P50 interactivity with its guards; the error rate the run reports is its accuracy."""

    accuracy_round = False

    async def first(self, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        return agentx.compare(agentx_reading(_result(record)), parent, record.request_id)

    async def retest(self, first: Judgement, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        return agentx.confirm(first, agentx_reading(_result(record)), parent, record.request_id)


class JudgeNotebook(Inbox, SnapshotQuery, BranchLogStore, CommitQuery, Protocol):
    """What judging a branch reads, and the branch log it writes as the branch's runtime."""


class BranchJudge:
    """Judges one branch's measured candidates; one task per branch, so its steps never race each other."""

    def __init__(self, branch: str, notebook: JudgeNotebook, gate: RuleGate, toolbox: Toolbox, rule: Rule) -> None:
        self._branch = branch
        self._notebook = notebook
        self._gate = gate
        self._toolbox = toolbox
        self._rule = rule

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
        retests = [retest for retest in retests if self._repeats(retest, first)]
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

    def _repeats(self, retest: TestRecord, first: TestRecord) -> bool:
        """Only a retest of the same change on the same parent, with the eval round the rule reads, settles a gain."""
        asked, wanted = retest.request, first.request
        same = (asked.parent_commit, asked.delta) == (wanted.parent_commit, wanted.delta)
        return same and asked.accuracy == self._rule.accuracy_round

    async def _settle(self, first: TestRecord, judgement: Judgement, logged: set[BranchLogEntry]) -> None:
        action = await self._gate.settle(first.request, judgement)
        if action is Action.RETEST:
            await self._toolbox.submit_test(self._retest_of(first))
        elif action is Action.DIAGNOSE:
            held_back_by = ", ".join(judgement.refused_by) or "the accuracy gate"
            content = f"{first.request_id} cleared the bar twice and is held back by {held_back_by}; diagnose it"
            await self._notice(logged, content, *judgement.request_ids)
        elif action is Action.REQUEST_FORK:
            raise RuntimeError(f"{first.request_id} calls for a fork, which the runtime does not request yet")
        elif action is Action.RECORD_ONLY and decide(judgement) is Action.COMMIT:
            moved = f"{first.request_id} improved on {first.request.parent_commit}, but the branch has moved on"
            await self._notice(logged, f"{moved}; submit it again on the head as a new test", *judgement.request_ids)

    def _retest_of(self, first: TestRecord) -> TestRequest:
        sandbox = _result(first).sandbox
        node = sandbox.node_name if sandbox is not None else None
        return TestRequest(
            kind=TestKind.TEST,
            purpose=Purpose.CONFIRM,
            branch_id=self._branch,
            parent_commit=first.request.parent_commit,
            delta=first.request.delta,
            retest_of=first.request_id,
            avoid_nodes=(node,) if node else (),
            accuracy=self._rule.accuracy_round,
            concurrency=first.request.concurrency,
        )

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


def _result(record: TestRecord) -> TestResult:
    if record.result is None:
        raise UnreadableResult(f"{record.request_id} has no result")
    return record.result


__all__ = ["NOTICE", "AgentXRule", "BranchJudge", "InferenceXRule", "JudgeNotebook", "Rule"]
