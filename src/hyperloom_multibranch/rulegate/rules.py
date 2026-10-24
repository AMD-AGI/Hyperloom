# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a session's workload is judged, read from the notebook's records.

The specialist runtime judges a branch's candidates with it, and the scheduler's gate the PRELUDE baseline, a merge and
a final candidate, so every test is held to the same bar, the same accuracy gate and the same confirming retest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from hyperloom.common.perf_metric import GRADED_ERROR_RATE
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest, TestResult
from hyperloom_multibranch.contracts.judgement import Judgement
from hyperloom_multibranch.contracts.notebook.interfaces import CommitQuery, GateRecords
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord, TestRecord
from hyperloom_multibranch.rulegate import agentx, inferencex
from hyperloom_multibranch.rulegate.accuracy import agentx_accuracy_ok, inferencex_accuracy_ok
from hyperloom_multibranch.rulegate.baseline import (
    read_inferencex_baseline,
    write_agentx_baseline,
    write_inferencex_baseline,
)
from hyperloom_multibranch.rulegate.readings import (
    UnreadableResult,
    agentx_reading,
    dropped_requests,
    eval_score,
    inferencex_reading,
)


class Rule(Protocol):
    """How the session's workload is judged; ``accuracy_round`` says whether its tests run the eval it reads."""

    accuracy_round: bool

    async def baseline(self, store: GateRecords, root_commit: str, records: Sequence[TestRecord]) -> BaselineRecord: ...

    async def first(self, record: TestRecord, parent: Mapping[str, float]) -> Judgement: ...

    async def retest(self, first: Judgement, record: TestRecord, parent: Mapping[str, float]) -> Judgement: ...

    async def accuracy_passes(self, record: TestRecord) -> bool: ...


@dataclass(frozen=True)
class InferenceXRule(Rule):
    """Throughput at the graded concurrency against the PRELUDE bar, the eval score against the baseline's, and no
    request the graded rounds sent left uncompleted."""

    notebook: CommitQuery
    concurrency: int
    accuracy_round = True

    async def baseline(self, store: GateRecords, root_commit: str, records: Sequence[TestRecord]) -> BaselineRecord:
        return await write_inferencex_baseline(store, root_commit, records, self.concurrency)

    async def first(self, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        reading = self._candidate(record)
        return inferencex.compare(reading, await read_inferencex_baseline(self.notebook), parent, record.request_id)

    async def retest(self, first: Judgement, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        reading = self._candidate(record)
        baseline = await read_inferencex_baseline(self.notebook)
        return inferencex.confirm(first, reading, baseline, parent, record.request_id)

    async def accuracy_passes(self, record: TestRecord) -> bool:
        baseline = await read_inferencex_baseline(self.notebook)
        return inferencex_accuracy_ok(baseline.accuracy, eval_score(_result(record)))

    def _candidate(self, record: TestRecord) -> inferencex.InferenceXReading:
        result = _result(record)
        dropped = dropped_requests(result, self.concurrency)
        return replace(inferencex_reading(result, self.concurrency), dropped=dropped)


class AgentXRule(Rule):
    """P50 interactivity with its guards; the error rate the run reports is its accuracy."""

    accuracy_round = False

    async def baseline(self, store: GateRecords, root_commit: str, records: Sequence[TestRecord]) -> BaselineRecord:
        return await write_agentx_baseline(store, root_commit, records)

    async def first(self, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        return agentx.compare(agentx_reading(_result(record)), parent, record.request_id)

    async def retest(self, first: Judgement, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        return agentx.confirm(first, agentx_reading(_result(record)), parent, record.request_id)

    async def accuracy_passes(self, record: TestRecord) -> bool:
        return agentx_accuracy_ok(agentx_reading(_result(record)).get(GRADED_ERROR_RATE))


def repeats(retest: TestRecord, first: TestRecord, rule: Rule) -> bool:
    """Only a retest of the same change and files on the same parent, with the rule's eval round, settles a gain."""
    return _candidate(retest.request) == _candidate(first.request) and retest.request.accuracy == rule.accuracy_round


def retest_request(first: TestRecord, rule: Rule) -> TestRequest:
    """The confirming retest of ``first``: the same change and files on the same parent, off the node that measured it."""
    sandbox = _result(first).sandbox
    node = sandbox.node_name if sandbox is not None else None
    return TestRequest(
        kind=TestKind.TEST,
        purpose=Purpose.CONFIRM,
        branch_id=first.request.branch_id,
        parent_commit=first.request.parent_commit,
        delta=first.request.delta,
        retest_of=first.request_id,
        avoid_nodes=(node,) if node else (),
        accuracy=rule.accuracy_round,
        concurrency=first.request.concurrency,
        attachments=first.request.attachments,
    )


def _candidate(request: TestRequest) -> tuple[object, ...]:
    return request.parent_commit, request.delta, request.attachments


def _result(record: TestRecord) -> TestResult:
    if record.result is None:
        raise UnreadableResult(f"{record.request_id} has no result")
    return record.result


__all__ = ["AgentXRule", "InferenceXRule", "Rule", "repeats", "retest_request"]
