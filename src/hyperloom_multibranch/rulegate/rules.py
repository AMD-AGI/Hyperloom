# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a session's workload is judged, read from the blackboard's records.

The specialist runtime judges a branch's candidates with it, and the scheduler's gate the PRELUDE baseline, a merge and
a final candidate, so every test is held to the same bar, the same accuracy gate and the same confirming retest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from hyperloom.common.perf_metric import GRADED_ERROR_RATE
from hyperloom.contracts.long_context import refuses
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest, TestResult
from hyperloom_multibranch.contracts.judgement import Judgement
from hyperloom_multibranch.contracts.blackboard.interfaces import CommitQuery, GateRecords
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, TestRecord
from hyperloom_multibranch.rulegate import agentx, inferencex
from hyperloom_multibranch.rulegate.accuracy import agentx_accuracy_ok, inferencex_accuracy_ok
from hyperloom_multibranch.rulegate.baseline import (
    read_inferencex_baseline,
    write_agentx_baseline,
    write_agentx_prelude_baseline,
    write_inferencex_baseline,
)
from hyperloom_multibranch.rulegate.readings import (
    UnreadableResult,
    agentx_reading,
    dropped_requests,
    eval_score,
    inferencex_reading,
    unscored_reading,
)


class Rule(Protocol):
    """How the session's workload is judged; ``accuracy_round`` says whether its tests run the eval it reads.

    PRELUDE submits ``baseline_tests`` root tests, and the baseline needs ``min_baselines`` of them measured. A rule
    that asks for none takes the PRELUDE child's own run of the root as the baseline (:meth:`prelude_baseline`), and
    ``min_baselines`` root tests where PRELUDE starts no child.
    """

    accuracy_round: bool
    baseline_tests: int
    min_baselines: int

    async def baseline(self, store: GateRecords, root_commit: str, records: Sequence[TestRecord]) -> BaselineRecord: ...

    async def prelude_baseline(
        self, store: GateRecords, root_commit: str, result: TestResult, *, source: str, wall_seconds: float
    ) -> BaselineRecord: ...

    async def first(self, record: TestRecord, parent: Mapping[str, float]) -> Judgement: ...

    async def retest(self, first: Judgement, record: TestRecord, parent: Mapping[str, float]) -> Judgement: ...

    async def accuracy_passes(self, record: TestRecord) -> bool: ...


@dataclass(frozen=True)
class InferenceXRule(Rule):
    """Throughput at the graded concurrency against the PRELUDE bar, the eval score against the baseline's, no request
    the graded rounds sent left uncompleted, and long-context answers no worse than the root's beyond the check's own noise.

    Only an explore whose accepted request records that the eval was skipped is judged without a score; any other test
    without one is unreadable.
    """

    blackboard: CommitQuery
    concurrency: int
    accuracy_round = True
    baseline_tests = inferencex.BASELINE_TESTS
    min_baselines = inferencex.MIN_BASELINES

    async def baseline(self, store: GateRecords, root_commit: str, records: Sequence[TestRecord]) -> BaselineRecord:
        return await write_inferencex_baseline(store, root_commit, records, self.concurrency)

    async def prelude_baseline(
        self, store: GateRecords, root_commit: str, result: TestResult, *, source: str, wall_seconds: float
    ) -> BaselineRecord:
        raise ValueError(f"an InferenceX baseline is measured by {self.baseline_tests} root tests, not the child's run")

    async def first(self, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        baseline = await read_inferencex_baseline(self.blackboard)
        return inferencex.compare(self._candidate(record), baseline, parent, record.request_id)

    async def retest(self, first: Judgement, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        baseline = await read_inferencex_baseline(self.blackboard)
        return inferencex.confirm(first, self._candidate(record), baseline, parent, record.request_id)

    async def accuracy_passes(self, record: TestRecord) -> bool:
        baseline = await read_inferencex_baseline(self.blackboard)
        return inferencex_accuracy_ok(baseline.accuracy, eval_score(_result(record)))

    def _candidate(self, record: TestRecord) -> inferencex.InferenceXReading:
        result = _result(record)
        dropped = dropped_requests(result, self.concurrency)
        long_context = result.long_context
        long_context_dropped = long_context is not None and refuses(long_context, long_context.root)
        skipped = record.request.purpose is Purpose.EXPLORE and not ran_accuracy(record.request, self)
        read = unscored_reading if skipped else inferencex_reading
        return replace(
            read(result, self.concurrency),
            dropped=dropped,
            long_context_dropped=long_context_dropped,
        )


class AgentXRule(Rule):
    """P50 interactivity with its guards; the error rate the run reports is its accuracy.

    The bar is fixed rather than drawn from the root's spread, so one run of the root anchors the parent values: the
    PRELUDE child's own, and PRELUDE submits no root test beside it, as one takes 75 to 90 minutes; one root test where
    PRELUDE starts no child.
    """

    accuracy_round = False
    baseline_tests = 0
    min_baselines = 1

    async def baseline(self, store: GateRecords, root_commit: str, records: Sequence[TestRecord]) -> BaselineRecord:
        if len(records) != self.min_baselines:
            raise ValueError(f"an AgentX baseline is one run of the root, not {len(records)} root tests")
        return await write_agentx_baseline(store, root_commit, records[0])

    async def prelude_baseline(
        self, store: GateRecords, root_commit: str, result: TestResult, *, source: str, wall_seconds: float
    ) -> BaselineRecord:
        return await write_agentx_prelude_baseline(store, root_commit, result, source=source, wall_seconds=wall_seconds)

    async def first(self, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        return agentx.compare(agentx_reading(_result(record)), parent, record.request_id)

    async def retest(self, first: Judgement, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        return agentx.confirm(first, agentx_reading(_result(record)), parent, record.request_id)

    async def accuracy_passes(self, record: TestRecord) -> bool:
        return agentx_accuracy_ok(agentx_reading(_result(record)).get(GRADED_ERROR_RATE))


def ran_accuracy(request: TestRequest, rule: Rule) -> bool:
    """Whether the test ran the eval the rule reads; an InferenceX explore may skip it, recorded as ``accuracy=False``
    on its accepted request."""
    return request.accuracy == rule.accuracy_round


def repeats(retest: TestRecord, first: TestRecord, rule: Rule) -> bool:
    """Only a retest of the same change and files on the same parent, with the rule's eval round, settles a gain."""
    return _candidate(retest.request) == _candidate(first.request) and ran_accuracy(retest.request, rule)


def retest_request(first: TestRecord, rule: Rule) -> TestRequest:
    """The confirming retest of ``first``: the same change and files on the same parent, off the node that measured it,
    with the short-prompt check ``first`` was forced to run, so both results record it. One asked before
    ``first`` has a result runs on any node."""
    sandbox = first.result.sandbox if first.result is not None else None
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
        consistency_check=first.request.consistency_check,
    )


def _candidate(request: TestRequest) -> tuple[object, ...]:
    return request.parent_commit, request.delta, request.attachments


def _result(record: TestRecord) -> TestResult:
    if record.result is None:
        raise UnreadableResult(f"{record.request_id} has no result")
    return record.result


__all__ = ["AgentXRule", "InferenceXRule", "Rule", "ran_accuracy", "repeats", "retest_request"]
