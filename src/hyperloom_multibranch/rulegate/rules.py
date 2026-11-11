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
from hyperloom.contracts.consistency import SHORT_PROMPTS_COMPLETE
from hyperloom.contracts.long_context import LONG_CONTEXT_COMPLETE, refuses
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest, TestResult
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.contracts.blackboard.interfaces import CommitQuery, GateRecords
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, TestRecord
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
    unscored_reading,
)


#: The attachment id of a test that undoes a change its own agent found invalid, a file saying which.
RETRACTION_ID = "retraction"


class Rule(Protocol):
    """How the session's workload is judged; ``accuracy_round`` says whether its tests run the eval it reads, and
    ``speed_gates`` are the refusals that say only that a candidate is not faster than its parent;
    ``runs_consistency_checks`` whether its tests can hold their server for the consistency checks.

    PRELUDE submits ``baseline_tests`` root tests, and the baseline needs ``min_baselines`` of them measured.
    ``confirms_ahead`` says whether a test is short enough to submit its confirming retest before the judgement asks
    for it.
    """

    accuracy_round: bool
    speed_gates: frozenset[str]
    runs_consistency_checks: bool
    baseline_tests: int
    min_baselines: int
    confirms_ahead: bool

    async def baseline(self, store: GateRecords, root_commit: str, records: Sequence[TestRecord]) -> BaselineRecord: ...

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
    speed_gates = inferencex.SPEED_GATES
    runs_consistency_checks = True
    baseline_tests = inferencex.BASELINE_TESTS
    min_baselines = inferencex.MIN_BASELINES
    confirms_ahead = True

    async def baseline(self, store: GateRecords, root_commit: str, records: Sequence[TestRecord]) -> BaselineRecord:
        return await write_inferencex_baseline(store, root_commit, records, self.concurrency)

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


@dataclass(frozen=True)
class AgentXRule(Rule):
    """The session's ``objective`` with its guards; the error rate the run reports is its accuracy.

    The bar is fixed rather than drawn from the root's spread, so one root test anchors the parent values. A test
    takes 75 to 90 minutes, too long to spend on a confirm the judgement may not need.
    """

    objective: agentx.AgentXObjective
    accuracy_round = False
    speed_gates = agentx.SPEED_GATES
    # An AgentX test holds no server for them (BenchmarkPoint.holds_accuracy_checks), so both checks end incomplete.
    runs_consistency_checks = False
    baseline_tests = 1
    min_baselines = 1
    confirms_ahead = False

    async def baseline(self, store: GateRecords, root_commit: str, records: Sequence[TestRecord]) -> BaselineRecord:
        if len(records) != self.min_baselines:
            raise ValueError(f"an AgentX baseline is one run of the root, not {len(records)} root tests")
        return await write_agentx_baseline(store, root_commit, records[0], self.objective)

    async def first(self, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        """A candidate the checkpoint stopped is judged on its ruling (:func:`stopped_at_checkpoint`)."""
        ruling = _result(record).agentx_checkpoint
        if ruling is not None and ruling.stopped:
            return agentx.trailed(ruling, record.request_id, self.objective)
        return agentx.compare(self._reading(record), parent, record.request_id, self.objective)

    async def retest(self, first: Judgement, record: TestRecord, parent: Mapping[str, float]) -> Judgement:
        return agentx.confirm(first, self._reading(record), parent, record.request_id, self.objective)

    async def accuracy_passes(self, record: TestRecord) -> bool:
        return agentx_accuracy_ok(self._reading(record).get(GRADED_ERROR_RATE))

    def _reading(self, record: TestRecord) -> dict[str, float]:
        return agentx_reading(_result(record), self.objective.metric)


def ran_accuracy(request: TestRequest, rule: Rule) -> bool:
    """Whether the test ran the eval the rule reads; an InferenceX explore may skip it, recorded as ``accuracy=False``
    on its accepted request."""
    return request.accuracy == rule.accuracy_round


def retracts(record: TestRecord, rule: Rule) -> bool:
    """Whether the test undoes a change its own agent found invalid: it carries an attachment under
    :data:`RETRACTION_ID`, forced both consistency checks, as an autonomous branch's import of a retraction does, and,
    under a rule whose tests can run them, both ran to the end. Only the judge of an autonomous branch reads it: a
    specialist's own request never counts."""
    request, result = record.request, record.result
    forced = request.forces_consistency_check and request.forces_long_context_check
    if not (_marked(request) and forced):
        return False
    if not rule.runs_consistency_checks:
        return True
    short, long = (None, None) if result is None else (result.short_prompt_consistency, result.long_context)
    checked = short is not None and short.status == SHORT_PROMPTS_COMPLETE
    return checked and long is not None and long.status == LONG_CONTEXT_COMPLETE


def _marked(request: TestRequest) -> bool:
    return any(attachment.id == RETRACTION_ID for attachment in request.attachments)


def lands_as_retraction(record: TestRecord, judgement: Judgement, rule: Rule) -> bool:
    """Whether a retraction the rule did not find faster lands all the same: undoing a change is slower by
    construction, so only the rule's errors, guards and accuracy, and a consistency check that ran, hold it back. Under
    a rule whose tests cannot run the consistency checks (AgentX) none ran, so consistency is not tested. A final verdict whose every refusal is one of the
    rule's speed gates, on measurements whose accuracy passed, and a long-context check that shows no drop, which the
    AgentX rule does not read for any other test."""
    final = judgement.verdict in (Verdict.FLAT, Verdict.REGRESSED)
    passed = judgement.accuracy is AccuracyVerdict.PASSED
    long_context = _result(record).long_context
    dropped = long_context is not None and refuses(long_context, long_context.root)
    return final and passed and not dropped and set(judgement.refused_by) <= rule.speed_gates


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


def stopped_at_checkpoint(record: TestRecord) -> bool:
    """Whether the AgentX checkpoint stopped the test: a ruling on the candidate, judged though no full measurement
    exists."""
    ruling = None if record.result is None else record.result.agentx_checkpoint
    return ruling is not None and ruling.stopped


def _candidate(request: TestRequest) -> tuple[object, ...]:
    return request.parent_commit, request.delta, request.attachments


def _result(record: TestRecord) -> TestResult:
    if record.result is None:
        raise UnreadableResult(f"{record.request_id} has no result")
    return record.result


__all__ = [
    "RETRACTION_ID",
    "AgentXRule",
    "InferenceXRule",
    "Rule",
    "lands_as_retraction",
    "ran_accuracy",
    "repeats",
    "retest_request",
    "retracts",
    "stopped_at_checkpoint",
]
