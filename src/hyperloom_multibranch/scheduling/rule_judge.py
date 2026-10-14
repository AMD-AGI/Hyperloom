# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How the scheduling side judges its own tests: the rule gate's InferenceX rule, applied to a settled test record.

The rule gate module ships the rule as pure functions (:mod:`hyperloom_multibranch.rulegate.inferencex` and
:mod:`~hyperloom_multibranch.rulegate.accuracy`) and the gate that lands a specialist's candidate; it has no service
that judges a test the scheduler issued, runs the confirming retest, or establishes the PRELUDE baseline.
:class:`MbFakeRuleJudge` does that until the rule gate module does. It judges on the InferenceX throughput alone:
which metric a workload is judged on (InferenceX or AgentX) is not settled yet.

The bar and the accuracy reference are derived from the notebook's baseline record each time, so the notebook stays
the one owner of the baseline.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import Notebook
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord, TestRecord
from hyperloom_multibranch.rulegate.accuracy import baseline_accuracy, inferencex_accuracy_ok
from hyperloom_multibranch.rulegate.inferencex import DispersionThreshold, compare, confirm
from hyperloom_multibranch.scheduling.requests import SessionTests

#: How many baseline tests PRELUDE measures; the rule gate needs at least ``rulegate.inferencex.MIN_BASELINES``, and
#: its noise simulation assumes three. Replaced by the rule gate's own count once it offers one.
MB_FAKE_BASELINE_TESTS = 3


class NotMeasured(Exception):
    """A settled test reported no value to judge: the candidate failed to serve, or the test measured nothing."""

    def __init__(self, record: TestRecord) -> None:
        validity = record.result.validity.value if record.result is not None else "no result"
        super().__init__(f"{record.request_id} measured no {GRADED_OUTPUT} ({validity})")
        self.request_id = record.request_id


def formal_mean(record: TestRecord, metric: str) -> float | None:
    """The mean of ``metric`` over the test's measured rounds; ``None`` when it measured none."""

    rounds = record.result.rounds if record.result is not None else ()
    values = [row.metrics[metric] for row in rounds if row.phase == "measure" and metric in row.metrics]
    return statistics.fmean(values) if values else None


def accuracy_score(record: TestRecord) -> float | None:
    """The mean of the scores the test's accuracy rounds reported; ``None`` when they reported none."""

    scores = [value.score for value in record.result.accuracy] if record.result is not None else []
    return statistics.fmean(scores) if scores else None


def _measured(record: TestRecord) -> float:
    value = formal_mean(record, GRADED_OUTPUT)
    if value is None:
        raise NotMeasured(record)
    return value


@dataclass(frozen=True)
class _Reference:
    threshold: DispersionThreshold
    accuracy: float


class MbFakeRuleJudge:
    """Judges the session's own tests as the rule gate judges a specialist's candidate.

    ``notebook`` holds the rule gate's role, the only one that may write the baseline; ``tests`` runs the retests.
    Stands in for the rule gate's judging service; replaced when ``hyperloom_multibranch.rulegate`` offers one.
    """

    def __init__(self, notebook: Notebook, tests: SessionTests) -> None:
        self._notebook = notebook
        self._tests = tests

    async def establish_baseline(self, root_commit: str, records: tuple[TestRecord, ...]) -> BaselineRecord:
        """Anchor the session to the baseline ``records`` measured on ``root_commit`` and write it to the notebook.

        Raises :class:`NotMeasured`, ``ValueError`` from the threshold, or
        :class:`~hyperloom_multibranch.rulegate.accuracy.BaselineAccuracyError` when the baseline cannot anchor anything.
        """

        reference = _reference(records)
        record = BaselineRecord(
            root_commit,
            {GRADED_OUTPUT: statistics.fmean(reference.threshold.baselines)},
            {GRADED_OUTPUT: reference.threshold.spread_pct},
            {GRADED_OUTPUT: reference.threshold.threshold_pct},
            tuple(record.request_id for record in records),
        )
        await self._notebook.write_baseline(record)
        return record

    async def judge(self, record: TestRecord) -> Judgement:
        """``record``'s test against its parent commit's recorded value; a gain past the bar is retested first, on
        another node, so the answer is never ``NEEDS_CONFIRM``. Raises :class:`NotMeasured`, and
        :class:`~hyperloom_multibranch.scheduling.requests.TestWithdrawn` when the retest is withdrawn."""

        reference = await self._reference()
        parent = await self._recorded(record.request.parent_commit)
        accurate = self._accurate(record, reference)
        first = compare(_measured(record), accurate, parent, reference.threshold, record.request_id)
        if first.verdict is not Verdict.NEEDS_CONFIRM:
            return first
        retest = await self._tests.run(_confirming(record))
        accurate = self._accurate(retest, reference)
        return confirm(first, _measured(retest), accurate, parent, reference.threshold, retest.request_id)

    async def reproduces(self, record: TestRecord) -> bool:
        """Whether ``record``, a test of a commit as it is, holds the commit's own recorded value: anything but a
        regression past the bar reproduces it, so a gain needs no retest."""

        reference = await self._reference()
        own = await self._recorded(record.request.parent_commit)
        judged = compare(_measured(record), True, own, reference.threshold, record.request_id)
        return judged.verdict is not Verdict.REGRESSED

    async def accuracy_passes(self, record: TestRecord) -> bool:
        """Whether an accuracy test clears the accuracy gate every commit clears."""

        return inferencex_accuracy_ok((await self._reference()).accuracy, accuracy_score(record))

    async def _reference(self) -> _Reference:
        baseline = await self._notebook.baseline()
        return _reference(tuple([await self._notebook.record(request_id) for request_id in baseline.request_ids]))

    async def _recorded(self, commit: str) -> dict[str, float]:
        return dict((await self._notebook.recorded_value(commit)).values)

    @staticmethod
    def _accurate(record: TestRecord, reference: _Reference) -> bool:
        """A test without accuracy rounds is held to no accuracy gate."""

        if not record.request.accuracy:
            return True
        return inferencex_accuracy_ok(reference.accuracy, accuracy_score(record))


def _reference(baselines: tuple[TestRecord, ...]) -> _Reference:
    threshold = DispersionThreshold(tuple(_measured(record) for record in baselines))
    return _Reference(threshold, baseline_accuracy([accuracy_score(record) for record in baselines]))


def _confirming(record: TestRecord) -> TestRequest:
    """The retest of ``record``'s test on another node; the toolbox runs it ahead of every other test."""

    tested = record.request
    sandbox = record.result.sandbox if record.result is not None else None
    node = sandbox.node_name if sandbox is not None else None
    return TestRequest(
        kind=TestKind.TEST,
        purpose=Purpose.CONFIRM,
        branch_id=tested.branch_id,
        parent_commit=tested.parent_commit,
        delta=tested.delta,
        retest_of=record.request_id,
        avoid_nodes=(node,) if node else (),
        accuracy=tested.accuracy,
    )


__all__ = ["MB_FAKE_BASELINE_TESTS", "MbFakeRuleJudge", "NotMeasured", "accuracy_score", "formal_mean"]
