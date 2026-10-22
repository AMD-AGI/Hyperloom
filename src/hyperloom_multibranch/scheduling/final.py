# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FINAL: which commit the session hands on, the integration head or the best single branch's head.

The candidates are the integration head and the specialist head with the highest recorded value, whatever stopped
its branch. Each is validated through the toolbox as a session measurement: one ``SWEEP`` test over the workload's
:attr:`~hyperloom_multibranch.scheduling.settings.Workload.final_concs`, read per conc from its rounds, and a
cold-start reproduction the rule gate measures against the commit's own recorded value and holds to the accuracy gate,
and the regression set of the commit's own ancestry run on the commit itself. A sweep that gives no value at the
operating point that decides does not void the candidate: the reproduction measured the same commit there, so the
candidate is ranked on that one point and the choice's reason says so. The toolbox measures a
sweep over the concs its request names and the primary one alone when it names none, so FINAL names them. FINAL
waits for every validation test it started, past its reserve and the session's deadline if need be.
The primary operating point (the workload's conc) decides; a candidate
whose other sweep points fall more than :data:`MB_SWEEP_DEGRADATION_PCT` below the other candidate's at the same conc
is passed over while the other one does not degrade in turn. An AgentX workload defines no other operating point, so
its candidates are measured at the primary one alone, and the choice's reason says so.
"""

from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from hyperloom.contracts.measurement import Purpose, TestKind, TestResult, Validity
from hyperloom_multibranch.contracts.gate import NotMeasured
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import BranchProgress
from hyperloom_multibranch.contracts.toolbox import RequestRejected, TestWithdrawn
from hyperloom_multibranch.scheduling.integrator import defects_beyond_gain
from hyperloom_multibranch.scheduling.requests import (
    NO_DELTA,
    SessionChecks,
    integration_test_request,
    together,
)
from hyperloom_multibranch.scheduling.settings import Workload

#: How far below the other candidate a non-primary sweep point may fall before the candidate is passed over.
MB_SWEEP_DEGRADATION_PCT = 5.0

#: How far below its commit's recorded value a cold start may measure and still reproduce it. The record is the best of
#: the measurements that won the commit, so a cold start of the same recipe lands a little under it, often past the
#: gain bar.
MB_REPRODUCTION_TOLERANCE_PCT = 5.0

MEASURE_PHASE = "measure"
PRIMARY_POINT_ONLY = "measured at the primary operating point alone: the workload defines no conc sweep"
#: A candidate whose sweep gave no value at the operating point that decides is ranked on its cold-start
#: reproduction, which measured the same commit there. It then has no other operating point, so it degrades nowhere
#: and the degradation rule holds nothing against it.
REPRODUCTION_POINT_ONLY = "ranked on its cold-start reproduction at the primary operating point"

INTEGRATION_LABEL = "integration"
BEST_BRANCH_LABEL = "best_branch"


def sweep_curve(result: TestResult, metric: str) -> dict[int, float]:
    """The mean of ``metric`` over a sweep's measure rounds at each conc; a conc it measured nothing at is left out."""

    values: defaultdict[int, list[float]] = defaultdict(list)
    for row in result.rounds:
        if row.phase == MEASURE_PHASE and row.concurrency is not None and metric in row.metrics:
            values[row.concurrency].append(row.metrics[metric])
    return {conc: statistics.fmean(measured) for conc, measured in values.items()}


@dataclass(frozen=True)
class FinalCandidate:
    """A commit FINAL validates; ``recorded_value`` is its recorded value of the primary metric, if it has one."""

    label: str
    branch: str
    commit: str
    recorded_value: float | None


@dataclass(frozen=True)
class FinalCandidates:
    integration: FinalCandidate
    best_branch: FinalCandidate | None

    @property
    def each(self) -> tuple[FinalCandidate, ...]:
        return (self.integration,) if self.best_branch is None else (self.integration, self.best_branch)


def _branch_candidate(row: BranchProgress, integration_head: str, metric: str) -> FinalCandidate | None:
    if row.branch == INTEGRATION_BRANCH or not row.commits:
        return None
    value = row.head_value.get(metric)
    if value is None or row.head_commit == integration_head:
        return None
    return FinalCandidate(BEST_BRANCH_LABEL, row.branch, row.head_commit, value)


def choose_candidates(
    progress: Sequence[BranchProgress],
    integration_head: str,
    integration_value: Mapping[str, float],
    metric: str,
) -> FinalCandidates:
    """The integration head and the specialist head with the highest recorded ``metric`` over every branch the
    notebook knows.

    A branch with no commit, or whose head is the integration head, offers no single-branch candidate.
    """

    offered = [_branch_candidate(row, integration_head, metric) for row in progress]
    ranked = [candidate for candidate in offered if candidate is not None]
    best = max(ranked, key=lambda candidate: candidate.recorded_value, default=None)
    integration = FinalCandidate(INTEGRATION_LABEL, INTEGRATION_BRANCH, integration_head, integration_value.get(metric))
    return FinalCandidates(integration, best)


@dataclass(frozen=True)
class CandidateCheck:
    """What validating one candidate found.

    ``curve`` is the sweep's primary metric per conc, without the concs it measured nothing at; ``degraded_concs``
    are the sweep points that fell below the other candidate's. ``note`` says what the choice's reason should carry
    about how this candidate was measured, for a candidate that was still ranked on something short of a whole sweep.
    """

    candidate: FinalCandidate
    curve: Mapping[int, float] = field(default_factory=dict)
    accuracy_passed: bool = False
    reproduced: bool = False
    regression_passed: bool = False
    degraded_concs: tuple[int, ...] = ()
    error: str = ""
    note: str = ""

    @property
    def eligible(self) -> bool:
        passed = self.accuracy_passed and self.reproduced and self.regression_passed
        return passed and not self.error


@dataclass(frozen=True)
class FinalChoice:
    """The pick among ``checks``, made on ``metric`` at ``primary_conc`` with sweep points held to
    ``degradation_pct``; the session breakdown records it as how the final commit was chosen."""

    chosen: FinalCandidate | None
    checks: tuple[CandidateCheck, ...]
    reason: str
    metric: str
    primary_conc: int
    degradation_pct: float = MB_SWEEP_DEGRADATION_PCT


def _degraded(mine: Mapping[int, float], other: Mapping[int, float], primary_conc: int) -> set[int]:
    floor = 1.0 - MB_SWEEP_DEGRADATION_PCT / 100.0
    return {conc for conc, value in mine.items() if conc != primary_conc and value < other.get(conc, 0.0) * floor}


def _with_degradation(checks: tuple[CandidateCheck, ...], conc: int) -> tuple[CandidateCheck, ...]:
    curves = {check.candidate.label: check.curve for check in checks if check.eligible}

    def marked(check: CandidateCheck) -> CandidateCheck:
        if not check.eligible:
            return check
        mine = curves[check.candidate.label]
        found = set().union(
            *(_degraded(mine, curve, conc) for label, curve in curves.items() if label != check.candidate.label)
        )
        return replace(check, degraded_concs=tuple(sorted(found)))

    return tuple(marked(check) for check in checks)


def pick_final(checks: tuple[CandidateCheck, ...], metric: str, conc: int) -> FinalChoice:
    """The eligible candidate with the higher ``metric`` at ``conc``, unless only it degrades elsewhere in the sweep."""

    marked = _with_degradation(checks, conc)
    ranked = sorted((check for check in marked if check.eligible), key=lambda check: -check.curve[conc])
    if not ranked:
        return FinalChoice(None, marked, "no candidate passed the final validation", metric, conc)
    chosen = next((check for check in ranked if not check.degraded_concs), ranked[0])
    if chosen is not ranked[0]:
        top = ranked[0]
        reason = (
            f"{top.candidate.label} is higher at conc {conc} but falls more than {MB_SWEEP_DEGRADATION_PCT}% "
            f"below at conc {list(top.degraded_concs)}"
        )
    elif chosen.degraded_concs:
        reason = f"every candidate degrades somewhere in the sweep; the higher {metric} at conc {conc} decides"
    else:
        reason = f"the higher {metric} at conc {conc}"
    notes = "; ".join(check.note for check in marked if check.note)
    return FinalChoice(chosen.candidate, marked, f"{reason}; {notes}" if notes else reason, metric, conc)


@dataclass(frozen=True)
class _Reproduction:
    """A cold start of a candidate: ``reproduced`` when it measures within :data:`MB_REPRODUCTION_TOLERANCE_PCT` of the
    commit's own recorded value and nothing but the missing gain refused it (not dropped requests, a broken guard, a
    pair that did not measure the same work), and ``accuracy_passed`` when it clears the accuracy gate.

    ``value`` is what it measured of the primary metric at the operating point that decides, which is the sweep's
    primary point: the candidate is ranked on it when the sweep gave nothing there.
    """

    reproduced: bool
    accuracy_passed: bool
    value: float | None


class FinalSelector:
    """Validates the candidates through the toolbox, the rule gate and the regression set."""

    def __init__(self, checks: SessionChecks, workload: Workload) -> None:
        self._tests = checks.tests
        self._gate = checks.gate
        self._regression = checks.regression
        self._workload = workload
        self._metric = workload.metric

    async def select(self, candidates: FinalCandidates) -> FinalChoice:
        """Validate every candidate concurrently and pick one, however long the validation runs: no clock bounds it,
        the toolbox's per-test timeout bounds a test that hangs, and only a cancel cuts it short."""

        checks = await asyncio.gather(*(self._validated(candidate) for candidate in candidates.each))
        choice = pick_final(tuple(checks), self._metric, self._workload.conc)
        if self._workload.final_concs == (self._workload.conc,):
            return replace(choice, reason=f"{choice.reason}; {PRIMARY_POINT_ONLY}")
        return choice

    async def _validated(self, candidate: FinalCandidate) -> CandidateCheck:
        try:
            return await self._check(candidate)
        except TestWithdrawn as exc:
            return CandidateCheck(candidate, error=f"a validation test was withdrawn: {exc}")
        except RequestRejected as exc:
            return CandidateCheck(candidate, error=f"the toolbox refused a validation test: {exc}")
        except NotMeasured as exc:
            return CandidateCheck(candidate, error=f"the gate had nothing to judge: {exc}")

    async def _check(self, candidate: FinalCandidate) -> CandidateCheck:
        sweep = asyncio.ensure_future(self._sweep(candidate.commit))
        repro = asyncio.ensure_future(self._reproduction(candidate.commit))
        regression = asyncio.ensure_future(self._regression.check(candidate.commit, NO_DELTA, sources=()))
        await together(sweep, repro, regression)
        swept, reproduction = sweep.result(), repro.result()
        check = CandidateCheck(
            candidate,
            sweep_curve(swept, self._metric) if swept.validity is Validity.SUCCEEDED else {},
            reproduction.accuracy_passed,
            reproduction.reproduced,
            regression.result().passed,
        )
        if self._workload.conc in check.curve:
            return check
        missing = self._no_primary_point(swept)
        if reproduction.value is None:
            return replace(check, error=missing)
        return replace(
            check, curve={self._workload.conc: reproduction.value}, note=f"{missing}, {REPRODUCTION_POINT_ONLY}"
        )

    def _no_primary_point(self, swept: TestResult) -> str:
        """Why the sweep gave no value of the primary metric at the conc that decides."""

        if swept.validity is not Validity.SUCCEEDED:
            return f"the sweep ended {swept.validity.value} ({swept.failure_reason or 'no reason given'})"
        return f"the sweep has no {self._metric} at the primary conc {self._workload.conc}"

    async def _sweep(self, commit: str) -> TestResult:
        concs = self._workload.final_concs
        return await self._tests.result(
            integration_test_request(TestKind.SWEEP, Purpose.FINAL, commit, concurrency=concs)
        )

    async def _reproduction(self, commit: str) -> _Reproduction:
        request = integration_test_request(TestKind.TEST, Purpose.FINAL, commit)
        return await self._tests.judged(request, self._judge_reproduction)

    async def _judge_reproduction(self, request_id: str) -> _Reproduction:
        judgement = await self._gate.compare(request_id)
        held = judgement.gain_pct >= -MB_REPRODUCTION_TOLERANCE_PCT
        reproduced = held and not defects_beyond_gain(judgement)
        accuracy = await self._gate.accuracy_passes(request_id)
        return _Reproduction(reproduced, accuracy, judgement.recorded_value.get(self._metric))


__all__ = [
    "MB_REPRODUCTION_TOLERANCE_PCT",
    "MB_SWEEP_DEGRADATION_PCT",
    "PRIMARY_POINT_ONLY",
    "REPRODUCTION_POINT_ONLY",
    "CandidateCheck",
    "FinalCandidate",
    "FinalCandidates",
    "FinalChoice",
    "FinalSelector",
    "choose_candidates",
    "pick_final",
    "sweep_curve",
]
