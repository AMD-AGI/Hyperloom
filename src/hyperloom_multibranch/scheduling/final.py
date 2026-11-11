# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FINAL: which commit the session hands on, the integration head or the best single branch's head.

The candidates are the integration head and the specialist head with the highest recorded value, whatever stopped
its branch, unless that head carries the recipe the integration head already carries, which a merge of its work
leaves behind. Each is validated through the toolbox as a session measurement: one ``SWEEP`` test over the workload's
:attr:`~hyperloom_multibranch.scheduling.settings.Workload.final_concs`, read per conc from its rounds, and a
cold-start reproduction the rule gate measures against the commit's own recorded value and holds to the accuracy gate,
and every case of the session's regression set run on the commit itself, whether it builds on the case's fix or not:
a case that reproduces there but not on the root disqualifies it, and the candidates that carry the fewest of the
root's defects are the ones ranked. A sweep that gives no value at the
operating point that decides does not void the candidate: the reproduction measured the same commit there, so the
candidate is ranked on that one point and the choice's reason says so. The toolbox measures a
sweep over the concs its request names and the primary one alone when it names none, so FINAL names them.
The session's root, the integration head of a session that merged nothing, is the reference every recorded value was
measured against, so a cold start of it is not held to the tolerance or the rule's speed gates: one that measures
slower than its baseline is noise in the reference, not a defect of the commit. Every other refusal still holds it
back, as dropped or failed requests do. Every check records what its reproduction measured against the commit's
record, what refused it and, for the root, the speed refusals it was not held to; a FINAL that chooses nothing names
each candidate's reason. FINAL waits for every validation test it started, past its reserve and the
session's deadline if need be.
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
from functools import partial

from hyperloom.contracts.measurement import Purpose, TestKind, TestResult, Validity
from hyperloom_multibranch.contracts.gate import NotMeasured
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Blackboard
from hyperloom_multibranch.contracts.blackboard.records import BranchProgress
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected, TestWithdrawn
from hyperloom_multibranch.blackboard.changes import same_recipe
from hyperloom_multibranch.scheduling.budget import FinalWork
from hyperloom_multibranch.scheduling.explore import integration_row
from hyperloom_multibranch.scheduling.integrator import defects_beyond_gain
from hyperloom_multibranch.scheduling.requests import SessionChecks, integration_test_request, together
from hyperloom_multibranch.scheduling.settings import Workload

#: How far below the other candidate a non-primary sweep point may fall before the candidate is passed over.
MB_SWEEP_DEGRADATION_PCT = 5.0

#: How far below its commit's recorded value a cold start may measure and still reproduce it. The record is the best of
#: the measurements that won the commit, so a cold start of the same recipe lands a little under it, often past the
#: gain bar.
MB_REPRODUCTION_TOLERANCE_PCT = 5.0
#: The refusal of a cold start that measured more than :data:`MB_REPRODUCTION_TOLERANCE_PCT` below its record.
BELOW_REPRODUCTION_TOLERANCE = "below_reproduction_tolerance"

MEASURE_PHASE = "measure"
PRIMARY_POINT_ONLY = "measured at the primary operating point alone: the workload defines no conc sweep"
#: A candidate whose sweep gave no value at the operating point that decides is ranked on its cold-start
#: reproduction, which measured the same commit there. It then has no other operating point, so it degrades nowhere
#: and the degradation rule holds nothing against it.
REPRODUCTION_POINT_ONLY = "ranked on its cold-start reproduction at the primary operating point"

NONE_PASSED = "no candidate passed the final validation"

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
    """A commit FINAL validates; ``recorded_value`` is its recorded value of the primary metric, if it has one, and
    ``root`` says it is the session's root."""

    label: str
    branch: str
    commit: str
    recorded_value: float | None
    root: bool = False


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


async def choose_candidates(
    progress: Sequence[BranchProgress], integration: BranchProgress, metric: str, repo: ExperimentRepo, root: str
) -> FinalCandidates:
    """The head of ``integration`` and the specialist head with the highest recorded ``metric`` over every branch the
    blackboard knows; ``root`` is the session's root commit, which the integration head still is when nothing merged.

    A branch with no commit offers no single-branch candidate, and neither does one whose head carries the recipe the
    integration head already carries: a merge writes the branch's work as a new commit, so the two differ by commit
    and are one recipe, which FINAL would otherwise validate twice. The next branch down then stands for the session's
    single-branch work instead.
    """

    head = integration.head_commit
    offered = [_branch_candidate(row, head, metric) for row in progress]
    ranked = sorted((row for row in offered if row is not None), key=lambda candidate: -candidate.recorded_value)
    candidate = FinalCandidate(
        INTEGRATION_LABEL, INTEGRATION_BRANCH, head, integration.head_value.get(metric), head == root
    )
    return FinalCandidates(candidate, await _best_of_its_own(ranked, head, repo))


async def session_candidates(blackboard: Blackboard, repo: ExperimentRepo, metric: str) -> FinalCandidates:
    """:func:`choose_candidates` over every branch ``blackboard`` holds progress of, against the session's root."""

    progress = await blackboard.progress()
    integration = integration_row(progress)
    if integration is None:
        raise RuntimeError("the blackboard has no integration branch after PRELUDE recorded its root")
    root = (await blackboard.baseline()).root_commit
    return await choose_candidates(progress, integration, metric, repo, root)


async def _best_of_its_own(
    ranked: Sequence[FinalCandidate], integration_head: str, repo: ExperimentRepo
) -> FinalCandidate | None:
    """The first of ``ranked`` whose commit is not the integration head's recipe over again."""

    if not ranked:
        return None
    head = await repo.read_recipe(integration_head)
    for candidate in ranked:
        if not same_recipe(await repo.read_recipe(candidate.commit), head):
            return candidate
    return None


@dataclass(frozen=True)
class CandidateCheck:
    """What validating one candidate found.

    ``curve`` is the sweep's primary metric per conc, without the concs it measured nothing at; ``degraded_concs``
    are the sweep points that fell below the other candidate's. ``regression_cases`` is how many regression cases ran
    on the commit, none when the session registered none, and ``carried`` names those that reproduced on it and on the
    root alike, the root's defects it still carries. ``note`` says what the choice's reason should carry
    about how this candidate was measured, for a candidate that was still ranked on something short of a whole sweep.
    ``gain_pct`` is how far its reproduction measured from the commit's record, ``None`` when none was judged,
    ``refused_by`` what kept the reproduction from holding, and ``waived`` the speed refusals a root's reproduction
    was not held to.
    """

    candidate: FinalCandidate
    curve: Mapping[int, float] = field(default_factory=dict)
    accuracy_passed: bool = False
    reproduced: bool = False
    regression_passed: bool = False
    regression_cases: int = 0
    carried: tuple[str, ...] = ()
    degraded_concs: tuple[int, ...] = ()
    error: str = ""
    note: str = ""
    gain_pct: float | None = None
    refused_by: tuple[str, ...] = ()
    waived: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        passed = self.accuracy_passed and self.reproduced and self.regression_passed
        return passed and not self.error

    @property
    def refusal(self) -> str:
        """Why the candidate is not eligible; empty when it is."""

        if self.error:
            return self.error
        reasons = []
        if not self.reproduced:
            gain = "not judged" if self.gain_pct is None else f"{self.gain_pct:.1f}% against its record"
            refused = f", refused by {', '.join(self.refused_by)}" if self.refused_by else ""
            reasons.append(f"not reproduced ({gain}{refused})")
        if not self.accuracy_passed:
            reasons.append("accuracy failed")
        if not self.regression_passed:
            reasons.append("the regression set failed")
        return ", ".join(reasons)


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


def _carried(reason: str, eligible: list[CandidateCheck], chosen: CandidateCheck, fewest: int) -> str:
    """``reason`` after the candidates passed over for carrying more of the root's defects, and before those the
    chosen one still carries."""

    over = [check for check in eligible if len(check.carried) > fewest]
    if over:
        said = "; ".join(
            f"{check.candidate.label} carries more of the root's defects ({', '.join(check.carried)})" for check in over
        )
        reason = f"{said}; among the rest, {reason}"
    if chosen.carried:
        reason = f"{reason}; {chosen.candidate.label} still carries the root's defects {', '.join(chosen.carried)}"
    return reason


def pick_final(checks: tuple[CandidateCheck, ...], metric: str, conc: int) -> FinalChoice:
    """The eligible candidate with the higher ``metric`` at ``conc``, unless only it degrades elsewhere in the sweep,
    among those that carry the fewest of the root's defects."""

    eligible = [check for check in checks if check.eligible]
    fewest = min((len(check.carried) for check in eligible), default=0)
    staying = {
        check.candidate.label: check
        for check in _with_degradation(tuple(check for check in eligible if len(check.carried) == fewest), conc)
    }
    marked = tuple(staying.get(check.candidate.label, check) for check in checks)
    ranked = sorted(staying.values(), key=lambda check: -check.curve[conc])
    if not ranked:
        refusals = "; ".join(f"{check.candidate.label} {check.refusal}" for check in marked)
        reason = f"{NONE_PASSED}: {refusals}" if refusals else NONE_PASSED
        return FinalChoice(None, marked, reason, metric, conc)
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
    reason = _carried(reason, eligible, chosen, fewest)
    notes = "; ".join(check.note for check in marked if check.note)
    return FinalChoice(chosen.candidate, marked, f"{reason}; {notes}" if notes else reason, metric, conc)


@dataclass(frozen=True)
class _Reproduction:
    """A cold start of a candidate: ``reproduced`` when it measures within :data:`MB_REPRODUCTION_TOLERANCE_PCT` of the
    commit's own recorded value and nothing but the missing gain refused it (not dropped requests, a broken guard, a
    pair that did not measure the same work), and ``accuracy_passed`` when it clears the accuracy gate. The root's
    reproduction is not held to the tolerance or the rule's speed gates; ``waived`` names those that refused it.
    ``refused_by`` is what kept it from holding and ``gain_pct`` how far it measured from the record.

    ``value`` is what it measured of the primary metric at the operating point that decides, which is the sweep's
    primary point: the candidate is ranked on it when the sweep gave nothing there.
    """

    reproduced: bool
    accuracy_passed: bool
    value: float | None
    gain_pct: float
    refused_by: tuple[str, ...]
    waived: tuple[str, ...]


class FinalSelector:
    """Validates the candidates through the toolbox, the rule gate and the regression set, ``parallel`` tests at
    once."""

    def __init__(self, checks: SessionChecks, workload: Workload, parallel: int) -> None:
        self._tests = checks.tests
        self._gate = checks.gate
        self._regression = checks.regression
        self._workload = workload
        self._metric = workload.metric
        self._parallel = parallel

    async def select(self, candidates: FinalCandidates) -> FinalChoice:
        """Validate every candidate concurrently and pick one, however long the validation runs: no clock bounds it,
        the toolbox's per-test timeout bounds a test that hangs, and only a cancel cuts it short."""

        checks = await asyncio.gather(*(self._validated(candidate) for candidate in candidates.each))
        choice = pick_final(tuple(checks), self._metric, self._workload.conc)
        if self._workload.final_concs == (self._workload.conc,):
            return replace(choice, reason=f"{choice.reason}; {PRIMARY_POINT_ONLY}")
        return choice

    async def planned_work(self, candidates: FinalCandidates, test_seconds: float) -> FinalWork:
        """What :meth:`select` would run to validate ``candidates``, ``parallel`` tests at once, one attempt of
        each, given ``test_seconds``, one test at the primary conc: for each candidate its sweep, a test per conc of
        :attr:`~hyperloom_multibranch.scheduling.settings.Workload.final_concs`, and its reproduction, and the
        regression runs still to make (:meth:`~hyperloom_multibranch.contracts.gate.RegressionCheck.final_case_seconds`)."""

        sweep = len(self._workload.final_concs) * test_seconds
        tests = [sweep, test_seconds] * len(candidates.each)
        tests.extend(await self._regression.final_case_seconds(tuple(each.commit for each in candidates.each)))
        return FinalWork(sum(tests), max(tests), test_seconds, self._parallel)

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
        repro = asyncio.ensure_future(self._reproduction(candidate))
        regression = asyncio.ensure_future(self._regression.check_every_case(candidate.commit))
        await together(sweep, repro, regression)
        swept, reproduction, regressed = sweep.result(), repro.result(), regression.result()
        check = CandidateCheck(
            candidate,
            sweep_curve(swept, self._metric) if swept.validity is Validity.SUCCEEDED else {},
            reproduction.accuracy_passed,
            reproduction.reproduced,
            regressed.passed,
            regression_cases=regressed.cases,
            carried=regressed.carried,
            gain_pct=reproduction.gain_pct,
            refused_by=reproduction.refused_by,
            waived=reproduction.waived,
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

    async def _reproduction(self, candidate: FinalCandidate) -> _Reproduction:
        request = integration_test_request(TestKind.TEST, Purpose.FINAL, candidate.commit)
        return await self._tests.judged(request, partial(self._judge_reproduction, candidate.root))

    async def _judge_reproduction(self, root: bool, request_id: str) -> _Reproduction:
        judgement = await self._gate.compare(request_id)
        accuracy = await self._gate.accuracy_passes(request_id)
        below = judgement.gain_pct < -MB_REPRODUCTION_TOLERANCE_PCT
        tolerance = (BELOW_REPRODUCTION_TOLERANCE,) if below else ()
        defects = defects_beyond_gain(judgement)
        refused_by: tuple[str, ...] = (*tolerance, *defects)
        waived: tuple[str, ...] = ()
        if root:
            speed = await self._gate.speed_gates()
            waived = (*tolerance, *(reason for reason in judgement.refused_by if reason in speed))
            refused_by = tuple(reason for reason in defects if reason not in speed)
        value = judgement.recorded_value.get(self._metric)
        return _Reproduction(not refused_by, accuracy, value, judgement.gain_pct, refused_by, waived)


__all__ = [
    "BELOW_REPRODUCTION_TOLERANCE",
    "MB_REPRODUCTION_TOLERANCE_PCT",
    "MB_SWEEP_DEGRADATION_PCT",
    "NONE_PASSED",
    "PRIMARY_POINT_ONLY",
    "REPRODUCTION_POINT_ONLY",
    "CandidateCheck",
    "FinalCandidate",
    "FinalCandidates",
    "FinalChoice",
    "FinalSelector",
    "choose_candidates",
    "pick_final",
    "session_candidates",
    "sweep_curve",
]
