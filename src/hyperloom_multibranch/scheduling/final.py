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
back, as dropped or failed requests do. Every check records its reproduction's gain over the root's PRELUDE baseline,
what that reproduction measured against the commit's record, what refused it and, for the root, the speed refusals it
was not held to; a FINAL that chooses nothing names each candidate's reason. FINAL waits for every validation test it
started, past its reserve and the session's deadline if need be.
The primary operating point (the workload's conc) decides; a candidate
whose other sweep points fall more than :data:`MB_SWEEP_DEGRADATION_PCT` below the other candidate's at the same conc
is passed over while the other one does not degrade in turn. An AgentX workload defines no other operating point, so
its candidates are measured at the primary one alone, and the choice's reason says so.

After CLOSE, :class:`ConcSweep` sweeps the root and the commit FINAL chose over the concurrencies the InferenceX arm
of the session's frozen base publishes, or the operator's ``--conc-sweep-concs`` for a base no InferenceX arm gave,
with the workload's CONC among them. It runs only when that commit is not the root and its reproduction gained over
the baseline. Both arms are submitted at once, so they run side by side where the session runs two tests at once. An
AgentX point is a test of its own, booted cold as every AgentX level is; an InferenceX arm is one sweep test over all
its points on one server, whose attempt timeout scales with them (:meth:`SweepPlan.test_points`). The report
(``MB-reports/conc_sweep_summary.json``) is rewritten after every test, and carried each time as ``conc_sweep_summary``
in ``MB-reports/session_breakdown.json``; a later run measures only the points not yet measured. These tests are
session measurements with no accuracy and no checks.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

from hyperloom.common.gain_math import conc_pair_comparison, gain_pct
from hyperloom.contracts.measurement import Purpose, TestKind, TestResult, Validity
from hyperloom.orchestrator.conc_sweep.plot import ConcCurveOptions
from hyperloom.orchestrator.conc_sweep.points import SCHEMA_VERSION, SweepPoints, ladder, sweep_points
from hyperloom.orchestrator.conc_sweep.report import SETTLED, read_summary, write_curve, write_report
from hyperloom_common.jsonio import read_json_strict
from hyperloom_multibranch.contracts.gate import NotMeasured
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Blackboard
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, BranchProgress
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected, TestWithdrawn
from hyperloom_multibranch.blackboard.changes import same_recipe
from hyperloom_multibranch.blackboard.codec import decode
from hyperloom_multibranch.blackboard.knowledge.breakdown import FILENAME as BREAKDOWN_FILENAME
from hyperloom_multibranch.scheduling.budget import FinalWork
from hyperloom_multibranch.scheduling.explore import integration_row
from hyperloom_multibranch.scheduling.integrator import defects_beyond_gain
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.requests import SessionChecks, SessionTests, integration_test_request, together
from hyperloom_multibranch.scheduling.settings import RunSettings, Workload

logger = logging.getLogger(__name__)

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
    """The candidates and ``baseline_value``, the root's PRELUDE baseline of the primary metric, which each gain is
    over."""

    integration: FinalCandidate
    best_branch: FinalCandidate | None
    baseline_value: float

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
    progress: Sequence[BranchProgress],
    integration: BranchProgress,
    metric: str,
    repo: ExperimentRepo,
    baseline: BaselineRecord,
) -> FinalCandidates:
    """The head of ``integration`` and the specialist head with the highest recorded ``metric`` over every branch the
    blackboard knows, against the session's PRELUDE ``baseline``; its root commit is the integration head still when
    nothing merged.

    A branch with no commit offers no single-branch candidate, and neither does one whose head carries the recipe the
    integration head already carries: a merge writes the branch's work as a new commit, so the two differ by commit
    and are one recipe, which FINAL would otherwise validate twice. The next branch down then stands for the session's
    single-branch work instead.
    """

    head, root = integration.head_commit, baseline.root_commit
    offered = [_branch_candidate(row, head, metric) for row in progress]
    ranked = sorted((row for row in offered if row is not None), key=lambda candidate: -candidate.recorded_value)
    candidate = FinalCandidate(
        INTEGRATION_LABEL, INTEGRATION_BRANCH, head, integration.head_value.get(metric), head == root
    )
    best = await _best_of_its_own(ranked, head, repo)
    return FinalCandidates(candidate, best, baseline.values[metric])


async def session_candidates(blackboard: Blackboard, repo: ExperimentRepo, metric: str) -> FinalCandidates:
    """:func:`choose_candidates` over every branch ``blackboard`` holds progress of, against the session's root."""

    progress = await blackboard.progress()
    integration = integration_row(progress)
    if integration is None:
        raise RuntimeError("the blackboard has no integration branch after PRELUDE recorded its root")
    return await choose_candidates(progress, integration, metric, repo, await blackboard.baseline())


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
    ``gain_pct`` is its reproduction's gain over the root's PRELUDE baseline and ``reproduction_pct`` how far that
    reproduction measured from the commit's record, each ``None`` when none was judged; ``refused_by`` is what kept the
    reproduction from holding, and ``waived`` the speed refusals a root's reproduction was not held to.
    ``cache_states_differ`` is its reproduction judgement's (:attr:`~..contracts.judgement.Judgement.cache_states_differ`).
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
    reproduction_pct: float | None = None
    refused_by: tuple[str, ...] = ()
    waived: tuple[str, ...] = ()
    cache_states_differ: tuple[str, ...] = ()

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
            gain = "not judged" if self.reproduction_pct is None else f"{self.reproduction_pct:.1f}% against its record"
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
    ``refused_by`` is what kept it from holding and ``reproduction_pct`` how far it measured from the record.

    ``value`` is what it measured of the primary metric at the operating point that decides, which is the sweep's
    primary point: the candidate is ranked on it when the sweep gave nothing there. ``cache_states_differ`` is the
    judgement's flag, kept for the candidate's check.
    """

    reproduced: bool
    accuracy_passed: bool
    value: float | None
    reproduction_pct: float
    refused_by: tuple[str, ...]
    waived: tuple[str, ...]
    cache_states_differ: tuple[str, ...] = ()


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

        baseline = candidates.baseline_value
        checks = await asyncio.gather(*(self._validated(candidate, baseline) for candidate in candidates.each))
        choice = pick_final(tuple(checks), self._metric, self._workload.conc)
        if self._workload.final_concs == (self._workload.conc,):
            return replace(choice, reason=f"{choice.reason}; {PRIMARY_POINT_ONLY}")
        return choice

    async def planned_work(self, candidates: FinalCandidates, test_seconds: float) -> FinalWork:
        """What :meth:`select` would run to validate ``candidates``, ``parallel`` tests at once, one attempt of
        each, given ``test_seconds``, one test at the primary conc: for each candidate its sweep, a test per conc of
        :attr:`~hyperloom_multibranch.scheduling.settings.Workload.final_concs`, and its reproduction, and the
        regression runs still to make (:meth:`~hyperloom_multibranch.contracts.gate.RegressionCheck.final_case_seconds`),
        a run of another session's case as long as one test."""

        sweep = len(self._workload.final_concs) * test_seconds
        tests = [sweep, test_seconds] * len(candidates.each)
        commits = tuple(each.commit for each in candidates.each)
        tests.extend(await self._regression.final_case_seconds(commits, test_seconds))
        return FinalWork(sum(tests), max(tests), test_seconds, self._parallel)

    async def _validated(self, candidate: FinalCandidate, baseline: float) -> CandidateCheck:
        try:
            return await self._check(candidate, baseline)
        except TestWithdrawn as exc:
            return CandidateCheck(candidate, error=f"a validation test was withdrawn: {exc}")
        except RequestRejected as exc:
            return CandidateCheck(candidate, error=f"the toolbox refused a validation test: {exc}")
        except NotMeasured as exc:
            return CandidateCheck(candidate, error=f"the gate had nothing to judge: {exc}")

    async def _check(self, candidate: FinalCandidate, baseline: float) -> CandidateCheck:
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
            gain_pct=gain_pct(reproduction.value, baseline),
            reproduction_pct=reproduction.reproduction_pct,
            refused_by=reproduction.refused_by,
            waived=reproduction.waived,
            cache_states_differ=reproduction.cache_states_differ,
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
        differ = judgement.cache_states_differ
        return _Reproduction(not refused_by, accuracy, value, judgement.gain_pct, refused_by, waived, differ)


BASELINE, OPTIMIZED = "baseline", "optimized"
NO_OPTIMIZATION = "no_optimization"
NO_SWEEP_POINTS = "no_sweep_points"
_SUCCEEDED = "succeeded"


@dataclass(frozen=True)
class SweepPlan:
    """The commit FINAL chose and the points both arms run."""

    chosen: str
    points: SweepPoints

    def test_points(self, workload: Workload) -> int:
        """How many points one test of the sweep measures: one in an AgentX test, every point in an InferenceX one."""
        return 1 if workload.inferencex is None else len(self.points.concs)


def sweep_plan(layout: SessionLayout, settings: RunSettings) -> SweepPlan | str:
    """What the sweep of the closed session in ``layout`` measures, or why it measures nothing."""
    document = read_json_strict(layout.report, require_dict=True)
    choice = None if document["final"] is None else decode(FinalChoice, document["final"])
    chosen = None if choice is None else choice.chosen
    if chosen is None or chosen.root:
        return NO_OPTIMIZATION
    check = next(check for check in choice.checks if check.candidate == chosen)
    if not (check.gain_pct or 0.0) > 0:
        return NO_OPTIMIZATION
    from hyperloom.orchestrator.state.base_launch import frozen_base

    frozen = frozen_base(layout.toolbox)
    workload = settings.workload
    points = sweep_points(None if frozen is None else frozen.launch, workload.conc, settings.conc_sweep_concs)
    return NO_SWEEP_POINTS if points is None else SweepPlan(chosen.commit, points)


def sweep_unfinished(layout: SessionLayout, settings: RunSettings) -> bool:
    """Whether the session in ``layout`` closed after a FINAL that chose, asked for a conc sweep, and that sweep has
    not run its course; a session that ended before FINAL chose is finished first."""
    if not (settings.conc_sweep and layout.report.is_file()):
        return False
    if not (read_json_strict(layout.report, require_dict=True).get("final") or {}).get("chosen"):
        return False
    summary = read_summary(layout.reports)
    return summary is None or summary["status"] not in SETTLED


def _header(layout: SessionLayout, workload: Workload) -> dict[str, Any]:
    shape = workload.inferencex
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": layout.session_id,
        "benchmark_mode": "agentx" if shape is None else "synthetic",
        "isl": None if shape is None else shape.isl,
        "osl": None if shape is None else shape.osl,
        "tp": workload.tp,
    }


def _write(layout: SessionLayout, payload: dict[str, Any]) -> None:
    """Write the report into ``MB-reports/`` and carry it in the session breakdown beside it."""
    write_report(payload, layout.reports, breakdown=layout.reports / BREAKDOWN_FILENAME)


def write_skipped(layout: SessionLayout, workload: Workload, reason: str) -> dict[str, Any]:
    """The report of a sweep that measured nothing, written where a sweep's report goes."""
    payload = {
        **_header(layout, workload),
        "status": "skipped",
        "skip_reason": reason,
        "baseline": {"points": []},
        "optimized": {"points": []},
        "summary": {},
    }
    _write(layout, payload)
    logger.info("conc_sweep: skipped (%s)", reason)
    return payload


def points_of(arm: str, concs: Sequence[int], result: TestResult) -> list[dict[str, Any]]:
    """One report row per concurrency of ``concs``: the mean of every metric over the test's measure rounds there."""
    measured: defaultdict[int, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in result.rounds:
        if row.phase == MEASURE_PHASE and row.concurrency in concs:
            for key, value in row.metrics.items():
                measured[row.concurrency][key].append(value)
    rows = []
    for conc in concs:
        metrics = {key: statistics.fmean(values) for key, values in measured[conc].items()}
        ok = result.validity is Validity.SUCCEEDED and bool(metrics)
        error = None if ok else result.failure_reason or "the test measured nothing at this concurrency"
        status = _SUCCEEDED if ok else "failed"
        rows.append({"arm": arm, "conc": conc, "status": status, "request_id": result.request_id, "error": error})
        rows[-1].update(metrics)
    return rows


@dataclass
class ConcSweep:
    """One sweep's inputs and the points both arms have measured so far, by arm and concurrency."""

    plan: SweepPlan
    root: str
    layout: SessionLayout
    workload: Workload
    tests: SessionTests
    parallel: int
    per_point_sec: float
    estimated_sec: float = 0.0
    started_at: float = field(default_factory=time.time)
    measured: dict[str, dict[int, dict[str, Any]]] = field(default_factory=lambda: {BASELINE: {}, OPTIMIZED: {}})
    cancel_reason: str = ""

    def pending(self, arm: str) -> list[int]:
        done = self.measured[arm]
        order = ladder(self.plan.points.concs, self.workload.conc)
        return [conc for conc in order if done.get(conc, {}).get("status") != _SUCCEEDED]

    async def run(self, stop: Awaitable[int], grant_end_unix: float | None = None) -> dict[str, Any]:
        """Run both arms until they end or ``stop`` returns a signal, which takes their tests back."""
        self._resume()
        per_arm = [self.per_point_sec * len(self.pending(arm)) for arm in (BASELINE, OPTIMIZED)]
        self.estimated_sec = max(per_arm) if self.parallel > 1 else sum(per_arm)
        self._log_plan(grant_end_unix)
        arms = asyncio.gather(self._arm(BASELINE, self.root), self._arm(OPTIMIZED, self.plan.chosen))
        stopped = asyncio.ensure_future(stop)
        await asyncio.wait({arms, stopped}, return_when=asyncio.FIRST_COMPLETED)
        if stopped.done():
            self.cancel_reason = f"signal {stopped.result()}"
            await self.tests.withdraw_integration(self.cancel_reason)
            await arms
        else:
            stopped.cancel()
        payload = self.payload(final=True)
        _write(self.layout, payload)
        shape = self.workload.inferencex
        options = ConcCurveOptions(
            self.workload.model_basename,
            self.workload.gpu_type,
            self.workload.tp,
            0 if shape is None else shape.isl,
            0 if shape is None else shape.osl,
        )
        write_curve(payload, self.layout.reports, options)
        logger.info(
            "conc_sweep: done status=%s successful_pairs=%s", payload["status"], payload["summary"]["successful_pairs"]
        )
        return payload

    async def _arm(self, arm: str, commit: str) -> None:
        pending = self.pending(arm)
        groups = [(conc,) for conc in pending] if self.workload.inferencex is None else [tuple(sorted(pending))]
        for concs in (group for group in groups if group):
            request = integration_test_request(TestKind.SWEEP, Purpose.HEALTH, commit, concurrency=concs)
            try:
                result = await self.tests.result(request)
            except TestWithdrawn:
                return
            except RequestRejected as exc:
                self._record([{"arm": arm, "conc": conc, "status": "failed", "error": str(exc)} for conc in concs])
                return
            self._record(points_of(arm, concs, result))

    def _record(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.measured[row["arm"]][row["conc"]] = row
        _write(self.layout, self.payload())

    def _resume(self) -> None:
        """Keep the points an earlier run of the same sweep measured on the same commit of each arm, so only the rest
        are run."""
        earlier = read_summary(self.layout.reports)
        if not earlier or earlier.get("concs") != list(self.plan.points.concs):
            return
        for arm, commit in ((BASELINE, self.root), (OPTIMIZED, self.plan.chosen)):
            if earlier[arm].get("commit") != commit:
                continue
            for point in earlier[arm]["points"]:
                if point["status"] == _SUCCEEDED:
                    self.measured[arm][point["conc"]] = point

    def _log_plan(self, grant_end_unix: float | None) -> None:
        logger.info(
            "conc_sweep: plan source=%s concs=%s arms=baseline,optimized %s pending_points=%d estimated=%.1fh "
            "(each point priced as the session's longest root test of %.0fs)",
            self.plan.points.source,
            list(self.plan.points.concs),
            "side by side" if self.parallel > 1 else "one after the other",
            len(self.pending(BASELINE)) + len(self.pending(OPTIMIZED)),
            self.estimated_sec / 3600.0,
            self.per_point_sec,
        )
        if grant_end_unix is None:
            return
        left = grant_end_unix - time.time()
        logger.info("conc_sweep: the run's grant has %.1fh left", left / 3600.0)
        if self.estimated_sec > left:
            logger.warning("conc_sweep: the estimate outlasts the run's grant; the platform may end the sweep early")

    def payload(self, *, final: bool = False) -> dict[str, Any]:
        sections = {
            arm: {"commit": commit, "points": [self.measured[arm][conc] for conc in sorted(self.measured[arm])]}
            for arm, commit in ((BASELINE, self.root), (OPTIMIZED, self.plan.chosen))
        }
        comparison, summary = conc_pair_comparison(
            sections[BASELINE]["points"], sections[OPTIMIZED]["points"], metric_key=self.workload.metric
        )
        return {
            **_header(self.layout, self.workload),
            "status": self._status(summary) if final else "in_progress",
            "concs": list(self.plan.points.concs),
            "concs_source": self.plan.points.source,
            BASELINE: sections[BASELINE],
            OPTIMIZED: sections[OPTIMIZED],
            "comparison": comparison,
            "summary": summary,
            "estimated_sec": round(self.estimated_sec),
            "elapsed_sec": round(time.time() - self.started_at, 2),
            "cancel_reason": self.cancel_reason,
        }

    def _status(self, summary: dict[str, Any]) -> str:
        if self.cancel_reason:
            return "cancelled"
        if not (self.pending(BASELINE) or self.pending(OPTIMIZED)):
            return _SUCCEEDED
        return "partial" if summary["successful_pairs"] else "failed"


__all__ = [
    "BASELINE",
    "BELOW_REPRODUCTION_TOLERANCE",
    "MB_REPRODUCTION_TOLERANCE_PCT",
    "MB_SWEEP_DEGRADATION_PCT",
    "NONE_PASSED",
    "NO_OPTIMIZATION",
    "NO_SWEEP_POINTS",
    "OPTIMIZED",
    "PRIMARY_POINT_ONLY",
    "REPRODUCTION_POINT_ONLY",
    "CandidateCheck",
    "ConcSweep",
    "FinalCandidate",
    "FinalCandidates",
    "FinalChoice",
    "FinalSelector",
    "SweepPlan",
    "choose_candidates",
    "pick_final",
    "points_of",
    "session_candidates",
    "sweep_curve",
    "sweep_plan",
    "sweep_unfinished",
    "write_skipped",
]
