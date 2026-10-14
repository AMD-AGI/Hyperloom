# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FINAL: which commit the session hands on, the integration head or the best single branch's head.

The candidates are the integration head and the specialist head with the highest recorded value, whatever stopped
its branch. Each is validated through the toolbox as a session measurement: a concurrency sweep around the workload's
conc (one test per conc, :data:`SWEEP_FACTORS`), an accuracy test held to the accuracy gate, and a cold-start
reproduction judged by the rule against the commit's own recorded value. The primary operating point (the workload's conc) decides; a candidate
whose other sweep points fall more than :data:`MB_SWEEP_DEGRADATION_PCT` below the other candidate's at the same conc
is passed over while the other one does not degrade in turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import BranchProgress, TestRecord
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_regression_passes
from hyperloom_multibranch.scheduling.requests import SessionTests, TestWithdrawn, integration_test_request
from hyperloom_multibranch.scheduling.roster import BranchEntry, BranchKind
from hyperloom_multibranch.scheduling.rule_judge import MbFakeRuleJudge, NotMeasured, formal_mean
from hyperloom_multibranch.scheduling.settings import Workload

#: How far below the other candidate a non-primary sweep point may fall before the candidate is passed over.
MB_SWEEP_DEGRADATION_PCT = 5.0

#: The sweep's concs as multiples of the workload's conc; the primary operating point is one of them.
SWEEP_FACTORS = (0.5, 1.0, 2.0)

INTEGRATION_LABEL = "integration"
BEST_BRANCH_LABEL = "best_branch"


def sweep_concs(conc: int) -> tuple[int, ...]:
    """The concs FINAL sweeps around ``conc``, ascending."""

    return tuple(sorted({max(round(conc * factor), 1) for factor in SWEEP_FACTORS}))


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


def _branch_candidate(
    entry: BranchEntry, row: BranchProgress | None, integration_head: str, metric: str
) -> FinalCandidate | None:
    if entry.kind is not BranchKind.SPECIALIST or row is None or not row.commits:
        return None
    value = row.head_value.get(metric)
    if value is None or row.head_commit == integration_head:
        return None
    return FinalCandidate(BEST_BRANCH_LABEL, entry.branch, row.head_commit, value)


def choose_candidates(
    roster_entries: Sequence[BranchEntry],
    progress: Sequence[BranchProgress],
    integration_head: str,
    integration_value: Mapping[str, float],
    metric: str,
) -> FinalCandidates:
    """The integration head and the specialist head with the highest recorded ``metric`` over every roster branch.

    A branch with no commit, or whose head is the integration head, offers no single-branch candidate.
    """

    rows = {row.branch: row for row in progress}
    offered = [_branch_candidate(entry, rows.get(entry.branch), integration_head, metric) for entry in roster_entries]
    ranked = [candidate for candidate in offered if candidate is not None]
    best = max(ranked, key=lambda candidate: candidate.recorded_value, default=None)
    integration = FinalCandidate(INTEGRATION_LABEL, INTEGRATION_BRANCH, integration_head, integration_value.get(metric))
    return FinalCandidates(integration, best)


@dataclass(frozen=True)
class CandidateCheck:
    """What validating one candidate found.

    ``curve`` is the sweep's primary metric per conc, without the concs it measured nothing at; ``degraded_concs``
    are the sweep points that fell below the other candidate's.
    """

    candidate: FinalCandidate
    curve: Mapping[int, float] = field(default_factory=dict)
    accuracy_passed: bool = False
    reproduced: bool = False
    regression_passed: bool = False
    degraded_concs: tuple[int, ...] = ()
    error: str = ""

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
    return FinalChoice(chosen.candidate, marked, reason, metric, conc)


async def _together(*measurements: asyncio.Future[Any]) -> None:
    """Await every measurement; once one fails, the others are cancelled, which takes their tests back, and its error
    is raised. Each is cancelled once, so the take-back is never cut short by a second cancel."""

    try:
        await asyncio.wait(measurements, return_when=asyncio.FIRST_EXCEPTION)
    finally:
        for measurement in measurements:
            measurement.cancel()
        await asyncio.gather(*measurements, return_exceptions=True)
    failures = [measurement.exception() for measurement in measurements if not measurement.cancelled()]
    failure = next((error for error in failures if error is not None), None)
    if failure is not None:
        raise failure


class FinalSelector:
    """Validates the candidates through the toolbox and the rule."""

    def __init__(
        self, tests: SessionTests, judge: MbFakeRuleJudge, workload: Workload, metric: str, clock: Clock
    ) -> None:
        self._tests = tests
        self._judge = judge
        self._workload = workload
        self._metric = metric
        self._clock = clock

    async def select(self, candidates: FinalCandidates, deadline: float) -> FinalChoice:
        """Validate every candidate concurrently by ``deadline`` (clock seconds) and pick one."""

        checks = await asyncio.gather(*(self._bounded(candidate, deadline) for candidate in candidates.each))
        return pick_final(tuple(checks), self._metric, self._workload.conc)

    async def _bounded(self, candidate: FinalCandidate, deadline: float) -> CandidateCheck:
        try:
            return await self._clock.wait_for(self._check(candidate), deadline - self._clock.now())
        except asyncio.TimeoutError:
            return CandidateCheck(candidate, error="the FINAL deadline passed before validation finished")
        except TestWithdrawn as exc:
            return CandidateCheck(candidate, error=f"a validation test was withdrawn: {exc}")
        except RequestRejected as exc:
            return CandidateCheck(candidate, error=f"the toolbox refused a validation test: {exc}")
        except NotMeasured as exc:
            return CandidateCheck(candidate, error=f"the reproduction measured nothing: {exc}")

    async def _check(self, candidate: FinalCandidate) -> CandidateCheck:
        sweep = asyncio.ensure_future(self._sweep(candidate.commit))
        accuracy = asyncio.ensure_future(self._accuracy_passes(candidate.commit))
        repro = asyncio.ensure_future(self._reproduces(candidate.commit))
        await _together(sweep, accuracy, repro)
        check = CandidateCheck(
            candidate, sweep.result(), accuracy.result(), repro.result(), mb_fake_regression_passes(candidate.commit)
        )
        if self._workload.conc not in check.curve:
            return replace(check, error=f"the sweep has no {self._metric} at the primary conc {self._workload.conc}")
        return check

    async def _sweep(self, commit: str) -> dict[int, float]:
        points = {
            conc: asyncio.ensure_future(self._measure(TestKind.TEST, commit, concurrency=(conc,)))
            for conc in sweep_concs(self._workload.conc)
        }
        await _together(*points.values())
        values = {conc: formal_mean(point.result(), self._metric) for conc, point in points.items()}
        return {conc: value for conc, value in values.items() if value is not None}

    async def _measure(self, kind: TestKind, commit: str, concurrency: tuple[int, ...] = ()) -> TestRecord:
        accuracy = kind is TestKind.ACCURACY
        request = integration_test_request(kind, Purpose.FINAL, commit, accuracy=accuracy, concurrency=concurrency)
        return await self._tests.run(request)

    async def _accuracy_passes(self, commit: str) -> bool:
        return await self._judge.accuracy_passes(await self._measure(TestKind.ACCURACY, commit))

    async def _reproduces(self, commit: str) -> bool:
        return await self._judge.reproduces(await self._measure(TestKind.TEST, commit))


__all__ = [
    "MB_SWEEP_DEGRADATION_PCT",
    "SWEEP_FACTORS",
    "CandidateCheck",
    "FinalCandidate",
    "FinalCandidates",
    "FinalChoice",
    "FinalSelector",
    "choose_candidates",
    "pick_final",
    "sweep_concs",
]
