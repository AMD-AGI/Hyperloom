# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FINAL: which commit the session hands on, the integration head or the best single branch's head.

The candidates are the integration head and the specialist head with the highest recorded value, whatever stopped
its branch. Each is validated through the toolbox as a session measurement: a full concurrency sweep, an accuracy
test, and a cold-start reproduction judged by the rule gate against the commit's own recorded value. The primary
operating point (the workload's conc) decides; a candidate whose other sweep points fall more than
:data:`MB_SWEEP_DEGRADATION_PCT` below the other candidate's at the same conc is passed over while the other one
does not degrade in turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom.common.io import atomic_write_json
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.peers import TestKind, Verdict
from hyperloom_multibranch.contracts.notebook.records import BranchProgress
from hyperloom_multibranch.contracts.scheduling import PriorityClass, RequestWithdrawn, RuleGate, SweepResult, Toolbox
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_regression_passes
from hyperloom_multibranch.scheduling.requests import integration_test_request
from hyperloom_multibranch.scheduling.roster import BranchEntry, BranchKind
from hyperloom_multibranch.scheduling.settings import Workload

#: How far below the other candidate a non-primary sweep point may fall before the candidate is passed over.
MB_SWEEP_DEGRADATION_PCT = 5.0

INTEGRATION_LABEL = "integration"
BEST_BRANCH_LABEL = "best_branch"
ACCURACY_INTENT = "final:accuracy"
REPRO_INTENT = "final:repro"


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
    """What validating one candidate found; ``degraded_concs`` are the sweep points that fell below the other's."""

    candidate: FinalCandidate
    sweep: SweepResult | None = None
    accuracy_passed: bool = False
    reproduced: bool = False
    regression_passed: bool = False
    degraded_concs: tuple[int, ...] = ()
    error: str = ""

    @property
    def eligible(self) -> bool:
        passed = self.accuracy_passed and self.reproduced and self.regression_passed
        return passed and self.sweep is not None and not self.error

    def curve(self, metric: str) -> dict[int, float]:
        """The sweep's ``metric`` per conc; empty without a sweep."""

        points = self.sweep.points if self.sweep is not None else ()
        return {point.conc: point.values[metric] for point in points if metric in point.values}


@dataclass(frozen=True)
class FinalChoice:
    chosen: FinalCandidate | None
    checks: tuple[CandidateCheck, ...]
    reason: str


def _degraded(mine: Mapping[int, float], other: Mapping[int, float], primary_conc: int) -> set[int]:
    floor = 1.0 - MB_SWEEP_DEGRADATION_PCT / 100.0
    return {conc for conc, value in mine.items() if conc != primary_conc and value < other.get(conc, 0.0) * floor}


def _with_degradation(checks: tuple[CandidateCheck, ...], metric: str, conc: int) -> tuple[CandidateCheck, ...]:
    curves = {check.candidate.label: check.curve(metric) for check in checks if check.eligible}

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

    marked = _with_degradation(checks, metric, conc)
    ranked = sorted((check for check in marked if check.eligible), key=lambda check: -check.curve(metric)[conc])
    if not ranked:
        return FinalChoice(None, marked, "no candidate passed the final validation")
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
    return FinalChoice(chosen.candidate, marked, reason)


class FinalSelector:
    """Validates the candidates through the toolbox and the rule gate and records the choice in ``record``."""

    def __init__(
        self, toolbox: Toolbox, gate: RuleGate, workload: Workload, metric: str, record: Path, clock: Clock
    ) -> None:
        self._toolbox = toolbox
        self._gate = gate
        self._workload = workload
        self._metric = metric
        self._record = record
        self._clock = clock

    async def select(self, candidates: FinalCandidates, deadline: float) -> FinalChoice:
        """Validate every candidate concurrently by ``deadline`` (clock seconds) and pick one."""

        checks = await asyncio.gather(*(self._bounded(candidate, deadline) for candidate in candidates.each))
        choice = pick_final(tuple(checks), self._metric, self._workload.conc)
        document = {
            "at": self._clock.now(),
            "deadline": deadline,
            "metric": self._metric,
            "primary_conc": self._workload.conc,
            "degradation_pct": MB_SWEEP_DEGRADATION_PCT,
            "candidates": encode(candidates.each),
            "choice": encode(choice),
        }
        atomic_write_json(self._record, document, indent=2, sort_keys=True, make_parents=True)
        return choice

    async def _bounded(self, candidate: FinalCandidate, deadline: float) -> CandidateCheck:
        try:
            return await self._clock.wait_for(self._check(candidate), deadline - self._clock.now())
        except asyncio.TimeoutError:
            return CandidateCheck(candidate, error="the FINAL deadline passed before validation finished")
        except RequestWithdrawn as exc:
            return CandidateCheck(candidate, error=f"a validation test was withdrawn: {exc}")

    async def _check(self, candidate: FinalCandidate) -> CandidateCheck:
        sweep, accuracy_passed, reproduced = await asyncio.gather(
            self._toolbox.sweep(INTEGRATION_BRANCH, candidate.commit),
            self._accuracy_passes(candidate.commit),
            self._reproduces(candidate.commit),
        )
        check = CandidateCheck(
            candidate, sweep, accuracy_passed, reproduced, mb_fake_regression_passes(candidate.commit)
        )
        if self._workload.conc not in check.curve(self._metric):
            return replace(check, error=f"the sweep has no {self._metric} at the primary conc {self._workload.conc}")
        return check

    async def _measure(self, kind: TestKind, commit: str, intent: str) -> str:
        purpose = f"final {kind.value} of {commit}"
        request = integration_test_request(kind, commit, intent, purpose, PriorityClass.SESSION_MEASUREMENT)
        request_id = await self._toolbox.submit(request)
        await self._toolbox.result(request_id)
        return request_id

    async def _accuracy_passes(self, commit: str) -> bool:
        return await self._gate.accuracy_passes(await self._measure(TestKind.ACCURACY, commit, ACCURACY_INTENT))

    async def _reproduces(self, commit: str) -> bool:
        judgement = await self._gate.compare(await self._measure(TestKind.TEST, commit, REPRO_INTENT))
        return judgement.verdict is not Verdict.REGRESSED


__all__ = [
    "MB_SWEEP_DEGRADATION_PCT",
    "CandidateCheck",
    "FinalCandidate",
    "FinalCandidates",
    "FinalChoice",
    "FinalSelector",
    "choose_candidates",
    "pick_final",
]
