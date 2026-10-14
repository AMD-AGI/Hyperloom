# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FINAL: which heads are candidates, how a candidate is validated, and which one is picked."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.contracts.notebook.peers import (
    TestKind,
)
from hyperloom_multibranch.contracts.notebook.records import BranchProgress
from hyperloom_multibranch.contracts.scheduling import OperatingPoint, PriorityClass, SweepResult
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.final import (
    CandidateCheck,
    FinalCandidate,
    FinalCandidates,
    FinalSelector,
    choose_candidates,
    pick_final,
)
from hyperloom_multibranch.scheduling.roster import BranchEntry, BranchKind, BranchOrigin, BranchStop, StopCause
from hyperloom_multibranch.scheduling.settings import Workload
from hyperloom_multibranch.scheduling.tests.peer_fakes import MbFakeGate, MbFakeToolbox

METRIC = "output_throughput"
CONC = 64
WORKLOAD = Workload("org/model", "vllm", 8, 1024, 1024, CONC, "fp8")
CLOCK = Clock(now=lambda: 100.0)
INTEGRATION = FinalCandidate("integration", INTEGRATION_BRANCH, "i1", 1040.0)
BRANCH = FinalCandidate("best_branch", "b02", "c7", 1060.0)


def _entry(branch: str, stop: BranchStop | None = None) -> BranchEntry:
    return BranchEntry(branch, BranchKind.SPECIALIST, BranchOrigin.FIRST_BATCH, "c0", 1, 0.0, stop=stop)


def _row(branch: str, head: str, value: float | None, commits: int = 1) -> BranchProgress:
    values = {} if value is None else {METRIC: value}
    return BranchProgress(branch, True, "c0", head, 3, commits, 0.0, values)


def test_the_best_single_branch_is_chosen_over_every_branch_whatever_stopped_it() -> None:
    judged = BranchStop(StopCause.JUDGE, "no progress", 5.0)
    ended = BranchStop(StopCause.SESSION_END, "time_exhausted", 9.0)
    entries = (_entry("b01", judged), _entry("b02", ended), _entry("b03"), _entry("b04"), _entry("b05"))
    progress = (
        _row("b01", "c5", 1100.0),
        _row("b02", "c7", 1060.0),
        _row("b03", "i1", 1200.0),
        _row("b04", "c0", 1300.0, commits=0),
        _row("b05", "c9", None),
    )

    candidates = choose_candidates(entries, progress, "i1", {METRIC: 1040.0}, METRIC)

    assert candidates.integration == INTEGRATION
    assert candidates.best_branch == FinalCandidate("best_branch", "b01", "c5", 1100.0)


def test_without_a_committed_branch_only_the_integration_head_is_a_candidate() -> None:
    integration = BranchEntry(INTEGRATION_BRANCH, BranchKind.INTEGRATION, BranchOrigin.ROOT, "c0", 1, 0.0)
    rows = (_row(INTEGRATION_BRANCH, "i1", 1040.0), _row("b01", "c0", None, commits=0))

    candidates = choose_candidates((integration, _entry("b01")), rows, "i1", {}, METRIC)

    assert candidates.best_branch is None
    assert candidates.each == (FinalCandidate("integration", INTEGRATION_BRANCH, "i1", None),)


def _sweep(commit: str, curve: dict[int, float]) -> SweepResult:
    return SweepResult(commit, tuple(OperatingPoint(conc, {METRIC: value}) for conc, value in curve.items()), ())


def _check(candidate: FinalCandidate, curve: dict[int, float], accuracy_passed: bool = True) -> CandidateCheck:
    return CandidateCheck(candidate, _sweep(candidate.commit, curve), accuracy_passed, True, True)


def test_the_higher_primary_point_wins() -> None:
    checks = (_check(INTEGRATION, {32: 900.0, CONC: 1040.0}), _check(BRANCH, {32: 880.0, CONC: 1060.0}))

    choice = pick_final(checks, METRIC, CONC)

    assert choice.chosen == BRANCH
    assert [check.degraded_concs for check in choice.checks] == [(), ()]


def test_a_higher_primary_point_that_degrades_elsewhere_in_the_sweep_is_passed_over() -> None:
    checks = (_check(INTEGRATION, {32: 900.0, CONC: 1040.0}), _check(BRANCH, {32: 850.0, CONC: 1060.0}))

    choice = pick_final(checks, METRIC, CONC)

    assert choice.chosen == INTEGRATION
    assert choice.checks[1].degraded_concs == (32,)
    assert "best_branch is higher" in choice.reason


def test_when_both_degrade_somewhere_the_primary_point_decides() -> None:
    checks = (
        _check(INTEGRATION, {16: 500.0, 32: 900.0, CONC: 1040.0}),
        _check(BRANCH, {16: 600.0, 32: 850.0, CONC: 1060.0}),
    )

    choice = pick_final(checks, METRIC, CONC)

    assert choice.chosen == BRANCH
    assert [check.degraded_concs for check in choice.checks] == [(16,), (32,)]


def test_an_ineligible_candidate_is_never_chosen_and_vetoes_nothing() -> None:
    checks = (_check(INTEGRATION, {32: 900.0, CONC: 1040.0}), _check(BRANCH, {32: 2000.0, CONC: 1060.0}, False))

    choice = pick_final(checks, METRIC, CONC)

    assert choice.chosen == INTEGRATION
    assert choice.checks[0].degraded_concs == ()
    assert pick_final(checks[1:], METRIC, CONC).chosen is None


class Bench:
    """The toolbox and rule gate stand-ins FINAL validates on: sweeps from ``curves``, regressions and withdrawals by
    commit."""

    def __init__(
        self, tmp_path: Path, curves: dict[str, dict[int, float]], regressed: frozenset[str] = frozenset()
    ) -> None:
        notebook = tmp_path / "notebook"
        self.toolbox = MbFakeToolbox(LocalNotebook(notebook, Writer(Role.TOOLBOX, "toolbox"), clock=CLOCK.now), curves)
        self.gate = MbFakeGate(
            LocalNotebook(notebook, Writer(Role.GATE, "gate"), clock=CLOCK.now), self.toolbox, regressed=regressed
        )

    def selector(self, record: Path) -> FinalSelector:
        return FinalSelector(self.toolbox, self.gate, WORKLOAD, METRIC, record, CLOCK)


async def test_each_candidate_is_swept_accuracy_checked_and_reproduced_as_a_session_measurement(
    tmp_path: Path,
) -> None:
    bench = Bench(
        tmp_path, {"i1": {32: 900.0, CONC: 1040.0}, "c7": {32: 890.0, CONC: 1060.0}}, regressed=frozenset({"c7"})
    )
    record = tmp_path / "MB-final.json"

    choice = await bench.selector(record).select(FinalCandidates(INTEGRATION, BRANCH), deadline=200.0)

    assert choice.chosen == INTEGRATION
    assert [check.reproduced for check in choice.checks] == [True, False]
    assert sorted(bench.toolbox.swept) == [(INTEGRATION_BRANCH, "c7"), (INTEGRATION_BRANCH, "i1")]
    requests = sorted(bench.toolbox.submitted, key=lambda request: (request.parent_commit, request.kind.value))
    assert [(request.parent_commit, request.kind) for request in requests] == [
        ("c7", TestKind.ACCURACY),
        ("c7", TestKind.TEST),
        ("i1", TestKind.ACCURACY),
        ("i1", TestKind.TEST),
    ]
    assert {request.priority for request in requests} == {PriorityClass.SESSION_MEASUREMENT.value}
    assert {request.branch for request in requests} == {INTEGRATION_BRANCH}
    assert {request.intent_fingerprint for request in requests} == {"final:accuracy", "final:repro"}
    written = json.loads(record.read_text())
    assert written["choice"]["chosen"]["commit"] == "i1"
    assert (written["primary_conc"], written["metric"]) == (CONC, METRIC)


async def test_a_withdrawn_test_a_missing_primary_point_or_a_failed_accuracy_disqualifies(tmp_path: Path) -> None:
    bench = Bench(tmp_path, {"i1": {32: 900.0}, "c7": {CONC: 1060.0}})
    bench.toolbox.withdrawing = frozenset({"c7"})

    choice = await bench.selector(tmp_path / "f.json").select(FinalCandidates(INTEGRATION, BRANCH), 200.0)

    assert choice.chosen is None
    assert "no output_throughput at the primary conc" in choice.checks[0].error
    assert "withdrawn" in choice.checks[1].error

    bench.gate.inaccurate = frozenset({"i1"})
    bench.toolbox.sweeps = {"i1": {CONC: 1040.0}}
    alone = await bench.selector(tmp_path / "g.json").select(FinalCandidates(INTEGRATION, None), 200.0)
    assert alone.chosen is None and not alone.checks[0].accuracy_passed


class StalledToolbox(MbFakeToolbox):
    """A toolbox whose sweep never finishes; a failure double, not a placeholder."""

    async def sweep(self, branch: str, commit: str) -> SweepResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_a_candidate_unfinished_at_the_deadline_is_not_eligible(tmp_path: Path) -> None:
    bench = Bench(tmp_path, {})
    bench.toolbox = bench.gate.toolbox = StalledToolbox(bench.toolbox.notebook)

    choice = await bench.selector(tmp_path / "f.json").select(FinalCandidates(INTEGRATION, None), deadline=100.0)

    assert choice.chosen is None
    assert "deadline" in choice.checks[0].error


class CountingStalledToolbox(StalledToolbox):
    """The stalled toolbox, counting the sweeps cancelled while they wait; a failure double, not a placeholder."""

    cancelled_sweeps = 0

    async def sweep(self, branch: str, commit: str) -> SweepResult:
        try:
            return await super().sweep(branch, commit)
        except asyncio.CancelledError:
            self.cancelled_sweeps += 1
            raise


async def test_a_withdrawn_test_cancels_the_candidates_other_measurements(tmp_path: Path) -> None:
    bench = Bench(tmp_path, {})
    toolbox = CountingStalledToolbox(bench.toolbox.notebook, withdrawing=frozenset({"i1"}))
    bench.toolbox = bench.gate.toolbox = toolbox

    choice = await bench.selector(tmp_path / "f.json").select(FinalCandidates(INTEGRATION, None), deadline=200.0)

    assert "withdrawn" in choice.checks[0].error
    assert toolbox.cancelled_sweeps == 1
