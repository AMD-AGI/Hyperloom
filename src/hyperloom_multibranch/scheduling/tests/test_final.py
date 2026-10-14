# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FINAL: which heads are candidates, how a candidate is validated, and which one is picked."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import BranchProgress
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.scheduling.final import (
    CandidateCheck,
    FinalCandidate,
    FinalCandidates,
    FinalSelector,
    choose_candidates,
    pick_final,
    sweep_concs,
)
from hyperloom_multibranch.scheduling.roster import BranchEntry, BranchKind, BranchOrigin, BranchStop, StopCause
from hyperloom_multibranch.scheduling.settings import Workload
from hyperloom_multibranch.scheduling.requests import SessionTests
from hyperloom_multibranch.scheduling.tests.peer_fakes import Curves, MbFakeGate, RecordingToolbox, VirtualClock
from hyperloom_multibranch.toolbox.fake_model import Fate

METRIC = "output_throughput"
CONC = 64
WORKLOAD = Workload("org/model", "vllm", 8, 1024, 1024, CONC, "fp8")
START = 100.0
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


def _check(candidate: FinalCandidate, curve: dict[int, float], accuracy_passed: bool = True) -> CandidateCheck:
    return CandidateCheck(candidate, curve, accuracy_passed, True, True)


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


def test_the_sweep_brackets_the_primary_conc() -> None:
    assert sweep_concs(CONC) == (32, CONC, 128)
    assert sweep_concs(1) == (1, 2)


class Bench:
    """The toolbox's fake and the rule gate stand-in FINAL validates on, on a virtual clock: each commit measures its
    value in ``curves`` at a conc, regressions are by commit."""

    def __init__(self, tmp_path: Path, curves: Curves, regressed: frozenset[str] = frozenset()) -> None:
        self.time = VirtualClock(START)
        notebook = tmp_path / "notebook"
        self.toolbox = RecordingToolbox(
            LocalNotebook(notebook, Writer(Role.TOOLBOX, "toolbox"), clock=self.time.now), curves
        )
        scheduler = LocalNotebook(notebook, Writer(Role.SCHEDULER, "scheduler"), clock=self.time.now)
        self.tests = SessionTests(self.toolbox, scheduler, self.time.clock)
        self.gate = MbFakeGate(
            LocalNotebook(notebook, Writer(Role.GATE, "gate"), clock=self.time.now), self.tests, regressed=regressed
        )

    def selector(self, record: Path) -> FinalSelector:
        return FinalSelector(self.tests, self.gate, WORKLOAD, METRIC, record, self.time.clock)

    def fail(self, commit: str, fate: Fate, conc: int | None = None) -> None:
        """Every test of ``commit`` (at ``conc`` when given) ends in ``fate``."""

        def matches(request: object) -> bool:
            at = getattr(request, "concurrency", ())
            return getattr(request, "parent_commit", "") == commit and (conc is None or at == (conc,))

        self.toolbox.script(matches, [fate] * 4)


async def test_each_candidate_is_swept_accuracy_checked_and_reproduced_as_a_session_measurement(
    tmp_path: Path,
) -> None:
    bench = Bench(
        tmp_path, {"i1": {32: 900.0, CONC: 1040.0}, "c7": {32: 890.0, CONC: 1060.0}}, regressed=frozenset({"c7"})
    )
    record = tmp_path / "MB-final.json"

    choice = await bench.selector(record).select(FinalCandidates(INTEGRATION, BRANCH), deadline=START + 3600.0)

    assert choice.chosen == INTEGRATION
    assert [check.reproduced for check in choice.checks] == [True, False]
    assert dict(choice.checks[0].curve) == {32: 900.0, CONC: 1040.0, 128: 1000.0}
    requests = sorted(
        bench.toolbox.requests.values(),
        key=lambda request: (request.parent_commit, request.kind.value, request.concurrency),
    )
    assert [(request.parent_commit, request.kind, request.concurrency) for request in requests] == [
        (commit, kind, concurrency)
        for commit in ("c7", "i1")
        for kind, concurrency in (
            (TestKind.ACCURACY, ()),
            (TestKind.TEST, ()),
            (TestKind.TEST, (32,)),
            (TestKind.TEST, (CONC,)),
            (TestKind.TEST, (128,)),
        )
    ]
    assert {(request.branch_id, request.purpose) for request in requests} == {(INTEGRATION_BRANCH, Purpose.FINAL)}
    assert {(request.intent_fingerprint, request.priority) for request in requests} == {(None, None)}
    assert all(request.accuracy == (request.kind is TestKind.ACCURACY) for request in requests)
    written = json.loads(record.read_text())
    assert written["choice"]["chosen"]["commit"] == "i1"
    assert (written["primary_conc"], written["metric"]) == (CONC, METRIC)


async def test_a_withdrawn_test_a_missing_primary_point_or_a_failed_accuracy_disqualifies(tmp_path: Path) -> None:
    bench = Bench(tmp_path, {"i1": {32: 900.0}, "c7": {CONC: 1060.0}})
    bench.fail("c7", Fate.MEASUREMENT)
    bench.fail("i1", Fate.CANDIDATE, CONC)

    choice = await bench.selector(tmp_path / "f.json").select(FinalCandidates(INTEGRATION, BRANCH), START + 3600.0)

    assert choice.chosen is None
    assert "no output_throughput at the primary conc" in choice.checks[0].error
    assert "withdrawn" in choice.checks[1].error

    bench = Bench(tmp_path / "again", {"i1": {CONC: 1040.0}})
    bench.gate.inaccurate = frozenset({"i1"})
    alone = await bench.selector(tmp_path / "g.json").select(FinalCandidates(INTEGRATION, None), START + 3600.0)
    assert alone.chosen is None and not alone.checks[0].accuracy_passed


async def test_a_candidate_unfinished_at_the_deadline_is_not_eligible_and_its_tests_are_taken_back(
    tmp_path: Path,
) -> None:
    bench = Bench(tmp_path, {})
    bench.toolbox.executor.gate.clear()

    choice = await bench.selector(tmp_path / "f.json").select(FinalCandidates(INTEGRATION, None), START + 600.0)

    assert choice.chosen is None
    assert "deadline" in choice.checks[0].error
    assert bench.toolbox.withdrawn.keys() == bench.toolbox.requests.keys() and bench.toolbox.requests


async def test_a_withdrawn_test_cancels_the_candidates_other_measurements(tmp_path: Path) -> None:
    bench = Bench(tmp_path, {})
    bench.toolbox.executor.gate.clear()
    selecting = asyncio.ensure_future(
        bench.selector(tmp_path / "f.json").select(FinalCandidates(INTEGRATION, None), START + 3600.0)
    )
    while len(bench.toolbox.requests) < len(sweep_concs(CONC)) + 2:
        await asyncio.sleep(0)
    [accuracy] = [rid for rid, request in bench.toolbox.requests.items() if request.kind is TestKind.ACCURACY]

    await bench.toolbox.cancel(accuracy)
    choice = await selecting

    assert "withdrawn" in choice.checks[0].error
    assert bench.toolbox.withdrawn.keys() == bench.toolbox.requests.keys()
