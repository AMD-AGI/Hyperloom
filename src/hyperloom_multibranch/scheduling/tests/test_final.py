# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FINAL: which heads are candidates, how a candidate is validated, and which one is picked."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.common.perf_metric import GRADED_INTVTY_P50

from hyperloom.contracts.measurement import (
    FailureClass,
    Purpose,
    RoundValue,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom.contracts.recipe import Recipe, RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import BranchProgress, LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.inferencex import DROPPED_REQUESTS
from hyperloom_multibranch.rulegate.rules import InferenceXRule
from hyperloom_multibranch.rulegate.service import BASELINE_TESTS, GateService, Polling
from hyperloom_multibranch.scheduling.final import (
    CandidateCheck,
    FinalCandidate,
    FinalCandidates,
    PRIMARY_POINT_ONLY,
    FinalSelector,
    choose_candidates,
    pick_final,
    sweep_curve,
)
from hyperloom_multibranch.scheduling.roster import BranchEntry, BranchKind, BranchOrigin, BranchStop, StopCause
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, Workload, sweep_concs
from hyperloom_multibranch.scheduling.requests import SessionChecks, SessionTests, integration_test_request
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    BASE_IMAGE_DIGEST,
    BASELINE_VALUE,
    GATE_POLL_SECONDS,
    MAIN_POINT,
    Curves,
    FakeRegressionCheck,
    NoForks,
    RecordingToolbox,
    VirtualClock,
    judged,
    recipe_effect,
    session_gate,
    tagged,
)
from hyperloom_multibranch.toolbox.fake_model import Fate

METRIC = "output_throughput"
CONC = 64
WORKLOAD = Workload("org/model", "vllm", 8, CONC, "fp8", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1024, 1024))
START = 100.0
#: Far longer than FINAL's reserve of any budget the tests set.
VALIDATION_SECONDS = 24 * 3600.0
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


def test_a_sweep_is_read_per_conc_from_its_measured_rounds() -> None:
    rounds = (
        RoundValue("warmup", 32, {METRIC: 1.0}),
        RoundValue("measure", 32, {METRIC: 900.0}),
        RoundValue("measure", 32, {METRIC: 910.0}),
        RoundValue("measure", CONC, {METRIC: 1040.0}),
        RoundValue("measure", 128, {"ttft": 3.0}),
        RoundValue("measure", None, {METRIC: 5.0}),
    )
    assert sweep_curve(TestResult("r1", Validity.SUCCEEDED, rounds=rounds), METRIC) == {32: 905.0, CONC: 1040.0}
    assert sweep_curve(TestResult("r2", Validity.SUCCEEDED), METRIC) == {}


class RefusingAccuracyGate(GateService):
    """The rule gate's session service, whose accuracy gate refuses every test; a failure double, not a placeholder."""

    async def accuracy_passes(self, request_id: str) -> bool:
        return False


class DroppingRequestsGate(GateService):
    """The rule gate's session service, which blocks every cold start for requests it dropped on both runs; a failure
    double, not a placeholder."""

    async def compare(self, request_id: str) -> Judgement:
        return Judgement(Verdict.BLOCKED, 0.0, 1.0, (request_id,), {}, True, (DROPPED_REQUESTS,))


class Bench:
    """FINAL's toolbox fake, git repository and rule gate, on a virtual clock and an anchored notebook.

    ``i1`` (recorded at 1040) is committed on ``b01`` and ``c7`` (recorded at 1060) on ``b02``; what a test of either
    measures is its curve in ``curves``.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.time = VirtualClock(START)
        self.commits: dict[str, str] = {}
        self.regression = FakeRegressionCheck()

    def handle(self, role: Role) -> LocalNotebook:
        return LocalNotebook(self.tmp_path / "notebook", Writer(role, role.value), clock=self.time.now)

    async def build(self, curves: Curves) -> Bench:
        self.repo = GitExperimentRepo(self.tmp_path / "repo", LocalBlobStore(self.tmp_path / "blobs"))
        root = await self.repo.init_root(Recipe(BASE_IMAGE_DIGEST))
        scheduler = self.handle(Role.SCHEDULER)
        await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root))
        self.toolbox = RecordingToolbox(self.handle(Role.TOOLBOX), recipe_effect(self.tmp_path / "repo", curves))
        self.tests = SessionTests(self.toolbox, self.time.clock)
        self.gate = session_gate(self.handle(Role.GATE), self.toolbox)
        baseline = integration_test_request(TestKind.TEST, Purpose.BASELINE, root)
        await self.gate.establish_baseline(root, [await self.tests.submit(baseline) for _ in range(BASELINE_TESTS)])
        settler = RuleGate(self.handle(Role.GATE), self.repo, NoForks(), self.toolbox)
        for branch, tag, value in (("b01", "i1", 1040.0), ("b02", "c7", 1060.0)):
            await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit=root, reason="first"))
            await self.repo.create_branch(branch, root)
            self.commits[tag] = await self._commit(settler, branch, tag, value)
        return self

    async def _commit(self, settler: RuleGate, branch: str, tag: str, value: float) -> str:
        """``tag``'s change measured on ``branch`` and settled as a gain recorded at ``value``, as the specialist
        runtime settles a confirmed one; returns the commit it made."""

        parent = await self.repo.head(branch)
        test = TestRequest(TestKind.TEST, Purpose.EXPLORE, branch, parent, tagged(tag))
        measured = await self.tests.result(test)
        request = (await self.handle(Role.OBSERVER).record(measured.request_id)).request
        gain = (value / BASELINE_VALUE - 1.0) * 100.0
        await settler.settle(request, judged(Verdict.IMPROVED, gain, measured.request_id, value))
        return await self.repo.head(branch)

    def candidates(self, both: bool = True) -> FinalCandidates:
        integration = replace(INTEGRATION, commit=self.commits["i1"])
        return FinalCandidates(integration, replace(BRANCH, commit=self.commits["c7"]) if both else None)

    def selector(self, workload: Workload = WORKLOAD) -> FinalSelector:
        return FinalSelector(SessionChecks(self.tests, self.gate, self.regression), workload)

    def fail(self, tag: str, fate: Fate, kind: TestKind | None = None) -> None:
        """Every FINAL test of ``tag``'s commit (of ``kind`` when given) ends in ``fate``."""

        commit = self.commits[tag]

        def matches(request: TestRequest) -> bool:
            final = request.purpose is Purpose.FINAL and request.parent_commit == commit
            return final and kind in (None, request.kind)

        self.toolbox.script(matches, [fate] * 4)

    def final_tests(self) -> dict[str, TestRequest]:
        return {rid: test for rid, test in self.toolbox.requests.items() if test.purpose is Purpose.FINAL}


async def test_each_candidate_is_swept_and_reproduced_with_accuracy_as_a_session_measurement(
    tmp_path: Path,
) -> None:
    curves = {
        frozenset({"i1"}): {MAIN_POINT: 1040.0, CONC: 1040.0},
        frozenset({"c7"}): {MAIN_POINT: 1000.0, CONC: 1060.0},
    }
    bench = await Bench(tmp_path).build(curves)
    choice = await bench.selector().select(bench.candidates())

    assert choice.chosen == bench.candidates().integration
    assert [check.reproduced for check in choice.checks] == [True, False]
    assert dict(choice.checks[0].curve) == {32: 1040.0, CONC: 1040.0, 128: 1040.0}
    expected = sorted(
        (kind.value, concurrency) for kind, concurrency in ((TestKind.TEST, ()), (TestKind.SWEEP, sweep_concs(CONC)))
    )
    tests = bench.final_tests().values()
    for commit in bench.commits.values():
        mine = sorted((test.kind.value, test.concurrency) for test in tests if test.parent_commit == commit)
        assert mine == expected
    assert {(test.branch_id, test.intent_fingerprint, test.priority) for test in tests} == {
        (INTEGRATION_BRANCH, None, None)
    }
    assert all(test.accuracy for test in tests) and all(check.accuracy_passed for check in choice.checks)
    assert (choice.metric, choice.primary_conc) == (METRIC, CONC)


async def test_an_agentx_candidate_is_measured_at_the_primary_operating_point_alone(tmp_path: Path) -> None:
    curves = {
        frozenset({"i1"}): {MAIN_POINT: 1040.0, CONC: 1040.0},
        frozenset({"c7"}): {MAIN_POINT: 1000.0, CONC: 1060.0},
    }
    bench = await Bench(tmp_path).build(curves)

    choice = await bench.selector(replace(WORKLOAD, inferencex=None)).select(bench.candidates())

    sweeps = [test.concurrency for test in bench.final_tests().values() if test.kind is TestKind.SWEEP]
    assert sweeps == [(CONC,), (CONC,)]
    assert dict(choice.checks[0].curve) == {CONC: pytest.approx(52.0)}
    assert (choice.chosen, choice.metric) == (bench.candidates().integration, GRADED_INTVTY_P50)
    assert choice.reason == f"the higher {GRADED_INTVTY_P50} at conc {CONC}; {PRIMARY_POINT_ONLY}"


async def test_an_unmeasured_test_a_failed_sweep_or_a_failed_accuracy_disqualifies(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).build({frozenset({"i1"}): {MAIN_POINT: 1040.0}})
    bench.fail("c7", Fate.MEASUREMENT)
    bench.fail("i1", Fate.CANDIDATE, TestKind.SWEEP)

    choice = await bench.selector().select(bench.candidates())

    assert choice.chosen is None
    assert choice.checks[0].error == "the sweep ended candidate_failed (serve_flag)"
    assert "measurement_failed" in choice.checks[1].error

    bench = await Bench(tmp_path / "again").build({frozenset({"i1"}): {MAIN_POINT: 1040.0, CONC: 1040.0}})
    gate = bench.handle(Role.GATE)
    rule = InferenceXRule(gate, CONC)
    bench.gate = RefusingAccuracyGate(gate, bench.toolbox, rule, bench.toolbox, polling=Polling(GATE_POLL_SECONDS))
    alone = await bench.selector().select(bench.candidates(both=False))
    assert alone.chosen is None and not alone.checks[0].accuracy_passed and alone.checks[0].reproduced


async def test_a_candidate_the_regression_set_fails_on_its_own_commit_is_not_eligible(tmp_path: Path) -> None:
    curves = {
        frozenset({"i1"}): {MAIN_POINT: 1080.0, CONC: 1080.0},
        frozenset({"c7"}): {MAIN_POINT: 1060.0, CONC: 1060.0},
    }
    bench = await Bench(tmp_path).build(curves)
    bench.regression.failures = {bench.commits["i1"]: "case kv-overflow reproduced"}

    choice = await bench.selector().select(bench.candidates())

    assert choice.chosen == bench.candidates().best_branch
    assert [check.regression_passed for check in choice.checks] == [False, True]
    assert sorted(bench.regression.calls) == sorted((bench.commits[tag], RecipeDelta(), ()) for tag in ("i1", "c7"))


async def test_a_cold_start_within_the_reproduction_tolerance_of_its_record_is_reproduced(tmp_path: Path) -> None:
    curves = {
        frozenset({"i1"}): {MAIN_POINT: 1040.0, CONC: 1040.0},
        frozenset({"c7"}): {MAIN_POINT: 1040.0, CONC: 1040.0},
    }
    bench = await Bench(tmp_path).build(curves)

    choice = await bench.selector().select(bench.candidates())

    best_branch = choice.checks[1]
    assert best_branch.reproduced and best_branch.accuracy_passed and best_branch.regression_passed
    assert choice.chosen is not None


async def test_a_cold_start_that_dropped_requests_is_not_reproduced(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).build({frozenset({"i1"}): {MAIN_POINT: 1040.0, CONC: 1040.0}})
    gate = bench.handle(Role.GATE)
    rule = InferenceXRule(gate, CONC)
    bench.gate = DroppingRequestsGate(gate, bench.toolbox, rule, bench.toolbox, polling=Polling(GATE_POLL_SECONDS))

    choice = await bench.selector().select(bench.candidates(both=False))

    check = choice.checks[0]
    assert check.reproduced is False and check.accuracy_passed and check.regression_passed
    assert choice.chosen is None


class CrashedSweepTests(SessionTests):
    """Session tests whose sweeps all end ``candidate_failed`` after measuring every conc; a failure double, not a
    placeholder."""

    async def result(self, request: TestRequest) -> TestResult:
        measured = await super().result(request)
        if request.kind is not TestKind.SWEEP:
            return measured
        return replace(
            measured,
            validity=Validity.CANDIDATE_FAILED,
            failure_class=FailureClass.CANDIDATE,
            failure_reason="server crashed",
        )


async def test_a_sweep_that_failed_after_measuring_the_primary_conc_disqualifies(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).build({frozenset({"i1"}): {MAIN_POINT: 1040.0}})
    bench.tests = CrashedSweepTests(bench.toolbox, bench.time.clock)

    choice = await bench.selector().select(bench.candidates(both=False))

    [check] = choice.checks
    assert choice.chosen is None and check.reproduced and check.accuracy_passed
    assert (check.curve, check.error) == ({}, "the sweep ended candidate_failed (server crashed)")


async def test_final_waits_for_its_validation_however_long_it_runs(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).build({frozenset({"i1"}): {MAIN_POINT: 1040.0, CONC: 1040.0}})
    bench.toolbox.executor.gate.clear()
    selecting = asyncio.ensure_future(bench.selector().select(bench.candidates(both=False)))

    await bench.time.sleep(VALIDATION_SECONDS)
    bench.toolbox.executor.gate.set()
    choice = await selecting

    assert choice.chosen == bench.candidates().integration and not choice.checks[0].error


async def test_a_cancelled_final_takes_its_tests_back(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).build({})
    bench.toolbox.executor.gate.clear()
    selecting = asyncio.ensure_future(bench.selector().select(bench.candidates(both=False)))
    while len(bench.final_tests()) < 2:
        await asyncio.sleep(0)

    selecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await selecting

    assert bench.toolbox.withdrawn.keys() == bench.final_tests().keys()


async def test_a_withdrawn_test_cancels_the_candidates_other_measurements(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).build({})
    bench.toolbox.executor.gate.clear()
    selecting = asyncio.ensure_future(bench.selector().select(bench.candidates(both=False)))
    while len(bench.final_tests()) < 2:
        await asyncio.sleep(0)
    [sweep] = [rid for rid, request in bench.final_tests().items() if request.kind is TestKind.SWEEP]

    await bench.toolbox.cancel(sweep)
    choice = await selecting

    assert "withdrawn" in choice.checks[0].error
    assert bench.toolbox.withdrawn.keys() == bench.final_tests().keys()
