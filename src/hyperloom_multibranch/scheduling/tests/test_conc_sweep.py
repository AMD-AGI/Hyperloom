# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The conc sweep of a closed session runs only past a FINAL that chose a commit that gained, sweeps the root and that
commit over the recipe's points, one test per AgentX point and one per InferenceX arm, keeps what an earlier run of it
measured, and ends cancelled on a stop signal with the points measured before it."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.contracts.measurement import RoundValue, TestRequest, TestResult, Validity
from hyperloom.orchestrator.state import base_launch
from hyperloom_multibranch import cli
from hyperloom_multibranch.blackboard.codec import encode
from hyperloom_multibranch.contracts.toolbox import TestWithdrawn
from hyperloom.orchestrator.conc_sweep.points import SweepPoints
from hyperloom_multibranch.scheduling import final
from hyperloom_multibranch.scheduling.final import CandidateCheck, FinalCandidate, FinalChoice
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, RunSettings, Workload
from hyperloom_multibranch.scheduling.stops import SessionStopReason
from inference_suite.contracts import SourceKind

WORKLOAD = Workload(
    "/models/Llama-3.1-8B-Instruct", "sglang", 2, 16, "fp8", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1024, 512)
)
AGENTX = replace(WORKLOAD, inferencex=None)
CHOSEN = FinalCandidate("integration", "integration", "c0ffee", 130.0)


def _settings(workload: Workload = WORKLOAD, *, parallel: int = 2, concs: tuple[int, ...] = ()) -> RunSettings:
    return RunSettings(
        workload, 3.0, 0.0, 0.0, RosterCaps(4, 2, parallel), 60.0, conc_sweep=True, conc_sweep_concs=concs
    )


def _layout(tmp_path: Path, chosen: FinalCandidate | None = CHOSEN, gain_pct: float | None = 4.0) -> SessionLayout:
    layout = SessionLayout(tmp_path / "MB-session")
    layout.reports.mkdir(parents=True)
    checks = () if chosen is None else (CandidateCheck(chosen, gain_pct=gain_pct),)
    final = FinalChoice(chosen, checks, "the higher output_throughput at conc 16", "output_throughput", 16)
    layout.report.write_text(json.dumps({"stop_reason": "time_exhausted", "final": encode(final)}))
    return layout


@pytest.fixture
def inferencex_base(monkeypatch: pytest.MonkeyPatch) -> None:
    point = SimpleNamespace(source_concs=(4, 32))
    launch = SimpleNamespace(source=SimpleNamespace(kind=SourceKind.INFERENCEX), point=point)
    monkeypatch.setattr(base_launch, "frozen_base", lambda _toolbox: SimpleNamespace(launch=launch))


class _Tests:
    """Answers each sweep test with a measure round per concurrency, and records every request."""

    def __init__(self, failing: frozenset[str] = frozenset()) -> None:
        self.requests: list[TestRequest] = []
        self.withdrawn: list[str] = []
        self.failing = failing
        self.hold = asyncio.Event()
        self.held = False

    async def result(self, request: TestRequest) -> TestResult:
        self.requests.append(request)
        if self.held and len(self.requests) > 1:
            await self.hold.wait()
            raise TestWithdrawn(f"t{len(self.requests)}", self.withdrawn[-1])
        scale = 100.0 if request.parent_commit == "root" else 130.0
        rounds = tuple(
            RoundValue("measure", conc, {"output_throughput": scale * conc, "total_token_throughput": 2 * scale * conc})
            for conc in request.concurrency
        )
        failed = request.parent_commit in self.failing
        validity = Validity.CANDIDATE_FAILED if failed else Validity.SUCCEEDED
        return TestResult(request_id=f"t{len(self.requests)}", validity=validity, rounds=() if failed else rounds)

    async def withdraw_integration(self, reason: str) -> tuple[str, ...]:
        self.withdrawn.append(reason)
        self.hold.set()
        return ()


def _sweep(layout: SessionLayout, settings: RunSettings, tests: _Tests) -> final.ConcSweep:
    plan = final.sweep_plan(layout, settings)
    assert isinstance(plan, final.SweepPlan)
    caps = settings.caps
    return final.ConcSweep(plan, "root", layout, settings.workload, tests, caps.max_parallel_tests, 600.0)


async def _never() -> int:
    await asyncio.Event().wait()
    return 0


def _summary(layout: SessionLayout) -> dict[str, Any]:
    return json.loads((layout.reports / "conc_sweep_summary.json").read_text())


@pytest.mark.parametrize(
    ("chosen", "gain_pct"),
    [(None, None), (replace(CHOSEN, root=True), 4.0), (CHOSEN, 0.0), (CHOSEN, None)],
)
def test_a_final_that_chose_no_gaining_commit_beside_the_root_gets_no_sweep(
    tmp_path: Path, inferencex_base: None, chosen: FinalCandidate | None, gain_pct: float | None
):
    assert final.sweep_plan(_layout(tmp_path, chosen, gain_pct), _settings()) == final.NO_OPTIMIZATION


def test_the_plan_sweeps_the_chosen_commit_over_the_recipes_points_with_the_workloads_conc(
    tmp_path: Path, inferencex_base: None
):
    plan = final.sweep_plan(_layout(tmp_path), _settings(concs=(2,)))

    assert plan == final.SweepPlan("c0ffee", SweepPoints((4, 16, 32), "inferencex"))
    assert (plan.test_points(WORKLOAD), plan.test_points(AGENTX)) == (3, 1)


def test_an_inferencex_arm_is_one_sweep_test_and_both_arms_run_at_once(tmp_path: Path, inferencex_base: None):
    layout, tests = _layout(tmp_path), _Tests()

    payload = asyncio.run(_sweep(layout, _settings(), tests).run(_never()))

    assert sorted((r.parent_commit, r.concurrency) for r in tests.requests) == [
        ("c0ffee", (4, 16, 32)),
        ("root", (4, 16, 32)),
    ]
    assert {r.purpose.value for r in tests.requests} == {"health"} and not any(r.accuracy for r in tests.requests)
    assert payload["status"] == "succeeded" and payload["summary"]["successful_pairs"] == 3
    assert _summary(layout)["optimized"]["commit"] == "c0ffee"
    assert payload["estimated_sec"] == 600 * 3


def test_the_session_breakdown_beside_the_report_carries_the_sweep(tmp_path: Path, inferencex_base: None):
    layout = _layout(tmp_path)
    breakdown = layout.reports / "session_breakdown.json"
    breakdown.write_text(json.dumps({"schema_version": "multibranch-2", "tests": []}))

    asyncio.run(_sweep(layout, _settings(), _Tests()).run(_never()))

    document = json.loads(breakdown.read_text())
    assert document["conc_sweep_summary"] == _summary(layout)
    assert (document["schema_version"], document["tests"]) == ("multibranch-2", [])


def test_each_agentx_test_lands_in_the_session_breakdown_as_it_ends(
    tmp_path: Path, inferencex_base: None, monkeypatch: pytest.MonkeyPatch
):
    layout = _layout(tmp_path)
    breakdown = layout.reports / "session_breakdown.json"
    breakdown.write_text(json.dumps({"schema_version": "multibranch-2"}))
    carried: list[int] = []
    record = final.ConcSweep._record

    def record_then_read(self: final.ConcSweep, rows: list[dict[str, Any]]) -> None:
        record(self, rows)
        section = json.loads(breakdown.read_text())["conc_sweep_summary"]
        carried.append(sum(len(section[arm]["points"]) for arm in ("baseline", "optimized")))

    monkeypatch.setattr(final.ConcSweep, "_record", record_then_read)

    asyncio.run(_sweep(layout, _settings(AGENTX, parallel=1), _Tests()).run(_never()))

    assert carried == [1, 2, 3, 4, 5, 6]


def test_a_skipped_sweep_is_carried_in_the_session_breakdown(tmp_path: Path):
    layout = _layout(tmp_path)
    breakdown = layout.reports / "session_breakdown.json"
    breakdown.write_text(json.dumps({"schema_version": "multibranch-2"}))

    final.write_skipped(layout, WORKLOAD, final.NO_OPTIMIZATION)

    assert json.loads(breakdown.read_text())["conc_sweep_summary"]["skip_reason"] == final.NO_OPTIMIZATION


def test_an_agentx_point_is_a_test_of_its_own(tmp_path: Path, inferencex_base: None):
    tests = _Tests()

    asyncio.run(_sweep(_layout(tmp_path), _settings(AGENTX, parallel=1), tests).run(_never()))

    assert [r.concurrency for r in tests.requests if r.parent_commit == "root"] == [(16,), (4,), (32,)]


def test_an_arm_that_failed_leaves_the_sweep_failed_and_a_later_run_measures_only_it(
    tmp_path: Path, inferencex_base: None
):
    layout = _layout(tmp_path)
    assert asyncio.run(_sweep(layout, _settings(), _Tests(frozenset({"c0ffee"}))).run(_never()))["status"] == "failed"

    tests = _Tests()
    payload = asyncio.run(_sweep(layout, _settings(), tests).run(_never()))

    assert [r.parent_commit for r in tests.requests] == ["c0ffee"]
    assert payload["status"] == "succeeded"


def test_a_later_run_measures_again_the_arm_whose_commit_changed(tmp_path: Path, inferencex_base: None):
    layout = _layout(tmp_path)
    asyncio.run(_sweep(layout, _settings(), _Tests()).run(_never()))
    settings, tests = _settings(), _Tests()
    plan = replace(final.sweep_plan(layout, settings), chosen="beef")
    sweep = final.ConcSweep(plan, "root", layout, settings.workload, tests, settings.caps.max_parallel_tests, 600.0)

    payload = asyncio.run(sweep.run(_never()))

    assert [r.parent_commit for r in tests.requests] == ["beef"]
    assert (payload["status"], payload["optimized"]["commit"]) == ("succeeded", "beef")


def test_a_stop_signal_takes_the_tests_back_and_keeps_what_landed(tmp_path: Path, inferencex_base: None):
    layout, tests = _layout(tmp_path), _Tests()
    tests.held = True

    async def stop_after_first() -> int:
        while not layout.reports.joinpath("conc_sweep_summary.json").is_file():
            await asyncio.sleep(0)
        return 15

    payload = asyncio.run(_sweep(layout, _settings(AGENTX, parallel=1), tests).run(stop_after_first()))

    assert (payload["status"], payload["cancel_reason"]) == ("cancelled", "signal 15")
    assert tests.withdrawn == ["signal 15"]
    measured = [p["conc"] for arm in ("baseline", "optimized") for p in _summary(layout)[arm]["points"]]
    assert measured == [WORKLOAD.conc]


def test_a_closed_session_whose_sweep_did_not_settle_is_swept_again(tmp_path: Path):
    layout = _layout(tmp_path)

    assert final.sweep_unfinished(layout, _settings())
    final.write_skipped(layout, WORKLOAD, final.NO_SWEEP_POINTS)
    assert not final.sweep_unfinished(layout, _settings())
    assert not final.sweep_unfinished(layout, replace(_settings(), conc_sweep=False))


@pytest.mark.parametrize(
    ("stop", "held", "swept"),
    [
        (SessionStopReason.TIME_EXHAUSTED, False, True),
        (SessionStopReason.SIGNAL, False, False),
        (SessionStopReason.TIME_EXHAUSTED, True, False),
    ],
    ids=["closed", "signalled", "lock_held"],
)
def test_a_new_session_sweeps_once_closed_unless_a_signal_stopped_it_or_another_run_holds_its_lock(
    stop: SessionStopReason, held: bool, swept: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setattr(cli, "require_specialist_reach", lambda branches: None)
    sweeps: list[Path] = []
    roots: list[Path] = []

    class Closed:
        def __init__(self, *_: object) -> None:
            pass

        async def run(self, cap: object) -> SimpleNamespace:
            return SimpleNamespace(stop_reason=stop, exit_code=0)

    async def sweep_closed(layout: SessionLayout, *_: object) -> None:
        sweeps.append(layout.root)

    def report_root(root: Path) -> None:
        roots.append(root)
        if held:
            SessionLayout(root).final_only_lock.write_text("1\n")

    monkeypatch.setattr(cli, "Session", Closed)
    monkeypatch.setattr(cli, "sweep_closed", sweep_closed)
    handlers = list(logging.getLogger().handlers)
    try:
        assert cli.run_session(_settings(), None, None, report_root) == 0  # type: ignore[arg-type]
    finally:
        for handler in set(logging.getLogger().handlers) - set(handlers):
            logging.getLogger().removeHandler(handler)
            handler.close()

    assert sweeps == (roots if swept else [])
    assert SessionLayout(roots[0]).final_only_lock.is_file() is held
