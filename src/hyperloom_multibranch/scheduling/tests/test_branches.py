# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch control: opening records ``OPEN`` and starts specialists, stopping withdraws everything once."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling import branches
from hyperloom_multibranch.scheduling.branches import BranchControl
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import AdmittedBranch
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_BASE_IMAGE_DIGEST
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import BranchOrigin, BranchStop, Roster, RosterCaps, StopCause
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    SPECIALIST_POLL_SECONDS,
    DownToolbox,
    RecordedSessions,
    RecordingToolbox,
    RefusingRuntime,
    specialist_runtime,
)

NOW = 1_700_000_000.0
OPEN_BY = NOW + 3600.0
CLOCK = Clock(now=lambda: NOW)
BASELINE = Recipe(MB_FAKE_BASE_IMAGE_DIGEST)


class BrokenRuntime(RefusingRuntime):
    """A runtime that cannot start ``b01`` and cannot re-size any specialist; a failure double, not a placeholder."""

    async def start(self, branch: str, start_commit: str) -> None:
        if branch == "b01":
            raise RuntimeError(f"no GPU left for {branch}")
        await super().start(branch, start_commit)

    async def set_k(self, branch: str, k: int) -> None:
        raise RuntimeError(f"{branch} did not take K {k}")


class HangingStartRuntime(RefusingRuntime):
    """A runtime whose start of ``b01`` never returns; a failure double, not a placeholder."""

    async def start(self, branch: str, start_commit: str) -> None:
        if branch == "b01":
            await asyncio.Event().wait()
        await super().start(branch, start_commit)


class StartedThenHangingRuntime(RefusingRuntime):
    """A runtime whose start of ``b01`` brings the specialist up and then never returns; a failure double, not a
    placeholder."""

    async def start(self, branch: str, start_commit: str) -> None:
        await super().start(branch, start_commit)
        if branch == "b01":
            await asyncio.Event().wait()


class HangingSetKRuntime(RefusingRuntime):
    """A runtime that never returns from a new K; a failure double, not a placeholder."""

    async def set_k(self, branch: str, k: int) -> None:
        await asyncio.Event().wait()


class Bench:
    def __init__(
        self,
        tmp_path: Path,
        refusing: frozenset[str] = frozenset(),
        runtime: type[RefusingRuntime] = RefusingRuntime,
    ) -> None:
        self.notebook = LocalNotebook(tmp_path / "nb", Writer(Role.SCHEDULER, "scheduler"), clock=CLOCK.now)
        self.roster = Roster(RosterCaps(3, 2, 8), tmp_path / "MB-roster.json", CLOCK)
        self.toolbox = RecordingToolbox(
            LocalNotebook(tmp_path / "nb", Writer(Role.TOOLBOX, "toolbox"), clock=CLOCK.now)
        )
        self.repo_path = tmp_path / "repo"
        self.repo = GitExperimentRepo(self.repo_path, LocalBlobStore(tmp_path / "blobs"))
        self.sessions = RecordedSessions(self.toolbox, self.repo, self.notebook)
        self.runtime = specialist_runtime(
            tmp_path / "nb", self.toolbox, self.repo, self.sessions, now=CLOCK.now, runtime=runtime
        )
        self.runtime.refusing = refusing
        self.root = ""
        self.control = self.controlling(self.toolbox, tmp_path)

    def controlling(self, toolbox: RecordingToolbox, tmp_path: Path) -> BranchControl:
        return BranchControl(
            self.roster,
            self.notebook,
            self.runtime,
            toolbox,
            self.repo,
            ProcessRegistry(tmp_path / "MB-pids.jsonl", CLOCK),
            DecisionLog(tmp_path / "MB-decisions.jsonl", CLOCK),
            CLOCK,
        )

    async def rooted(self) -> Bench:
        self.root = await self.repo.init_root(BASELINE)
        await self.notebook.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=self.root))
        self.roster.add_integration(self.root, k=1)
        return self


async def _runs_started() -> None:
    """Let the runtime start the runs of the branches it was asked to start; it starts each in a task of its own."""

    await asyncio.sleep(SPECIALIST_POLL_SECONDS)


async def test_the_opening_batch_is_recorded_started_and_given_quota(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()

    opened = await bench.control.open_branches(2, bench.root, 5, BranchOrigin.FIRST_BATCH, OPEN_BY)
    await _runs_started()

    assert [entry.branch for entry in opened] == ["b01", "b02"]
    assert bench.sessions.started == {"b01": bench.root, "b02": bench.root}
    assert bench.sessions.k == {"b01": 2, "b02": 2}
    assert [await bench.repo.head(branch) for branch in ("b01", "b02")] == [bench.root, bench.root]
    view = await bench.notebook.lineage()
    assert {branch.branch: branch.start_commit for branch in view.branches}["b02"] == bench.root
    assert dict(bench.toolbox.policies[-1].branch_k) == {INTEGRATION_BRANCH: 1, "b01": 2, "b02": 2}
    assert bench.sessions.quota_at_start == {"b01": 2, "b02": 2}


async def test_an_admitted_repair_is_cut_and_opens_at_its_commit_with_its_k_and_its_quota(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)
    fix = await bench.repo.commit("b01", bench.root, BASELINE, "a fix")
    bench.roster.open("b02", origin=BranchOrigin.REPAIR, start_commit="", k=2, from_branch="b01")

    assert await bench.control.start_admitted(AdmittedBranch("b02", fix), OPEN_BY)
    await _runs_started()

    assert await bench.repo.head("b02") == fix
    assert (bench.sessions.started["b02"], bench.sessions.k["b02"]) == (fix, 2)
    assert bench.sessions.quota_at_start["b02"] == 2
    view = {branch.branch: branch for branch in (await bench.notebook.lineage()).branches}
    assert (view["b02"].start_commit, view["b02"].origin) == (fix, LineageKind.OPEN)
    logged = json.loads((tmp_path / "MB-decisions.jsonl").read_text().splitlines()[-1])
    assert (logged["kind"], logged["branch"]) == ("repair", "b02")


async def test_a_fork_waits_for_the_gate_to_land_its_fix_and_opens_at_it(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)
    bench.roster.open("b02", origin=BranchOrigin.FORK, start_commit="", k=2, from_branch="b01")
    admitted = AdmittedBranch("b02", bench.root)

    assert not await bench.control.start_admitted(admitted, OPEN_BY)
    await bench.repo.create_branch("b02", bench.root)
    assert not await bench.control.start_admitted(admitted, OPEN_BY)
    fix = await bench.repo.commit("b02", bench.root, BASELINE, "the fix")
    assert await bench.control.start_admitted(admitted, OPEN_BY)
    await _runs_started()

    assert (bench.sessions.started["b02"], bench.sessions.k["b02"]) == (fix, 2)
    view = {branch.branch: branch for branch in (await bench.notebook.lineage()).branches}
    assert view["b02"].start_commit == fix
    logged = json.loads((tmp_path / "MB-decisions.jsonl").read_text().splitlines()[-1])
    assert (logged["kind"], logged["branch"], logged["detail"]["from_branch"]) == ("fork", "b02", "b01")


async def test_a_fork_whose_fix_does_not_land_in_time_is_withdrawn_as_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(branches, "SPECIALIST_START_SECONDS", 0.0)
    bench = await Bench(tmp_path).rooted()
    bench.roster.open("b01", origin=BranchOrigin.FIRST_BATCH, start_commit=bench.root, k=1)
    bench.roster.open("b02", origin=BranchOrigin.FORK, start_commit="", k=1, from_branch="b01")

    assert await bench.control.start_admitted(AdmittedBranch("b02", bench.root), OPEN_BY)

    stop = bench.roster.entry("b02").stop
    assert stop is not None and stop.cause is StopCause.START_FAILED
    await _runs_started()
    assert "b02" not in bench.sessions.started and bench.toolbox.cancelled == ["b02"]


async def test_a_failing_specialist_stop_is_returned_and_still_withdraws_and_records_the_branch(tmp_path: Path) -> None:
    bench = await Bench(tmp_path, refusing=frozenset({"b01"})).rooted()
    await bench.control.open_branches(1, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)

    failures = await bench.control.stop_branch("b01", BranchStop(StopCause.JUDGE, "flat", NOW))

    assert [str(failure) for failure in failures] == ["the runtime refused to stop b01"]

    assert bench.toolbox.cancelled == ["b01"]
    assert "b01" not in bench.toolbox.policies[-1].branch_k
    stops = {branch.branch: branch.stop_reason for branch in (await bench.notebook.lineage()).branches}
    assert stops["b01"] == "judge: flat"
    logged = json.loads((tmp_path / "MB-decisions.jsonl").read_text().splitlines()[-1])
    assert logged["detail"]["failures"] == ["RuntimeError: the runtime refused to stop b01"]


async def test_stopping_a_branch_withdraws_it_once_and_its_admitted_fork_never_starts(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)
    bench.roster.open("b02", origin=BranchOrigin.FORK, start_commit="", k=1, from_branch="b01")
    stop = BranchStop(StopCause.JUDGE, "flat", NOW)

    await bench.control.stop_branch("b01", stop)
    await bench.control.stop_branch("b01", BranchStop(StopCause.SESSION_END, "time_exhausted", NOW))

    assert bench.sessions.stopped == ["b01"]
    assert bench.toolbox.cancelled == ["b02", "b01"]
    assert bench.roster.entry("b01").stop == stop
    assert bench.roster.entry("b02").stop == BranchStop(StopCause.JUDGE, "parent b01 stopped: flat", NOW)
    assert await bench.control.start_admitted(AdmittedBranch("b02", bench.root), OPEN_BY)
    await _runs_started()
    assert "b02" not in bench.sessions.started
    assert "b02" not in {branch.branch for branch in (await bench.notebook.lineage()).branches}
    stops = {branch.branch: branch.stop_reason for branch in (await bench.notebook.lineage()).branches}
    assert stops["b01"] == "judge: flat"
    assert "b01" not in bench.toolbox.policies[-1].branch_k


async def test_k_orders_skip_stopped_branches_and_clamp(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(2, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)
    await _runs_started()
    await bench.control.stop_branch("b02", BranchStop(StopCause.JUDGE, "flat", NOW))

    await bench.control.apply_k({"b01": 9, "b02": 2, INTEGRATION_BRANCH: 2})

    assert bench.sessions.running == {"b01"} and bench.sessions.k == {"b01": 2, "b02": 1}
    assert bench.toolbox.policies[-1].k_for(INTEGRATION_BRANCH) == 2


async def test_a_specialist_that_does_not_start_is_withdrawn_without_a_stop_and_the_others_still_open(
    tmp_path: Path,
) -> None:
    bench = await Bench(tmp_path, runtime=BrokenRuntime).rooted()

    opened = await bench.control.open_branches(2, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)
    await bench.control.apply_k({"b02": 2})
    await _runs_started()

    assert [entry.branch for entry in opened] == ["b01", "b02"]
    assert bench.sessions.stopped == [] and bench.sessions.running == {"b02"}
    stop = bench.roster.entry("b01").stop
    assert stop is not None and (stop.cause, stop.reason) == (
        StopCause.START_FAILED,
        "RuntimeError: no GPU left for b01",
    )
    assert bench.roster.entry("b02").stop is None and bench.roster.entry("b02").k == 2
    stops = {branch.branch: branch.stop_reason for branch in (await bench.notebook.lineage()).branches}
    assert stops["b01"] == "start_failed: RuntimeError: no GPU left for b01" and stops["b02"] is None
    logged = [json.loads(line) for line in (tmp_path / "MB-decisions.jsonl").read_text().splitlines()]
    assert [row["detail"]["failure"] for row in logged if row["kind"] == "set_k"] == [
        "RuntimeError: b02 did not take K 2"
    ]


async def test_a_start_that_does_not_finish_in_time_is_withdrawn_and_the_others_still_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(branches, "SPECIALIST_START_SECONDS", 0.2)
    bench = await Bench(tmp_path, runtime=HangingStartRuntime).rooted()

    await bench.control.open_branches(2, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)
    await _runs_started()

    stop = bench.roster.entry("b01").stop
    assert stop is not None and stop.cause is StopCause.START_FAILED
    assert bench.toolbox.cancelled == ["b01"]
    assert bench.sessions.running == {"b02"} and bench.roster.entry("b02").stop is None


async def test_a_toolbox_failing_while_a_branch_opens_fails_that_branch_alone(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    toolbox = DownToolbox(LocalNotebook(tmp_path / "nb", Writer(Role.TOOLBOX, "toolbox"), clock=CLOCK.now))
    toolbox.down = True
    bench.control = bench.controlling(toolbox, tmp_path)

    opened = await bench.control.open_branches(1, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)

    assert [entry.branch for entry in opened] == ["b01"]
    stop = bench.roster.entry("b01").stop
    assert stop is not None and (stop.cause, stop.reason) == (
        StopCause.START_FAILED,
        "ConnectionError: toolbox service down",
    )
    await _runs_started()
    assert bench.sessions.started == {}


async def test_a_specialist_that_came_up_before_its_start_timed_out_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(branches, "SPECIALIST_START_SECONDS", 0.2)
    bench = await Bench(tmp_path, runtime=StartedThenHangingRuntime).rooted()

    await bench.control.open_branches(1, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)

    stop = bench.roster.entry("b01").stop
    assert stop is not None and stop.cause is StopCause.START_FAILED
    assert bench.sessions.stopped == ["b01"] and bench.sessions.running == set()
    assert await bench.control.stop_unconfirmed_specialists() == ()


async def test_a_start_still_running_at_the_callers_deadline_is_withdrawn(tmp_path: Path) -> None:
    bench = await Bench(tmp_path, runtime=HangingStartRuntime).rooted()

    await bench.control.open_branches(1, bench.root, 1, BranchOrigin.FIRST_BATCH, NOW + 0.2)

    stop = bench.roster.entry("b01").stop
    assert stop is not None and stop.cause is StopCause.START_FAILED and stop.reason.startswith("TimeoutError")
    assert bench.toolbox.cancelled == ["b01"]


async def test_a_new_k_the_specialist_never_takes_is_a_failed_step_after_its_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(branches, "SPECIALIST_CALL_SECONDS", 0.2)
    bench = await Bench(tmp_path, runtime=HangingSetKRuntime).rooted()
    await bench.control.open_branches(1, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)

    await bench.control.apply_k({"b01": 2})

    logged = [json.loads(line) for line in (tmp_path / "MB-decisions.jsonl").read_text().splitlines()]
    assert [row["detail"]["failure"] for row in logged if row["kind"] == "set_k"] == [
        "TimeoutError: no result within 0.2 clock seconds"
    ]
    assert bench.roster.entry("b01").stop is None


async def test_a_specialist_stop_that_fails_is_asked_again_until_the_runtime_confirms_it(tmp_path: Path) -> None:
    bench = await Bench(tmp_path, refusing=frozenset({"b01"})).rooted()
    await bench.control.open_branches(1, bench.root, 1, BranchOrigin.FIRST_BATCH, OPEN_BY)
    await bench.control.stop_branch("b01", BranchStop(StopCause.JUDGE, "flat", NOW))

    assert await bench.control.stop_unconfirmed_specialists() == ("b01",)
    bench.runtime.refusing = frozenset()
    assert await bench.control.stop_unconfirmed_specialists() == ()
    assert bench.sessions.stopped == ["b01"] and bench.sessions.running == set()
