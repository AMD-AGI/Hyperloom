# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch control: opening records ``OPEN`` and starts specialists, stopping withdraws everything once."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ConfigDelta, Recipe, RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.fork import AdmissionStatus, RepairRequest
from hyperloom_multibranch.contracts.intent import Intent, IntentLayer
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent
from hyperloom_multibranch.contracts.blackboard.vocab import BranchDriver, LineageKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.rsi.briefing import SessionBriefs
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling import branches
from hyperloom_multibranch.scheduling.branches import BatchStart, BranchControl
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.context import SchedulerContext
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import AdmittedBranch, ForkDesk
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import BranchOrigin, BranchStop, Roster, RosterCaps, StopCause
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    BASE_IMAGE_DIGEST,
    SPECIALIST_POLL_SECONDS,
    DownToolbox,
    RecordedReservations,
    RecordedSessions,
    RecordingToolbox,
    RefusingRuntime,
    SessionStep,
    specialist_runtime,
    tagged,
)
from hyperloom_multibranch.specialist.directions import DIRECTIONS

NOW = 1_700_000_000.0
OPEN_BY = NOW + 3600.0
CLOCK = Clock(now=lambda: NOW)
BASELINE = Recipe(BASE_IMAGE_DIGEST)


class BrokenRuntime(RefusingRuntime):
    """A runtime that cannot start ``b01`` and cannot re-size any specialist; a failure double, not a placeholder."""

    async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
        if branch == "b01":
            raise RuntimeError(f"no GPU left for {branch}")
        await super().start(branch, start_commit, task)

    async def set_k(self, branch: str, k: int) -> None:
        raise RuntimeError(f"{branch} did not take K {k}")


class HangingStartRuntime(RefusingRuntime):
    """A runtime whose start of ``b01`` never returns; a failure double, not a placeholder."""

    async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
        if branch == "b01":
            await asyncio.Event().wait()
        await super().start(branch, start_commit, task)


class StartedThenHangingRuntime(RefusingRuntime):
    """A runtime whose start of ``b01`` brings the specialist up and then never returns; a failure double, not a
    placeholder."""

    async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
        await super().start(branch, start_commit, task)
        if branch == "b01":
            await asyncio.Event().wait()


class CrowdingRuntime(RefusingRuntime):
    """A runtime during whose start of ``b01`` the fork desk admits two forks of it onto ``roster``, taking the last
    free slots; a test double for the admissions that race an opening batch, not a placeholder."""

    roster: Roster

    async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
        await super().start(branch, start_commit, task)
        if branch == "b01":
            for _ in range(2):
                self.roster.open(
                    self.roster.next_branch_id(), origin=BranchOrigin.FORK, start_commit="", k=1, from_branch=branch
                )


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
        clock: Clock = CLOCK,
        steps: Mapping[str, SessionStep] | None = None,
    ) -> None:
        self.clock = clock
        self.blackboard = LocalBlackboard(tmp_path / "nb", Writer(Role.SCHEDULER, "scheduler"), clock=clock.now)
        self.gate_blackboard = LocalBlackboard(tmp_path / "nb", Writer(Role.GATE, "rule-gate"), clock=clock.now)
        self.roster = Roster(RosterCaps(3, 2, 8), tmp_path / "MB-roster.json", clock)
        self.desk = ForkDesk(self.roster, DecisionLog(tmp_path / "MB-decisions.jsonl", clock), clock)
        self.reserved = RecordedReservations()
        self.repo_path = tmp_path / "repo"
        self.repo = GitExperimentRepo(self.repo_path, LocalBlobStore(tmp_path / "blobs"))
        self.toolbox = RecordingToolbox(
            LocalBlackboard(tmp_path / "nb", Writer(Role.TOOLBOX, "toolbox"), clock=clock.now), repo=self.repo
        )
        self.sessions = RecordedSessions(self.toolbox, self.repo, self.blackboard, steps=steps)
        self.runtime = specialist_runtime(
            tmp_path / "nb", self.toolbox, self.repo, self.sessions, now=clock.now, runtime=runtime
        )
        self.runtime.refusing = refusing
        self.root = ""
        self.control = self.controlling(self.toolbox, tmp_path)

    def context(self, tmp_path: Path) -> SchedulerContext:
        decisions = DecisionLog(tmp_path / "MB-decisions.jsonl", self.clock)
        return SchedulerContext(
            SessionLayout(tmp_path),
            self.roster,
            self.blackboard,
            decisions,
            ProcessRegistry(tmp_path / "MB-pids.jsonl", self.clock),
            self.clock,
            self.desk,
            self.reserved,
        )

    def controlling(self, toolbox: RecordingToolbox, tmp_path: Path) -> BranchControl:
        return BranchControl(self.context(tmp_path), self.runtime, toolbox, self.repo, SessionBriefs(self.blackboard))

    async def rooted(self) -> Bench:
        self.root = await self.repo.init_root(BASELINE)
        await self.blackboard.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=self.root))
        self.roster.add_integration(self.root, k=1)
        return self


async def _runs_started() -> None:
    """Let the runtime start the runs of the branches it was asked to start; it starts each in a task of its own."""

    await asyncio.sleep(SPECIALIST_POLL_SECONDS)


async def _relaunched(sessions: RecordedSessions, branch: str) -> None:
    while len(sessions.directions.get(branch, ())) < 2:
        await _runs_started()


async def test_the_opening_batch_is_recorded_started_and_given_quota(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()

    opened = await bench.control.open_branches(2, 5, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    await _runs_started()

    assert [entry.branch for entry in opened] == ["b01", "b02"]
    assert bench.sessions.started == {"b01": bench.root, "b02": bench.root}
    assert bench.sessions.k == {"b01": 2, "b02": 2}
    assert [await bench.repo.head(branch) for branch in ("b01", "b02")] == [bench.root, bench.root]
    view = await bench.blackboard.lineage()
    assert {branch.branch: branch.start_commit for branch in view.branches}["b02"] == bench.root
    assert dict(bench.toolbox.policies[-1].branch_k) == {INTEGRATION_BRANCH: 1, "b01": 2, "b02": 2}
    assert bench.sessions.quota_at_start == {"b01": 2, "b02": 2}


async def test_a_batch_stops_at_the_first_slot_an_admission_took_and_spends_no_branch_id_on_it(
    tmp_path: Path,
) -> None:
    bench = await Bench(tmp_path, runtime=CrowdingRuntime).rooted()
    assert isinstance(bench.runtime, CrowdingRuntime)
    bench.runtime.roster = bench.roster

    opened = await bench.control.open_branches(3, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))

    assert [entry.branch for entry in opened] == ["b01"]
    assert [entry.branch for entry in bench.roster.entries() if entry.origin is BranchOrigin.FORK] == ["b02", "b03"]
    await bench.control.stop_branch("b02", BranchStop(StopCause.JUDGE, "flat", NOW))
    assert [
        entry.branch
        for entry in await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.REFILL, bench.root, OPEN_BY))
    ] == ["b04"]
    assert json.loads((tmp_path / "MB-roster.json").read_text())["issued_branch_ids"] == 4


async def test_a_batch_keeps_each_branch_focus_unrendered_and_names_it_in_its_open_decision(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    briefs = SessionBriefs(bench.blackboard)
    bench.control = BranchControl(bench.context(tmp_path), bench.runtime, bench.toolbox, bench.repo, briefs)

    start = BatchStart(BranchOrigin.REFILL, await bench.control.integration_head(), OPEN_BY)
    await bench.control.open_branches(2, 1, start, ("attention",))
    await _runs_started()

    assert start.start_commit == bench.root
    assert {branch: briefs.for_branch(branch).focus for branch in ("b01", "b02")} == {"b01": "attention", "b02": ""}
    assert bench.sessions.briefs == {"b01": "", "b02": ""}
    opened = [json.loads(line) for line in (tmp_path / "MB-decisions.jsonl").read_text().splitlines()]
    assert [(row["branch"], row["detail"]["focus"]) for row in opened if row["kind"] == "open"] == [
        ("b01", "attention"),
        ("b02", ""),
    ]


async def test_user_branches_open_at_the_root_at_k_one_on_their_directions_up_to_the_free_slots(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    directions = ("Fuse the MoE gate.", "Tune attention.", "Try a bigger batch.", "One too many.")

    opened = await bench.control.open_user_branches(directions, bench.root, OPEN_BY)
    await _runs_started()

    assert [(entry.branch, entry.origin, entry.k) for entry in opened] == [
        (branch, BranchOrigin.USER, 1) for branch in ("b01", "b02", "b03")
    ]
    assert bench.sessions.started == {"b01": bench.root, "b02": bench.root, "b03": bench.root}
    rows = [json.loads(line) for line in (tmp_path / "MB-decisions.jsonl").read_text().splitlines()]
    assert [(row["branch"], row["reason"], row["detail"]["direction"]) for row in rows if row["kind"] == "open"] == [
        ("b01", "user branch started", "Fuse the MoE gate."),
        ("b02", "user branch started", "Tune attention."),
        ("b03", "user branch started", "Try a bigger batch."),
    ]
    assert json.loads((tmp_path / "MB-roster.json").read_text())["issued_branch_ids"] == 3


async def test_agent_branches_open_at_the_root_at_k_one_after_the_user_branches_each_on_its_agent(
    tmp_path: Path,
) -> None:
    bench = await Bench(tmp_path).rooted()

    await bench.control.open_user_branches(("Fuse the MoE gate.",), bench.root, OPEN_BY)
    opened = await bench.control.open_agent_branches(("https://github.com/acme/agent@v1",), bench.root, OPEN_BY)
    await _runs_started()

    assert [(entry.branch, entry.origin, entry.k) for entry in opened] == [("b02", BranchOrigin.AGENT, 1)]
    assert bench.sessions.agents == {"b02": "https://github.com/acme/agent@v1"}
    assert bench.sessions.deadlines["b02"] == OPEN_BY
    assert bench.sessions.started["b02"] == bench.root
    rows = [json.loads(line) for line in (tmp_path / "MB-decisions.jsonl").read_text().splitlines()]
    assert ("b02", "agent branch started") in [(row["branch"], row["reason"]) for row in rows if row["kind"] == "open"]
    assert bench.sessions.plans == {"b02": ""}


async def test_each_agent_branch_starts_on_its_own_share_of_the_directions_in_order(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()

    agents = ("/agents/arbor", "/agents/other", "/agents/arbor")
    await bench.control.open_agent_branches(
        agents, bench.root, OPEN_BY, ("/shares/share-1.json", "", "/shares/share-2.json")
    )
    await _runs_started()

    assert bench.sessions.agents == {"b01": "/agents/arbor", "b02": "/agents/other", "b03": "/agents/arbor"}
    assert bench.sessions.plans == {"b01": "/shares/share-1.json", "b02": "", "b03": "/shares/share-2.json"}


async def test_the_blackboard_records_an_agent_branch_as_autonomous_and_a_user_branch_as_specialist(
    tmp_path: Path,
) -> None:
    bench = await Bench(tmp_path).rooted()

    await bench.control.open_user_branches(("Fuse the MoE gate.",), bench.root, OPEN_BY)
    await bench.control.open_agent_branches(("https://github.com/acme/agent@v1",), bench.root, OPEN_BY)
    await _runs_started()

    drivers = {node.branch: node.driver for node in (await bench.blackboard.lineage()).branches}
    assert (drivers["b01"], drivers["b02"]) == (BranchDriver.SPECIALIST, BranchDriver.AUTONOMOUS)


async def test_each_user_branchs_specialist_runs_on_its_own_direction_in_every_launch_and_a_judge_opened_one_on_a_built_in_one(
    tmp_path: Path,
) -> None:
    crashed: list[str] = []

    async def crash_once(branch: str) -> None:
        if not crashed:
            crashed.append(branch)
            raise RuntimeError(f"the specialist CLI of {branch} died")

    bench = await Bench(tmp_path, steps={"b01": crash_once}).rooted()

    await bench.control.open_user_branches(("Fuse the MoE gate.", "Tune attention."), bench.root, OPEN_BY)
    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    await asyncio.wait_for(_relaunched(bench.sessions, "b01"), 5)

    directions = dict(bench.sessions.directions)
    assert directions.pop("b01") == ["Fuse the MoE gate.", "Fuse the MoE gate."]
    assert directions.pop("b02") == ["Tune attention."]
    [built_in] = directions.pop("b03")
    assert built_in in DIRECTIONS and not directions


def _repair_of(parent_commit: str) -> RepairRequest:
    intent = Intent("b01", parent_commit, "attention", "fp8 kv cache", IntentLayer.CONFIG)
    blocked = Judgement(
        Verdict.BLOCKED, 6.0, 2.0, ("r1", "r2"), {"output_throughput": 1060.0}, accuracy=AccuracyVerdict.FAILED
    )
    return RepairRequest(intent, tagged("fp8-kv"), blocked, "per-tensor KV scales; scale per head")


async def _admit_repair(bench: Bench) -> RepairRequest:
    """Open ``b01``, then admit and start ``b02`` as a repair of a candidate of ``b01`` measured on a fix of it."""

    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    fix = await bench.repo.commit("b01", bench.root, BASELINE, "a fix")
    bench.roster.open("b02", origin=BranchOrigin.REPAIR, start_commit="", k=2, from_branch="b01")
    repair = _repair_of(fix)
    await bench.control.start_admitted(AdmittedBranch("b02", fix, repair), OPEN_BY)
    await _runs_started()
    return repair


async def test_an_admitted_repair_is_cut_and_opens_at_its_commit_with_its_k_its_quota_and_its_request(
    tmp_path: Path,
) -> None:
    bench = await Bench(tmp_path).rooted()

    repair = await _admit_repair(bench)

    fix = repair.intent.parent_commit
    assert await bench.repo.head("b02") == fix
    assert (bench.sessions.started["b02"], bench.sessions.k["b02"]) == (fix, 2)
    assert bench.sessions.repairs == {"b02": repair}
    assert bench.sessions.quota_at_start["b02"] == 2
    view = {branch.branch: branch for branch in (await bench.blackboard.lineage()).branches}
    assert (view["b02"].start_commit, view["b02"].origin, view["b02"].repairs) == (fix, LineageKind.OPEN, "r1")
    logged = json.loads((tmp_path / "MB-decisions.jsonl").read_text().splitlines()[-1])
    assert (logged["kind"], logged["branch"]) == ("repair", "b02")


async def test_an_admitted_repair_is_cut_at_its_candidates_parent_though_its_branch_committed_past_it(
    tmp_path: Path,
) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    measured = await bench.repo.commit("b01", bench.root, BASELINE, "a fix")
    bench.roster.open("b02", origin=BranchOrigin.REPAIR, start_commit="", k=2, from_branch="b01")
    await bench.repo.commit("b01", measured, BASELINE, "a later fix")

    await bench.control.start_admitted(AdmittedBranch("b02", measured, _repair_of(measured)), OPEN_BY)
    await _runs_started()

    assert await bench.repo.head("b02") == measured
    assert bench.sessions.started["b02"] == measured
    assert bench.roster.entry("b02").stop is None


async def test_a_repair_branch_is_stopped_for_its_time_limit_two_hours_after_its_admission(tmp_path: Path) -> None:
    now = [NOW]
    bench = await Bench(tmp_path, clock=Clock(now=lambda: now[0])).rooted()
    await _admit_repair(bench)

    now[0] = NOW + branches.REPAIR_BRANCH_SECONDS - 1
    assert await bench.control.stop_overrun_repairs() == ()
    assert bench.roster.entry("b02").stop is None

    now[0] = NOW + branches.REPAIR_BRANCH_SECONDS
    assert await bench.control.stop_overrun_repairs() == ("b02",)
    assert await bench.control.stop_overrun_repairs() == ()

    expected = BranchStop(StopCause.REPAIR_TIME_LIMIT, "a repair branch runs at most 2 hours", now[0])
    assert bench.roster.entry("b02").stop == expected
    assert bench.roster.entry("b01").stop is None
    assert bench.sessions.stopped == ["b02"] and bench.toolbox.tool_cancels == ["b02"]
    stops = {branch.branch: branch.stop_reason for branch in (await bench.blackboard.lineage()).branches}
    assert stops["b02"] == "repair_time_limit: a repair branch runs at most 2 hours"
    logged = json.loads((tmp_path / "MB-decisions.jsonl").read_text().splitlines()[-1])
    assert (logged["kind"], logged["branch"], logged["reason"]) == ("stop", "b02", expected.summary)


async def test_a_fork_the_gate_landed_opens_at_its_fix(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    bench.roster.open("b02", origin=BranchOrigin.FORK, start_commit="", k=2, from_branch="b01")
    await bench.repo.create_branch("b02", bench.root)
    fix = await bench.repo.commit("b02", bench.root, BASELINE, "the fix")
    split = LineageEvent(LineageKind.SPLIT, "b02", commit_id=fix, from_commit=bench.root, from_branch="b01")
    await bench.gate_blackboard.record_lineage(split)

    await bench.control.start_admitted(AdmittedBranch("b02", bench.root), OPEN_BY)
    await _runs_started()

    assert (bench.sessions.started["b02"], bench.sessions.k["b02"]) == (fix, 2)
    view = {branch.branch: branch for branch in (await bench.blackboard.lineage()).branches}
    assert (view["b02"].origin, view["b02"].from_branch) == (LineageKind.SPLIT, "b01")
    logged = json.loads((tmp_path / "MB-decisions.jsonl").read_text().splitlines()[-1])
    assert (logged["kind"], logged["branch"], logged["detail"]["from_branch"]) == ("fork", "b02", "b01")


async def test_a_fork_the_gate_did_not_land_is_stopped_as_not_started_frees_its_slot_and_is_handed_back(
    tmp_path: Path,
) -> None:
    bench = await Bench(tmp_path).rooted()
    bench.roster.open("b01", origin=BranchOrigin.FIRST_BATCH, start_commit=bench.root, k=1)
    bench.roster.open("b02", origin=BranchOrigin.FORK, start_commit="", k=1, from_branch="b01")

    await bench.control.void_admitted("b02", "the rule gate did not land the fork")

    assert bench.roster.entry("b02").stop == BranchStop(
        StopCause.START_FAILED, "the rule gate did not land the fork", NOW
    )
    assert bench.roster.free_slots() == 2
    assert bench.reserved.voided == [("b02", "the rule gate did not land the fork")]
    await _runs_started()
    assert "b02" not in bench.sessions.started and bench.toolbox.tool_cancels == ["b02"]


async def test_stopping_a_branch_voids_the_forks_and_repairs_it_queued(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(3, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    queued = await bench.desk.admit_repair(_repair_of(bench.root))

    await bench.control.stop_branch("b01", BranchStop(StopCause.JUDGE, "flat", NOW))
    bench.desk.fill_free_slots()

    assert (queued.status, queued.branch) == (AdmissionStatus.PENDING, "b04")
    assert bench.desk.pending() == () and bench.desk.waiting() == ()
    logged = [json.loads(line) for line in (tmp_path / "MB-decisions.jsonl").read_text().splitlines()]
    [voided] = [row for row in logged if row["kind"] == "repair" and row["detail"].get("status") == "voided"]
    assert (voided["branch"], voided["reason"]) == ("b04", "voided: parent b01 stopped: flat")
    assert bench.reserved.voided == [("b04", "parent b01 stopped: flat")]


async def test_stopping_a_branch_hands_its_admitted_fork_or_repair_back_to_the_gate(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(2, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    admitted = await bench.desk.admit_repair(_repair_of(bench.root))

    await bench.control.stop_branch("b01", BranchStop(StopCause.JUDGE, "flat", NOW))

    assert (admitted.status, admitted.branch) == (AdmissionStatus.ACCEPTED, "b03")
    assert bench.roster.entry("b03").stop == BranchStop(StopCause.JUDGE, "parent b01 stopped: flat", NOW)
    assert bench.reserved.voided == [("b03", "parent b01 stopped: flat")]


async def test_a_branch_that_started_is_not_handed_back_to_the_gate_when_it_stops(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))

    await bench.control.stop_branch("b01", BranchStop(StopCause.JUDGE, "flat", NOW))

    assert bench.reserved.voided == []


async def test_a_repair_whose_specialist_does_not_start_is_handed_back_to_the_gate(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    fix = await bench.repo.commit("b01", bench.root, BASELINE, "a fix")
    toolbox = DownToolbox(
        LocalBlackboard(tmp_path / "nb", Writer(Role.TOOLBOX, "toolbox"), clock=CLOCK.now), repo=bench.repo
    )
    toolbox.down = True
    bench.control = bench.controlling(toolbox, tmp_path)
    bench.roster.open("b02", origin=BranchOrigin.REPAIR, start_commit="", k=2, from_branch="b01")

    await bench.control.start_admitted(AdmittedBranch("b02", fix, _repair_of(fix)), OPEN_BY)

    stop = bench.roster.entry("b02").stop
    assert stop is not None and stop.cause is StopCause.START_FAILED
    assert bench.reserved.voided == [("b02", "ConnectionError: toolbox service down")]


async def test_a_failing_specialist_stop_is_returned_and_still_withdraws_and_records_the_branch(tmp_path: Path) -> None:
    bench = await Bench(tmp_path, refusing=frozenset({"b01"})).rooted()
    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))

    failures = await bench.control.stop_branch("b01", BranchStop(StopCause.JUDGE, "flat", NOW))

    assert [str(failure) for failure in failures] == ["the runtime refused to stop b01"]

    assert bench.toolbox.tool_cancels == ["b01"]
    assert "b01" not in bench.toolbox.policies[-1].branch_k
    stops = {branch.branch: branch.stop_reason for branch in (await bench.blackboard.lineage()).branches}
    assert stops["b01"] == "judge: flat"
    logged = json.loads((tmp_path / "MB-decisions.jsonl").read_text().splitlines()[-1])
    assert logged["detail"]["failures"] == ["RuntimeError: the runtime refused to stop b01"]


async def test_stopping_a_branch_withdraws_it_once_and_its_admitted_fork_never_starts(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    bench.roster.open("b02", origin=BranchOrigin.FORK, start_commit="", k=1, from_branch="b01")
    stop = BranchStop(StopCause.JUDGE, "flat", NOW)

    await bench.control.stop_branch("b01", stop)
    await bench.control.stop_branch("b01", BranchStop(StopCause.SESSION_END, "time_exhausted", NOW))

    assert bench.sessions.stopped == ["b01"]
    assert bench.toolbox.tool_cancels == ["b02", "b01"]
    assert bench.roster.entry("b01").stop == stop
    assert bench.roster.entry("b02").stop == BranchStop(StopCause.JUDGE, "parent b01 stopped: flat", NOW)
    await bench.control.start_admitted(AdmittedBranch("b02", bench.root), OPEN_BY)
    await _runs_started()
    assert "b02" not in bench.sessions.started
    assert "b02" not in {branch.branch for branch in (await bench.blackboard.lineage()).branches}
    stops = {branch.branch: branch.stop_reason for branch in (await bench.blackboard.lineage()).branches}
    assert stops["b01"] == "judge: flat"
    assert "b01" not in bench.toolbox.policies[-1].branch_k


async def _held_tests(bench: Bench, count: int) -> list[str]:
    """``count`` explore tests of ``b01``, held by the toolbox's executor until it drains."""

    bench.toolbox.executor.gate.clear()
    return [
        await bench.toolbox.submit_test(
            TestRequest(
                TestKind.TEST,
                Purpose.EXPLORE,
                "b01",
                bench.root,
                delta=RecipeDelta(config=ConfigDelta(server_args=f"--n {n}")),
            )
        )
        for n in range(count)
    ]


async def test_a_judge_stop_leaves_the_branchs_tests_to_finish_under_its_k_and_records_their_results(
    tmp_path: Path,
) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, 2, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    tests = await _held_tests(bench, 3)

    await bench.control.stop_branch("b01", BranchStop(StopCause.JUDGE, "flat", NOW))

    assert bench.toolbox.cancelled == [] and bench.toolbox.tool_cancels == ["b01"]
    assert bench.toolbox.policies[-1].k_for("b01") == 2
    assert (await bench.toolbox.queue_state()).running_by_branch == {"b01": 2}
    await bench.toolbox.drain()
    assert all([(await bench.blackboard.record(test)).result is not None for test in tests])
    await bench.control.push_quota()
    assert "b01" not in bench.toolbox.policies[-1].branch_k


async def test_a_session_end_stop_withdraws_the_branchs_tests(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    tests = await _held_tests(bench, 2)

    await bench.control.stop_branch("b01", BranchStop(StopCause.SESSION_END, "time_exhausted", NOW))

    assert bench.toolbox.cancelled == ["b01"]
    assert set(bench.toolbox.withdrawn) == set(tests)
    assert "b01" not in bench.toolbox.policies[-1].branch_k


async def test_the_quota_gives_each_listed_branch_the_deadline_it_was_started_with_and_none_once_it_drops_out(
    tmp_path: Path,
) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(2, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    assert dict(bench.toolbox.policies[-1].branch_deadline) == {"b01": OPEN_BY, "b02": OPEN_BY}

    await bench.control.stop_branch("b02", BranchStop(StopCause.JUDGE, "flat", NOW))

    assert dict(bench.toolbox.policies[-1].branch_deadline) == {"b01": OPEN_BY}


async def test_k_orders_skip_stopped_branches_and_clamp(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    await bench.control.open_branches(2, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    await _runs_started()
    await bench.control.stop_branch("b02", BranchStop(StopCause.JUDGE, "flat", NOW))

    await bench.control.apply_k({"b01": 9, "b02": 2, INTEGRATION_BRANCH: 2})

    assert bench.sessions.running == {"b01"} and bench.sessions.k == {"b01": 2, "b02": 1}
    assert bench.toolbox.policies[-1].k_for(INTEGRATION_BRANCH) == 2


async def test_a_specialist_that_does_not_start_is_withdrawn_without_a_stop_and_the_others_still_open(
    tmp_path: Path,
) -> None:
    bench = await Bench(tmp_path, runtime=BrokenRuntime).rooted()

    opened = await bench.control.open_branches(2, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
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
    stops = {branch.branch: branch.stop_reason for branch in (await bench.blackboard.lineage()).branches}
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

    await bench.control.open_branches(2, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    await _runs_started()

    stop = bench.roster.entry("b01").stop
    assert stop is not None and stop.cause is StopCause.START_FAILED
    assert bench.toolbox.tool_cancels == ["b01"]
    assert bench.sessions.running == {"b02"} and bench.roster.entry("b02").stop is None


async def test_a_toolbox_failing_while_a_branch_opens_fails_that_branch_alone(tmp_path: Path) -> None:
    bench = await Bench(tmp_path).rooted()
    toolbox = DownToolbox(
        LocalBlackboard(tmp_path / "nb", Writer(Role.TOOLBOX, "toolbox"), clock=CLOCK.now), repo=bench.repo
    )
    toolbox.down = True
    bench.control = bench.controlling(toolbox, tmp_path)

    opened = await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))

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

    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))

    stop = bench.roster.entry("b01").stop
    assert stop is not None and stop.cause is StopCause.START_FAILED
    assert bench.sessions.stopped == ["b01"] and bench.sessions.running == set()
    assert await bench.control.stop_unconfirmed_specialists() == ()


async def test_a_start_still_running_at_the_callers_deadline_is_withdrawn(tmp_path: Path) -> None:
    bench = await Bench(tmp_path, runtime=HangingStartRuntime).rooted()

    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, NOW + 0.2))

    stop = bench.roster.entry("b01").stop
    assert stop is not None and stop.cause is StopCause.START_FAILED and stop.reason.startswith("TimeoutError")
    assert bench.toolbox.tool_cancels == ["b01"]


async def test_a_new_k_the_specialist_never_takes_is_a_failed_step_after_its_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(branches, "SPECIALIST_CALL_SECONDS", 0.2)
    bench = await Bench(tmp_path, runtime=HangingSetKRuntime).rooted()
    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))

    await bench.control.apply_k({"b01": 2})

    logged = [json.loads(line) for line in (tmp_path / "MB-decisions.jsonl").read_text().splitlines()]
    assert [row["detail"]["failure"] for row in logged if row["kind"] == "set_k"] == [
        "TimeoutError: no result within 0.2 clock seconds"
    ]
    assert bench.roster.entry("b01").stop is None


async def test_a_specialist_stop_that_fails_is_asked_again_until_the_runtime_confirms_it(tmp_path: Path) -> None:
    bench = await Bench(tmp_path, refusing=frozenset({"b01"})).rooted()
    await bench.control.open_branches(1, 1, BatchStart(BranchOrigin.FIRST_BATCH, bench.root, OPEN_BY))
    await bench.control.stop_branch("b01", BranchStop(StopCause.JUDGE, "flat", NOW))

    assert await bench.control.stop_unconfirmed_specialists() == ("b01",)
    bench.runtime.refusing = frozenset()
    assert await bench.control.stop_unconfirmed_specialists() == ()
    assert bench.sessions.stopped == ["b01"] and bench.sessions.running == set()
