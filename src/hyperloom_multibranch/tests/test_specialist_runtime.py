# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import logging
import itertools
import json

import pytest

from hyperloom.common.gain_math import ACCURACY_METRICS
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import LaunchConfig, Recipe
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.fork import AdmissionStatus, RepairRequest
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.contracts.lifecycle import BranchTask, SpecialistStatus
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH, ClaimStatus, Conclusion, LineageKind
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import delta, intent
from hyperloom_multibranch.rulegate.baseline import write_inferencex_baseline
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.rules import InferenceXRule
from hyperloom_multibranch.specialist.directions import DIRECTIONS
from hyperloom_multibranch.specialist.exhausted import EXHAUSTED_PREFIX
from hyperloom_multibranch.specialist.judging import NOTICE, Judging
import hyperloom_multibranch.specialist.runtime as runtime_module
from hyperloom_multibranch.specialist.runtime import (
    MAX_LOSS_RESTARTS,
    MAX_RESTARTS,
    RESTART_WINDOW_S,
    RuntimePacing,
    SessionCrashed,
    SessionFinished,
    PreflightFailed,
    SessionIdle,
    SessionStart,
    SpecialistRuntime,
    restart_notice,
)
from hyperloom_multibranch.specialist.scripted import ScriptedSession, ScriptedSetup
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingScheduler, RecordingVerifier
from hyperloom_multibranch.toolbox.fake import FakeToolbox
from hyperloom_multibranch.toolbox.fake_model import FakeToolboxConfig, NoiseModel

BRANCH = "s1"
RECIPE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
SCRIPT = tuple(delta(f"--max-num-seqs {n}") for n in (256, 384, 512))


class _Runs:
    """Every run the runtime starts: the first ones raise ``errors`` in turn, the next ``crashes`` crash, the others
    idle until stopped."""

    def __init__(
        self, crashes: int = 0, errors: tuple[Exception, ...] = (), preflight: PreflightFailed | None = None
    ) -> None:
        self.crashes, self.errors, self.preflight = crashes, list(errors), preflight
        self.started: list[_Run] = []
        self.preflights = 0
        self.released: list[str] = []
        #: Set, a run's stop waits for it.
        self.stops_after: asyncio.Event | None = None

    def __call__(self, start: SessionStart) -> "_Run":
        run = _Run(self, start)
        self.started.append(run)
        return run


class _Run:
    def __init__(self, runs: _Runs, start: SessionStart) -> None:
        self._runs, self.start, self.stopped = runs, start, asyncio.Event()
        self.k = start.k

    async def preflight(self) -> None:
        self._runs.preflights += 1
        if self._runs.preflight is not None:
            raise self._runs.preflight

    async def run(self) -> None:
        if self._runs.errors:
            raise self._runs.errors.pop(0)
        if self._runs.crashes:
            self._runs.crashes -= 1
            raise RuntimeError("the agent CLI exited 1")
        await self.stopped.wait()

    async def set_k(self, k: int) -> None:
        self.k = k

    async def stop(self) -> None:
        if self._runs.stops_after is not None:
            await self._runs.stops_after.wait()
        self.stopped.set()

    async def release(self) -> None:
        self._runs.released.append(self.start.branch)


class _JudgeFails(LocalBlackboard):
    async def poll(self, branch: str, cursor: str = ""):
        raise RuntimeError("the inbox is corrupt")


class _Session:
    """The branch ``s1`` opened at the root, on the fake toolbox, which measures nothing until it is drained."""

    def __init__(self, root) -> None:
        self.root = root
        self.toolbox = FakeToolbox(
            FakeToolboxConfig(), blackboard=LocalBlackboard(root, Writer(Role.TOOLBOX, "toolbox")), autorun=False
        )
        self.repo = InMemoryRepo(RECIPE)
        self.reader = LocalBlackboard(root, Writer(Role.OBSERVER, "observer"))
        self.gate = RuleGate(
            LocalBlackboard(root, Writer(Role.GATE, "gate")),
            self.repo,
            RecordingScheduler(AdmissionStatus.ACCEPTED),
            RecordingVerifier(),
        )

    async def open(self) -> str:
        scheduler = LocalBlackboard(self.root, Writer(Role.SCHEDULER, "scheduler"))
        start = await self.repo.head(INTEGRATION_BRANCH)
        await scheduler.record_lineage(
            LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=start, recipe_digest=RECIPE.digest())
        )
        await self.repo.create_branch(BRANCH, start)
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, BRANCH, from_commit=start))
        return start

    def runtime(self, runs, *, blackboard=LocalBlackboard, clock=lambda: 0.0, claim_ttl=600.0) -> SpecialistRuntime:
        def blackboard_for(branch: str) -> LocalBlackboard:
            return blackboard(self.root, Writer(Role.SPECIALIST_RUNTIME, branch, branch))

        judging = Judging(self.gate, self.toolbox, InferenceXRule(self.reader, 64))
        pacing = RuntimePacing(poll_seconds=0.01, claim_ttl_seconds=claim_ttl, clock=clock)
        return SpecialistRuntime(blackboard_for, judging, runs, pacing=pacing)

    async def claim(self, key) -> ClaimStatus:
        (found,) = [claim for claim in (await self.reader.snapshot()).claims if claim.key == key]
        return found.status

    def scripted(self, k: int) -> ScriptedSession:
        setup = ScriptedSetup(SCRIPT, self.toolbox, self.repo, self.reader, poll_seconds=0.01)
        return ScriptedSession(SessionStart(BRANCH, start_commit="", k=k), setup)

    async def notices(self) -> list[str]:
        return [
            record.entry.content
            for record in await self.reader.branch_log(BRANCH)
            if record.entry.content.startswith(NOTICE)
        ]

    async def explored(self) -> int:
        return len([record for record in (await self.reader.snapshot()).records if record.request.branch_id == BRANCH])


async def _until(condition, what: str) -> None:
    for _ in range(500):
        if await condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
async def session(tmp_path):
    session = _Session(tmp_path / "blackboard")
    await session.open()
    yield session
    await session.toolbox.close()


async def test_a_started_branch_runs_its_specialist_with_the_k_it_is_given(session):
    runs = _Runs()
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _true(runs.started), "the specialist to start")
    await runtime.set_k(BRANCH, 3)
    await runtime.stop(BRANCH)

    (run,) = runs.started
    assert (run.k, run.stopped.is_set()) == (3, True)
    with pytest.raises(KeyError):
        await runtime.set_k(BRANCH, 2)


async def test_a_crashed_specialist_is_restarted_with_its_task_and_a_notice_each_time(session):
    runs = _Runs(crashes=2)
    runtime = session.runtime(runs)
    blocked = Judgement(
        Verdict.BLOCKED, 3.0, 2.0, ("s0-4", "s0-5"), {"output_throughput": 1030.0}, AccuracyVerdict.FAILED
    )
    repair = RepairRequest(intent("s0", "c0"), delta(), blocked, "per-tensor KV scales; scale per head")
    task = BranchTask("leads: fp8 kv cache", repair, "try a larger max_num_seqs first")

    await runtime.start(BRANCH, await session.repo.head(BRANCH), task)
    await _until(lambda: _true(len(runs.started) == 3), "two restarts")

    assert all(run.stopped.is_set() for run in runs.started[:2])
    assert [run.start.task for run in runs.started] == [task] * 3
    assert len([notice for notice in await session.notices() if "is restarted" in notice]) == 2
    await runtime.stop(BRANCH)


async def test_branches_started_with_no_direction_take_the_session_s_directions_in_order(session):
    runs = _Runs()
    runtime = session.runtime(runs)
    head = await session.repo.head(BRANCH)
    blocked = Judgement(
        Verdict.BLOCKED, 3.0, 2.0, ("s0-4", "s0-5"), {"output_throughput": 1030.0}, AccuracyVerdict.FAILED
    )
    repair = RepairRequest(intent("s0", "c0"), delta(), blocked, "per-tensor KV scales; scale per head")
    handed = [f"b{index}" for index in range(len(DIRECTIONS) + 1)]

    await runtime.start(handed[0], head)
    await runtime.start("user", head, BranchTask(direction="the user's own"))
    await runtime.start("repair", head, BranchTask(repair=repair))
    await runtime.start("agent", head, BranchTask(agent="https://example.com/agent.git"))
    for name in handed[1:]:
        await runtime.start(name, head)
    await _until(lambda: _true(len(runs.started) == len(handed) + 3), "every specialist to start")
    for name in [*handed, "user", "repair", "agent"]:
        await runtime.stop(name)

    directions = {run.start.branch: run.start.task.direction for run in runs.started}
    assert [directions[name] for name in handed] == [*DIRECTIONS, DIRECTIONS[0]]
    assert (directions["user"], directions["repair"], directions["agent"]) == ("the user's own", "", "")


async def test_a_direction_no_branch_held_goes_first_even_after_earlier_branches_stopped(session):
    runs = _Runs()
    runtime = session.runtime(runs)
    head = await session.repo.head(BRANCH)

    await runtime.start("b0", head)
    await runtime.start("b1", head)
    await _until(lambda: _true(len(runs.started) == 2), "the first specialists to start")
    await runtime.stop("b0")
    await runtime.start("b2", head)
    await _until(lambda: _true(len(runs.started) == 3), "the last specialist to start")
    for name in ("b1", "b2"):
        await runtime.stop(name)

    directions = {run.start.branch: run.start.task.direction for run in runs.started}
    assert [directions[name] for name in ("b0", "b1", "b2")] == list(DIRECTIONS[:3])


async def test_once_every_direction_was_held_the_least_held_over_the_session_wins_then_the_least_running(session):
    runs = _Runs()
    runtime = session.runtime(runs)
    head = await session.repo.head(BRANCH)
    first = [f"b{index}" for index in range(len(DIRECTIONS) + 1)]

    for name in first:
        await runtime.start(name, head)
    await _until(lambda: _true(len(runs.started) == len(first)), "the first specialists to start")
    stopped = ("b0", first[-1], "b5")
    for name in stopped:
        await runtime.stop(name)
    await runtime.start("late", head)
    await _until(lambda: _true(len(runs.started) == len(first) + 1), "the last specialist to start")
    for name in [*first, "late"]:
        if name not in stopped:
            await runtime.stop(name)

    directions = {run.start.branch: run.start.task.direction for run in runs.started}
    assert directions[first[-1]] == DIRECTIONS[0]
    assert directions["late"] == DIRECTIONS[5]


async def test_a_direction_given_to_a_branch_counts_as_held(session):
    runs = _Runs()
    runtime = session.runtime(runs)
    head = await session.repo.head(BRANCH)

    await runtime.start("user", head, BranchTask(direction=DIRECTIONS[0]))
    await _until(lambda: _true(len(runs.started) == 1), "the user's specialist to start")
    await runtime.stop("user")
    await runtime.start("b0", head)
    await _until(lambda: _true(len(runs.started) == 2), "the next specialist to start")
    await runtime.stop("b0")

    assert runs.started[1].start.task.direction == DIRECTIONS[1]


async def test_a_direction_a_specialist_gave_up_on_is_not_dealt_again_on_the_same_start_commit(session):
    runs = _Runs(errors=(_idle(), _idle()))
    runtime = session.runtime(runs)
    head = await session.repo.head(BRANCH)
    others = [f"b{index}" for index in range(1, len(DIRECTIONS))]

    await runtime.start(BRANCH, head)
    await _until(lambda: _any_notice(session, "not restarted"), "the first specialist to give up")
    for name in others:
        await runtime.start(name, head)
    for name in (BRANCH, *others):
        await runtime.stop(name)
    await runtime.start("same", head)
    await runtime.start("moved", "f" * 40)
    await _until(lambda: _true(len(runs.started) == len(others) + 4), "the late specialists to start")
    for name in ("same", "moved"):
        await runtime.stop(name)

    directions = {run.start.branch: run.start.task.direction for run in runs.started}
    assert directions[BRANCH] == DIRECTIONS[0]
    assert directions["same"] == DIRECTIONS[1]
    assert directions["moved"] == DIRECTIONS[0]


async def test_past_the_restarts_a_window_allows_the_specialist_stays_down(session):
    runs = _Runs(crashes=10)
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _any_notice(session, "not restarted"), "the runtime to give up")

    assert len(runs.started) == 1 + MAX_RESTARTS
    assert await runtime.status(BRANCH) == SpecialistStatus(False, 1 + MAX_RESTARTS, 0, True, False)
    await runtime.set_k(BRANCH, 2)
    await runtime.stop(BRANCH)


def _lost(*, worked: bool = False) -> SessionCrashed:
    return SessionCrashed("the session was lost (no heartbeat for more than 300s)", lost=True, worked=worked)


async def test_losses_in_a_row_stop_the_restarts_however_far_apart_they_are(session):
    losses = MAX_LOSS_RESTARTS + 1
    runs = _Runs(errors=tuple(_lost() for _ in range(losses + 2)))
    ticks = itertools.count(step=RESTART_WINDOW_S + 1)
    runtime = session.runtime(runs, clock=lambda: next(ticks))

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _any_notice(session, "not restarted"), "the runtime to give up")

    assert len(runs.started) == losses == 3
    assert any(f"lost {losses} sessions in a row" in notice for notice in await session.notices())
    assert await runtime.status(BRANCH) == SpecialistStatus(False, losses, losses, True, False)
    await runtime.stop(BRANCH)


async def test_losses_while_waiting_on_a_long_dev_lease_never_give_the_specialist_up(session):
    waiting = SessionCrashed("the session was lost (heartbeat stale for 601s)", lost=True, waiting=True)
    runs = _Runs(errors=(*(waiting for _ in range(MAX_LOSS_RESTARTS + 2)), _lost(), _lost()))
    ticks = itertools.count(step=RESTART_WINDOW_S + 1)
    runtime = session.runtime(runs, clock=lambda: next(ticks))

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _true(len(runs.started) == MAX_LOSS_RESTARTS + 5), "every loss to be restarted")

    assert not any("not restarted" in notice for notice in await session.notices())
    assert (await runtime.status(BRANCH)).gave_up is False
    await runtime.stop(BRANCH)


async def test_a_session_that_worked_between_losses_counts_them_again_from_its_own(session):
    runs = _Runs(errors=(_lost(), _lost(), _lost(worked=True), _lost()))
    ticks = itertools.count(step=RESTART_WINDOW_S + 1)
    runtime = session.runtime(runs, clock=lambda: next(ticks))

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _true(len(runs.started) == 5), "every loss to be restarted")

    assert not any("not restarted" in notice for notice in await session.notices())
    assert await runtime.status(BRANCH) == SpecialistStatus(True, 4, 4, False, False)
    await runtime.stop(BRANCH)


async def test_a_branch_preflights_once_before_its_first_run_and_not_on_a_restart(session):
    runs = _Runs(crashes=1)
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _true(len(runs.started) == 2), "the crashed run to be restarted")

    assert runs.preflights == 1
    await runtime.stop(BRANCH)


async def test_a_failed_preflight_gives_up_the_specialist_with_every_reason_and_runs_nothing(session):
    reasons = ("attempt 1: launch exited 127", "attempt 2: no keep file", "attempt 3: no server")
    runs = _Runs(preflight=PreflightFailed(reasons))
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _any_notice(session, "not started"), "the runtime to give up")

    assert len(runs.started) == 1 and runs.started[0].stopped.is_set()
    [notice] = [notice for notice in await session.notices() if "preflight failed" in notice]
    assert all(reason in notice for reason in reasons)
    assert await runtime.status(BRANCH) == SpecialistStatus(False, 0, 0, True, False, unready=str(runs.preflight))
    await runtime.stop(BRANCH)


async def test_a_finished_session_ends_its_specialist_with_no_restart_and_no_crash(session):
    runs = _Runs(errors=(SessionFinished("the agent finished its run"),))
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH), BranchTask(agent="https://example.com/agent.git"))
    await _until(lambda: _any_notice(session, "not restarted"), "the runtime to end the specialist")

    assert len(runs.started) == 1 and runs.started[0].stopped.is_set()
    assert any("the agent finished its run" in notice for notice in await session.notices())
    assert await runtime.status(BRANCH) == SpecialistStatus(False, 0, 0, True, False)
    await runtime.stop(BRANCH)


@pytest.mark.parametrize("where", ["factory", "preflight"])
async def test_a_session_that_cannot_be_built_or_made_ready_gives_its_specialist_up_with_the_reason(session, where):
    runs = _Runs(preflight=OSError("no space left on device"))

    def factory(start: SessionStart) -> _Run:
        if where == "factory":
            raise ValueError("branch s1 runs the agent a, which no specialist stands for")
        return runs(start)

    runtime = session.runtime(factory)

    await runtime.start(BRANCH, await session.repo.head(BRANCH), BranchTask(agent="https://example.com/agent.git"))
    await _until(lambda: _any_notice(session, "not started"), "the runtime to give up")

    why = "no specialist stands for" if where == "factory" else "no space left"
    assert any(why in notice for notice in await session.notices())
    status = await runtime.status(BRANCH)
    assert (status.gave_up, status.crashes) == (True, 0) and why in status.unready
    await runtime.stop(BRANCH)


@pytest.mark.parametrize("crashes", [0, 2])
async def test_what_a_branch_s_runs_hold_is_given_back_once_when_the_branch_stops(session, crashes):
    runs = _Runs(crashes=crashes)
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH), BranchTask(agent="https://example.com/agent.git"))
    await _until(lambda: _true(len(runs.started) == crashes + 1), "the last run")
    assert runs.released == []
    await runtime.stop(BRANCH)

    assert runs.released == [BRANCH]


async def test_a_stop_its_caller_gave_up_waiting_for_is_confirmed_by_the_next_stop(session):
    runs = _Runs()
    runs.stops_after = asyncio.Event()
    runtime = session.runtime(runs)
    await runtime.start(BRANCH, await session.repo.head(BRANCH), BranchTask(agent="https://example.com/agent.git"))
    await _until(lambda: _true(len(runs.started) == 1), "the run")

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(runtime.stop(BRANCH), 0.2)
    runs.stops_after.set()
    await asyncio.wait_for(runtime.stop(BRANCH), 5)
    await runtime.stop(BRANCH)

    assert runs.started[0].stopped.is_set() and runs.released == [BRANCH]


async def test_a_slow_stop_is_logged_with_the_time_each_of_its_steps_took(session, monkeypatch, caplog):
    monkeypatch.setattr(runtime_module, "SLOW_STOP_SECONDS", 0.05)
    runs = _Runs()
    runs.stops_after = asyncio.Event()
    runtime = session.runtime(runs)
    await runtime.start(BRANCH, await session.repo.head(BRANCH), BranchTask(agent="https://example.com/agent.git"))
    await _until(lambda: _true(len(runs.started) == 1), "the run")

    asyncio.get_running_loop().call_later(0.2, runs.stops_after.set)
    with caplog.at_level(logging.WARNING, logger=runtime_module.__name__):
        await runtime.stop(BRANCH)

    [slow] = [record.getMessage() for record in caplog.records if "stop of s1 took" in record.getMessage()]
    assert "session stop" in slow and "supervisor" in slow and "release" in slow


async def test_a_branch_whose_preflight_failed_gives_back_what_it_holds_when_it_stops(session):
    runs = _Runs(preflight=PreflightFailed(("no card is left",)))
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH), BranchTask(agent="https://example.com/agent.git"))
    await _until(lambda: _any_notice(session, "not started"), "the runtime to give up")
    await runtime.stop(BRANCH)

    assert runs.released == [BRANCH]


def _idle(tested: bool = False, resumable: bool = False) -> SessionIdle:
    return SessionIdle("its model ended 3 turns in a row with nothing to test", tested=tested, resumable=resumable)


@pytest.mark.parametrize("kind", ["", "direction", "repair"])
async def test_two_idle_sessions_in_a_row_end_the_specialist_after_one_fresh_session_and_no_crash(session, kind):
    blocked = Judgement(
        Verdict.BLOCKED, 3.0, 2.0, ("s0-4", "s0-5"), {"output_throughput": 1030.0}, AccuracyVerdict.FAILED
    )
    repair = RepairRequest(intent("s0", "c0"), delta(), blocked, "per-tensor KV scales; scale per head")
    tasks = {"": BranchTask(), "direction": BranchTask(direction="the user's own"), "repair": BranchTask(repair=repair)}
    runs = _Runs(errors=(_idle(), _idle()))
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH), tasks[kind])
    await _until(lambda: _any_notice(session, "not restarted"), "the runtime to give up")

    assert len(runs.started) == 2
    notices = await session.notices()
    assert any("starts a fresh session" in notice for notice in notices)
    assert any("found nothing to test" in notice and "not restarted" in notice for notice in notices)
    assert await runtime.status(BRANCH) == SpecialistStatus(False, 0, 0, True, False)
    await runtime.stop(BRANCH)


async def test_a_specialist_s_exhausted_record_is_in_its_give_up_notice_and_closes_its_direction_on_that_head(session):
    runs = _Runs(errors=(_idle(), _idle()))
    runtime = session.runtime(runs)
    head, moved = await session.repo.head(BRANCH), "e" * 40
    record = {"direction": DIRECTIONS[0], "head": moved, "tried": ["r1"], "why": "blocked_by_tool"}
    record |= {"detail": "start_analysis refused the head's profile", "reopen_if": "the analysis is allowed"}
    entry = BranchLogEntry(BRANCH, BranchLogType.RESULT, EXHAUSTED_PREFIX + json.dumps(record))
    await LocalBlackboard(session.root, Writer(Role.SPECIALIST_RUNTIME, BRANCH, BRANCH)).append_branch_log(entry)

    await runtime.start(BRANCH, head)
    await _until(lambda: _any_notice(session, "not restarted"), "the specialist to give up")
    await runtime.stop(BRANCH)
    await runtime.start("next", moved)
    await _until(lambda: _true(len(runs.started) == 3), "the next specialist to start")
    await runtime.stop("next")

    (given_up,) = [notice for notice in await session.notices() if "not restarted" in notice]
    assert "blocked_by_tool: start_analysis refused the head's profile" in given_up
    assert runs.started[-1].start.task.direction == DIRECTIONS[1]


async def test_a_direction_a_running_or_stopped_branch_recorded_exhausted_on_a_head_is_not_dealt_there(session):
    runs = _Runs()
    runtime = session.runtime(runs)
    head = await session.repo.head(BRANCH)
    fusion = DIRECTIONS[2].split(":")[0].upper()
    record = {"direction": f" {fusion} ", "head": head, "tried": ["r1"], "why": "no_ceiling_above_bar"}
    record |= {"detail": "no elementwise op above the bar is left", "reopen_if": "the head moves"}

    await runtime.start(BRANCH, head)
    entry = BranchLogEntry(BRANCH, BranchLogType.RESULT, EXHAUSTED_PREFIX + json.dumps(record))
    await LocalBlackboard(session.root, Writer(Role.SPECIALIST_RUNTIME, BRANCH, BRANCH)).append_branch_log(entry)
    await runtime.start("b1", head)
    await runtime.start("next", head)
    await _until(lambda: _true(len(runs.started) == 3), "the first specialists to start")
    for name in (BRANCH, "b1"):
        await runtime.stop(name)
    await runtime.start("later", head)
    await _until(lambda: _true(len(runs.started) == 4), "the specialists to start")
    for name in ("next", "later"):
        await runtime.stop(name)

    directions = {run.start.branch: run.start.task.direction for run in runs.started}
    assert DIRECTIONS[2] not in (directions["next"], directions["later"])
    assert (directions["next"], directions["later"]) == (DIRECTIONS[3], DIRECTIONS[4])


async def test_an_exhausted_record_naming_no_session_direction_closes_nothing_and_is_logged(session, caplog):
    runs = _Runs()
    runtime = session.runtime(runs)
    head = await session.repo.head(BRANCH)
    record = {"direction": "rope and norm fusions", "head": head, "tried": ["r1"], "why": "no_ceiling_above_bar"}
    record |= {"detail": "no elementwise op above the bar is left", "reopen_if": "the head moves"}

    await runtime.start(BRANCH, head)
    entry = BranchLogEntry(BRANCH, BranchLogType.RESULT, EXHAUSTED_PREFIX + json.dumps(record))
    await LocalBlackboard(session.root, Writer(Role.SPECIALIST_RUNTIME, BRANCH, BRANCH)).append_branch_log(entry)
    await runtime.start("b1", head)
    await runtime.start("next", head)
    await _until(lambda: _true(len(runs.started) == 3), "the specialists to start")
    for name in (BRANCH, "b1", "next"):
        await runtime.stop(name)

    assert runs.started[-1].start.task.direction == DIRECTIONS[2]
    assert "names none of the session's directions" in caplog.text


async def test_an_idle_session_whose_conversation_is_resumed_says_so_and_still_counts_as_idle(session):
    runs = _Runs(errors=(_idle(resumable=True), _idle(resumable=True)))
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _any_notice(session, "not restarted"), "the runtime to give up")
    await runtime.stop(BRANCH)

    assert len(runs.started) == 2
    notices = await session.notices()
    assert any("found nothing to test" in n and "resumes its conversation" in n for n in notices)
    assert not any("starts a fresh session" in n for n in notices)


async def test_an_idle_session_that_submitted_a_test_starts_the_count_again(session):
    runs = _Runs(errors=(_idle(), _idle(tested=True), _idle()))
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _true(len(runs.started) == 4), "a fresh session after each idle one")
    status = await runtime.status(BRANCH)
    await runtime.stop(BRANCH)

    assert not status.gave_up


async def test_the_status_is_known_only_while_the_runtime_runs_the_branch(session):
    runs = _Runs(crashes=1)
    runtime = session.runtime(runs)
    assert await runtime.status(BRANCH) is None

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _true(len(runs.started) == 2), "the restart")

    assert await runtime.status(BRANCH) == SpecialistStatus(True, 1, 0, False, False)
    await runtime.stop(BRANCH)
    assert await runtime.status(BRANCH) is None


async def test_crashes_further_apart_than_the_window_do_not_add_up(session):
    runs = _Runs(crashes=MAX_RESTARTS + 2)
    ticks = itertools.count(step=RESTART_WINDOW_S + 1)
    runtime = session.runtime(runs, clock=lambda: next(ticks))

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _true(len(runs.started) == MAX_RESTARTS + 3), "every crash to be restarted")

    assert not any("not restarted" in notice for notice in await session.notices())
    await runtime.stop(BRANCH)


async def test_a_failed_judge_stops_the_specialist_and_is_not_restarted(session):
    runs = _Runs()
    runtime = session.runtime(runs, blackboard=_JudgeFails)

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _any_notice(session, "judging this branch failed"), "the judge to fail")

    (run,) = runs.started
    assert run.stopped.is_set()
    assert await runtime.status(BRANCH) == SpecialistStatus(False, 0, 0, False, True)
    await runtime.stop(BRANCH)


async def test_a_branch_has_one_specialist_and_keeps_room_for_a_test(session):
    runtime = session.runtime(_Runs())
    await runtime.start(BRANCH, await session.repo.head(BRANCH))

    with pytest.raises(ValueError, match="already has a specialist"):
        await runtime.start(BRANCH, await session.repo.head(BRANCH))
    with pytest.raises(ValueError, match="stop the branch"):
        await runtime.set_k(BRANCH, 0)
    await runtime.stop(BRANCH)
    await runtime.stop(BRANCH)
    assert await runtime.status(BRANCH) is None


async def test_the_claims_a_specialist_holds_are_renewed_until_its_branch_stops(session):
    runtime = session.runtime(_Runs(), claim_ttl=0.3)
    start = await session.repo.head(BRANCH)
    await runtime.start(BRANCH, start)
    holder = LocalBlackboard(session.root, Writer(Role.SPECIALIST_RUNTIME, BRANCH, BRANCH))
    claim = await holder.acquire(intent(BRANCH, start), SCRIPT[0].fingerprint(), ttl_seconds=0.3)

    await asyncio.sleep(0.9)
    assert await session.claim(claim.key) is ClaimStatus.HELD
    await runtime.stop(BRANCH)
    await asyncio.sleep(0.6)
    assert await session.claim(claim.key) is ClaimStatus.EXPIRED


async def test_a_new_k_is_noted_once_in_the_branch_log(session):
    runtime = session.runtime(_Runs())
    await runtime.start(BRANCH, await session.repo.head(BRANCH))

    await runtime.set_k(BRANCH, 3)
    await runtime.set_k(BRANCH, 3)

    assert [notice for notice in await session.notices() if "at once" in notice] == [
        f"{NOTICE} you may now have 3 tests running at once"
    ]
    await runtime.stop(BRANCH)


async def test_a_scripted_specialist_keeps_k_of_its_tests_running(session):
    scripted = session.scripted(k=2)
    running = asyncio.create_task(scripted.run())

    await _until(lambda: _equal(session.explored(), 2), "two tests")
    await asyncio.sleep(0.05)
    assert await session.explored() == 2
    await scripted.set_k(3)
    await _until(lambda: _equal(session.explored(), 3), "the third test")

    await scripted.stop()
    await asyncio.wait_for(running, 1)


async def test_a_restarted_scripted_specialist_measures_no_change_twice(session):
    first = session.scripted(k=2)
    running = asyncio.create_task(first.run())
    await _until(lambda: _equal(session.explored(), 2), "two tests")
    await first.stop()
    await asyncio.wait_for(running, 1)

    second = session.scripted(k=3)
    running = asyncio.create_task(second.run())
    await _until(lambda: _equal(session.explored(), 3), "the last change")
    await asyncio.sleep(0.05)

    deltas = [record.request.delta for record in (await session.reader.snapshot()).records]
    assert sorted(change.fingerprint() for change in deltas) == sorted(change.fingerprint() for change in SCRIPT)
    await second.stop()
    await asyncio.wait_for(running, 1)


async def _true(value: object) -> bool:
    return bool(value)


async def _equal(count, expected: int) -> bool:
    return await count == expected


async def _any_notice(session: _Session, text: str) -> bool:
    return any(text in notice for notice in await session.notices())


async def test_a_crash_notice_says_whether_the_conversation_is_resumed(session):
    runs = _Runs(errors=(SessionCrashed("the agent CLI exited 1", resumable=True), RuntimeError("boom")))
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _true(len(runs.started) == 3), "both restarts")

    restarted = [notice for notice in await session.notices() if "is restarted" in notice]
    assert ["resuming its conversation" in n for n in restarted] == [True, False]
    assert "with a fresh session" in restarted[1]
    await runtime.stop(BRANCH)


async def test_the_notices_of_a_restart_are_told_apart_from_the_branch_s_other_notices(session):
    errors = (_idle(resumable=True), SessionCrashed("the agent CLI exited 1", resumable=True), _idle(tested=True))
    runs = _Runs(errors=errors)
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _true(len(runs.started) == 4), "every restart")
    await runtime.set_k(BRANCH, 3)
    await runtime.stop(BRANCH)

    notices = await session.notices()
    restarts = [notice for notice in notices if "is restarted" in notice or "the branch" in notice]
    assert len(restarts) == 3 and all(restart_notice(notice) for notice in restarts)
    assert [restart_notice(notice) for notice in notices if notice not in restarts] == [False]


FAST = delta("--max-num-seqs 256")


class _Judged(_Session):
    """``_Session`` with its baseline measured and a quiet toolbox on which ``FAST`` gains 5%."""

    def __init__(self, root) -> None:
        super().__init__(root)
        quiet = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
        config = FakeToolboxConfig(noise=quiet, accuracy_metric=ACCURACY_METRICS[0], effects={FAST.fingerprint(): 0.05})
        self.toolbox = FakeToolbox(
            config, blackboard=LocalBlackboard(root, Writer(Role.TOOLBOX, "toolbox")), autorun=False
        )

    async def open(self) -> str:
        start = await super().open()
        baseline = TestRequest(
            kind=TestKind.TEST, purpose=Purpose.BASELINE, branch_id=INTEGRATION_BRANCH, parent_commit=start
        )
        ids = [await self.toolbox.submit_test(baseline) for _ in range(3)]
        await self.toolbox.drain()
        gate = LocalBlackboard(self.root, Writer(Role.GATE, "gate"))
        await write_inferencex_baseline(gate, start, [await gate.record(request_id) for request_id in ids], 64)
        return start

    async def explore(self, change) -> str:
        request = TestRequest(
            kind=TestKind.TEST,
            purpose=Purpose.EXPLORE,
            branch_id=BRANCH,
            parent_commit=await self.repo.head(BRANCH),
            delta=change,
        )
        return await self.toolbox.submit_test(request)


@pytest.fixture
async def judged(tmp_path):
    session = _Judged(tmp_path / "blackboard")
    await session.open()
    yield session
    await session.toolbox.close()


def _judges() -> list[asyncio.Task]:
    return [task for task in asyncio.all_tasks() if task.get_name() == f"judge of {BRANCH}" and not task.done()]


async def test_a_branch_stopped_to_finish_its_tests_still_commits_the_one_that_wins_then_stops_judging(judged):
    runs = _Runs()
    runtime = judged.runtime(runs)
    start = await judged.repo.head(BRANCH)
    await runtime.start(BRANCH, start)
    first = await judged.explore(FAST)

    await runtime.stop(BRANCH, finish_tests=True)
    assert runs.started[0].stopped.is_set() and runs.released == [BRANCH]
    assert await runtime.status(BRANCH) is None
    with pytest.raises(ValueError, match="already has a specialist"):
        await runtime.start(BRANCH, start)
    await judged.toolbox.drain()
    await _until(lambda: _true(not _judges()), "the judge to end once no test of the branch is in flight")

    record = await judged.reader.record(first)
    assert (record.conclusion, record.commit_id) == (Conclusion.COMMITTED, await judged.repo.head(BRANCH))
    assert record.commit_id != start and len(record.judgement.request_ids) == 2


async def test_a_plain_stop_ends_the_judge_at_once_and_ends_one_left_finishing_tests(judged):
    runtime = judged.runtime(_Runs())
    start = await judged.repo.head(BRANCH)
    await runtime.start(BRANCH, start)
    first = await judged.explore(FAST)
    await runtime.stop(BRANCH)
    assert not _judges()

    await runtime.start("s2", start)
    await runtime.stop("s2", finish_tests=True)
    await runtime.stop("s2")
    assert not [task for task in asyncio.all_tasks() if task.get_name() == "judge of s2" and not task.done()]

    await judged.toolbox.drain()
    await asyncio.sleep(0.1)
    assert (await judged.reader.record(first)).judgement is None


async def test_a_refused_retest_of_a_finishing_branch_is_noted_and_its_judge_ends_without_failing(judged):
    runtime = judged.runtime(_Runs())
    await runtime.start(BRANCH, await judged.repo.head(BRANCH))
    first = await judged.explore(FAST)
    stopped = RequestRejected(f"branch {BRANCH!r} has no quota", code="branch_stopped")
    judged.toolbox.reject_if(lambda request: stopped if request.purpose is Purpose.CONFIRM else None)

    await runtime.stop(BRANCH, finish_tests=True)
    await judged.toolbox.drain()
    await _until(lambda: _true(not _judges()), "the judge to end")

    assert (await judged.reader.record(first)).judgement is None
    notices = await judged.notices()
    assert f"{NOTICE} the retest of {first} was refused ({stopped}); nothing was judged" in notices
    assert not any("judging this branch failed" in notice for notice in notices)
