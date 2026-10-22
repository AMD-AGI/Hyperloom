# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import itertools

import pytest

from hyperloom.contracts.recipe import LaunchConfig, Recipe
from hyperloom_multibranch.contracts.fork import AdmissionStatus, RepairRequest
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.lifecycle import BranchTask, SpecialistStatus
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, ClaimStatus, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import delta, intent
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.rules import InferenceXRule
from hyperloom_multibranch.specialist.directions import DIRECTIONS
from hyperloom_multibranch.specialist.judging import NOTICE, Judging
from hyperloom_multibranch.specialist.runtime import (
    MAX_LOSS_RESTARTS,
    MAX_RESTARTS,
    RESTART_WINDOW_S,
    RuntimePacing,
    SessionCrashed,
    SessionIdle,
    SessionStart,
    SpecialistRuntime,
)
from hyperloom_multibranch.specialist.scripted import ScriptedSession, ScriptedSetup
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingScheduler, RecordingVerifier
from hyperloom_multibranch.toolbox.fake import FakeToolbox
from hyperloom_multibranch.toolbox.fake_model import FakeToolboxConfig

BRANCH = "s1"
RECIPE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
SCRIPT = tuple(delta(f"--max-num-seqs {n}") for n in (256, 384, 512))


class _Runs:
    """Every run the runtime starts: the first ones raise ``errors`` in turn, the next ``crashes`` crash, the others
    idle until stopped."""

    def __init__(self, crashes: int = 0, errors: tuple[Exception, ...] = ()) -> None:
        self.crashes, self.errors = crashes, list(errors)
        self.started: list[_Run] = []

    def __call__(self, start: SessionStart) -> "_Run":
        run = _Run(self, start)
        self.started.append(run)
        return run


class _Run:
    def __init__(self, runs: _Runs, start: SessionStart) -> None:
        self._runs, self.start, self.stopped = runs, start, asyncio.Event()
        self.k = start.k

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
        self.stopped.set()


class _JudgeFails(LocalNotebook):
    async def poll(self, branch: str, cursor: str = ""):
        raise RuntimeError("the inbox is corrupt")


class _Session:
    """The branch ``s1`` opened at the root, on the fake toolbox, which measures nothing until it is drained."""

    def __init__(self, root) -> None:
        self.root = root
        self.toolbox = FakeToolbox(
            FakeToolboxConfig(), notebook=LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox")), autorun=False
        )
        self.repo = InMemoryRepo(RECIPE)
        self.reader = LocalNotebook(root, Writer(Role.OBSERVER, "observer"))
        self.gate = RuleGate(
            LocalNotebook(root, Writer(Role.GATE, "gate")),
            self.repo,
            RecordingScheduler(AdmissionStatus.ACCEPTED),
            RecordingVerifier(),
        )

    async def open(self) -> str:
        scheduler = LocalNotebook(self.root, Writer(Role.SCHEDULER, "scheduler"))
        start = await self.repo.head(INTEGRATION_BRANCH)
        await scheduler.record_lineage(
            LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=start, recipe_digest=RECIPE.digest())
        )
        await self.repo.create_branch(BRANCH, start)
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, BRANCH, from_commit=start))
        return start

    def runtime(self, runs, *, notebook=LocalNotebook, clock=lambda: 0.0, claim_ttl=600.0) -> SpecialistRuntime:
        def notebook_for(branch: str) -> LocalNotebook:
            return notebook(self.root, Writer(Role.SPECIALIST_RUNTIME, branch, branch))

        judging = Judging(self.gate, self.toolbox, InferenceXRule(self.reader, 64))
        pacing = RuntimePacing(poll_seconds=0.01, claim_ttl_seconds=claim_ttl, clock=clock)
        return SpecialistRuntime(notebook_for, judging, runs, pacing=pacing)

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
    session = _Session(tmp_path / "notebook")
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
    blocked = Judgement(Verdict.BLOCKED, 3.0, 2.0, ("s0-4", "s0-5"), {"output_throughput": 1030.0}, False)
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
    blocked = Judgement(Verdict.BLOCKED, 3.0, 2.0, ("s0-4", "s0-5"), {"output_throughput": 1030.0}, False)
    repair = RepairRequest(intent("s0", "c0"), delta(), blocked, "per-tensor KV scales; scale per head")
    handed = [f"b{index}" for index in range(len(DIRECTIONS) + 1)]

    await runtime.start(handed[0], head)
    await runtime.start("user", head, BranchTask(direction="the user's own"))
    await runtime.start("repair", head, BranchTask(repair=repair))
    for name in handed[1:]:
        await runtime.start(name, head)
    await _until(lambda: _true(len(runs.started) == len(handed) + 2), "every specialist to start")
    for name in [*handed, "user", "repair"]:
        await runtime.stop(name)

    directions = {run.start.branch: run.start.task.direction for run in runs.started}
    assert [directions[name] for name in handed] == [*DIRECTIONS, DIRECTIONS[0]]
    assert (directions["user"], directions["repair"]) == ("the user's own", "")


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


async def test_a_session_that_worked_between_losses_counts_them_again_from_its_own(session):
    runs = _Runs(errors=(_lost(), _lost(), _lost(worked=True), _lost()))
    ticks = itertools.count(step=RESTART_WINDOW_S + 1)
    runtime = session.runtime(runs, clock=lambda: next(ticks))

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _true(len(runs.started) == 5), "every loss to be restarted")

    assert not any("not restarted" in notice for notice in await session.notices())
    assert await runtime.status(BRANCH) == SpecialistStatus(True, 4, 4, False, False)
    await runtime.stop(BRANCH)


def _idle(tested: bool = False) -> SessionIdle:
    return SessionIdle("its model ended 3 turns in a row with nothing to test", tested=tested)


@pytest.mark.parametrize("kind", ["", "direction", "repair"])
async def test_two_idle_sessions_in_a_row_end_the_specialist_after_one_fresh_session_and_no_crash(session, kind):
    blocked = Judgement(Verdict.BLOCKED, 3.0, 2.0, ("s0-4", "s0-5"), {"output_throughput": 1030.0}, False)
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
    runtime = session.runtime(runs, notebook=_JudgeFails)

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
    with pytest.raises(KeyError):
        await runtime.stop(BRANCH)


async def test_the_claims_a_specialist_holds_are_renewed_until_its_branch_stops(session):
    runtime = session.runtime(_Runs(), claim_ttl=0.3)
    start = await session.repo.head(BRANCH)
    await runtime.start(BRANCH, start)
    holder = LocalNotebook(session.root, Writer(Role.SPECIALIST_RUNTIME, BRANCH, BRANCH))
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
