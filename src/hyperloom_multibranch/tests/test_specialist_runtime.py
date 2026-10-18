# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import itertools

import pytest

from hyperloom.contracts.recipe import LaunchConfig, Recipe
from hyperloom_multibranch.contracts.fork import AdmissionStatus, RepairRequest
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, ClaimStatus, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import delta, intent
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.rules import InferenceXRule
from hyperloom_multibranch.specialist.judging import NOTICE, Judging
from hyperloom_multibranch.specialist.runtime import (
    MAX_RESTARTS,
    RESTART_WINDOW_S,
    RuntimePacing,
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
    """Every run the runtime starts: the first ``crashes`` of them crash, the others idle until stopped."""

    def __init__(self, crashes: int = 0) -> None:
        self.crashes = crashes
        self.started: list[_Run] = []

    def __call__(self, start: SessionStart) -> "_Run":
        run = _Run(self, start)
        self.started.append(run)
        return run


class _Run:
    def __init__(self, runs: _Runs, start: SessionStart) -> None:
        self._runs, self.start, self.stopped = runs, start, asyncio.Event()
        self.k, self.brief = start.k, start.brief

    async def run(self) -> None:
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
        return ScriptedSession(SessionStart(BRANCH, start_commit="", brief="", k=k), setup)

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


async def test_a_crashed_specialist_is_restarted_with_its_brief_its_repair_and_a_notice_each_time(session):
    runs = _Runs(crashes=2)
    runtime = session.runtime(runs)
    blocked = Judgement(Verdict.BLOCKED, 3.0, 2.0, ("s0-4", "s0-5"), {"output_throughput": 1030.0}, False)
    repair = RepairRequest(intent("s0", "c0"), delta(), blocked, "per-tensor KV scales; scale per head")

    await runtime.start(BRANCH, await session.repo.head(BRANCH), brief="leads: fp8 kv cache", repair=repair)
    await _until(lambda: _true(len(runs.started) == 3), "two restarts")

    assert all(run.stopped.is_set() for run in runs.started[:2])
    assert [(run.brief, run.start.repair) for run in runs.started] == [("leads: fp8 kv cache", repair)] * 3
    assert len([notice for notice in await session.notices() if "is restarted" in notice]) == 2
    await runtime.stop(BRANCH)


async def test_past_the_restarts_a_window_allows_the_specialist_stays_down(session):
    runs = _Runs(crashes=10)
    runtime = session.runtime(runs)

    await runtime.start(BRANCH, await session.repo.head(BRANCH))
    await _until(lambda: _any_notice(session, "not restarted"), "the runtime to give up")

    assert len(runs.started) == 1 + MAX_RESTARTS
    await runtime.set_k(BRANCH, 2)
    await runtime.stop(BRANCH)


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
