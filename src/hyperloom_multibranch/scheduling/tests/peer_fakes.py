# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stand-ins for the peers a session runs on, the toolbox's own fake, and a virtual clock, for the tests.

Each ``MbFake`` class stands in for another owner's module until it lands: the specialist lifecycle and a
specialist's candidate loop (``hyperloom_multibranch.specialist``), plus the main-flow child of PRELUDE. Every
scheduling test uses these, scripted through their fields, and subclasses them only for a failure double; they are
replaced by the real modules once those land. The toolbox is the toolbox module's
:class:`~hyperloom_multibranch.toolbox.fake.FakeToolbox`, noise-free and noting what the scheduler asked of it; what a
test measures follows from the recipe it measures (:func:`recipe_effect`). The experiment repository and the rule gate
are their owners' own.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ConfigDelta, Recipe, RecipeDelta, apply
from hyperloom_multibranch.contracts.fork import Admission
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.lifecycle import SpecialistLaunch
from hyperloom_multibranch.contracts.notebook import peers as notebook_peers
from hyperloom_multibranch.contracts.notebook.interfaces import RecordSearch
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import CancelOutcome, QueueState
from hyperloom_multibranch.repo.git_repo import RECIPE_FILE, GitExperimentRepo
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.peers import RootWriter, git_root_writer
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, RegisteredProcess
from hyperloom_multibranch.scheduling.requests import SessionTests, integration_test_request
from hyperloom_multibranch.scheduling.rule_judge import MB_FAKE_BASELINE_TESTS, MbFakeRuleJudge
from hyperloom_multibranch.toolbox.fake import FakeToolbox, ToolboxNotebook
from hyperloom_multibranch.toolbox.fake_model import DEFAULT_BASELINE, FakeToolboxConfig, NoiseModel
from hyperloom_multibranch.toolbox.queue import QueuePolicy

LAUNCH_FLAG = "--launch-info-file"
BASELINE_VALUE = DEFAULT_BASELINE[GRADED_OUTPUT]
TOOLBOX_DOWN = "toolbox service down"
#: The server argument a scripted change carries its tag in; :func:`recipe_effect` reads a recipe's tags from it.
TAG_FLAG = "--mb-tag-"
REPRO_CASE = "repro/case.py"
#: Loop turns and real seconds the virtual clock lets pass before it decides the loop is quiet and jumps.
SETTLE_YIELDS = 20
SETTLE_REAL_SECONDS = 0.0005
#: Writes the launch info naming argv[2] (when set), lingers argv[4] seconds with a grandchild, exits with argv[3].
CHILD_SCRIPT = """
import json, subprocess, sys, time
launch, session, code, linger = sys.argv[1:5]
if session:
    open(launch, "w").write(json.dumps({"session_dir": session}))
if float(linger):
    subprocess.Popen(["sleep", linger])
    time.sleep(float(linger))
sys.exit(int(code))
"""

SpecialistScript = Callable[[str, str], Awaitable[None]]


class _HoldingExecutor(ThreadPoolExecutor):
    """A default executor whose every job holds a :class:`VirtualClock` until the job is done."""

    def __init__(self, time: VirtualClock, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._time = time
        self._loop = loop

    def submit(self, fn: Callable[..., object], /, *args: object, **kwargs: object) -> Future[object]:
        self._time.hold()
        future = super().submit(fn, *args, **kwargs)
        future.add_done_callback(lambda _: self._loop.call_soon_threadsafe(self._time.release))
        return future


class VirtualClock:
    """Virtual time for an in-process session: it jumps to the earliest sleeper's wake time once the loop is quiet.

    The loop is quiet when a whole settle window passed with no hold, release or new sleeper. Real work the event loop
    cannot see through, a child process or a thread, holds the clock while it is in flight, so virtual time never
    runs past a deadline while that work is still going.
    """

    def __init__(self, start: float) -> None:
        self._now = start
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._order = itertools.count()
        self._holds = 0
        self._activity = 0
        self._driver: asyncio.Task[None] | None = None

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        wake: asyncio.Future[None] = loop.create_future()
        heapq.heappush(self._sleepers, (self._now + max(seconds, 0.0), next(self._order), wake))
        self._activity += 1
        if self._driver is None or self._driver.done():
            self._driver = loop.create_task(self._drive())
        await wake

    def hold(self) -> None:
        self._holds += 1
        self._activity += 1

    def release(self) -> None:
        self._holds -= 1
        self._activity += 1

    @contextlib.asynccontextmanager
    async def held(self) -> AsyncIterator[None]:
        self.hold()
        try:
            yield
        finally:
            self.release()

    @property
    def clock(self) -> Clock:
        return Clock(now=self.now, sleep=self.sleep)

    def hold_threads(self) -> None:
        """Make every :func:`asyncio.to_thread` call on the running loop hold the clock while its thread works."""

        loop = asyncio.get_running_loop()
        loop.set_default_executor(_HoldingExecutor(self, loop))

    async def _drive(self) -> None:
        while self._sleepers:
            seen = self._activity
            for _ in range(SETTLE_YIELDS):
                await asyncio.sleep(0)
            await asyncio.sleep(SETTLE_REAL_SECONDS)
            if not self._holds and seen == self._activity:
                self._wake_earliest()

    def _wake_earliest(self) -> None:
        while self._sleepers and self._sleepers[0][2].done():
            heapq.heappop(self._sleepers)
        if self._sleepers:
            self._now = max(self._now, self._sleepers[0][0])
        while self._sleepers and self._sleepers[0][0] <= self._now:
            wake = heapq.heappop(self._sleepers)[2]
            if not wake.done():
                wake.set_result(None)


class HeldGitRepo(GitExperimentRepo):
    """The git experiment repository on a :class:`VirtualClock`: every call holds the clock while git runs, which the
    event loop cannot see through. Not a placeholder: the repository is its owner's own."""

    def __init__(self, path: Path, time: VirtualClock) -> None:
        super().__init__(path)
        self.time = time

    async def read_recipe(self, commit: str) -> Recipe:
        async with self.time.held():
            return await super().read_recipe(commit)

    async def create_branch(self, branch: str, at_commit: str) -> None:
        async with self.time.held():
            await super().create_branch(branch, at_commit)

    async def head(self, branch: str) -> str:
        async with self.time.held():
            return await super().head(branch)

    async def commit(self, branch: str, parent: str, recipe: Recipe, message: str) -> str:
        async with self.time.held():
            return await super().commit(branch, parent, recipe, message)

    async def rewind(self, branch: str, commit: str, parent: str) -> None:
        async with self.time.held():
            await super().rewind(branch, commit, parent)


def held_root_writer(path: Path, time: VirtualClock) -> RootWriter:
    """:func:`~hyperloom_multibranch.scheduling.peers.git_root_writer` holding ``time`` while git runs."""

    write_root = git_root_writer(path)

    async def held(baseline: Recipe) -> str:
        async with time.held():
            return await write_root(baseline)

    return held


class MbFakeChildRegistry(ProcessRegistry):
    """Runs a short script in place of the main-flow child of PRELUDE; every other process is spawned as asked.

    Stands in for the ``hyperloom optimize`` child; deleted once the skeleton runs PRELUDE on a hosted main flow.
    """

    def __init__(self, pids_file: Path, clock: Clock, *, session: str, exit_code: int, linger: float) -> None:
        super().__init__(pids_file, clock)
        self._script_args = (session, str(exit_code), str(linger))
        self.argv: tuple[str, ...] = ()

    async def spawn(self, argv: Sequence[str], *, branch: str, log: Path) -> RegisteredProcess:
        if LAUNCH_FLAG not in argv:
            return await super().spawn(argv, branch=branch, log=log)
        self.argv = tuple(argv)
        launch = argv[argv.index(LAUNCH_FLAG) + 1]
        script = (sys.executable, "-c", CHILD_SCRIPT, launch, *self._script_args)
        return await super().spawn(script, branch=branch, log=log)


class MbFakeHeldRegistry(MbFakeChildRegistry):
    """The PRELUDE child stand-in on a :class:`VirtualClock`: process work holds the clock while it runs.

    The PRELUDE child holds it until it exits; every other spawn and every kill holds it for the call alone.
    Deleted together with :class:`MbFakeChildRegistry`.
    """

    def __init__(self, pids_file: Path, time: VirtualClock, *, session: str, exit_code: int) -> None:
        super().__init__(pids_file, time.clock, session=session, exit_code=exit_code, linger=0.0)
        self._time = time
        self._watchers: list[asyncio.Task[None]] = []

    async def spawn(self, argv: Sequence[str], *, branch: str, log: Path) -> RegisteredProcess:
        async with self._time.held():
            spawned = await super().spawn(argv, branch=branch, log=log)
        if LAUNCH_FLAG in argv:
            self._time.hold()
            self._watchers.append(asyncio.ensure_future(self._release_on_exit(spawned)))
        return spawned

    async def kill_branch(self, branch: str) -> bool:
        async with self._time.held():
            return await super().kill_branch(branch)

    async def kill_all(self) -> bool:
        async with self._time.held():
            return await super().kill_all()

    async def _release_on_exit(self, spawned: RegisteredProcess) -> None:
        try:
            await spawned.process.wait()
        finally:
            self._time.release()


@dataclass
class MbFakeLifecycle:
    """Stands in for the specialist lifecycle: a started branch runs its scripted specialist as a task.

    Stopping a branch in ``refusing`` raises, and stopping one in ``hanging`` never returns. Deleted once
    ``hyperloom_multibranch.specialist`` ships a real lifecycle and these tests run on it.
    """

    scripts: Mapping[str, SpecialistScript] = field(default_factory=dict)
    refusing: frozenset[str] = frozenset()
    hanging: frozenset[str] = frozenset()
    started: dict[str, str] = field(default_factory=dict)
    briefs: dict[str, str | None] = field(default_factory=dict)
    k: dict[str, int] = field(default_factory=dict)
    stopped: list[str] = field(default_factory=list)
    tasks: dict[str, asyncio.Future[None]] = field(default_factory=dict)

    async def start(self, launch: SpecialistLaunch) -> None:
        self.started[launch.branch] = launch.start_commit
        self.briefs[launch.branch] = launch.brief
        self.k[launch.branch] = launch.k
        if launch.branch in self.scripts:
            script = self.scripts[launch.branch](launch.branch, launch.start_commit)
            self.tasks[launch.branch] = asyncio.ensure_future(script)

    async def stop(self, branch: str) -> None:
        self.stopped.append(branch)
        if branch in self.refusing:
            raise RuntimeError(f"the runtime refused to stop {branch}")
        if branch in self.hanging:
            await asyncio.Event().wait()
        task = self.tasks.pop(branch, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def set_k(self, branch: str, k: int) -> None:
        self.k[branch] = k


#: What a recipe measures, by the tags its server arguments carry: its primary-metric value per conc, the main
#: operating point under :data:`MAIN_POINT`; a recipe or conc not named measures :data:`BASELINE_VALUE`.
Curves = Mapping[frozenset[str], Mapping[int, float]]
MAIN_POINT = 0


def tagged(*tags: str) -> RecipeDelta:
    """A change that carries ``tags`` and nothing else that matters."""

    return RecipeDelta(config=ConfigDelta(server_args=" ".join(f"{TAG_FLAG}{tag}" for tag in tags)))


def recipe_tags(recipe: Recipe) -> frozenset[str]:
    return frozenset(arg[len(TAG_FLAG) :] for arg in recipe.launch.server_args.split() if arg.startswith(TAG_FLAG))


def _committed_recipe(repo: Path, commit: str) -> Recipe:
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    shown = subprocess.run(
        ("git", f"--git-dir={repo}", "show", f"{commit}:{RECIPE_FILE}"),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return Recipe.from_document(json.loads(shown.stdout))


def recipe_effect(repo: Path, curves: Curves) -> Callable[[TestRequest], float]:
    """The fake toolbox's effect of a test: what ``curves`` gives the tags of the recipe it measures, its parent
    commit's recipe in the git repository at ``repo`` with its change applied."""

    def effect(request: TestRequest) -> float:
        tags = recipe_tags(apply(_committed_recipe(repo, request.parent_commit), request.delta))
        conc = request.concurrency[0] if request.concurrency else MAIN_POINT
        return curves.get(tags, {}).get(conc, BASELINE_VALUE) / BASELINE_VALUE - 1.0

    return effect


class RecordingToolbox(FakeToolbox):
    """The toolbox module's fake, without noise, noting every queue policy, branch cancel and close it was given.

    ``effect`` sets what a test measures (see :func:`recipe_effect`); without one every test measures the baseline.
    ``accuracy`` is the accuracy score of a change by its fingerprint. Not a placeholder: the fake is the toolbox owner's own.
    """

    def __init__(
        self,
        notebook: ToolboxNotebook,
        effect: Callable[[TestRequest], float] | None = None,
        accuracy: Mapping[str, float] | None = None,
    ):
        quiet = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
        config = FakeToolboxConfig(noise=quiet, effect_fn=effect, accuracy_scores=dict(accuracy or {}))
        super().__init__(config, notebook=notebook)
        self.policies: list[QueuePolicy] = []
        self.cancelled: list[str] = []
        self.closed = 0

    async def set_policy(self, policy: QueuePolicy) -> None:
        self.policies.append(policy)
        await super().set_policy(policy)

    async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
        self.cancelled.append(branch_id)
        return await super().cancel_branch(branch_id, reason)

    async def close(self) -> None:
        self.closed += 1
        await super().close()


class DownToolbox(RecordingToolbox):
    """A toolbox whose service fails every call while ``down``; a failure double, not a placeholder."""

    down = False

    def _serving(self) -> None:
        if self.down:
            raise ConnectionError(TOOLBOX_DOWN)

    async def submit_test(self, request: TestRequest) -> str:
        self._serving()
        return await super().submit_test(request)

    async def cancel(self, request_id: str) -> CancelOutcome:
        self._serving()
        return await super().cancel(request_id)

    async def queue_state(self) -> QueueState:
        self._serving()
        return await super().queue_state()

    async def set_policy(self, policy: QueuePolicy) -> None:
        self._serving()
        await super().set_policy(policy)

    async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
        self._serving()
        return await super().cancel_branch(branch_id, reason)

    async def close(self) -> None:
        self._serving()
        await super().close()


class StuckToolbox(RecordingToolbox):
    """A toolbox that never finishes closing, nor cancelling a branch in ``stuck``; a failure double, not a
    placeholder."""

    stuck: frozenset[str] = frozenset()

    async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
        if branch_id in self.stuck:
            self.cancelled.append(branch_id)
            await asyncio.Event().wait()
        return await super().cancel_branch(branch_id, reason)

    async def close(self) -> None:
        await asyncio.Event().wait()


async def measure_baseline(judge: MbFakeRuleJudge, tests: SessionTests, root: str) -> None:
    """Measure the baseline on ``root`` as PRELUDE does and anchor the session to it."""

    request = integration_test_request(TestKind.TEST, Purpose.BASELINE, root, accuracy=True)
    records = [await tests.run(request) for _ in range(MB_FAKE_BASELINE_TESTS)]
    await judge.establish_baseline(root, tuple(records))


def judged(verdict: Verdict, gain_pct: float, request_id: str, value: float) -> Judgement:
    return Judgement(verdict, gain_pct, 1.0, (request_id,), {GRADED_OUTPUT: value}, accuracy_ok=True)


@dataclass
class MbFakeSpecialist:
    """Stands in for a specialist's candidate loop: measures a scripted change on its branch through the toolbox and
    lands the judgement scripted for it through the rule gate, which commits, records or asks for a fork.

    Deleted once ``hyperloom_multibranch.specialist`` ships a real specialist and the skeleton runs on it.
    """

    gate: RuleGate
    tests: SessionTests
    notebook: RecordSearch
    repo: ExperimentRepo

    async def measure(self, branch: str, change: RecipeDelta) -> notebook_peers.TestRequest:
        parent = await self.repo.head(branch)
        test = TestRequest(
            kind=TestKind.TEST, purpose=Purpose.EXPLORE, branch_id=branch, parent_commit=parent, delta=change
        )
        return (await self.tests.run(test)).request

    async def commit(self, branch: str, change: RecipeDelta, value: float) -> str:
        """``change`` on ``branch``, confirmed at ``value``; returns the commit it made."""

        measured = await self.measure(branch, change)
        gain = (value / BASELINE_VALUE - 1.0) * 100.0
        await self.gate.settle(measured, judged(Verdict.IMPROVED, gain, measured.request_id, value))
        return await self.repo.head(branch)

    async def fork(self, branch: str, fix: RecipeDelta, reason: str) -> Admission:
        """A correctness fix on ``branch`` that regresses there, asked onto a branch of its own."""

        measured = await self.measure(branch, fix)
        slower = replace(judged(Verdict.REGRESSED, -2.0, measured.request_id, 980.0), fixes_correctness=True)
        return await self.gate.fork(measured, slower, REPRO_CASE, reason)


@dataclass
class ScriptedCompletion:
    """A model for the tests: raises ``error`` when it is set, else the scripted replies in order, then ``fallback``.

    Not a placeholder: a test never calls a real model.
    """

    replies: list[str] = field(default_factory=list)
    fallback: str = '{"open": 0, "branches": [], "stop": [], "stop_session": null}'
    error: Exception | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self.replies.pop(0) if self.replies else self.fallback


__all__ = [
    "BASELINE_VALUE",
    "CHILD_SCRIPT",
    "LAUNCH_FLAG",
    "MAIN_POINT",
    "REPRO_CASE",
    "TAG_FLAG",
    "TOOLBOX_DOWN",
    "Curves",
    "DownToolbox",
    "HeldGitRepo",
    "MbFakeChildRegistry",
    "MbFakeHeldRegistry",
    "MbFakeLifecycle",
    "MbFakeSpecialist",
    "RecordingToolbox",
    "ScriptedCompletion",
    "SpecialistScript",
    "StuckToolbox",
    "VirtualClock",
    "held_root_writer",
    "judged",
    "measure_baseline",
    "recipe_effect",
    "recipe_tags",
    "tagged",
]
