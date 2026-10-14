# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""In-memory stand-ins for the peers a session runs on, the toolbox's own fake, and a virtual clock, for the tests.

Each ``MbFake`` class stands in for another owner's module until it lands, one per peer: the specialist lifecycle
(``hyperloom_multibranch.specialist``), the rule gate (``.rulegate``) and the experiment repository (``.repo``), plus
the main-flow child of PRELUDE. Every scheduling test uses these, scripted through their fields, and subclasses them
only for a failure double; they are replaced by the real modules once those land. The toolbox is the toolbox
module's :class:`~hyperloom_multibranch.toolbox.fake.FakeToolbox`, noise-free and noting what the scheduler asked of
it; the rule gate judges from scripted values.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import heapq
import itertools
import shutil
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ConfigDelta, LaunchConfig, PatchRef, Recipe, RecipeDelta, RecipeError
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Notebook
from hyperloom_multibranch.contracts.notebook.peers import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord, LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, LineageKind
from hyperloom_multibranch.contracts.scheduling import (
    AdmissionStatus,
    ForkAdmission,
    ForkAdmissionService,
    ForkRequest,
    RepoMerge,
)
from hyperloom_multibranch.contracts.toolbox import CancelOutcome, QueueState
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_PRIMARY_METRIC
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, RegisteredProcess
from hyperloom_multibranch.scheduling.requests import SessionTests
from hyperloom_multibranch.toolbox.fake import FakeToolbox, ToolboxNotebook
from hyperloom_multibranch.toolbox.fake_model import DEFAULT_BASELINE, FakeToolboxConfig, Fate, NoiseModel
from hyperloom_multibranch.toolbox.queue import QueuePolicy

LAUNCH_FLAG = "--launch-info-file"
ROOT_COMMIT = "c0"
BASELINE_TESTS = 3
BASELINE_VALUE = DEFAULT_BASELINE[MB_FAKE_PRIMARY_METRIC]
TOOLBOX_DOWN = "toolbox service down"
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
    k: dict[str, int] = field(default_factory=dict)
    stopped: list[str] = field(default_factory=list)
    tasks: dict[str, asyncio.Future[None]] = field(default_factory=dict)

    async def start(self, branch: str, start_commit: str) -> None:
        self.started[branch] = start_commit
        if branch in self.scripts:
            self.tasks[branch] = asyncio.ensure_future(self.scripts[branch](branch, start_commit))

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


#: How a test of a commit fares: its primary-metric value per conc (the main operating point under conc 0).
Curves = Mapping[str, Mapping[int, float]]
MAIN_POINT = 0


def curve_effect(curves: Curves) -> Callable[[TestRequest], float]:
    """The fake toolbox's effect of a test: its parent commit's value at the test's conc, else the baseline."""

    def effect(request: TestRequest) -> float:
        conc = request.concurrency[0] if request.concurrency else MAIN_POINT
        return curves.get(request.parent_commit, {}).get(conc, BASELINE_VALUE) / BASELINE_VALUE - 1.0

    return effect


class RecordingToolbox(FakeToolbox):
    """The toolbox module's fake, without noise, noting every queue policy, branch cancel and close it was given.

    ``curves`` sets what a commit measures; a test whose parent commit is in ``failing`` fails its measurement, which
    the toolbox writes to the notebook as a withdrawal. Not a placeholder: the fake is the toolbox owner's own.
    """

    def __init__(self, notebook: ToolboxNotebook, curves: Curves | None = None, failing: frozenset[str] = frozenset()):
        quiet = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
        super().__init__(FakeToolboxConfig(noise=quiet, effect_fn=curve_effect(curves or {})), notebook=notebook)
        self.policies: list[QueuePolicy] = []
        self.cancelled: list[str] = []
        self.closed = 0
        self.script(lambda request: request.parent_commit in failing, [Fate.MEASUREMENT])

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


def judged(verdict: Verdict, gain_pct: float, request_id: str, value: float) -> Judgement:
    return Judgement(verdict, gain_pct, 1.0, (request_id,), {MB_FAKE_PRIMARY_METRIC: value}, accuracy_ok=True)


@dataclass
class MbFakeGate:
    """Stands in for the rule gate: scripted judgements, and commits and forks made for a scripted specialist.

    ``compare`` answers ``verdicts`` by the fingerprint of the test's delta, else ``default``; a test whose parent
    commit is in ``regressed`` regressed. No judgement is given before ``held_until`` tests are being compared at once.
    Accuracy fails for a parent commit in ``inaccurate``. The scripted specialist's candidates are measured through
    ``tests``. Deleted once ``hyperloom_multibranch.rulegate`` ships a real rule gate and these tests run on it.
    """

    notebook: Notebook
    tests: SessionTests
    desk: ForkAdmissionService | None = None
    verdicts: Mapping[str, Judgement] = field(default_factory=dict)
    default: Judgement = field(default_factory=lambda: judged(Verdict.FLAT, 0.0, "", BASELINE_VALUE))
    regressed: frozenset[str] = frozenset()
    inaccurate: frozenset[str] = frozenset()
    held_until: int = 0
    established: list[BaselineRecord] = field(default_factory=list)
    forks: list[ForkAdmission] = field(default_factory=list)
    comparing: asyncio.Event = field(default_factory=asyncio.Event)
    _in_flight: int = 0
    _all_in: asyncio.Event = field(default_factory=asyncio.Event)

    async def baseline_test_count(self) -> int:
        return BASELINE_TESTS

    async def establish_baseline(self, root_commit: str, request_ids: tuple[str, ...]) -> BaselineRecord:
        record = BaselineRecord(
            root_commit, {MB_FAKE_PRIMARY_METRIC: BASELINE_VALUE}, {MB_FAKE_PRIMARY_METRIC: 1.0}, {}, request_ids
        )
        await self.notebook.write_baseline(record)
        self.established.append(record)
        return record

    async def compare(self, request_id: str) -> Judgement:
        self.comparing.set()
        self._in_flight += 1
        if self._in_flight >= self.held_until:
            self._all_in.set()
        await self._all_in.wait()
        test = (await self.notebook.record(request_id)).request
        made = self.verdicts.get(test.delta.fingerprint(), self.default)
        if test.parent_commit in self.regressed:
            made = replace(made, verdict=Verdict.REGRESSED)
        return replace(made, request_ids=(request_id,))

    async def accuracy_passes(self, request_id: str) -> bool:
        return (await self.notebook.record(request_id)).request.parent_commit not in self.inaccurate

    async def commit(self, branch: str, parent: str, commit_id: str, value: float) -> None:
        """A specialist's candidate on ``branch`` measured, confirmed and committed as ``commit_id``."""

        request_id = await self._measure(branch, parent, commit_id)
        gain = (value / BASELINE_VALUE - 1.0) * 100.0
        await self.notebook.write_judgement(
            request_id, judged(Verdict.IMPROVED, gain, request_id, value), Conclusion.COMMITTED, commit_id
        )

    async def fork(self, from_branch: str, parent: str, fix_commit: str) -> ForkAdmission:
        """A correctness fix on ``from_branch`` that regresses there, asked onto a branch of its own."""

        assert self.desk is not None
        admission = await self.desk.admit(ForkRequest(from_branch, parent, f"fix {fix_commit} regresses"))
        self.forks.append(admission)
        if admission.status is not AdmissionStatus.ACCEPTED:
            return admission
        request_id = await self._measure(from_branch, parent, fix_commit)
        slower = replace(judged(Verdict.REGRESSED, -2.0, request_id, 980.0), fixes_correctness=True)
        await self.notebook.write_judgement(request_id, slower, Conclusion.SPLIT_REQUESTED)
        await self.notebook.record_lineage(
            LineageEvent(
                LineageKind.SPLIT, admission.branch, commit_id=fix_commit, from_commit=parent, from_branch=from_branch
            )
        )
        await self.notebook.write_judgement(request_id, slower, Conclusion.SPLIT_COMMITTED, fix_commit)
        await self.desk.fork_committed(admission.branch, fix_commit)
        return admission

    async def _measure(self, branch: str, parent: str, commit_id: str) -> str:
        change = RecipeDelta(config=ConfigDelta(server_args=f"--candidate {commit_id}"))
        test = TestRequest(
            kind=TestKind.TEST, purpose=Purpose.EXPLORE, branch_id=branch, parent_commit=parent, delta=change
        )
        return (await self.tests.run(test)).request_id


def merged_delta(candidate: str) -> RecipeDelta:
    """The change :class:`MbFakeRepo` merges ``candidate`` with when ``merges`` scripts none."""

    return RecipeDelta(config=ConfigDelta(server_args=f"--merged {candidate}"))


@dataclass
class MbFakeRepo:
    """Stands in for the experiment repository: merges are clean unless scripted in ``merges``.

    A merge of a candidate in ``invalid_merges`` raises the recipe contract's error. ``patches`` holds the patch texts
    by ``(commit, path)``; a checkout copies ``checkout_files`` when it is set. Deleted once
    ``hyperloom_multibranch.repo`` ships a real repository and these tests run on it.
    """

    recipes: dict[str, Recipe] = field(default_factory=dict)
    integration_head: str = ""
    merges: Mapping[str, RepoMerge] = field(default_factory=dict)
    invalid_merges: frozenset[str] = frozenset()
    patches: Mapping[tuple[str, str], str] = field(default_factory=dict)
    checkout_files: Path | None = None
    stored: list[tuple[PatchRef, str]] = field(default_factory=list)
    integrated: list[tuple[str, RecipeDelta, tuple[str, ...]]] = field(default_factory=list)
    checkouts: list[str] = field(default_factory=list)
    committing: asyncio.Event = field(default_factory=asyncio.Event)
    commit_released: asyncio.Event | None = None

    async def init_root(self, recipe: Recipe) -> str:
        self.recipes[ROOT_COMMIT] = recipe
        self.integration_head = ROOT_COMMIT
        return ROOT_COMMIT

    async def recipe_at(self, commit: str) -> Recipe:
        return self.recipes.get(commit, self.recipes[ROOT_COMMIT])

    async def head(self, branch: str) -> str:
        assert branch == INTEGRATION_BRANCH
        return self.integration_head

    async def merge(self, candidate: str, onto: str) -> RepoMerge:
        if candidate in self.invalid_merges:
            raise RecipeError(f"the merge of {candidate} pins two refs of one component")
        return replace(self.merges.get(candidate, RepoMerge("", merged_delta(candidate))), onto=onto)

    async def patch_text(self, commit: str, path: str) -> str:
        return self.patches.get((commit, path), "")

    async def store_patch(self, component: str, text: str) -> PatchRef:
        digest = hashlib.sha256(text.encode()).hexdigest()
        ref = PatchRef(f"patch/{component}/{len(self.stored):04d}-{digest[:8]}.patch", digest)
        self.stored.append((ref, text))
        return ref

    async def commit_integration(self, onto: str, delta: RecipeDelta, merged_from: tuple[str, ...]) -> str:
        assert onto == self.integration_head
        self.committing.set()
        if self.commit_released is not None:
            await self.commit_released.wait()
        self.integrated.append((onto, delta, merged_from))
        head = f"i{len(self.integrated)}"
        base = await self.recipe_at(onto)
        self.recipes[head] = replace(base, launch=LaunchConfig(server_args=f"--integrated {len(self.integrated)}"))
        self.integration_head = head
        return head

    async def checkout(self, commit: str, destination: Path) -> None:
        self.checkouts.append(commit)
        destination.mkdir(parents=True, exist_ok=True)
        if self.checkout_files is not None:
            shutil.copytree(self.checkout_files, destination, dirs_exist_ok=True)


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
    "BASELINE_TESTS",
    "BASELINE_VALUE",
    "CHILD_SCRIPT",
    "LAUNCH_FLAG",
    "MAIN_POINT",
    "ROOT_COMMIT",
    "TOOLBOX_DOWN",
    "Curves",
    "DownToolbox",
    "MbFakeChildRegistry",
    "MbFakeGate",
    "MbFakeHeldRegistry",
    "MbFakeLifecycle",
    "MbFakeRepo",
    "RecordingToolbox",
    "ScriptedCompletion",
    "SpecialistScript",
    "StuckToolbox",
    "VirtualClock",
    "curve_effect",
    "judged",
    "merged_delta",
]
