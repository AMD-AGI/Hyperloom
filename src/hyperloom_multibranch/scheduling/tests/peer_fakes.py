# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""In-memory stand-ins for the peers a session runs on, and a virtual clock, for the scheduling tests.

Each ``MbFake`` class stands in for another owner's module until it lands, one per peer: the specialist lifecycle
(``hyperloom_multibranch.specialist``), the toolbox (``.toolbox``), the rule gate (``.rulegate``) and the experiment
repository (``.repo``), plus the main-flow child of PRELUDE. Every scheduling test uses these, scripted through their
fields, and subclasses them only for a failure double; they are replaced by the real modules once those land. The
toolbox writes every request and its result to the notebook at once; the rule gate judges from scripted values.
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

from hyperloom.contracts.recipe import ConfigDelta, LaunchConfig, PatchRef, Recipe, RecipeDelta, RecipeError
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Notebook
from hyperloom_multibranch.contracts.notebook.peers import (
    Judgement,
    SandboxFacts,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
    Verdict,
)
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord, LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, LineageKind
from hyperloom_multibranch.contracts.scheduling import (
    AdmissionStatus,
    ForkAdmission,
    ForkAdmissionService,
    ForkRequest,
    OperatingPoint,
    QueueStatus,
    QuotaPolicy,
    RepoMerge,
    RequestWithdrawn,
    SweepResult,
)
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_PRIMARY_METRIC
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, RegisteredProcess

LAUNCH_FLAG = "--launch-info-file"
ROOT_COMMIT = "c0"
BASELINE_TESTS = 3
BASELINE_VALUE = 1000.0
#: What the toolbox stand-in records for a withdrawal of every branch.
ALL_BRANCHES = "*"
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


@dataclass
class MbFakeToolbox:
    """Stands in for the toolbox: every test is written to the notebook with its result at once.

    Once a quota policy was applied, a test of a branch without quota is refused as withdrawn, the way the toolbox
    treats a stopped branch. ``sweeps`` gives a commit's primary-metric value per conc; a commit without one sweeps
    no point. The results of tests whose parent commit is in ``withdrawing`` are withdrawn, and withdrawing a branch in
    ``stuck`` never returns. While ``broken`` every call fails as a toolbox whose service is down. Deleted once ``hyperloom_multibranch.toolbox`` ships a real
    toolbox and these tests run on it.
    """

    notebook: Notebook
    sweeps: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    withdrawing: frozenset[str] = frozenset()
    queue: QueueStatus = field(default_factory=QueueStatus)
    broken: bool = False
    stuck: frozenset[str] = frozenset()
    submitted: list[TestRequest] = field(default_factory=list)
    results: dict[str, TestResult] = field(default_factory=dict)
    quotas: list[QuotaPolicy] = field(default_factory=list)
    withdrawn: list[str] = field(default_factory=list)
    swept: list[tuple[str, str]] = field(default_factory=list)

    async def submit(self, request: TestRequest) -> str:
        self._serving()
        if self.quotas and self.quotas[-1].k_of(request.branch) is None:
            raise RequestWithdrawn(f"{request.request_id}: {request.branch} has no quota")
        made = TestResult(request.request_id, Validity.SUCCESS, SandboxFacts("node-0"), "m" * 64, gpu_seconds=60.0)
        await self.notebook.write_request(request)
        await self.notebook.write_result(made)
        self.submitted.append(request)
        self.results[request.request_id] = made
        return request.request_id

    async def result(self, request_id: str) -> TestResult:
        self._serving()
        if self.request(request_id).parent_commit in self.withdrawing:
            raise RequestWithdrawn(request_id)
        return self.results[request_id]

    async def sweep(self, branch: str, commit: str) -> SweepResult:
        self._serving()
        self.swept.append((branch, commit))
        curve = self.sweeps.get(commit, {})
        points = tuple(OperatingPoint(conc, {MB_FAKE_PRIMARY_METRIC: value}) for conc, value in curve.items())
        return SweepResult(commit, points, ())

    async def withdraw(self, branch: str) -> tuple[str, ...]:
        self._serving()
        self.withdrawn.append(branch)
        if branch in self.stuck:
            await asyncio.Event().wait()
        return ()

    async def withdraw_all(self) -> tuple[str, ...]:
        self._serving()
        self.withdrawn.append(ALL_BRANCHES)
        return ()

    async def queue_status(self) -> QueueStatus:
        self._serving()
        return self.queue

    async def apply_quota(self, policy: QuotaPolicy) -> None:
        self._serving()
        self.quotas.append(policy)

    def request(self, request_id: str) -> TestRequest:
        return next(test for test in self.submitted if test.request_id == request_id)

    def _serving(self) -> None:
        if self.broken:
            raise ConnectionError(TOOLBOX_DOWN)


def judged(verdict: Verdict, gain_pct: float, request_id: str, value: float) -> Judgement:
    return Judgement(verdict, gain_pct, 1.0, True, (request_id,), {MB_FAKE_PRIMARY_METRIC: value})


@dataclass
class MbFakeGate:
    """Stands in for the rule gate: scripted judgements, and commits and forks made for a scripted specialist.

    ``compare`` waits for the test's result, then answers ``verdicts`` by the test's intent fingerprint, else
    ``default``; a test whose parent commit is in ``regressed`` regressed. No judgement is given before
    ``held_until`` tests are being compared at once. Accuracy fails for a parent commit in ``inaccurate``. Deleted
    once ``hyperloom_multibranch.rulegate`` ships a real rule gate and these tests run on it.
    """

    notebook: Notebook
    toolbox: MbFakeToolbox
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
        await self.toolbox.result(request_id)
        test = self.toolbox.request(request_id)
        made = self.verdicts.get(test.intent_fingerprint, self.default)
        if test.parent_commit in self.regressed:
            made = replace(made, verdict=Verdict.REGRESSED)
        return replace(made, request_ids=(request_id,))

    async def accuracy_passes(self, request_id: str) -> bool:
        return self.toolbox.request(request_id).parent_commit not in self.inaccurate

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
        test = TestRequest(f"r-{commit_id}", TestKind.TEST, branch, parent, f"intent:{commit_id}", change)
        return await self.toolbox.submit(test)


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
        clean = RepoMerge("", RecipeDelta(config=ConfigDelta(server_args=f"--merged {candidate}")))
        return replace(self.merges.get(candidate, clean), onto=onto)

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
    "ALL_BRANCHES",
    "BASELINE_TESTS",
    "BASELINE_VALUE",
    "CHILD_SCRIPT",
    "LAUNCH_FLAG",
    "ROOT_COMMIT",
    "TOOLBOX_DOWN",
    "MbFakeChildRegistry",
    "MbFakeGate",
    "MbFakeHeldRegistry",
    "MbFakeLifecycle",
    "MbFakeRepo",
    "MbFakeToolbox",
    "ScriptedCompletion",
    "SpecialistScript",
    "VirtualClock",
    "judged",
]
