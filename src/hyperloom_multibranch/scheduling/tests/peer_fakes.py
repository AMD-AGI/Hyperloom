# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stand-ins for the peers a session runs on, the toolbox's own fake, and a virtual clock, for the tests.

Each ``MbFake`` class stands in for another owner's module until it lands: the main-flow child of PRELUDE. The
specialist lifecycle is the specialist runtime itself on scripted sessions (:class:`RecordedSessions`,
:func:`specialist_runtime`). The toolbox is the toolbox module's
:class:`~hyperloom_multibranch.toolbox.fake.FakeToolbox`, noise-free and noting what the scheduler asked of it; what a
test measures follows from the recipe it measures (:func:`recipe_effect`). The regression set is scripted per base
commit (:class:`FakeRegressionCheck`). The
experiment repository and the rule gate are their owners' own; :func:`session_gate` is the gate's session service as
the session's peers build it.
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
from typing import Any

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.measurement import TestRequest, TestResult
from hyperloom.contracts.recipe import ConfigDelta, PatchRef, Recipe, RecipeDelta, apply
from hyperloom.measure.compile import BlobStore
from hyperloom.orchestrator.knowledge.remote_recipe.client import KBStoreError
from hyperloom.orchestrator.knowledge.remote_recipe.models import RecipeScope
from hyperloom.orchestrator.knowledge.remote_recipe.values import (
    CURRENT_KNOWLEDGE_SCHEMA_VERSION,
    RECORD_KIND_HYPERLOOM_RECIPE,
)
from hyperloom_multibranch.contracts.fork import (
    ForkAdmission,
    ForkAdmissionService,
    ForkRequest,
    RepairRequest,
)
from hyperloom_multibranch.contracts.gate import RegressionOutcome
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.interfaces import SnapshotQuery
from hyperloom_multibranch.contracts.repo import ExperimentRepo, RepoMerge
from hyperloom_multibranch.contracts.toolbox import CancelOutcome, QueueState, RequestRejected
from hyperloom_multibranch.repo.git_repo import RECIPE_FILE, GitExperimentRepo
from hyperloom_multibranch.rulegate import baseline as rulegate_baseline
from hyperloom_multibranch.rulegate import rules as rulegate_rules
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.inferencex import InferenceXReading
from hyperloom_multibranch.rulegate.readings import inferencex_reading
from hyperloom_multibranch.rulegate.rules import InferenceXRule, Rule
from hyperloom_multibranch.rulegate.service import GateNotebook, GateService, Polling
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.local_peers import GATE_WRITER
from hyperloom_multibranch.scheduling.llm import CompletionReply
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, RegisteredProcess
from hyperloom_multibranch.specialist.judging import Judging
from hyperloom_multibranch.specialist.runtime import RuntimePacing, SessionFactory, SessionStart, SpecialistRuntime
from hyperloom_multibranch.specialist.scripted import BranchHeads, ScriptedSession, ScriptedSetup
from hyperloom_multibranch.toolbox.fake import FakeToolbox, FakeWiring
from hyperloom_multibranch.toolbox.fake_model import DEFAULT_BASELINE, FakeToolboxConfig, NoiseModel
from hyperloom_multibranch.toolbox.queue import QueuePolicy

LAUNCH_FLAG = "--launch-info-file"
BASELINE_VALUE = DEFAULT_BASELINE[GRADED_OUTPUT]
#: The digest of the image the tests' recipes are based on.
BASE_IMAGE_DIGEST = "sha256:" + "b" * 64
TOOLBOX_DOWN = "toolbox service down"
#: The server argument a scripted change carries its tag in; :func:`recipe_effect` reads a recipe's tags from it.
TAG_FLAG = "--mb-tag-"
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

#: What a test's specialist does on its branch before it proposes its script.
SessionStep = Callable[[str], Awaitable[None]]


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

    def __init__(self, path: Path, blobs: BlobStore, time: VirtualClock) -> None:
        super().__init__(path, blobs)
        self.time = time

    async def init_root(self, recipe: Recipe) -> str:
        async with self.time.held():
            return await super().init_root(recipe)

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

    async def merge(self, candidate: str, onto: str) -> RepoMerge:
        async with self.time.held():
            return await super().merge(candidate, onto)

    async def commit_integration(self, onto: str, delta: RecipeDelta, merged_from: Sequence[str]) -> str:
        async with self.time.held():
            return await super().commit_integration(onto, delta, merged_from)

    async def rewind(self, branch: str, commit: str, parent: str) -> None:
        async with self.time.held():
            await super().rewind(branch, commit, parent)

    async def store_patch(self, component: str, text: str) -> PatchRef:
        async with self.time.held():
            return await super().store_patch(component, text)

    async def patch_text(self, commit: str, path: str) -> str:
        async with self.time.held():
            return await super().patch_text(commit, path)

    async def checkout(self, commit: str, destination: Path) -> None:
        async with self.time.held():
            await super().checkout(commit, destination)


#: Seconds the session gate of the tests waits between two reads of a test's record; the tests' toolbox is the fake.
GATE_POLL_SECONDS = 0.001
#: Virtual seconds every poller of a session on a :class:`VirtualClock` waits between two reads, as in production: a
#: poll on real time is invisible to the clock, which then runs virtual time past deadlines while the poll waits.
VIRTUAL_POLL_SECONDS = 5.0


def session_gate(
    notebook: GateNotebook,
    toolbox: FakeToolbox,
    rule: Rule | None = None,
    *,
    polling: Polling = Polling(GATE_POLL_SECONDS),
) -> GateService:
    """The rule gate's session service on ``rule``, by default the InferenceX rule at the conc ``toolbox`` measures a
    test at when its request names none, the workload's; ``notebook`` writes as the gate."""

    (conc,) = toolbox.config.main_concurrency
    judged_on = InferenceXRule(notebook, conc) if rule is None else rule
    return GateService(notebook, toolbox, judged_on, toolbox, polling=polling)


def measure_no_throughput(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test the rule gate reads, baseline or candidate, reports a graded throughput of 0, which the fake
    toolbox itself cannot measure."""

    def zeroed(result: TestResult, concurrency: int) -> InferenceXReading:
        rounds = tuple(
            replace(row, metrics={**row.metrics, GRADED_OUTPUT: 0.0}) if GRADED_OUTPUT in row.metrics else row
            for row in result.rounds
        )
        return inferencex_reading(replace(result, rounds=rounds), concurrency)

    monkeypatch.setattr(rulegate_baseline, "inferencex_reading", zeroed)
    monkeypatch.setattr(rulegate_rules, "inferencex_reading", zeroed)


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


#: Real seconds the specialist runtime of the tests and its scripted sessions wait between two reads of the notebook.
SPECIALIST_POLL_SECONDS = 0.01


class RecordedSession:
    """One run of :class:`RecordedSessions`: its branch's ``step``, if any, then its scripted session."""

    def __init__(self, owner: RecordedSessions, branch: str, inner: ScriptedSession, step: SessionStep | None) -> None:
        self._owner = owner
        self._branch = branch
        self._inner = inner
        self._step = step

    async def run(self) -> None:
        if self._step is not None:
            await self._step(self._branch)
        await self._inner.run()

    async def set_k(self, k: int) -> None:
        self._owner.k[self._branch] = k
        await self._inner.set_k(k)

    async def stop(self) -> None:
        self._owner.stopped.append(self._branch)
        if self._branch in self._owner.hanging:
            await asyncio.Event().wait()
        await self._inner.stop()
        self._owner.running.discard(self._branch)


class RecordedSessions:
    """The specialist runtime's session factory in the tests: every run is a
    :class:`~hyperloom_multibranch.specialist.scripted.ScriptedSession` of its branch's script, after the branch's
    ``step`` when one is given; a branch without a script idles until it is stopped.

    ``started`` keeps the start commit of every branch a run started on, ``briefs`` the brief its first run started
    on, ``repairs`` the repair a repair branch's first run started on, ``directions`` the direction of each of a
    branch's runs in order, ``k`` its last K and ``quota_at_start`` the K the toolbox's policy gave it as its first
    run started; ``running`` holds the branches whose last run was not stopped, and ``stopped`` every stop a run was
    asked for. A run of a branch in ``hanging`` never returns from its stop, a failure double. Every scripted session
    waits ``poll_seconds`` with ``sleep`` between two reads. Not a placeholder: the scripted session is the specialist
    runtime's own.
    """

    def __init__(
        self,
        toolbox: RecordingToolbox,
        repo: BranchHeads,
        notebook: SnapshotQuery,
        *,
        scripts: Mapping[str, Sequence[RecipeDelta]] | None = None,
        steps: Mapping[str, SessionStep] | None = None,
        hanging: frozenset[str] = frozenset(),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        poll_seconds: float = SPECIALIST_POLL_SECONDS,
    ) -> None:
        self._toolbox = toolbox
        self._sleep = sleep
        self._poll_seconds = poll_seconds
        self._repo = repo
        self._notebook = notebook
        self._scripts = dict(scripts or {})
        self._steps = dict(steps or {})
        self.hanging = hanging
        self.started: dict[str, str] = {}
        self.briefs: dict[str, str] = {}
        self.repairs: dict[str, RepairRequest] = {}
        self.directions: dict[str, list[str]] = {}
        self.k: dict[str, int] = {}
        self.quota_at_start: dict[str, int | None] = {}
        self.running: set[str] = set()
        self.stopped: list[str] = []

    def __call__(self, start: SessionStart) -> RecordedSession:
        branch = start.branch
        if branch not in self.started:
            policies = self._toolbox.policies
            self.quota_at_start[branch] = policies[-1].branch_k.get(branch) if policies else None
        self.started.setdefault(branch, start.start_commit)
        self.briefs.setdefault(branch, start.task.brief)
        if start.task.repair is not None:
            self.repairs.setdefault(branch, start.task.repair)
        self.directions.setdefault(branch, []).append(start.task.direction)
        self.k[branch] = start.k
        self.running.add(branch)
        script = tuple(self._scripts.get(branch, ()))
        setup = ScriptedSetup(script, self._toolbox, self._repo, self._notebook, self._poll_seconds, self._sleep)
        return RecordedSession(self, branch, ScriptedSession(start, setup), self._steps.get(branch))


class RefusingRuntime(SpecialistRuntime):
    """The specialist runtime, refusing to stop a branch while it is in ``refusing`` and leaving it running; a
    failure double, not a placeholder."""

    refusing: frozenset[str] = frozenset()

    async def stop(self, branch: str) -> None:
        if branch in self.refusing:
            raise RuntimeError(f"the runtime refused to stop {branch}")
        await super().stop(branch)


class NoForks:
    """Branch requests no specialist of these tests makes; a test double, not a placeholder."""

    async def admit(self, request: ForkRequest) -> ForkAdmission:
        raise AssertionError("no fork is asked for here")

    async def admit_repair(self, request: RepairRequest) -> ForkAdmission:
        raise AssertionError("no repair is asked for here")


@dataclass
class RecordedReservations:
    """A :class:`~hyperloom_multibranch.contracts.fork.ReservedBranches` for the tests: answers that it landed a fork
    exactly for the branches ``landed`` names, cutting nothing, and notes every ``(branch, reason)`` it is told was
    voided in ``voided``."""

    landed: frozenset[str] = frozenset()
    voided: list[tuple[str, str]] = field(default_factory=list)

    async def land_split(self, branch: str) -> bool:
        return branch in self.landed

    async def settle_voided(self, branch: str, reason: str) -> None:
        self.voided.append((branch, reason))


def specialist_runtime(
    notebook: Path,
    toolbox: RecordingToolbox,
    repo: ExperimentRepo,
    sessions: SessionFactory,
    *,
    now: Callable[[], float],
    desk: ForkAdmissionService | None = None,
    runtime: type[RefusingRuntime] = RefusingRuntime,
    rule: Rule | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    poll_seconds: float = SPECIALIST_POLL_SECONDS,
) -> RefusingRuntime:
    """The specialist runtime of the notebook at ``notebook`` on the tests' toolbox, judging on ``rule``, by default
    the InferenceX rule at the conc ``toolbox`` measures at, as the session's peers build it; ``desk`` answers its
    branch requests. Its judges poll every ``poll_seconds`` and its claims renew on ``sleep`` and ``now``."""

    gate_notebook = LocalNotebook(notebook, Writer(Role.GATE, GATE_WRITER), clock=now)
    (conc,) = toolbox.config.main_concurrency

    def notebook_for(branch: str) -> LocalNotebook:
        return LocalNotebook(notebook, Writer(Role.SPECIALIST_RUNTIME, branch, branch), clock=now)

    gate = RuleGate(gate_notebook, repo, desk or NoForks(), toolbox)
    judging = Judging(gate, toolbox, InferenceXRule(gate_notebook, conc) if rule is None else rule)
    pacing = RuntimePacing(poll_seconds=poll_seconds, clock=now, sleep=sleep)
    return runtime(notebook_for, judging, sessions, pacing=pacing)


#: What a recipe measures, by the tags its server arguments carry: its primary-metric value per conc, the main
#: operating point under :data:`MAIN_POINT`; a recipe or conc not named measures :data:`BASELINE_VALUE`. The fake
#: toolbox measures one value per request, so a test over several concs measures, at each, the value of its middle conc.
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
        concs = sorted(request.concurrency)
        conc = concs[len(concs) // 2] if concs else MAIN_POINT
        return curves.get(tags, {}).get(conc, BASELINE_VALUE) / BASELINE_VALUE - 1.0

    return effect


@dataclass(frozen=True)
class RootProfileCall:
    """The arguments of one ``analyze_root_profile`` call."""

    commit_id: str
    trace_dir: Path
    point_config: Path
    source_id: str


class RecordingToolbox(FakeToolbox):
    """The toolbox module's fake, without noise, noting every queue policy, withdrawal and close it was given.

    ``withdrawals`` holds the reason of every session-wide withdrawal. ``effect`` sets what a test measures (see :func:`recipe_effect`); without one every test measures the baseline.
    ``accuracy`` is the accuracy score of a change by its fingerprint, and ``conc`` the workload's conc, which a test
    whose request names none is measured at. ``log`` places the submit log, which the hosted toolbox keeps in the
    session directory, so a second toolbox over that directory hands out request ids where the first one left off;
    without one it goes with the toolbox. ``root_profiles`` holds every :meth:`analyze_root_profile` call, which
    raises ``refusal`` when one is set. Not a placeholder: the fake is the toolbox owner's own, which has no
    ``analyze_root_profile`` of its own yet.
    """

    def __init__(
        self,
        notebook: LocalNotebook,
        effect: Callable[[TestRequest], float] | None = None,
        accuracy: Mapping[str, float] | None = None,
        conc: int = FakeToolboxConfig.main_concurrency[0],
        log: Path | None = None,
    ):
        quiet = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
        config = FakeToolboxConfig(
            noise=quiet, effect_fn=effect, accuracy_scores=dict(accuracy or {}), main_concurrency=(conc,)
        )
        super().__init__(config, notebook=notebook, wiring=FakeWiring(records=notebook, log_path=log))
        self.serving = False
        self.policies: list[QueuePolicy] = []
        self.cancelled: list[str] = []
        self.withdrawals: list[str] = []
        self.queue_reads = 0
        self.closed = 0
        self.root_profiles: list[RootProfileCall] = []
        self.refusal: RequestRejected | None = None

    async def start(self) -> None:
        """Starts once: the fake also starts itself on its first use, which may come after the session's start."""
        if not self.serving:
            self.serving = True
            await super().start()

    async def analyze_root_profile(self, commit_id: str, trace_dir: Path, point_config: Path, *, source_id: str) -> str:
        if self.refusal is not None:
            raise self.refusal
        self.root_profiles.append(RootProfileCall(commit_id, trace_dir, point_config, source_id))
        return f"prelude:{source_id}"

    async def set_policy(self, policy: QueuePolicy) -> None:
        self.policies.append(policy)
        await super().set_policy(policy)

    async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
        self.cancelled.append(branch_id)
        return await super().cancel_branch(branch_id, reason)

    async def withdraw_all(self, reason: str = "session stopped") -> tuple[str, ...]:
        self.withdrawals.append(reason)
        return await super().withdraw_all(reason)

    async def queue_state(self) -> QueueState:
        self.queue_reads += 1
        return await super().queue_state()

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

    async def withdraw_all(self, reason: str = "session stopped") -> tuple[str, ...]:
        self._serving()
        return await super().withdraw_all(reason)

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


@dataclass
class FakeRegressionCheck:
    """A :class:`~hyperloom_multibranch.contracts.gate.RegressionCheck` for the tests: fails on a base commit
    ``failures`` names, with its reason, and passes on any other; ``calls`` notes every
    ``(base_commit, candidate_delta, sources)`` it was asked about."""

    failures: Mapping[str, str] = field(default_factory=dict)
    calls: list[tuple[str, RecipeDelta, tuple[str, ...]]] = field(default_factory=list)

    async def check(
        self, base_commit: str, candidate_delta: RecipeDelta, *, sources: tuple[str, ...]
    ) -> RegressionOutcome:
        self.calls.append((base_commit, candidate_delta, sources))
        reason = self.failures.get(base_commit)
        return RegressionOutcome(reason is None, reason or "")


def judged(verdict: Verdict, gain_pct: float, request_id: str, value: float) -> Judgement:
    return Judgement(verdict, gain_pct, 1.0, (request_id,), {GRADED_OUTPUT: value}, accuracy_ok=True)


@dataclass
class ScriptedCompletion:
    """A model for the tests: raises ``error`` when it is set, else the scripted replies in order, then ``fallback``,
    each reply using ``tokens`` that cost ``cost_usd``.

    Not a placeholder: a test never calls a real model.
    """

    replies: list[str] = field(default_factory=list)
    fallback: str = '{"open": 0, "branches": [], "stop": [], "stop_session": null}'
    error: Exception | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)
    tokens: TokenCounts = TokenCounts()
    cost_usd: float = 0.0

    async def __call__(self, system: str, user: str) -> CompletionReply:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return CompletionReply(self.replies.pop(0) if self.replies else self.fallback, self.tokens, self.cost_usd)


@dataclass
class OneChampionKB:
    """A Recipe KB holding one champion in whatever scope it is asked for, or one that cannot be reached; it takes no
    write."""

    unreachable: bool = False
    searched: list[dict[str, Any]] = field(default_factory=list)

    def search_identities(
        self,
        *,
        scheme: str,
        match: dict[str, str],
        hardware_in: list[str] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        if self.unreachable:
            raise KBStoreError("connection refused")
        self.searched.append({"scheme": scheme, "match": match})
        return {"items": [{"canonical_id": f"{scheme}:champion"}], "next_offset": None}

    def get_view(self, canonical_id: str, scope: RecipeScope) -> dict[str, Any] | None:
        knowledge = {
            "knowledge_schema_version": CURRENT_KNOWLEDGE_SCHEMA_VERSION,
            "record_kind": RECORD_KIND_HYPERLOOM_RECIPE,
            "validated_e2e_gain": 12.0,
            "value": {"config": {"extra_server_args": "--enable-expert-parallel"}, "patch": {}, "kernel": {}},
        }
        return {"canonical_id": canonical_id, "session_id": "earlier", "knowledge": knowledge}

    def write_if_better(self, canonical_id: str, session_id: str, bundle: object, **options: object) -> object:
        raise KBStoreError("this Recipe KB takes no write")


__all__ = [
    "BASE_IMAGE_DIGEST",
    "BASELINE_VALUE",
    "GATE_POLL_SECONDS",
    "CHILD_SCRIPT",
    "LAUNCH_FLAG",
    "MAIN_POINT",
    "TAG_FLAG",
    "TOOLBOX_DOWN",
    "VIRTUAL_POLL_SECONDS",
    "Curves",
    "DownToolbox",
    "FakeRegressionCheck",
    "HeldGitRepo",
    "MbFakeChildRegistry",
    "MbFakeHeldRegistry",
    "NoForks",
    "OneChampionKB",
    "RecordedReservations",
    "RecordedSession",
    "RecordedSessions",
    "RecordingToolbox",
    "RefusingRuntime",
    "SPECIALIST_POLL_SECONDS",
    "ScriptedCompletion",
    "SessionStep",
    "StuckToolbox",
    "VirtualClock",
    "session_gate",
    "specialist_runtime",
    "judged",
    "recipe_effect",
    "recipe_tags",
    "tagged",
]
