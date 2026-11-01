# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The end-to-end skeleton: one session driven in-process on the specialist runtime, peer stand-ins and a virtual
clock.

PRELUDE roots the session on a scripted child session in a git repository; the judge opens two branches, whose scripted
specialists the runtime runs and judges; on ``b01`` the runtime confirms and commits ``c1``, and two fixes of it are
asked for as forks: the first takes the last slot as ``b03``, the second waits in the fork desk's queue as ``b04`` and
takes the slot the judge's stop of ``b02`` frees; the rule gate lands each fix on its fork, with the fix's case as a
regression case of the fork's first commit, and the scheduler starts each at its fix; on ``b02`` it commits ``c2``, the
best single-branch value, whose specialist holds a claim and a process the AI work socket started, and the judge stops
``b02``; the integrator merges ``c1`` and refuses ``c2`` and the fixes; FINAL weighs the judge-stopped ``b02`` against
the integration head and passes over it, because its head does not reproduce its recorded value; CLOSE leaves nothing
running.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NoReturn

import pytest

from hyperloom.common.jsonio import read_jsonl
from hyperloom.common.perf_metric import GRADED_INTVTY_P50, GRADED_OUTPUT
from hyperloom.common.proctree import running
from hyperloom.common.outside_facts.image_check import FactResult, ImageCheckFailed, ImageCheckMode, ImageCheckReport
from hyperloom.common.outside_facts.image_check_eval import Status
from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.contracts.measurement import (
    NOT_REPRODUCED,
    REPRODUCED,
    AttachmentRef,
    Purpose,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
)
from hyperloom.contracts.recipe import Recipe, RecipeDelta
from hyperloom.hosted.mcp_server import MIN_TOKEN_LENGTH, McpServer, McpServerConfig
from hyperloom.measure.versions import socket_versions
from hyperloom_multibranch.contracts.fork import AdmissionStatus, ForkAdmission, ForkRequest, RepairRequest
from hyperloom_multibranch.contracts.intent import Intent, IntentLayer
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.contracts.lifecycle import BranchTask
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.interfaces import UsageSink
from hyperloom_multibranch.contracts.blackboard.records import (
    LeaseStart,
    LlmUsage,
    BlackboardSnapshot,
    OpShares,
    PendingCandidate,
    QuarantineEvent,
    RegressionCase,
    ToolResultPointer,
)
from hyperloom_multibranch.contracts.blackboard.vocab import (
    ClaimStatus,
    Conclusion,
    LearningKind,
    LeaseKind,
    LineageKind,
    QuarantineAction,
    UsageSource,
)
from hyperloom_multibranch.contracts.toolbox import CancelOutcome, MeasurementCapacity
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.knowledge.learning_recall import LearningRecall
from hyperloom_multibranch.blackboard.knowledge.reference_leads import INFERENCEX_REFERENCE
from hyperloom_multibranch.blackboard.rsi.archive import ARCHIVE_ENV, SKIPPED, TreeArchive
from hyperloom_multibranch.blackboard.rsi.briefing import SessionBriefs
from hyperloom_multibranch.blackboard.rsi.decisions import decision_calls
from hyperloom_multibranch.blackboard.rsi.playbook import (
    FALLBACK_KEY,
    H0,
    PLAYBOOK_KEY,
)
from hyperloom_multibranch.blackboard.rsi.authoring import NO_ARCHIVE
from hyperloom_multibranch.blackboard.rsi.frontier import (
    FRONTIER_V1_LABEL,
    SETTLED_TESTS,
    FrontierPolicy,
)
from hyperloom_multibranch.blackboard.rsi.lesson_recall import LessonRecall
from hyperloom_multibranch.blackboard.rsi.lessons import LessonPool, LessonRevision, lesson_id_for, revision_id_for
from hyperloom_multibranch.blackboard.rsi.session import FRONTIER_LOG
from hyperloom_multibranch.blackboard.rsi.similarity import TaskConditions
from hyperloom_multibranch.blackboard.tests.factories import delta, intent, request
from hyperloom_multibranch.rulegate.critic import (
    FixReview,
    FixReviewer,
    LlmCritic,
    NoCritic,
    ReproVerdicts,
)
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID, FixCase
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.regression import FixProof, ReproRun
from hyperloom_multibranch.rulegate.service import BASELINE_TESTS, Polling
from hyperloom_multibranch.scheduling.budget import (
    MB_SECONDS_PER_HOUR,
    SessionBudget,
    explore_gpu_seconds,
)
from hyperloom_multibranch.scheduling import branches
from hyperloom_multibranch.scheduling.events import PROFILE_UNAVAILABLE, PROFILED
from hyperloom_multibranch.scheduling.explore import (
    EXPLORE_TICK_SECONDS,
    FORK_NOT_LANDED,
    OPEN_CUT_REASON,
)
from hyperloom_multibranch.scheduling.exports import IMAGE_CHECK_FILE
from hyperloom_multibranch.scheduling.final import INTEGRATION_LABEL, PRIMARY_POINT_ONLY
from hyperloom_multibranch.scheduling import judge
from hyperloom_multibranch.scheduling.judge import GRACE_KEY, HoldReason
from hyperloom_multibranch.scheduling.judge_prompt import JUDGE_PROMPT_VERSION
from hyperloom_multibranch.scheduling.integrator import ADOPTING_LEAD
from hyperloom_multibranch.scheduling.layout import SessionLayout, create_session
from hyperloom_multibranch.scheduling import lifecycle
from hyperloom_multibranch.scheduling.lifecycle import (
    CUT_SHORT,
    LESSONS_KEY,
    NO_CARDS_LEFT,
    Session,
    SessionHost,
    SessionOutcome,
    SessionServices,
    persisted_run,
)
from hyperloom_multibranch.scheduling.local_peers import workload_rule
from hyperloom_multibranch.contracts.llm import CompletionFailed, CompletionReply
from hyperloom_multibranch.scheduling.peers import Peers, PeerSetup
from hyperloom_multibranch.scheduling.prelude import CHILD_CARDS_ENV
from hyperloom_multibranch.scheduling.prelude_reader import MB_FRAMEWORK_REPO_ENV
from hyperloom_multibranch.scheduling.policy import SCHEDULING_POLICY, policy_version
from hyperloom_multibranch.scheduling.roster import BranchOrigin, BranchStatus, RosterCaps, StopCause, final_quota
from hyperloom_multibranch.scheduling.reference import ReferenceSource
from hyperloom_multibranch.scheduling.settings import (
    InferenceXShape,
    KernelOptimizer,
    RunSettings,
    Workload,
)
from hyperloom_multibranch.scheduling.stops import (
    STOP_OVERRAN,
    TOOLBOX_CLOSE_OVERRAN,
    TOOLBOX_CLOSE_SECONDS,
    SessionStopReason,
)
from hyperloom_multibranch.scheduling.tests.peer_fakes import (
    MAIN_POINT,
    VIRTUAL_POLL_SECONDS,
    Curves,
    FakeRegressionCheck,
    HeldGitRepo,
    LeasingToolbox,
    MbFakeHeldRegistry,
    OneChampionKB,
    RecordedSessions,
    RecordingToolbox,
    RefusingRuntime,
    ScriptedCompletion,
    SessionStep,
    StuckToolbox,
    VirtualClock,
    recipe_effect,
    recipe_tags,
    session_gate,
    specialist_runtime,
    tagged,
)
from hyperloom_multibranch.scheduling.tests.child_sessions import (
    BASELINE_ERROR_CLASS,
    child_with_patch,
    failed_child,
    with_agentx_baseline_run,
)
from hyperloom_multibranch.tests.fakes import RecordingVerifier
from hyperloom_multibranch.toolbox.attempts import TestJob
from hyperloom_multibranch.toolbox.fake_model import DEFAULT_BASELINE
from hyperloom_multibranch.toolbox.tools.base import JobStatus
from hyperloom_multibranch.scheduling.agent_plans import (
    DOSSIER_ENDED,
    DOSSIER_SKIPPED,
    DOSSIER_STARTED,
    KEPT_DOSSIER,
    KEPT_MAP,
    KEPT_SUMMARY,
    MADE_BY_PASS,
    MADE_BY_SHARES,
    MAP_KEPT,
    ROOT_SHARES_READ,
)
from hyperloom_multibranch.scheduling.lifecycle import NO_MAP
from hyperloom_multibranch.scheduling.settings import DOSSIER_MINUTES
from hyperloom_multibranch.specialist.arbor_adapter import MAPPED, SplitFailed
from hyperloom_multibranch.specialist.directions import DIRECTIONS
from hyperloom_multibranch.specialist.dossier_pass import DossierOutcome, DossierPass, DossierStart
from hyperloom_multibranch.specialist.prompts import prompt_versions
from hyperloom_multibranch.specialist.runtime import MAX_LOSS_RESTARTS, SessionCrashed

START = 1_700_000_000.0
#: Real seconds a PRELUDE child lingers so that what PRELUDE submits beside it is submitted while it runs.
CHILD_BESIDE_SECONDS = 2.0
CONC = 16
#: An ownership scope of a pod that is gone: another boot, so none of its pids is a process of this environment.
ANOTHER_POD = "0" * 32 + ":1:1"
WORKLOAD = Workload(
    "/models/Llama-3-8B", "sglang", 1, CONC, "bf16", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1024, 1024)
)
#: The same workload as an AgentX trace replay.
AGENTX = replace(WORKLOAD, inferencex=None)
#: The PRELUDE child's AgentX baseline run, launch to exit, which FINAL keeps in reserve.
AGENTX_CHILD_RUN_SECONDS = 600.0
#: What the session's image check finds: one fact of the rule pack that does not hold here, under the default mode.
IMAGE_CHECK = ImageCheckReport(
    ImageCheckMode.WARN,
    rule_pack_identity(),
    "2023-11-14T22:13:20Z",
    {"python": "/usr/bin/python3"},
    (FactResult("sglang.renamed-flag", "sglang", Status.FAIL, (), "the flag is gone"),),
)
SETTINGS = RunSettings(
    WORKLOAD,
    max_hours=0.5,
    max_gpu_hours=0.0,
    max_llm_usd=0.0,
    caps=RosterCaps(3, 2, 8),
    judge_interval_seconds=600.0,
)
#: An InferenceX matrix with one single-node entry for the workload's model.
LLAMA_MATRIX = """\
llama3-bf16-mi355x-sglang:
  image: lmsysorg/sglang:v0.5.12-rocm700-mi35x
  model: meta-llama/Llama-3-8B
  model-prefix: llama3
  runner: mi355x
  precision: bf16
  framework: sglang
  multinode: false
  scenarios:
    fixed-seq-len:
    - { isl: 1024, osl: 1024, search-space: [{ tp: 1, conc-start: 4, conc-end: 64 }] }
"""
#: Where the hosted toolbox keeps its submit log inside the session's toolbox directory.
SUBMISSIONS = "submissions.jsonl"
STEP_SECONDS = 30.0
CLAIM_TTL_SECONDS = 1e6
STOP_REASON = "no gain left"
#: A GPU-hour budget the baseline on the root alone uses up.
TINY_GPU_HOURS = 0.01
#: A GPU-hour budget the session's tests, FINAL's included, never use up, and the card time a lease of
#: :data:`LEASED_CARDS` cards held through a few judge calls of :data:`LEASED_JUDGE_INTERVAL_SECONDS` does.
LEASED_GPU_HOURS = 1.0
LEASED_CARDS = 4
LEASED_JUDGE_INTERVAL_SECONDS = 120.0
#: An LLM budget one specialist launch of :data:`LAUNCH_USD` uses up.
SMALL_LLM_USD = 2.5
LAUNCH_USD = 3.0
#: What each judge call uses and costs: most of its prompt read from the cache.
JUDGE_TOKENS = TokenCounts(input=400, output=120, cache_read=3600)
JUDGE_USD = 0.04
CONVERGED_REASON = "every branch is flat"
#: What the scheduler's own tests measure: the root, ``c1`` alone (the integration head once merged) and ``c2``, which
#: sweeps higher than the integration head but measures the root's value when reproduced; every other recipe measures
#: 1000.
CURVES = {
    frozenset(): {MAIN_POINT: 1000.0, CONC: 1000.0},
    frozenset({"c1"}): {MAIN_POINT: 1040.0, CONC: 1040.0},
    frozenset({"c2"}): {MAIN_POINT: 1000.0, CONC: 1065.0},
}
#: What the specialists' tests and their retests measure: ``c2`` at 1060, which FINAL does not reproduce.
EXPLORED = {**CURVES, frozenset({"c2"}): {MAIN_POINT: 1060.0}}
SCRIPTS = {"b01": (tagged("c1"),), "b02": (tagged("c2"),)}
FIX_REASON = "fix f1 regresses"
#: The fix each fork of ``c1`` carries, by the reason it is asked for, in the order the forks are asked for.
FIXES = {FIX_REASON: tagged("f1"), "fix f2 regresses": tagged("f2")}
FIX_REASONS = tuple(FIXES)
#: The case a fix's proof ran: reproduced on ``c1``, clean on the fix.
FIX_CASE = AttachmentRef(FIX_CASE_ID, "dir", "a" * 64)
#: An agent CLI as the AI work socket starts one: leading a session of its own.
AGENT_ARGV = ("sleep", "600")
#: What a FINAL that cannot read the regression set fails with.
REGRESSION_SET_GONE = "the regression set is gone"
#: How long each FINAL test of a slow toolbox runs: longer than the whole budget.
FINAL_TEST_SECONDS = 2 * SETTINGS.max_hours * 3600


def _reply(document: dict[str, object]) -> str:
    return json.dumps(document)


AUTHOR_MODEL = "author-model"
RESOLVER_MODEL = "resolver-model"
OPENING = _reply({"open": 2, "open_k": 9, "branches": [{"branch": INTEGRATION_BRANCH, "k": 2}], "stop": []})
STEER = _reply({"open": 0, "branches": [{"branch": "b01", "k": 7}], "stop": [{"branch": "b02", "reason": STOP_REASON}]})


@pytest.fixture(autouse=True)
def no_knowledge_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLOSE's knowledge steps find no archive: a test session leaves no knowledge behind."""

    monkeypatch.delenv(ARCHIVE_ENV, raising=False)


@pytest.fixture(autouse=True)
def child_framework(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The scheduler, and so the PRELUDE child, is launched on the framework checkout the child sessions are built
    on."""

    monkeypatch.setenv(MB_FRAMEWORK_REPO_ENV, str(tmp_path / "framework"))


@pytest.fixture(autouse=True)
def no_branch_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scripted branches end fewer explore tests than a grace lasts, so the judge's grace would refuse the stops
    these sessions script; the test of the grace sets it back."""

    monkeypatch.setattr(judge, "SETTLED_TESTS", 0)


def _measured(repo: Path, explored_curves: Curves, measured_curves: Curves) -> Callable[[TestRequest], float]:
    """What a test measures: ``explored_curves`` for the specialists' tests and retests, ``measured_curves``
    otherwise."""

    explored, measured = recipe_effect(repo, explored_curves), recipe_effect(repo, measured_curves)

    def effect(request: TestRequest) -> float:
        return (explored if request.purpose in (Purpose.EXPLORE, Purpose.CONFIRM) else measured)(request)

    return effect


class InsistingRuntime(RefusingRuntime):
    """A runtime whose stop hangs after the operator signals again; a failure double, not a placeholder."""

    async def stop(self, branch: str) -> None:
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.Event().wait()


class SignallingToolbox(StuckToolbox):
    """The stuck toolbox, signalling the scheduler once as the session stop starts to withdraw the integration
    branch's tests; a failure double, not a placeholder."""

    stuck = frozenset({INTEGRATION_BRANCH})

    async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
        if branch_id in self.stuck:
            os.kill(os.getpid(), signal.SIGTERM)
        return await super().cancel_branch(branch_id, reason)


class NoObjectStore(HeldGitRepo):
    """A repository that fails on the root commit with an error the session does not handle; a failure double."""

    async def init_root(self, recipe: Recipe) -> str:
        raise LookupError(f"no object store at {self.time}")


class SilentModel(ScriptedCompletion):
    """A judge model that never answers; a failure double, not a placeholder."""

    async def __call__(self, system: str, user: str) -> CompletionReply:
        self.calls.append((system, user))
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@dataclass
class Stage:
    """One session on the specialist runtime: ``scripts`` and ``steps`` are what each branch's specialist does,
    ``explored`` what its tests and retests measure and ``measured`` what every other test does, the runtime refuses
    to stop a branch in ``refusing`` and a run of a branch in ``hanging`` never confirms its stop. ``on_this_machine``
    says whether the toolbox measures on this machine's cards, leasing PRELUDE's child its cards, or in sandboxes
    elsewhere, where PRELUDE starts no child."""

    layout: SessionLayout
    time: VirtualClock
    processes: MbFakeHeldRegistry
    judge: ScriptedCompletion
    pids: list[int]
    scripts: Mapping[str, Sequence[RecipeDelta]] = field(default_factory=dict)
    explored: Curves = field(default_factory=lambda: EXPLORED)
    measured: Curves = field(default_factory=lambda: CURVES)
    steps: Mapping[str, SessionStep] = field(default_factory=dict)
    refusing: frozenset[str] = frozenset()
    hanging: frozenset[str] = frozenset()
    runtime_type: type[RefusingRuntime] = RefusingRuntime
    recipe_kb: OneChampionKB | None = None
    stuck: frozenset[str] = frozenset()
    toolbox_type: type[RecordingToolbox] = LeasingToolbox
    on_this_machine: bool = True
    settings: RunSettings = SETTINGS
    repo_type: type[HeldGitRepo] = HeldGitRepo
    critic: Callable[[UsageSink], FixReviewer] = lambda _usage: NoCritic()
    author: ScriptedCompletion = field(default_factory=ScriptedCompletion)
    learnings: LearningRecall | None = None
    image_check: Callable[[], ImageCheckReport] = lambda: IMAGE_CHECK
    dossier_pass: DossierPass | None = None
    toolbox: RecordingToolbox = field(init=False)
    sessions: RecordedSessions = field(init=False)
    repo: HeldGitRepo = field(init=False)
    setup: PeerSetup = field(init=False)
    gate: RuleGate = field(init=False)
    regression: FakeRegressionCheck = field(default_factory=FakeRegressionCheck)
    briefs: SessionBriefs = field(init=False)
    forks: list[ForkAdmission] = field(default_factory=list)
    watchers: list[asyncio.Future[None]] = field(default_factory=list)
    signalled_at: float = 0.0

    def handle(self, role: Role, branch: str = "") -> LocalBlackboard:
        return LocalBlackboard(self.layout.blackboard, Writer(role, branch or role.value, branch), clock=self.time.now)

    def factory(self, setup: PeerSetup) -> Peers:
        repo = self.repo_type(setup.layout.repo, setup.blobs, self.time)
        toolbox = self.toolbox_type(
            self.handle(Role.TOOLBOX),
            _measured(setup.layout.repo, self.explored, self.measured),
            conc=CONC,
            log=setup.layout.toolbox / SUBMISSIONS,
            repo=repo,
        )
        if isinstance(toolbox, StuckToolbox) and self.stuck:
            toolbox.stuck = self.stuck
        if isinstance(toolbox, LeasingToolbox):
            toolbox.clock = self.time.now
        rule = workload_rule(setup.workload, self.handle(Role.GATE))
        self.sessions = RecordedSessions(
            toolbox,
            repo,
            self.handle(Role.OBSERVER),
            scripts=self.scripts,
            steps=self.steps,
            hanging=self.hanging,
            sleep=self.time.sleep,
            poll_seconds=VIRTUAL_POLL_SECONDS,
        )
        self.gate = RuleGate(self.handle(Role.GATE), repo, setup.desk, RecordingVerifier())
        runtime = specialist_runtime(
            setup.layout.blackboard,
            toolbox,
            repo,
            self.sessions,
            now=self.time.now,
            desk=setup.desk,
            runtime=self.runtime_type,
            rule=rule,
            sleep=self.time.sleep,
            poll_seconds=VIRTUAL_POLL_SECONDS,
        )
        runtime.refusing = self.refusing
        self.toolbox, self.repo, self.setup = toolbox, repo, setup
        self.briefs = SessionBriefs(self.handle(Role.OBSERVER), repo=repo, learnings=self.learnings)
        mcp = McpServer(McpServerConfig(host_token="t" * MIN_TOKEN_LENGTH), campaign=setup.layout.session_id)
        gate = session_gate(
            self.handle(Role.GATE), toolbox, rule, polling=Polling(VIRTUAL_POLL_SECONDS, self.time.sleep)
        )
        digest = toolbox.config.image_digest
        return Peers(
            runtime,
            toolbox,
            repo,
            gate,
            mcp,
            self.briefs,
            digest,
            self.gate,
            self.regression,
            self.on_this_machine,
            self.dossier_pass,
        )

    async def root(self) -> str:
        return (await self.handle(Role.OBSERVER).baseline()).root_commit

    async def ask_fork(self, branch: str, parent: str, reason: str) -> None:
        """Have the rule gate ask for a fork of ``branch`` at ``parent`` for the fix :data:`FIXES` names by
        ``reason``, as the runtime asks for one: on a retested candidate that regresses but fixes a correctness defect,
        with the proof that its case runs clean on it. The candidate is the retest ``fix-<n>`` of a first test the
        blackboard never holds, and its clean case run is ``fix-<n>-case``: the runtime judges neither."""

        fix = request(f"fix-{len(self.forks) + 1}", intent(branch, parent), FIXES[reason])
        candidate = replace(fix, purpose=Purpose.CONFIRM, retest_of=f"{fix.request_id}-first")
        case = f"{candidate.request_id}-case"
        clean = request(
            case, intent(branch, parent), kind=TestKind.REPRO, purpose=Purpose.REPRO, attachments=(FIX_CASE,)
        )
        toolbox = self.handle(Role.TOOLBOX)
        await toolbox.write_request(candidate)
        await toolbox.write_result(TestResult(candidate.request_id, Validity.SUCCEEDED))
        await toolbox.write_request(clean)
        await toolbox.write_result(TestResult(case, Validity.SUCCEEDED, repro_verdict=NOT_REPRODUCED))
        regressed = Judgement(
            Verdict.REGRESSED,
            -2.0,
            2.0,
            (candidate.request_id,),
            {GRADED_OUTPUT: 980.0},
            AccuracyVerdict.PASSED,
            fixes_correctness=True,
        )
        on_parent = ReproRun(f"{candidate.request_id}-parent", REPRODUCED)
        proof = FixProof(on_parent, ReproRun(case, NOT_REPRODUCED), on_parent)
        self.forks.append(await self.gate.fork(candidate, regressed, reason, fix=proof))

    async def start_agent(self, branch: str) -> None:
        """An agent CLI of ``branch``'s specialist, started the way the AI work socket starts one and adopted by the
        session's process registry, as the runtime adopts every process it has started."""

        async with self.time.held():
            agent = await asyncio.create_subprocess_exec(*AGENT_ARGV, start_new_session=True)
        self.setup.processes.adopt(agent.pid, branch=branch, argv=AGENT_ARGV)
        self.pids.append(agent.pid)

    def once_committed(self, then: Callable[[str, str], Awaitable[None]]) -> SessionStep:
        """A step that waits until the runtime has committed on the branch, then runs ``then`` with the branch and its
        new head."""

        async def step(branch: str) -> None:
            self.watchers.append(asyncio.ensure_future(self._after_commit(branch, then)))

        return step

    async def _after_commit(self, branch: str, then: Callable[[str, str], Awaitable[None]]) -> None:
        start = await self.repo.head(branch)
        while (head := await self.repo.head(branch)) == start:
            await self.time.sleep(VIRTUAL_POLL_SECONDS)
        await then(branch, head)

    async def run(self) -> SessionOutcome:
        """Run the session, its image check answered by ``image_check`` instead of this environment."""

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(lifecycle, "run_image_check", self.image_check)
            return await self._session().run()

    async def finish(self, ended: SessionStopReason) -> SessionOutcome:
        """FINAL and CLOSE alone over the session directory this stage was built on, as ``final-only`` runs them."""

        return await self._session().finish(ended)

    def _session(self) -> Session:
        self.time.hold_threads()
        host = SessionHost(self.layout, self.processes, self.time.clock)
        return Session(
            self.settings,
            self.factory,
            host,
            SessionServices(
                self.judge,
                ScriptedCompletion(),
                self.recipe_kb,
                self.critic,
                self.author,
                AUTHOR_MODEL,
                RESOLVER_MODEL,
            ),
        )


def _stage(tmp_path: Path, *, child: Path | None, exit_code: int = 0, linger: float = 0.0) -> Stage:
    """A session on ``tmp_path`` whose PRELUDE child roots on ``child``, runs ``linger`` real seconds and exits with
    ``exit_code``."""

    time = VirtualClock(START)
    layout = create_session(tmp_path / "workspace", WORKLOAD.model_basename, time.clock)
    session = "" if child is None else str(child)
    processes = MbFakeHeldRegistry(layout.pids, time, session=session, exit_code=exit_code, linger=linger)
    judge = ScriptedCompletion([OPENING, STEER])
    return Stage(layout, time, processes, judge, [])


def _recovery(stage: Stage, settings: RunSettings) -> Stage:
    """A second stage over the directory ``stage`` left, as ``final-only`` builds one: the same session directory and
    the settings read back from its manifest, every peer fresh and no specialist script."""

    time = VirtualClock(stage.time.now())
    processes = MbFakeHeldRegistry(stage.layout.pids, time, session="", exit_code=0)
    return Stage(stage.layout, time, processes, ScriptedCompletion(), [], settings=settings)


def _script(stage: Stage, forks: int = len(FIX_REASONS)) -> None:
    """``b01`` commits ``c1`` and asks for ``forks`` forks of a fix of it; ``b02`` commits ``c2``, then holds a claim
    on it and a process."""

    async def ask_forks(branch: str, c1: str) -> None:
        for reason in FIX_REASONS[:forks]:
            await stage.ask_fork(branch, c1, reason)

    async def holds(branch: str, c2: str) -> None:
        runtime = stage.handle(Role.SPECIALIST_RUNTIME, branch)
        await runtime.acquire(intent(branch, c2), delta().fingerprint(), ttl_seconds=CLAIM_TTL_SECONDS)
        await stage.start_agent(branch)

    stage.scripts = SCRIPTS
    stage.steps = {"b01": stage.once_committed(ask_forks), "b02": stage.once_committed(holds)}


async def test_a_session_runs_every_phase_and_leaves_nothing_behind(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge.tokens, stage.judge.cost_usd = JUDGE_TOKENS, JUDGE_USD
    _script(stage)

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    assert (stage.setup.workload, stage.setup.processes) == (WORKLOAD, stage.processes)
    observer = stage.handle(Role.OBSERVER)
    lineage = await observer.lineage()
    branches = {branch.branch: branch for branch in lineage.branches}
    root, c1, c2 = await stage.root(), await stage.repo.head("b01"), await stage.repo.head("b02")
    assert lineage.commit(root) is not None and (await stage.repo.read_recipe(root)).components
    assert (await observer.session_profile()).session_id == stage.layout.session_id
    [fingerprint] = await observer.manifest_fingerprints()
    assert fingerprint.policy_versions["judge_prompt"] == JUDGE_PROMPT_VERSION
    assert fingerprint.policy_versions["scheduling"] == policy_version(SCHEDULING_POLICY)
    assert fingerprint.policy_versions[GRACE_KEY] == str(judge.session_grace(None))
    assert fingerprint.policy_versions[LESSONS_KEY] == ""
    [policy] = [row for row in _decisions(stage) if row["kind"] == "prelude" and row["reason"] == "scheduling policy"]
    assert policy["detail"] == {"values": dict(SCHEDULING_POLICY)}
    assert dict(fingerprint.sockets) == socket_versions()
    assert dict(fingerprint.prompt_versions) == prompt_versions()

    fix, queued_fix = await stage.repo.head("b03"), await stage.repo.head("b04")
    assert stage.sessions.started == {"b01": root, "b02": root, "b03": fix, "b04": queued_fix}
    assert recipe_tags(await stage.repo.read_recipe(c1)) == {"c1"}
    assert recipe_tags(await stage.repo.read_recipe(c2)) == {"c2"}
    assert recipe_tags(await stage.repo.read_recipe(fix)) == {"c1", "f1"}
    assert recipe_tags(await stage.repo.read_recipe(queued_fix)) == {"c1", "f2"}
    assert [(admission.status, admission.branch) for admission in stage.forks] == [
        (AdmissionStatus.ACCEPTED, "b03"),
        (AdmissionStatus.PENDING, "b04"),
    ]
    for fork in ("b03", "b04"):
        assert (branches[fork].origin, branches[fork].start_commit, branches[fork].from_branch) == (
            LineageKind.SPLIT,
            c1,
            "b01",
        )
    for candidate, head in (("fix-1", fix), ("fix-2", queued_fix)):
        record = await observer.record(candidate)
        assert (record.conclusion, record.commit_id) == (Conclusion.SPLIT_COMMITTED, head)
        assert await observer.regression_cases_for(head) == (RegressionCase(FIX_CASE_ID, head, f"{candidate}-case"),)
    [dequeued] = [
        row for row in _decisions(stage) if row["kind"] == "fork" and row["detail"].get("status") == "dequeued"
    ]
    [stopped] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b02"]
    assert dequeued["branch"] == "b04" and stopped["at"] <= dequeued["at"] < stopped["at"] + EXPLORE_TICK_SECONDS

    assert stage.sessions.k["b01"] == 2 and stage.sessions.running == set()
    specialist_ks = [
        k for policy in stage.toolbox.policies for b, k in policy.branch_k.items() if b != INTEGRATION_BRANCH
    ]
    assert max(specialist_ks) == SETTINGS.caps.max_k_per_branch
    steering = [policy for policy in stage.toolbox.policies if "b01" in policy.branch_k]
    assert steering[-1].k_for(INTEGRATION_BRANCH) == 2 and "b02" not in steering[-1].branch_k
    assert all(policy.listed_branches_only for policy in stage.toolbox.policies)
    assert stage.toolbox.policies[-1].branch_k == {}

    roster = json.loads(stage.layout.roster.read_text())
    assert {
        entry["branch"]: entry["origin"] for entry in roster["entries"] if entry["branch"] != INTEGRATION_BRANCH
    } == {
        "b01": BranchOrigin.FIRST_BATCH.value,
        "b02": BranchOrigin.FIRST_BATCH.value,
        "b03": BranchOrigin.FORK.value,
        "b04": BranchOrigin.FORK.value,
    }
    assert not [row for row in _decisions(stage) if row["kind"] == "open" and not row["branch"]]
    b02 = next(entry for entry in roster["entries"] if entry["branch"] == "b02")
    assert b02["stop"] == {"cause": StopCause.JUDGE.value, "reason": STOP_REASON, "at": b02["stop"]["at"]}
    assert branches["b02"].stop_reason == f"judge: {STOP_REASON}"
    assert branches[INTEGRATION_BRANCH].stopped_at is not None
    assert branches[INTEGRATION_BRANCH].stop_reason == f"session_end: {SessionStopReason.TIME_EXHAUSTED.value}"
    assert "b02" in stage.sessions.stopped and "b02" in stage.toolbox.tool_cancels
    assert stage.toolbox.serving and stage.toolbox.closed == 1 and stage.toolbox.results
    assert not any(running(pid) for pid in stage.pids)
    claims = (await observer.snapshot()).claims
    assert [claim.status for claim in claims if claim.holder == "b02"] == [ClaimStatus.RELEASED]

    merged = {item.commit_id: item.last_merge.outcome.value for item in await observer.committed() if item.last_merge}
    assert merged == {c1: "merged", c2: "not_stackable", fix: "not_stackable", queued_fix: "not_stackable"}
    head = await stage.repo.head(INTEGRATION_BRANCH)
    assert recipe_tags(await stage.repo.read_recipe(head)) == {"c1"}

    assert outcome.final is not None and outcome.final.chosen is not None
    assert outcome.final.chosen.label == INTEGRATION_LABEL and outcome.final.chosen.commit == head
    single = next(check for check in outcome.final.checks if check.candidate.label != INTEGRATION_LABEL)
    assert (single.candidate.branch, single.candidate.commit) == ("b02", c2)
    assert single.curve[CONC] > outcome.final.checks[0].curve[CONC] and not single.reproduced
    assert {check.candidate.commit for check in outcome.final.checks} == {head, c2}
    breakdown = json.loads((stage.layout.reports / "session_breakdown.json").read_text())
    assert breakdown["close"]["final"]["commit_id"] == head
    assert breakdown["close"]["selection"]["chosen"]["commit"] == head
    attempts = breakdown["close"]["merges"]["attempts"]
    assert (attempts["merged"], attempts["not_stackable"], attempts["conflict_skipped"]) == (1, 4, 0)
    merge_log = [json.loads(line) for line in stage.layout.merges.read_text().splitlines()]
    [adoption] = [row for row in merge_log if row["detail"].get("adopts")]
    assert (adoption["detail"]["adopts"], adoption["detail"]["candidate"]) == ("b02", c2)
    assert adoption["reason"].startswith(f"{ADOPTING_LEAD.format('b02')}: regressed ")

    assert all(entry["status"] == BranchStatus.STOPPED.value for entry in roster["entries"])
    assert outcome.close is not None and outcome.close.stop.clean and outcome.close.exports is not None
    manifest = json.loads(stage.layout.manifest.read_text())
    assert set(manifest["code_version"]) == {"package", "git_sha"} and manifest["metric"] == GRADED_OUTPUT
    assert manifest["rule_pack"] == rule_pack_identity()
    assert manifest["image_check"] == IMAGE_CHECK.summary()
    assert json.loads((stage.layout.reports / IMAGE_CHECK_FILE).read_text()) == IMAGE_CHECK.to_json()
    judge_calls = len(stage.layout.judge.read_text().splitlines())
    spends = [_judge_state(prompt)["spend"] for prompt in _judge_prompts(stage)]
    assert [spend["scheduler_llm_usd"] for spend in spends[:2]] == [0.0, JUDGE_USD]
    usages = [record.usage for record in (await observer.snapshot()).llm_usage]
    assert usages == [LlmUsage("", UsageSource.JUDGE, JUDGE_USD, JUDGE_TOKENS)] * judge_calls
    assert all(spend["explore_gpu_hours"] is None for spend in spends)
    [rooted] = [row for row in _decisions(stage) if row["reason"] == "root committed"]
    prelude_hours = round(rooted["detail"]["child_gpu_seconds_upper_bound"] / MB_SECONDS_PER_HOUR, 3)
    assert all(spend["prelude_gpu_hours_upper_bound"] == prelude_hours for spend in spends)
    # Each judge call's view reads the queue twice, for its widening signals and the builds a branch waits for, and the
    # dashboard rebuild after it once; so do the stop and every quota push.
    assert judge_calls >= 2 and stage.toolbox.queue_reads == 3 * judge_calls + 1 + len(stage.toolbox.policies)
    assert await stage.processes.leftovers() == ()
    for path in (
        stage.layout.manifest,
        stage.layout.decisions,
        stage.layout.judge,
        stage.layout.prelude_child,
        stage.layout.root_recipe,
        stage.layout.report,
        stage.layout.reports / "session_breakdown.json",
        stage.layout.merges,
        outcome.close.exports.dashboard.html,
    ):
        assert path.is_file(), path
    assert all(entry.name.startswith("MB-") for entry in stage.layout.root.iterdir())


async def test_an_agentx_session_is_judged_on_p50_interactivity_and_validated_at_its_conc_alone(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    run = with_agentx_baseline_run(child, DEFAULT_BASELINE, runtime_seconds=AGENTX_CHILD_RUN_SECONDS)
    stage = _stage(tmp_path, child=child)
    stage.settings = replace(SETTINGS, workload=AGENTX)
    stage.scripts = SCRIPTS

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    assert not [request for request in stage.toolbox.requests.values() if request.purpose is Purpose.BASELINE]
    assert isinstance(stage.toolbox, LeasingToolbox) and stage.toolbox.leased == [("prelude", AGENTX.tp)]
    assert stage.processes.env is not None and stage.processes.env[CHILD_CARDS_ENV] == "6"
    baseline = await stage.handle(Role.OBSERVER).baseline()
    assert (baseline.request_ids, baseline.values[GRADED_INTVTY_P50]) == ((), DEFAULT_BASELINE[GRADED_INTVTY_P50])
    assert baseline.source.startswith(f"prelude-child:{run} ")
    [established] = [row for row in _decisions(stage) if row["reason"] == "baseline established"]
    assert established["detail"]["longest_test_seconds"] == AGENTX_CHILD_RUN_SECONDS
    observer = stage.handle(Role.OBSERVER)
    profile = await observer.session_profile()
    assert (profile.isl, profile.osl, dict(profile.roofline_ceiling)) == (None, None, {})
    [child] = [row for row in _decisions(stage) if row["kind"] == "prelude" and row["reason"] == "child started"]
    assert not {"--isl", "--osl"} & set(child["detail"]["argv"])
    assert GRADED_INTVTY_P50 in (await observer.baseline()).threshold_pct
    c1, c2 = await stage.repo.head("b01"), await stage.repo.head("b02")
    merged = {item.commit_id: item.last_merge.outcome.value for item in await observer.committed() if item.last_merge}
    assert merged == {c1: "merged", c2: "not_stackable"}
    opening = _judge_state(_judge_prompts(stage)[0])
    assert opening["metric"] == GRADED_INTVTY_P50
    assert (opening["integration"]["roofline_ceiling"], opening["integration"]["headroom_pct"]) == ({}, None)

    head = await stage.repo.head(INTEGRATION_BRANCH)
    assert outcome.final is not None and outcome.final.chosen is not None and outcome.final.chosen.commit == head
    assert outcome.final.metric == GRADED_INTVTY_P50 and outcome.final.reason.endswith(PRIMARY_POINT_ONLY)
    sweeps = [request for request in stage.toolbox.requests.values() if request.kind is TestKind.SWEEP]
    assert sorted(request.parent_commit for request in sweeps) == sorted((head, c2))
    assert {request.concurrency for request in sweeps} == {(CONC,)}
    assert json.loads(stage.layout.manifest.read_text())["metric"] == GRADED_INTVTY_P50


async def test_the_specialists_are_briefed_on_the_warm_start_leads_prelude_read_and_replay_them_on_their_head(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.recipe_kb = OneChampionKB()

    await stage.run()

    [lead] = stage.briefs.warm
    assert (lead.canonical_id, lead.server_args) == ("inference:champion", "--enable-expert-parallel")
    assert stage.recipe_kb.searched
    replay = await stage.briefs.for_branch(INTEGRATION_BRANCH).warm_replay(lead.canonical_id)
    assert replay is not None and replay.parent_commit == await stage.repo.head(INTEGRATION_BRANCH)


def _frontier_lines(stage: Stage) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (stage.layout.reports / FRONTIER_LOG).read_text().splitlines()]


async def test_a_session_scores_its_frontier_with_the_defaults_for_its_budget(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage)

    outcome = await stage.run()

    defaults = FrontierPolicy.for_budget(GRADED_OUTPUT, SETTINGS.max_hours * MB_SECONDS_PER_HOUR)
    lines = _frontier_lines(stage)
    assert outcome.exit_code == 0 and lines
    assert all(line["scorer"] == FRONTIER_V1_LABEL for line in lines)
    assert all(line["new_branch_floor_pct"] == defaults.new_branch_floor_pct for line in lines)
    assert all(line["window_seconds"] >= defaults.window_seconds for line in lines)


async def test_close_asks_no_llm_for_a_playbook_change_without_an_archive_and_records_why(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage)

    outcome = await stage.run()

    assert outcome.exit_code == 0 and stage.author.calls == []
    breakdown = json.loads((stage.layout.reports / "session_breakdown.json").read_text())
    playbook = breakdown["close"]["knowledge"]["playbook"]
    assert (playbook["status"], playbook["reason"]) == (SKIPPED, NO_ARCHIVE)


async def test_a_session_with_an_rsi_archive_plays_the_champion_playbook_and_close_keeps_the_judges_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "rsi"
    monkeypatch.setenv(ARCHIVE_ENV, str(archive))
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage)

    outcome = await stage.run()

    assert outcome.exit_code == 0
    [fingerprint] = await stage.handle(Role.OBSERVER).manifest_fingerprints()
    assert fingerprint.policy_versions[PLAYBOOK_KEY] == H0.label and FALLBACK_KEY not in fingerprint.policy_versions
    [chosen] = [row for row in _decisions(stage) if row["reason"] == "scheduling playbook"]
    assert chosen["detail"] == {"playbook": H0.label, "champion": H0.label, "challenger": False}
    prompts = _judge_prompts(stage)
    assert prompts and all("Scheduling playbook h0:" in prompt for prompt in prompts)
    assert len(decision_calls(TreeArchive(archive), stage.layout.session_id)) == len(prompts)


async def test_the_manifest_names_the_lesson_pool_as_the_specialists_recall_read_it(tmp_path: Path) -> None:
    pool = LessonPool(tmp_path / "pool")
    pool.write(
        LessonRevision(
            revision_id_for("past"),
            lesson_id_for("past"),
            LearningKind.PITFALL,
            "the bare --cuda-graph-max-bs is ambiguous; use the -decode form",
            TaskConditions(),
            "specialist:past-1/b05",
            0.0,
            session_id="past-1",
        )
    )
    read = pool.digest()
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.learnings = LessonRecall(pool=pool)
    _script(stage)

    outcome = await stage.run()

    assert outcome.exit_code == 0 and read != ""
    assert json.loads(stage.layout.manifest.read_text())["policy_versions"][LESSONS_KEY] == read


async def test_a_session_copies_the_reference_material_it_was_given_and_its_manifest_pins_it(tmp_path: Path) -> None:
    knowledge = tmp_path / "GEAK" / "tuning"
    knowledge.mkdir(parents=True)
    (knowledge / "gemm.md").write_text("pad M to the next multiple of 64", encoding="utf-8")
    source = ReferenceSource("geak-tuning", str(knowledge), "c" * 40)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(stage.settings, references=(source,))
    _script(stage)

    outcome = await stage.run()

    assert outcome.exit_code == 0
    copied = stage.layout.reference / "geak-tuning" / "gemm.md"
    assert copied.read_text(encoding="utf-8") == "pad M to the next multiple of 64"
    assert json.loads(stage.layout.manifest.read_text())["settings"]["references"] == [
        {"name": "geak-tuning", "path": str(knowledge), "revision": "c" * 40}
    ]
    assert stage.setup.references == {"geak-tuning": stage.layout.reference / "geak-tuning"}


async def test_a_run_given_the_inferencex_checkout_briefs_its_launches_for_the_session_model_and_one_without_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "InferenceX"
    (checkout / "configs").mkdir(parents=True)
    (checkout / "configs" / "amd-master.yaml").write_text(LLAMA_MATRIX, encoding="utf-8")
    given = ReferenceSource(INFERENCEX_REFERENCE, str(checkout), "d" * 40)
    briefed = {}
    for name, references in (("given", (given,)), ("without", ())):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(MB_FRAMEWORK_REPO_ENV, str(tmp_path / name / "framework"))
        stage = _stage(tmp_path / name, child=child_with_patch(tmp_path / name))
        stage.settings = replace(stage.settings, references=references)
        await stage.run()
        briefed[name] = stage.briefs.references

    [lead] = briefed["given"]
    assert (lead.config, lead.model, lead.revision) == ("llama3-bf16-mi355x-sglang", "meta-llama/Llama-3-8B", "d" * 40)
    assert briefed["without"] == ()


async def test_the_dev_leases_of_the_run_reach_the_peers(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=None)
    stage.settings = replace(stage.settings, dev_leases=2)

    stage._session()

    assert stage.setup.dev_leases == 2


async def test_the_agent_port_of_the_run_reaches_the_peers(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=None)
    stage.settings = replace(stage.settings, agent_port=32100)

    stage._session()

    assert stage.setup.agent_port == 32100


@pytest.mark.parametrize(("review_gate", "handed"), [(True, "host"), (False, "off")])
async def test_the_session_s_review_stands_in_for_each_arbor_s_own(
    tmp_path: Path, review_gate: bool, handed: str
) -> None:
    stage = _stage(tmp_path, child=None)
    stage.settings = replace(stage.settings, review_gate=review_gate)

    stage._session()

    assert stage.setup.agent_review == handed


async def test_a_failed_prelude_skips_to_close(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=None, exit_code=1)

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.PRELUDE_FAILED and outcome.exit_code == 1
    assert "exited with 1" in outcome.error
    assert outcome.final is None and outcome.close is not None and outcome.close.stop.clean
    assert stage.judge.calls == []
    assert json.loads(stage.layout.report.read_text())["stop_reason"] == SessionStopReason.PRELUDE_FAILED.value


async def test_a_session_on_this_machine_runs_its_root_tests_and_champion_beside_the_prelude_child_on_leased_cards(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path), linger=CHILD_BESIDE_SECONDS)
    stage.recipe_kb = OneChampionKB()
    stage.judge = ScriptedCompletion([_reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.JUDGE_STOPPED, "")
    toolbox = stage.toolbox
    assert isinstance(toolbox, LeasingToolbox)
    assert (toolbox.leased, toolbox.released) == ([("prelude", WORKLOAD.tp)], ["lease-1"])
    beside = sorted(purpose.value for purpose, held in toolbox.beside_a_lease if held)
    prelude = [Purpose.BASELINE] * BASELINE_TESTS + [Purpose.INTEGRATION, Purpose.CONFIRM]
    assert beside == sorted(purpose.value for purpose in prelude)
    assert stage.processes.env is not None and stage.processes.env[CHILD_CARDS_ENV] == "6"
    baseline = await stage.handle(Role.OBSERVER).baseline()
    assert sorted(baseline.request_ids) == sorted(
        request_id for request_id, request in toolbox.requests.items() if request.purpose is Purpose.BASELINE
    )


class SandboxToolbox(RecordingToolbox):
    """The recording toolbox of a backend measuring in sandboxes elsewhere, so it leases no card, with a TraceLens route
    whose job ends at once with :data:`SANDBOX_SHARES` on the commit its profile test measured; a double for a hosted
    backend, not a placeholder."""

    async def submit_analysis(
        self, branch: str, tool: str, params: Mapping[str, Any], profile_request_id: str = ""
    ) -> str:
        commit = self.requests[profile_request_id].parent_commit
        pointer = ToolResultPointer(branch, "tracelens-1", "/a", "trace analysed", commit, profile_request_id)
        await self.blackboard.notify_tool_result(replace(pointer, op_shares=SANDBOX_SHARES))
        return pointer.job_id

    def tool_status(self, job_id: str) -> JobStatus:
        return JobStatus.SUCCEEDED


SANDBOX_SHARES = {"moe_gemm": 41.5, "attention": 22.0}


async def test_a_session_measuring_in_sandboxes_starts_no_child_and_opens_explore_on_the_roots_own_profile(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=None, exit_code=1)
    stage.toolbox_type, stage.on_this_machine = SandboxToolbox, False
    stage.judge = ScriptedCompletion([_reply({"open": 1, "branches": [], "stop": []})])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    assert stage.processes.child is None
    root = await stage.root()
    prelude = [request for request in stage.toolbox.requests.values() if request.parent_commit == root]
    purposes = sorted(request.purpose.value for request in prelude if request.purpose is not Purpose.FINAL)
    assert purposes == sorted([Purpose.BASELINE.value] * BASELINE_TESTS + [Purpose.PROFILE.value])
    shares = await stage.handle(Role.OBSERVER).op_shares(root)
    assert shares is not None and dict(shares.shares) == SANDBOX_SHARES
    rows = _decisions(stage)
    profiled = next(index for index, row in enumerate(rows) if row["kind"] == "profile")
    assert rows[profiled]["reason"] == PROFILED and rows[profiled]["detail"]["commit"] == root
    assert profiled < next(index for index, row in enumerate(rows) if row["kind"] == "judge")
    [rooted] = [row for row in rows if row["reason"] == "root committed"]
    assert rooted["detail"] == {"root_commit": root}
    assert stage.sessions.started == {"b01": root}


async def test_an_agentx_champion_replay_asks_for_no_confirm_ahead_of_its_judgement(tmp_path: Path) -> None:
    child = child_with_patch(tmp_path)
    with_agentx_baseline_run(child, DEFAULT_BASELINE, runtime_seconds=AGENTX_CHILD_RUN_SECONDS)
    stage = _stage(tmp_path, child=child)
    stage.settings = replace(SETTINGS, workload=AGENTX)
    stage.recipe_kb = OneChampionKB()
    stage.judge = ScriptedCompletion([_reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.JUDGE_STOPPED, "")
    requests = stage.toolbox.requests
    [replay] = [request_id for request_id, request in requests.items() if request.purpose is Purpose.INTEGRATION]
    assert not [request for request in requests.values() if request.retest_of == replay]


async def test_a_child_whose_baseline_failed_fails_prelude_at_once_with_its_reason(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=failed_child(tmp_path), exit_code=1)

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.PRELUDE_FAILED and outcome.exit_code == 1
    assert "stop_reason='baseline_failed'" in outcome.error and repr(BASELINE_ERROR_CLASS) in outcome.error
    assert outcome.final is None and outcome.close is not None and outcome.close.stop.clean
    assert stage.judge.calls == []
    assert {request.purpose for request in stage.toolbox.requests.values()} == {Purpose.BASELINE}
    assert stage.time.now() - START < 60.0
    assert json.loads(stage.layout.report.read_text())["stop_reason"] == SessionStopReason.PRELUDE_FAILED.value


def _fails_strictly() -> ImageCheckReport:
    raise ImageCheckFailed(replace(IMAGE_CHECK, mode=ImageCheckMode.STRICT))


async def test_a_strict_image_check_that_fails_stops_the_session_before_its_manifest_and_any_phase(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.image_check = _fails_strictly

    with pytest.raises(ImageCheckFailed, match="sglang.renamed-flag"):
        await stage.run()

    assert not stage.layout.manifest.exists() and not stage.layout.report.exists()
    assert stage.judge.calls == [] and not stage.layout.prelude_child.exists()


async def test_a_stop_signal_skips_to_close_and_stops_every_branch(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))

    async def interrupted(branch: str) -> None:
        await stage.time.sleep(STEP_SECONDS)
        os.kill(os.getpid(), signal.SIGTERM)

    stage.steps = {"b01": interrupted}

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.exit_code == 1
    assert outcome.final is None and outcome.close is not None
    assert sorted(stage.sessions.stopped) == ["b01", "b02"]
    assert set(stage.toolbox.cancelled) == {"b01", "b02", INTEGRATION_BRANCH}
    assert stage.toolbox.closed == 1
    late = await stage.setup.desk.admit(ForkRequest("b01", await stage.root(), "late fix"))
    assert late.status is AdmissionStatus.REJECTED
    roster = json.loads(stage.layout.roster.read_text())
    assert roster["closed"] and all(
        entry["stop"]["cause"] == StopCause.SESSION_END.value for entry in roster["entries"]
    )


def _signalling(stage: Stage) -> dict[str, SessionStep]:
    """``b01``'s specialist has an agent CLI started, then the operator stops the session."""

    async def interrupted(branch: str) -> None:
        await stage.time.sleep(STEP_SECONDS)
        await stage.start_agent(branch)
        stage.signalled_at = stage.time.now()
        os.kill(os.getpid(), signal.SIGTERM)

    return {"b01": interrupted}


async def test_close_after_a_signal_waits_the_close_grace_not_the_budget(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.steps, stage.hanging = _signalling(stage), frozenset({"b01", "b02"})

    outcome = await stage.run()

    assert outcome.close is not None and STOP_OVERRAN in outcome.close.stop.failures
    assert stage.toolbox.closed == 1
    assert stage.time.now() < START + SETTINGS.max_hours * 3600
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_a_toolbox_that_cannot_stop_after_a_signal_does_not_hold_close_up(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type, stage.stuck = StuckToolbox, frozenset({INTEGRATION_BRANCH})
    stage.steps = _signalling(stage)

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.close is not None
    assert STOP_OVERRAN in outcome.close.stop.failures
    assert json.loads(stage.layout.report.read_text())["stop_reason"] == SessionStopReason.SIGNAL.value
    assert stage.time.now() < START + SETTINGS.max_hours * 3600
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_after_a_signal_the_session_ends_within_one_close_grace(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type, stage.stuck = StuckToolbox, frozenset({INTEGRATION_BRANCH})
    stage.steps, stage.hanging = _signalling(stage), frozenset({"b01", "b02"})

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.close is not None
    grace = SessionBudget.from_hours(SETTINGS.max_hours, START).close_grace_seconds
    assert stage.signalled_at and stage.time.now() <= stage.signalled_at + grace
    assert stage.pids and not any(running(pid) for pid in stage.pids)


async def test_a_signal_during_a_hanging_close_cuts_it_short_and_leaves_no_process(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.steps, stage.runtime_type = _signalling(stage), InsistingRuntime

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.exit_code == 1
    assert "cut short" in outcome.error
    assert outcome.close is not None and outcome.close.exports is None
    assert outcome.close.stop.failures == (CUT_SHORT,) and outcome.close.stop.leftovers == ()
    written = json.loads(stage.layout.report.read_text())
    assert (written["stop_reason"], written["stop"]["failures"]) == ("signal", [CUT_SHORT])
    assert stage.pids and not any(running(pid) for pid in stage.pids)
    assert await stage.processes.leftovers() == ()
    assert stage.toolbox.closed == 1


async def test_a_judge_that_never_answers_gives_up_at_the_explore_deadline_and_opens_nothing(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = SilentModel()

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert outcome.final is not None and stage.sessions.started == {}
    decisions = [json.loads(line) for line in stage.layout.decisions.read_text().splitlines()]
    assert [row["reason"] for row in decisions if row["kind"] == "judge_failed"] == [
        "no answer by the EXPLORE deadline"
    ]


def _decisions(stage: Stage) -> list[dict[str, object]]:
    return [json.loads(line) for line in stage.layout.decisions.read_text().splitlines()]


def _judge_prompts(stage: Stage) -> list[str]:
    return [json.loads(line)["prompt"] for line in stage.layout.judge.read_text().splitlines()]


def _judge_state(prompt: str) -> dict[str, Any]:
    return json.loads(prompt[prompt.index("{") : prompt.index("\n\n", prompt.index("{"))])


async def test_spent_gpu_hours_end_explore_as_a_completed_session_that_still_runs_final(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, max_gpu_hours=TINY_GPU_HOURS)
    stage.judge = ScriptedCompletion([OPENING])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.GPU_HOURS_EXHAUSTED, "", 0)
    explore_end = SessionBudget.from_hours(SETTINGS.max_hours, START).explore_end(START, 0.0)
    [final_phase] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "final"]
    assert float(str(final_phase["at"])) < explore_end
    assert outcome.final is not None and outcome.final.chosen is not None
    assert sorted(stage.sessions.stopped) == ["b01", "b02"]
    [close] = [row for row in _decisions(stage) if row["kind"] == "session_stop"]
    assert close["reason"] == SessionStopReason.GPU_HOURS_EXHAUSTED.value

    # The opening call comes after PRELUDE and before any branch ran a test: what it was shown is the baseline on the
    # root, which the PRELUDE child's own measurements never reach.
    records = (await stage.handle(Role.OBSERVER).snapshot()).records
    baseline = [record for record in records if record.request.purpose is Purpose.BASELINE]
    assert baseline and all(record.result is not None for record in baseline)
    baseline_seconds = sum(record.result.gpu_seconds for record in baseline if record.result is not None)
    [opening] = _judge_prompts(stage)
    spend = _judge_state(opening)["spend"]
    assert spend["gpu_hours_used"] == round(baseline_seconds / MB_SECONDS_PER_HOUR, 3) > 0
    assert spend["explore_gpu_hours"] == round(explore_gpu_seconds(TINY_GPU_HOURS) / MB_SECONDS_PER_HOUR, 3)


async def test_card_time_a_lease_holds_ends_explore_on_the_gpu_hours_and_final_still_runs(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(
        SETTINGS, max_gpu_hours=LEASED_GPU_HOURS, judge_interval_seconds=LEASED_JUDGE_INTERVAL_SECONDS
    )

    async def hold_cards(branch: str) -> None:
        cards = tuple(range(LEASED_CARDS))
        lease = LeaseStart(
            f"{branch}-dev", branch, branch, LeaseKind.DEV, LEASED_CARDS, stage.time.now(), gpu_ids=cards
        )
        await stage.handle(Role.TOOLBOX).record_lease_start(lease)

    stage.steps = {"b01": hold_cards}

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.GPU_HOURS_EXHAUSTED, "", 0)
    explore_end = SessionBudget.from_hours(SETTINGS.max_hours, START).explore_end(START, 0.0)
    [final_phase] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "final"]
    assert float(str(final_phase["at"])) < explore_end
    assert outcome.final is not None and outcome.final.chosen is not None
    snapshot = await stage.handle(Role.OBSERVER).snapshot()
    assert snapshot.gpu_hours < explore_gpu_seconds(LEASED_GPU_HOURS) / MB_SECONDS_PER_HOUR

    # A steering call made while the lease was open was shown its card time, on b01's row and in the session's spend.
    steering = _judge_state(_judge_prompts(stage)[-1])
    [b01] = [row for row in steering["branches"] if row["branch"] == "b01"]
    spent = sum(row["gpu_seconds"] + row["held_card_seconds"] for row in steering["branches"])
    assert b01["held_card_seconds"] > 0
    assert steering["spend"]["gpu_hours_used"] == round(spent / MB_SECONDS_PER_HOUR, 3)


async def test_spent_specialist_llm_dollars_end_explore_as_a_completed_session_that_still_runs_final(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, max_llm_usd=SMALL_LLM_USD)
    stage.judge = ScriptedCompletion([OPENING])

    async def one_launch(branch: str) -> None:
        usage = LlmUsage(branch, UsageSource.SPECIALIST, LAUNCH_USD, launch=f"{branch}-launch")
        await stage.handle(Role.SPECIALIST_RUNTIME, branch).record_llm_usage(usage)

    stage.steps = {"b01": one_launch}

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.LLM_BUDGET_EXHAUSTED, "", 0)
    explore_end = SessionBudget.from_hours(SETTINGS.max_hours, START).explore_end(START, 0.0)
    [final_phase] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "final"]
    assert float(str(final_phase["at"])) < explore_end
    assert outcome.final is not None and outcome.final.chosen is not None
    [close] = [row for row in _decisions(stage) if row["kind"] == "session_stop"]
    assert close["reason"] == SessionStopReason.LLM_BUDGET_EXHAUSTED.value
    [opening] = _judge_prompts(stage)
    spend = _judge_state(opening)["spend"]
    assert (spend["specialist_llm_usd"], spend["max_llm_usd"]) == (0.0, SMALL_LLM_USD)


async def test_the_judge_s_own_llm_dollars_count_against_the_llm_budget(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, max_llm_usd=SMALL_LLM_USD)
    stage.judge = ScriptedCompletion([OPENING], tokens=JUDGE_TOKENS, cost_usd=LAUNCH_USD)

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.LLM_BUDGET_EXHAUSTED, "", 0)
    [opening] = _judge_prompts(stage)
    spend = _judge_state(opening)["spend"]
    assert (spend["specialist_llm_usd"], spend["scheduler_llm_usd"]) == (0.0, 0.0)
    snapshot = await stage.handle(Role.OBSERVER).snapshot()
    assert snapshot.llm_usd == LAUNCH_USD
    assert [record.usage.source for record in snapshot.llm_usage] == [UsageSource.JUDGE]


async def test_the_fix_critic_s_llm_dollars_count_against_the_llm_budget_as_the_scheduler_s(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, max_llm_usd=SMALL_LLM_USD)
    stage.judge = ScriptedCompletion([OPENING])

    async def critic_model(system: str, user: str) -> CompletionReply:
        return CompletionReply(
            json.dumps({"reachable": True, "reason": "the workload sends it"}), JUDGE_TOKENS, LAUNCH_USD
        )

    stage.critic = lambda usage: LlmCritic(critic_model, usage)
    fixed = Judgement(
        Verdict.REGRESSED, -2.0, 2.0, ("fix-1",), {GRADED_OUTPUT: 980.0}, AccuracyVerdict.PASSED, fixes_correctness=True
    )
    review = FixReview(
        delta(), {}, FixCase("exit 0\n", "a defect"), "the workload", fixed, ReproVerdicts(REPRODUCED, NOT_REPRODUCED)
    )

    async def review_a_fix(branch: str) -> None:
        await stage.setup.critic.review(review)

    stage.steps = {"b01": review_a_fix}

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.LLM_BUDGET_EXHAUSTED, "", 0)
    snapshot = await stage.handle(Role.OBSERVER).snapshot()
    critic = [record.usage for record in snapshot.llm_usage if record.usage.source is UsageSource.CRITIC]
    assert critic == [LlmUsage("", UsageSource.CRITIC, LAUNCH_USD, JUDGE_TOKENS)]


async def test_a_judge_soft_stop_ends_explore_early_and_still_runs_final(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([OPENING, _reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.JUDGE_STOPPED, "", 0)
    explore_end = SessionBudget.from_hours(SETTINGS.max_hours, START).explore_end(START, 0.0)
    [final_phase] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "final"]
    assert float(str(final_phase["at"])) < explore_end
    root = await stage.root()
    assert [row["commit"] for row in final_phase["detail"]["candidates"]] == [root]
    assert outcome.final is not None and outcome.final.chosen is not None
    assert outcome.final.chosen.commit == root
    assert sorted(stage.sessions.stopped) == ["b01", "b02"]


async def test_a_specialist_that_never_stops_at_the_end_of_explore_does_not_hold_final_up(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([OPENING, _reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})])
    stage.hanging = frozenset({"b01"})

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.JUDGE_STOPPED, "")
    assert outcome.final is not None and outcome.final.chosen is not None
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b01"]
    assert [failure.split(":")[0] for failure in stop["detail"]["failures"]] == ["TimeoutError"]
    assert stage.sessions.stopped.count("b01") == 1 and "b01" in stage.sessions.running
    assert "the specialist of b01 did not confirm its stop" in outcome.close.stop.failures
    assert not outcome.completed and outcome.exit_code == 1


async def test_a_judge_stop_that_never_returns_does_not_freeze_explore(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage)
    stage.hanging = frozenset({"b02"})

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert outcome.final is not None and outcome.final.chosen is not None
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b02"]
    assert stop["reason"] == f"judge: {STOP_REASON}" and "b02" in stage.toolbox.tool_cancels
    assert len(stage.judge.calls) > 2


async def test_a_first_signal_during_close_cuts_it_short_after_one_close_grace(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([OPENING, _reply({"open": 0, "stop_session": {"reason": CONVERGED_REASON}})])
    stage.toolbox_type = SignallingToolbox

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.JUDGE_STOPPED and outcome.final is not None
    assert CUT_SHORT in outcome.error and outcome.close is not None
    assert outcome.close.stop.failures == (CUT_SHORT, TOOLBOX_CLOSE_OVERRAN)
    [close] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "close"]
    grace = SessionBudget.from_hours(SETTINGS.max_hours, START).close_grace_seconds
    assert stage.time.now() <= float(str(close["at"])) + grace + TOOLBOX_CLOSE_SECONDS


async def test_an_unhandled_error_bounds_close_by_one_close_grace(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type, stage.repo_type = StuckToolbox, NoObjectStore

    with pytest.raises(LookupError, match="no object store"):
        await stage.run()

    report = json.loads(stage.layout.report.read_text())
    assert report["stop_reason"] == SessionStopReason.ABORTED.value
    assert report["stop"]["failures"] == [TOOLBOX_CLOSE_OVERRAN]
    [close] = [row for row in _decisions(stage) if row["kind"] == "phase" and row["reason"] == "close"]
    grace = SessionBudget.from_hours(SETTINGS.max_hours, START).close_grace_seconds
    assert stage.time.now() <= float(str(close["at"])) + grace


async def test_a_soft_stop_on_the_opening_call_opens_no_branch(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([_reply({"open": 2, "stop_session": {"reason": CONVERGED_REASON}})])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.JUDGE_STOPPED, "")
    assert stage.sessions.started == {} and outcome.final is not None


async def test_a_failed_opening_judge_call_opens_one_branch_at_k_one(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion(error=CompletionFailed("judge call failed: RuntimeError: status=529"))

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert stage.sessions.started == {"b01": await stage.root()}
    assert stage.sessions.k == {"b01": 1}
    failed = [row["reason"] for row in _decisions(stage) if row["kind"] == "judge_failed"]
    assert failed and set(failed) == {"kept the status quo"}
    assert not [
        row for row in _decisions(stage) if row["kind"] == "judge" or (row["kind"] == "open" and not row["branch"])
    ]


def _crowding(stage: Stage) -> type[RefusingRuntime]:
    """A runtime during whose start of ``b01`` two forks of it are asked for, which the fork desk admits
    onto the last free slots; a test double for the admissions that race the opening batch, not a placeholder."""

    class CrowdingRuntime(RefusingRuntime):
        async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
            await super().start(branch, start_commit, task)
            if branch == "b01":
                for reason in (FIX_REASON, "fix f2 regresses"):
                    stage.forks.append(await stage.setup.desk.admit(ForkRequest(branch, start_commit, reason)))

    return CrowdingRuntime


async def test_forks_admitted_while_the_opening_batch_starts_cut_the_batch_and_the_session_goes_on(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([_reply({"open": 3, "branches": [], "stop": []})])
    stage.runtime_type = _crowding(stage)

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert [(admission.status, admission.branch) for admission in stage.forks] == [
        (AdmissionStatus.ACCEPTED, "b02"),
        (AdmissionStatus.ACCEPTED, "b03"),
    ]
    root = await stage.root()
    assert stage.sessions.started == {"b01": root}
    roster = json.loads(stage.layout.roster.read_text())
    assert roster["issued_branch_ids"] == 3
    decisions = _decisions(stage)
    sessionwide = [row for row in decisions if row["kind"] in ("judge", "open") and not row["branch"]]
    assert [row["kind"] for row in sessionwide][:2] == ["judge", "open"]
    [cut] = [row for row in sessionwide if row["kind"] == "open"]
    assert cut["reason"] == OPEN_CUT_REASON
    assert cut["detail"] == {"requested": 3, "opened": 1, "open_cut": 2}


async def test_explore_ends_the_longest_baseline_test_and_one_close_grace_before_the_deadline_then_final_and_close_run(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    budget = SessionBudget.from_hours(SETTINGS.max_hours, START)
    decisions = _decisions(stage)
    [baseline] = [row for row in decisions if row["kind"] == "prelude" and row["reason"] == "baseline established"]
    phases = {str(row["reason"]): row for row in decisions if row["kind"] == "phase"}
    longest = float(str(baseline["detail"]["longest_test_seconds"]))
    explore_end = float(str(phases["explore"]["detail"]["explore_end"]))
    assert longest > 0
    assert explore_end == pytest.approx(budget.deadline - longest - budget.close_grace_seconds)
    assert explore_end <= float(str(phases["final"]["at"])) < budget.deadline
    assert outcome.final is not None and outcome.final.chosen is not None
    assert "close" in phases and outcome.close is not None


def _slow_final(stage: Stage, *, interrupt: bool = False) -> type[RecordingToolbox]:
    """A toolbox each of whose FINAL tests runs :data:`FINAL_TEST_SECONDS` before it measures; with ``interrupt``, the
    operator stops the session as the first one would end, every one runs as long again, and the integration branch's
    tests are never all withdrawn. A test double for validation that outlasts the budget, not a placeholder."""

    class SlowFinalToolbox(LeasingToolbox):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            measure = self.executor.run

            async def run(job: TestJob) -> TestResult:
                if job.request.purpose is Purpose.FINAL:
                    await stage.time.sleep(FINAL_TEST_SECONDS)
                    if interrupt and not stage.signalled_at:
                        stage.signalled_at = stage.time.now()
                        os.kill(os.getpid(), signal.SIGTERM)
                    if interrupt:
                        await stage.time.sleep(FINAL_TEST_SECONDS)
                return await measure(job)

            self.executor.run = run  # type: ignore[method-assign]

        async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
            if interrupt and branch_id == INTEGRATION_BRANCH:
                await asyncio.Event().wait()
            return await super().cancel_branch(branch_id, reason)

    return SlowFinalToolbox


async def test_a_final_whose_validation_outlasts_the_budget_still_chooses_and_close_follows_it(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type = _slow_final(stage)

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    assert outcome.final is not None and outcome.final.chosen is not None
    assert outcome.final.chosen.commit == await stage.root()
    assert outcome.close is not None and outcome.close.stop.clean
    budget = SessionBudget.from_hours(SETTINGS.max_hours, START)
    [final] = [row for row in _decisions(stage) if row["kind"] == "final"]
    assert float(str(final["at"])) > budget.hard_stop_at
    assert stage.time.now() <= float(str(final["at"])) + budget.close_grace_seconds


async def test_a_stop_signal_cuts_a_final_past_the_hard_stop_short_and_close_gets_one_close_grace(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.toolbox_type = _slow_final(stage, interrupt=True)

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.exit_code == 1
    assert outcome.final is None and outcome.close is not None
    assert STOP_OVERRAN in outcome.close.stop.failures
    budget = SessionBudget.from_hours(SETTINGS.max_hours, START)
    grace = budget.close_grace_seconds
    assert budget.hard_stop_at < stage.signalled_at
    assert stage.signalled_at + grace - TOOLBOX_CLOSE_SECONDS <= stage.time.now() <= stage.signalled_at + grace
    final_tests = {rid for rid, test in stage.toolbox.requests.items() if test.purpose is Purpose.FINAL}
    assert final_tests and final_tests <= stage.toolbox.withdrawn.keys()


class StuckStopToolbox(LeasingToolbox):
    """A toolbox whose cancel of ``b01`` never returns, so the end of EXPLORE hangs and the session deadline passes
    before FINAL: the state an operator is left in with a session worth validating and no final selection."""

    async def cancel_branch(self, branch_id: str, reason: str = "branch stopped") -> int:
        if branch_id == "b01":
            await asyncio.Event().wait()
        return await super().cancel_branch(branch_id, reason)


async def test_final_only_validates_the_branches_a_session_that_never_reached_final_left(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.scripts = SCRIPTS
    stage.toolbox_type = StuckStopToolbox

    cut_short = await stage.run()

    assert (cut_short.stop_reason, cut_short.final) == (SessionStopReason.TIME_EXHAUSTED, None)
    assert json.loads(stage.layout.report.read_text())["final"] is None

    persisted = persisted_run(stage.layout.root)
    assert (persisted.ended, persisted.settings, persisted.layout) == (
        SessionStopReason.TIME_EXHAUSTED,
        SETTINGS,
        stage.layout,
    )
    before = json.loads(stage.layout.roster.read_text())
    policies = json.loads(stage.layout.manifest.read_text())["policy_versions"]
    recovery = _recovery(stage, persisted.settings)
    outcome = await recovery.finish(persisted.ended)

    head, c2 = await stage.repo.head(INTEGRATION_BRANCH), await stage.repo.head("b02")
    assert outcome.final is not None and outcome.final.chosen is not None
    # The branch set FINAL weighs comes from the blackboard, which holds every branch the session explored.
    assert {check.candidate.commit for check in outcome.final.checks} == {head, c2}
    assert (outcome.final.chosen.label, outcome.final.chosen.commit) == (INTEGRATION_LABEL, head)
    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    assert outcome.close is not None and outcome.close.stop.clean

    report = json.loads(stage.layout.report.read_text())
    assert report["final"]["chosen"]["commit"] == head
    assert report["stop_reason"] == SessionStopReason.TIME_EXHAUSTED.value
    [final] = [row for row in _decisions(stage) if row["kind"] == "final"]
    assert float(str(final["at"])) > SessionBudget.from_hours(SETTINGS.max_hours, START).deadline
    # The session that opened the integration branch recorded its STOP; a second one is a conflicting write.
    assert [row["branch"] for row in _decisions(stage) if row["kind"] == "stop"].count(INTEGRATION_BRANCH) == 1
    branches_by_name = {branch.branch: branch for branch in (await recovery.handle(Role.OBSERVER).lineage()).branches}
    assert branches_by_name[INTEGRATION_BRANCH].stop_reason == f"session_end: {SessionStopReason.TIME_EXHAUSTED.value}"
    assert json.loads(stage.layout.roster.read_text()) == before
    assert json.loads(stage.layout.manifest.read_text())["policy_versions"] == policies
    # The tests the dead session left are withdrawn before FINAL's quota, which admits the integration branch alone.
    assert recovery.toolbox.policies == [final_quota(SETTINGS.caps)]
    assert recovery.toolbox.withdrawals == [SessionStopReason.TIME_EXHAUSTED.value] * 2

    with pytest.raises(ValueError, match="already chose"):
        persisted_run(stage.layout.root)


def _as_another_pod(pids: Path) -> None:
    """Move every record of ``MB-pids.jsonl`` to a pod that is gone, as the kill of a session's pod leaves them."""

    rows = [{**row, "owner_scope": ANOTHER_POD} for row in read_jsonl(pids)]
    pids.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


async def test_final_only_finishes_a_session_that_was_killed_before_it_could_close(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.scripts = SCRIPTS
    stage.toolbox_type = StuckStopToolbox
    await stage.run()
    # The SIGKILL leaves the directory without the report CLOSE would have written.
    stage.layout.report.unlink()

    # The scheduler recorded itself when the run started, so the directory names the process the session ran as.
    assert os.getpid() in [row["pid"] for row in read_jsonl(stage.layout.pids)]

    _as_another_pod(stage.layout.pids)
    persisted = persisted_run(stage.layout.root)
    assert persisted.ended is SessionStopReason.KILLED

    recovery = _recovery(stage, persisted.settings)
    outcome = await recovery.finish(persisted.ended)

    assert outcome.final is not None and outcome.final.chosen is not None
    assert outcome.final.chosen.commit == await stage.repo.head(INTEGRATION_BRANCH)
    # Finishing a dead session does not make it one that ran its course.
    assert (outcome.stop_reason, outcome.exit_code) == (SessionStopReason.KILLED, 1)
    report = json.loads(stage.layout.report.read_text())
    assert report["stop_reason"] == SessionStopReason.KILLED.value
    assert report["final"]["chosen"]["commit"] == outcome.final.chosen.commit

    with pytest.raises(ValueError, match="already chose"):
        persisted_run(stage.layout.root)


async def test_a_stop_signal_during_a_final_only_still_leaves_a_report_within_one_close_grace(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.scripts = SCRIPTS
    stage.toolbox_type = StuckStopToolbox
    await stage.run()

    persisted = persisted_run(stage.layout.root)
    recovery = _recovery(stage, persisted.settings)
    recovery.toolbox_type = _slow_final(recovery, interrupt=True)

    outcome = await recovery.finish(persisted.ended)

    assert outcome.stop_reason is SessionStopReason.SIGNAL and outcome.exit_code == 1
    assert outcome.final is None and outcome.close is not None
    grace = SessionBudget.from_hours(SETTINGS.max_hours, START).close_grace_seconds
    assert recovery.signalled_at <= recovery.time.now() <= recovery.signalled_at + grace
    assert json.loads(stage.layout.report.read_text())["final"] is None
    # The signal is how this one ended, so the session it reports ends there too and is still open to a later FINAL.
    assert json.loads(stage.layout.report.read_text())["stop_reason"] == SessionStopReason.SIGNAL.value
    assert persisted_run(stage.layout.root).ended is SessionStopReason.SIGNAL


class NoRegressionSet(FakeRegressionCheck):
    """A regression set FINAL cannot read, as a source tree that has gone leaves it; a failure double, not a
    placeholder."""

    async def check(self, base_commit: str, candidate_delta: RecipeDelta, *, sources: tuple[str, ...]) -> NoReturn:
        raise LookupError(REGRESSION_SET_GONE)


async def test_a_final_only_that_fails_reports_the_session_as_it_ended_and_leaves_it_open_to_another(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.scripts = SCRIPTS
    stage.toolbox_type = StuckStopToolbox
    await stage.run()
    persisted = persisted_run(stage.layout.root)
    recovery = _recovery(stage, persisted.settings)
    recovery.regression = NoRegressionSet()

    with pytest.raises(LookupError, match=REGRESSION_SET_GONE):
        await recovery.finish(persisted.ended)

    report = json.loads(stage.layout.report.read_text())
    assert (report["stop_reason"], report["final"]) == (SessionStopReason.TIME_EXHAUSTED.value, None)
    assert REGRESSION_SET_GONE in str(report["error"])
    assert persisted_run(stage.layout.root).ended is SessionStopReason.TIME_EXHAUSTED


async def test_an_unreadable_steering_reply_keeps_every_branch_and_k_as_they_were(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([OPENING], fallback="Stop b02 and give b01 one test.")
    stage.settings = replace(SETTINGS, max_hours=1.0)

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert len(stage.judge.calls) > 2
    decisions = _decisions(stage)
    assert [row["kind"] for row in decisions if row["kind"] in ("judge", "judge_failed")] == ["judge"] + [
        "judge_failed"
    ] * (len(stage.judge.calls) - 1)
    assert {row["reason"] for row in decisions if row["kind"] == "judge_failed"} == {"kept the status quo"}
    assert not [row for row in decisions if row["kind"] == "stop" and str(row["reason"]).startswith("judge:")]
    assert stage.sessions.k == {"b01": 2, "b02": 2}


async def test_a_judge_stop_the_runtime_refuses_does_not_end_the_session(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage)
    stage.refusing = frozenset({"b02"})

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert outcome.final is not None and outcome.final.chosen is not None
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b02"]
    assert stop["detail"]["failures"] == ["RuntimeError: the runtime refused to stop b02"]
    assert stage.pids and not any(running(pid) for pid in stage.pids)


def _refill_steer(open_count: int) -> str:
    """A steering reply on a full roster: stop ``b02`` and open ``open_count`` branches, the first on attention."""

    stop = [{"branch": "b02", "reason": STOP_REASON}]
    return _reply({"open": open_count, "focus": ["attention"], "branches": [], "stop": stop})


def _origins(stage: Stage) -> dict[str, str]:
    roster = json.loads(stage.layout.roster.read_text())
    return {entry["branch"]: entry["origin"] for entry in roster["entries"] if entry["branch"] != INTEGRATION_BRANCH}


def _judge_times(stage: Stage) -> list[float]:
    return [json.loads(line)["at"] for line in stage.layout.judge.read_text().splitlines()]


async def _refilled_on_a_full_roster(stage: Stage, open_count: int) -> str:
    """Run the session in which the judge, on a roster ``b01``, ``b02`` and the fork ``b03`` fill, stops ``b02`` and
    asks for ``open_count`` branches; the one branch opened, which must have started at the integration head."""

    _script(stage, forks=1)
    stage.judge = ScriptedCompletion([OPENING, _refill_steer(open_count)])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    [refill] = [branch for branch, origin in _origins(stage).items() if origin == BranchOrigin.REFILL.value]
    start = stage.sessions.started[refill]
    assert recipe_tags(await stage.repo.read_recipe(start)) == {"c1"}
    [opened] = [row for row in _decisions(stage) if row["kind"] == "open" and row["branch"] == refill]
    assert stage.briefs.for_branch(refill).focus == "attention" and opened["detail"]["focus"] == "attention"
    return refill


async def test_a_judge_stop_on_a_full_roster_frees_a_slot_refilled_at_once_from_the_integration_head(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))

    refill = await _refilled_on_a_full_roster(stage, 1)

    assert refill == "b04"
    assert not [row for row in _decisions(stage) if row["kind"] == "open" and not row["branch"]]


async def test_the_judge_sees_each_live_specialist_as_the_runtime_reports_it(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage, forks=1)
    stage.steps = {**stage.steps, "b02": _loses_its_heartbeat}
    stage.judge = ScriptedCompletion([OPENING, _refill_steer(1)])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    [_, steering, *later] = [
        _judged_rows(json.loads(line)["prompt"]) for line in stage.layout.judge.read_text().splitlines()
    ]
    assert "specialist" not in steering["b02"]
    assert steering["b01"]["specialist"] == {
        "running": True,
        "crashes": 0,
        "lost": 0,
        "gave_up": False,
        "judge_failed": False,
        "unready": "",
    }
    assert "specialist" not in steering[INTEGRATION_BRANCH]
    assert later and all("specialist" not in rows.get("b02", {}) for rows in later)


async def test_a_branch_whose_specialist_gave_up_is_stopped_before_the_next_judge_call(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage, forks=1)
    stage.steps = {**stage.steps, "b02": _loses_its_heartbeat}
    stage.judge = ScriptedCompletion([OPENING, _refill_steer(1)])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    rows = _decisions(stage)
    [stop] = [index for index, row in enumerate(rows) if row["kind"] == "stop" and row["branch"] == "b02"]
    assert str(rows[stop]["reason"]).startswith(StopCause.GAVE_UP.value)
    judged = [index for index, row in enumerate(rows) if row["kind"] == "judge"]
    assert judged[0] < stop < judged[1]
    roster = json.loads(stage.layout.roster.read_text())
    [b02] = [entry for entry in roster["entries"] if entry["branch"] == "b02"]
    assert b02["stop"]["cause"] == StopCause.GAVE_UP.value
    lost = MAX_LOSS_RESTARTS + 1
    assert b02["stop"]["reason"].endswith(f"{lost} crashes, {lost} of them with a lost heartbeat")


async def _loses_its_heartbeat(branch: str) -> None:
    raise SessionCrashed(f"{branch} lost its heartbeat", lost=True)


def _judged_rows(prompt: str) -> dict[str, dict[str, Any]]:
    """The branch rows of the session state a judge ``prompt`` shows, by branch."""

    state = json.loads(prompt[prompt.index("{") : prompt.index("\n\nReply with")])
    return {row["branch"]: row for row in state["branches"]}


async def test_two_agents_each_start_at_the_root_on_the_share_of_the_session_s_directions_split_after_prelude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The split's own subprocesses are tested with a fake Arbor in scheduling/tests/test_agent_plans.py; here a
    # stand-in, since real subprocesses would let the virtual clock run past EXPLORE.
    asked: list[tuple[tuple[str, ...], str, Path]] = []
    given: list[OpShares | None] = []

    async def split(
        agents: tuple[str, ...], map_file: str, work_dir: Path, root_shares: OpShares | None = None, *_source: str
    ) -> tuple[str, ...]:
        asked.append((agents, map_file, work_dir))
        given.append(root_shares)
        return ("/shares/share-1.json", "/shares/share-2.json")

    monkeypatch.setattr(lifecycle, "agent_plans", split)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, agents=("/agents/arbor", "/agents/arbor"), agent_directions="/maps/d.json")
    stage.judge = ScriptedCompletion([])

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.TIME_EXHAUSTED
    assert asked == [(("/agents/arbor", "/agents/arbor"), "/maps/d.json", stage.layout.agent_plans)]
    assert {branch for branch, origin in _origins(stage).items() if origin == "agent"} == {"b01", "b02"}
    root = await stage.root()
    assert given == [await stage.handle(Role.OBSERVER).op_shares(root)]
    assert stage.sessions.started["b01"] == stage.sessions.started["b02"] == root
    assert stage.sessions.plans == {"b01": "/shares/share-1.json", "b02": "/shares/share-2.json"}
    assert _agent_map(stage) == []  # a given map: no shares waited for and no dossier pass


#: Two agents' keeps, each a gain alone and more together: ``a1`` and ``a2`` both land on the integration branch.
AGENT_CURVES = {
    frozenset(): {MAIN_POINT: 1000.0, CONC: 1000.0},
    frozenset({"a1"}): {MAIN_POINT: 1040.0, CONC: 1040.0},
    frozenset({"a2"}): {MAIN_POINT: 1030.0, CONC: 1030.0},
    frozenset({"a1", "a2"}): {MAIN_POINT: 1075.0, CONC: 1075.0},
}


async def test_two_agents_started_on_their_shares_both_have_their_keeps_merged_into_the_integration_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator's first work item (plan, two Arbors, merge), with each agent's keep as its branch's scripted change.
    async def split(
        agents: tuple[str, ...], map_file: str, work_dir: Path, root_shares: OpShares | None = None, *_source: str
    ) -> tuple[str, ...]:
        return ("/shares/share-1.json", "/shares/share-2.json")

    monkeypatch.setattr(lifecycle, "agent_plans", split)
    monkeypatch.setattr(lifecycle, "arbor_agents", _two_arbors)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, agents=("/agents/arbor", "/agents/arbor"))
    stage.scripts = {"b01": (tagged("a1"),), "b02": (tagged("a2"),)}
    stage.explored = stage.measured = AGENT_CURVES
    stage.judge = ScriptedCompletion([])

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.TIME_EXHAUSTED
    assert {branch for branch, origin in _origins(stage).items() if origin == "agent"} == {"b01", "b02"}
    assert stage.sessions.plans == {"b01": "/shares/share-1.json", "b02": "/shares/share-2.json"}
    a1, a2 = await stage.repo.head("b01"), await stage.repo.head("b02")
    observer = stage.handle(Role.OBSERVER)
    merged = {item.commit_id: item.last_merge.outcome.value for item in await observer.committed() if item.last_merge}
    assert merged == {a1: "merged", a2: "merged"}
    head = await stage.repo.head(INTEGRATION_BRANCH)
    assert recipe_tags(await stage.repo.read_recipe(head)) == {"a1", "a2"}


async def _two_arbors(agents: tuple[str, ...], work_dir: Path) -> dict[int, Path]:
    """Every agent an Arbor checkout, with none fetched: real subprocesses would let the virtual clock run on."""

    return dict.fromkeys(range(len(agents)), work_dir / "sources" / "0")


def _agent_map(stage: Stage) -> list[dict[str, Any]]:
    """The decisions of the session's agent map, in order."""

    return [row for row in _decisions(stage) if str(row["reason"]).startswith("agent map:")]


async def test_without_a_map_two_arbors_split_the_map_of_the_session_s_dossier_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[tuple[str, OpShares | None]] = []
    started: list[DossierStart] = []

    async def split(
        agents: tuple[str, ...], map_file: str, work_dir: Path, root_shares: OpShares | None = None, *_source: str
    ) -> tuple[str, ...]:
        asked.append((map_file, root_shares))
        return ("/shares/share-1.json", "/shares/share-2.json")

    async def dossier(start: DossierStart) -> DossierOutcome:
        started.append(start)
        return DossierOutcome(Path("/pass/directions.json"), Path("/pass/docs/dossier/qwen3-8b"), MAPPED)

    monkeypatch.setattr(lifecycle, "agent_plans", split)
    monkeypatch.setattr(lifecycle, "arbor_agents", _two_arbors)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, agents=("/agents/arbor", "/agents/arbor"))
    stage.dossier_pass = dossier
    stage.judge = ScriptedCompletion([])

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.TIME_EXHAUSTED
    [start] = started
    assert (start.source, start.start_commit) == ("/agents/arbor", await stage.root())
    assert asked == [("/pass/directions.json", None)]
    read, began, ended = _agent_map(stage)
    assert [read["reason"], began["reason"], ended["reason"]] == [ROOT_SHARES_READ, DOSSIER_STARTED, DOSSIER_ENDED]
    assert read["detail"]["job"] == "no root profile was started"
    assert start.deadline == pytest.approx(began["at"] + DOSSIER_MINUTES * 60)
    assert ended["detail"] == {
        "reason": MAPPED,
        "map": "/pass/directions.json",
        "dossier": "/pass/docs/dossier/qwen3-8b",
    }
    assert stage.sessions.plans == {"b01": "/shares/share-1.json", "b02": "/shares/share-2.json"}


@pytest.mark.parametrize("minutes", [DOSSIER_MINUTES, 0.0])
async def test_a_dossier_pass_with_no_map_or_no_minutes_leaves_the_split_to_the_generic_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minutes: float
) -> None:
    asked: list[str] = []
    started: list[DossierStart] = []

    async def split(
        agents: tuple[str, ...], map_file: str, work_dir: Path, root_shares: OpShares | None = None, *_source: str
    ) -> tuple[str, ...]:
        asked.append(map_file)
        return ("", "")

    async def dossier(start: DossierStart) -> DossierOutcome:
        started.append(start)
        return DossierOutcome(None, None, "it refused its config: the pack has no dossier skill")

    monkeypatch.setattr(lifecycle, "agent_plans", split)
    monkeypatch.setattr(lifecycle, "arbor_agents", _two_arbors)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, agents=("/agents/arbor", "/agents/arbor"), agent_dossier_minutes=minutes)
    stage.dossier_pass = dossier
    stage.judge = ScriptedCompletion([])

    outcome = await stage.run()

    assert outcome.stop_reason is SessionStopReason.TIME_EXHAUSTED
    assert asked == [""]
    steps = [row["reason"] for row in _agent_map(stage)]
    if minutes:
        assert (steps, len(started)) == ([ROOT_SHARES_READ, DOSSIER_STARTED, DOSSIER_ENDED], 1)
        assert _agent_map(stage)[-1]["detail"]["map"] == ""
    else:
        assert (steps, started) == ([ROOT_SHARES_READ, DOSSIER_SKIPPED], [])


async def _one_arbor(agents: tuple[str, ...], work_dir: Path) -> dict[int, Path]:
    return {0: work_dir / "sources" / "0"}


def _pass_wrote(tmp_path: Path) -> tuple[Path, Path]:
    """A map and a dossier as a dossier pass leaves them in its checkout."""

    map_file, dossier = tmp_path / "pass" / "directions.json", tmp_path / "pass" / "docs" / "dossier" / "qwen3-8b"
    dossier.mkdir(parents=True)
    (dossier / "index.md").write_text("# dossier\n")
    map_file.write_text(json.dumps({"directions": [{"id": "m-attn"}]}))
    return map_file, dossier


async def test_a_run_that_stops_after_its_map_maps_one_arbor_keeps_map_and_dossier_and_opens_no_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    map_file, dossier = _pass_wrote(tmp_path)
    started: list[DossierStart] = []

    async def mapping(start: DossierStart) -> DossierOutcome:
        started.append(start)
        return DossierOutcome(map_file, dossier, MAPPED)

    monkeypatch.setattr(lifecycle, "arbor_agents", _one_arbor)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, agents=("/agents/arbor",), stop_after_map=True)
    stage.dossier_pass = mapping

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.MAPPED, "", 0)
    [start] = started
    assert start.start_commit == await stage.root()
    kept = stage.layout.agent_map
    summary = json.loads((kept / KEPT_SUMMARY).read_text())
    assert summary == {
        "conc": SETTINGS.workload.conc,
        "source": "/agents/arbor",
        "start_commit": start.start_commit,
        "pass": MAPPED,
        "made_by": MADE_BY_PASS,
        "map": str(kept / KEPT_MAP),
        "dossier": str(kept / KEPT_DOSSIER / "qwen3-8b"),
    }
    assert (kept / KEPT_MAP).read_text() == map_file.read_text()
    assert (kept / KEPT_DOSSIER / "qwen3-8b" / "index.md").is_file()
    steps = [row["reason"] for row in _agent_map(stage)]
    assert steps == [ROOT_SHARES_READ, DOSSIER_STARTED, DOSSIER_ENDED, MAP_KEPT]
    assert not stage.sessions.plans
    assert not [branch for branch, origin in _origins(stage).items() if origin == "agent"]


async def test_a_run_that_stops_after_its_map_keeps_a_map_generated_from_the_root_when_the_pass_set_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "generated-directions.json"
    generated.write_text(json.dumps({"directions": [{"id": "m-gemm"}]}))

    async def mapping(_start: DossierStart) -> DossierOutcome:
        return DossierOutcome(None, None, "it set no map by its deadline and was stopped")

    async def generate(*_args: object) -> Path:
        return generated

    monkeypatch.setattr(lifecycle, "arbor_agents", _one_arbor)
    monkeypatch.setattr(lifecycle, "generated_map", generate)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, agents=("/agents/arbor",), stop_after_map=True)
    stage.dossier_pass = mapping

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.exit_code) == (SessionStopReason.MAPPED, 0)
    summary = json.loads((stage.layout.agent_map / KEPT_SUMMARY).read_text())
    assert (summary["made_by"], summary["pass"], summary["dossier"]) == (
        MADE_BY_SHARES,
        "it set no map by its deadline and was stopped",
        "",
    )
    assert (stage.layout.agent_map / KEPT_MAP).read_text() == generated.read_text()


async def test_a_run_that_stops_after_its_map_and_gets_none_fails_with_why_and_keeps_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def mapping(_start: DossierStart) -> DossierOutcome:
        return DossierOutcome(None, None, "it refused its config: the pack has no dossier skill")

    async def generate(*_args: object) -> Path:
        raise SplitFailed("the root has no shares to generate a map from")

    monkeypatch.setattr(lifecycle, "arbor_agents", _one_arbor)
    monkeypatch.setattr(lifecycle, "generated_map", generate)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, agents=("/agents/arbor",), stop_after_map=True)
    stage.dossier_pass = mapping

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.exit_code) == (SessionStopReason.MAPPED, 1)
    assert outcome.error.startswith(NO_MAP)
    assert "the pack has no dossier skill" in outcome.error
    assert not stage.layout.agent_map.exists()


USER_DIRECTION = "Fuse the MoE gate into the router kernel."


async def test_a_user_branch_opens_first_at_the_root_and_the_judge_neither_stops_a_branch_in_its_grace_nor_reopens_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(judge, "SETTLED_TESTS", SETTLED_TESTS)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, user_directions=(USER_DIRECTION,))
    stops = [{"branch": "b01", "reason": STOP_REASON}, {"branch": "b02", "reason": STOP_REASON}]
    stage.judge = ScriptedCompletion([OPENING, _reply({"open": 1, "branches": [], "stop": stops})])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert _origins(stage) == {"b01": "user", "b02": "first_batch", "b03": "first_batch"}
    root = await stage.root()
    assert stage.sessions.started["b01"] == stage.sessions.started["b02"] == root
    directions = {branch: set(runs) for branch, runs in stage.sessions.directions.items()}
    assert directions.pop("b01") == {USER_DIRECTION}
    assert sorted(directions) == ["b02", "b03"]
    assert all(len(runs) == 1 and runs <= set(DIRECTIONS) for runs in directions.values())
    opening, steering, *_ = [_judge_state(prompt) for prompt in _judge_prompts(stage)]
    assert opening["free_branch_slots"] == 2
    assert [(row["origin"], row["held"]) for row in opening["branches"] if row["branch"] == "b01"] == [
        ("user", HoldReason.GRACE.value)
    ]
    assert {row["branch"]: row.get("held") for row in steering["branches"]} == {
        INTEGRATION_BRANCH: None,
        "b01": HoldReason.GRACE.value,
        "b02": HoldReason.GRACE.value,
        "b03": HoldReason.GRACE.value,
    }
    decisions = _decisions(stage)
    [opened] = [row for row in decisions if row["kind"] == "open" and row["branch"] == "b01"]
    assert (opened["reason"], opened["detail"]["direction"]) == ("user branch started", USER_DIRECTION)
    steered = [row for row in decisions if row["kind"] == "judge"][1]
    assert steered["detail"]["held"] == dict.fromkeys(("b01", "b02"), HoldReason.GRACE.value)
    stopped = {row["branch"]: row["reason"] for row in decisions if row["kind"] == "stop"}
    assert stopped["b01"].startswith("session_end") and stopped["b02"].startswith("session_end")


def _candidate_waits_on_b02(monkeypatch: pytest.MonkeyPatch, waits: Callable[[], bool]) -> None:
    """Every blackboard snapshot shows a candidate of ``b02`` waiting for its confirm while ``waits()`` holds."""

    candidate = PendingCandidate("b02-r1", "b02-r1-confirm", "c-b02", MAIN_POINT, 1060.0, 6.0)
    taken = LocalBlackboard.snapshot

    async def snapshot(self: LocalBlackboard) -> BlackboardSnapshot:
        state = await taken(self)
        if not waits():
            return state
        rows = tuple(replace(row, pending=(candidate,)) if row.branch == "b02" else row for row in state.progress)
        return replace(state, progress=rows)

    monkeypatch.setattr(LocalBlackboard, "snapshot", snapshot)


async def test_the_judge_cannot_stop_a_branch_whose_candidate_waits_for_its_confirm_but_the_session_end_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _candidate_waits_on_b02(monkeypatch, lambda: True)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    _opening, steering, *_ = [_judge_state(prompt) for prompt in _judge_prompts(stage)]
    shown = {row["branch"]: row["pending_candidates"] for row in steering["branches"]}
    assert (shown["b01"], shown["b02"]) == ([], [{"request_id": "b02-r1", "gain_over_head_pct": 6.0}])
    [_, steered] = [row for row in _decisions(stage) if row["kind"] == "judge"]
    assert (steered["detail"]["stop"], steered["detail"]["held"]) == ({}, {"b02": HoldReason.PENDING_CONFIRM.value})
    assert steered["detail"]["k"] == {"b01": SETTINGS.caps.max_k_per_branch}
    [stopped] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b02"]
    assert str(stopped["reason"]).startswith(StopCause.SESSION_END.value)


async def test_a_candidate_that_starts_waiting_while_the_judge_answers_holds_its_branch_against_that_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _candidate_waits_on_b02(monkeypatch, lambda: len(stage.judge.calls) >= 2)

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    _opening, steering, *_ = [_judge_state(prompt) for prompt in _judge_prompts(stage)]
    assert {row["branch"]: row["pending_candidates"] for row in steering["branches"]}["b02"] == []
    [_, steered] = [row for row in _decisions(stage) if row["kind"] == "judge"]
    assert (steered["detail"]["stop"], steered["detail"]["held"]) == ({}, {"b02": HoldReason.PENDING_CONFIRM.value})
    [stopped] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b02"]
    assert str(stopped["reason"]).startswith(StopCause.SESSION_END.value)


async def test_a_branch_whose_specialist_gave_up_is_not_stopped_while_its_candidate_waits_for_its_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _candidate_waits_on_b02(monkeypatch, lambda: True)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.steps = {**stage.steps, "b02": _loses_its_heartbeat}

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    [_, steering] = [row for row in _decisions(stage) if row["kind"] == "judge"]
    assert steering["detail"]["held"] == {"b02": HoldReason.PENDING_CONFIRM.value}
    [stopped] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b02"]
    assert str(stopped["reason"]).startswith(StopCause.SESSION_END.value)


async def test_a_failed_opening_judge_call_opens_no_branch_beside_the_user_branches(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, user_directions=(USER_DIRECTION, "Tune the attention block size."))
    stage.judge = ScriptedCompletion(error=CompletionFailed("judge call failed: RuntimeError: status=529"))

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert _origins(stage) == {"b01": "user", "b02": "user"}
    assert stage.sessions.started == {"b01": await stage.root(), "b02": await stage.root()}


class NoFirstBranchRuntime(RefusingRuntime):
    """A runtime that cannot start ``b01``; a failure double, not a placeholder."""

    async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
        if branch == "b01":
            raise RuntimeError(f"no specialist CLI for {branch}")
        await super().start(branch, start_commit, task)


async def test_a_failed_opening_judge_call_opens_one_branch_when_no_user_branch_started(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.settings = replace(SETTINGS, user_directions=(USER_DIRECTION,))
    stage.judge = ScriptedCompletion(error=CompletionFailed("judge call failed: RuntimeError: status=529"))
    stage.runtime_type = NoFirstBranchRuntime

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert _origins(stage) == {"b01": "user", "b02": "first_batch"}
    assert stage.sessions.started == {"b02": await stage.root()}
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b01"]
    assert str(stop["reason"]).startswith(StopCause.START_FAILED.value)


async def test_a_judge_asking_for_more_than_its_stops_free_opens_what_fits_and_records_the_cut(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))

    await _refilled_on_a_full_roster(stage, 2)

    sessionwide = [row for row in _decisions(stage) if row["kind"] in ("judge", "open") and not row["branch"]]
    assert [row["kind"] for row in sessionwide][:3] == ["judge", "judge", "open"]
    assert sessionwide[1]["detail"]["open"] == 2
    assert (sessionwide[2]["reason"], sessionwide[2]["detail"]) == (
        OPEN_CUT_REASON,
        {"requested": 2, "opened": 1, "open_cut": 1},
    )


async def test_a_judge_opens_a_later_batch_without_stopping_a_branch_on_the_largest_op_of_the_head(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    opening = _reply({"open": 1, "focus": ["attention"], "branches": [], "stop": []})
    stage.judge = ScriptedCompletion([opening, _reply({"open": 1, "branches": [], "stop": []})])

    async def profiled(branch: str) -> None:
        shares = {"moe_gemm": 41.5, "attention": 22.0}
        pointer = ToolResultPointer(branch, "job-1", "/a", "trace analysed", await stage.root(), "", shares)
        await stage.handle(Role.TOOLBOX).notify_tool_result(pointer)

    stage.steps = {"b01": profiled}

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert _origins(stage) == {"b01": BranchOrigin.FIRST_BATCH.value, "b02": BranchOrigin.REFILL.value}
    root = await stage.root()
    assert stage.sessions.started == {"b01": root, "b02": root}
    assert stage.briefs.for_branch("b01").focus == "" and stage.briefs.for_branch("b02").focus == "moe_gemm"


def _admit_on_start(stage: Stage, admit: Callable[[str, str], Awaitable[ForkAdmission]]) -> type[RefusingRuntime]:
    """A runtime that, once ``b01`` has started, has ``admit`` ask for a fork or repair of it; a test double for a
    specialist's branch request, not a placeholder."""

    class AskingRuntime(RefusingRuntime):
        async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
            await super().start(branch, start_commit, task)
            if branch == "b01":
                stage.forks.append(await admit(branch, start_commit))

    return AskingRuntime


def _repair(branch: str, parent_commit: str, mechanism: str = "fp8 kv cache") -> RepairRequest:
    """A repair of the blocked change ``mechanism`` names, one change per mechanism."""

    intent = Intent(branch, parent_commit, "attention", mechanism, IntentLayer.CONFIG)
    blocked = Judgement(
        Verdict.BLOCKED, 6.0, 2.0, ("r1", "r2"), {GRADED_OUTPUT: 1060.0}, accuracy=AccuracyVerdict.FAILED
    )
    change = tagged(mechanism.replace(" ", "-"))
    return RepairRequest(intent, change, blocked, "per-tensor KV scales; scale per head")


def _judged_within_a_tick_of(stage: Stage, freed_at: float) -> None:
    """The judge was asked within one tick of ``freed_at``, well before its interval would have asked it."""

    times = _judge_times(stage)
    assert any(freed_at <= at <= freed_at + EXPLORE_TICK_SECONDS for at in times), (freed_at, times)
    assert freed_at + EXPLORE_TICK_SECONDS < times[0] + SETTINGS.judge_interval_seconds


async def test_a_repair_stopped_at_its_time_limit_has_the_judge_asked_at_the_next_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(branches, "REPAIR_BRANCH_SECONDS", 60.0)
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.runtime_type = _admit_on_start(stage, lambda b, c: stage.setup.desk.admit_repair(_repair(b, c)))

    await stage.run()

    [(status, repair)] = [(admission.status, admission.branch) for admission in stage.forks]
    assert status is AdmissionStatus.ACCEPTED
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == repair]
    assert str(stop["reason"]).startswith(StopCause.REPAIR_TIME_LIMIT.value)
    _judged_within_a_tick_of(stage, float(str(stop["at"])))


async def test_a_fork_the_gate_does_not_land_is_voided_and_has_the_judge_asked_at_the_next_tick(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.runtime_type = _admit_on_start(stage, lambda b, c: stage.setup.desk.admit(ForkRequest(b, c, FIX_REASON)))

    await stage.run()

    [(status, fork)] = [(admission.status, admission.branch) for admission in stage.forks]
    assert status is AdmissionStatus.ACCEPTED and fork not in stage.sessions.started
    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == fork]
    assert stop["reason"] == f"{StopCause.START_FAILED.value}: {FORK_NOT_LANDED}"
    _judged_within_a_tick_of(stage, float(str(stop["at"])))


async def test_a_fork_voided_in_the_queue_is_recorded_only_and_never_lands(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage)
    stop_b01 = _reply({"open": 0, "branches": [], "stop": [{"branch": "b01", "reason": STOP_REASON}]})
    stage.judge = ScriptedCompletion([OPENING, stop_b01])

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert [(admission.status, admission.branch) for admission in stage.forks] == [
        (AdmissionStatus.ACCEPTED, "b03"),
        (AdmissionStatus.PENDING, "b04"),
    ]
    observer = stage.handle(Role.OBSERVER)
    assert (await observer.record("fix-1")).conclusion is Conclusion.SPLIT_COMMITTED
    assert (await observer.record("fix-2")).conclusion is Conclusion.RECORDED_ONLY
    assert "b04" not in _origins(stage) and "b04" not in stage.sessions.started
    assert not await stage.gate.land_split("b04")
    [voided] = [row for row in _decisions(stage) if row["kind"] == "fork" and row["detail"].get("status") == "voided"]
    assert (voided["branch"], voided["reason"]) == ("b04", f"voided: parent b01 stopped: {STOP_REASON}")


def _queueing(stage: Stage, failing: str = "") -> type[RefusingRuntime]:
    """A runtime that, once ``b02`` has started, has two repairs of it asked for: the first takes the last slot as
    ``b03``, the second is queued as ``b04``; it cannot start ``failing``. A test double for a specialist's repair
    requests, not a placeholder."""

    class QueueingRuntime(RefusingRuntime):
        async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
            if branch == failing:
                raise RuntimeError(f"no specialist CLI for {branch}")
            await super().start(branch, start_commit, task)
            if branch == "b02":
                for mechanism in ("fp8 kv cache", "bf16 kv cache"):
                    stage.forks.append(await stage.setup.desk.admit_repair(_repair(branch, start_commit, mechanism)))

    return QueueingRuntime


async def _queued_repair_session(stage: Stage, stops: str, failing: str = "") -> None:
    """Run the session in which ``b02`` queues the repair ``b04`` on a full roster and the judge then stops ``stops``
    and asks for one branch."""

    steer = _reply({"open": 1, "branches": [], "stop": [{"branch": stops, "reason": STOP_REASON}]})
    stage.judge = ScriptedCompletion([OPENING, steer])
    stage.runtime_type = _queueing(stage, failing)

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert [(admission.status, admission.branch) for admission in stage.forks] == [
        (AdmissionStatus.ACCEPTED, "b03"),
        (AdmissionStatus.PENDING, "b04"),
    ]


async def test_a_queued_repair_takes_the_slot_a_judge_stop_frees_before_the_judge_opens_a_branch(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))

    await _queued_repair_session(stage, stops="b01")

    root = await stage.root()
    assert stage.sessions.started == {"b01": root, "b02": root, "b03": root, "b04": root}
    assert stage.sessions.repairs["b04"].intent.mechanism == "bf16 kv cache"
    assert _origins(stage) == {
        "b01": BranchOrigin.FIRST_BATCH.value,
        "b02": BranchOrigin.FIRST_BATCH.value,
        "b03": BranchOrigin.REPAIR.value,
        "b04": BranchOrigin.REPAIR.value,
    }
    [cut] = [row for row in _decisions(stage) if row["kind"] == "open" and not row["branch"]]
    assert (cut["reason"], cut["detail"]) == (OPEN_CUT_REASON, {"requested": 1, "opened": 0, "open_cut": 1})


async def test_a_queued_repair_is_voided_when_the_branch_that_asked_for_it_stops(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))

    await _queued_repair_session(stage, stops="b02")

    assert "b04" not in _origins(stage) and "b04" not in stage.sessions.started
    assert _origins(stage)["b05"] == BranchOrigin.REFILL.value
    [voided] = [row for row in _decisions(stage) if row["kind"] == "repair" and row["detail"].get("status") == "voided"]
    assert (voided["branch"], voided["reason"]) == ("b04", f"voided: parent b02 stopped: {STOP_REASON}")


async def test_a_dequeued_repair_that_fails_to_start_has_the_judge_asked_at_the_next_tick(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))

    await _queued_repair_session(stage, stops="b01", failing="b04")

    [stop] = [row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] == "b04"]
    assert str(stop["reason"]).startswith(StopCause.START_FAILED.value)
    freed_at = float(str(stop["at"]))
    steered_at = _judge_times(stage)[1]
    assert any(freed_at <= at <= freed_at + EXPLORE_TICK_SECONDS for at in _judge_times(stage))
    assert freed_at + EXPLORE_TICK_SECONDS < steered_at + SETTINGS.judge_interval_seconds


class NoRefillRuntime(RefusingRuntime):
    """A runtime that cannot start any branch past ``b01``; a failure double, not a placeholder."""

    async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
        if branch != "b01":
            raise RuntimeError(f"no specialist CLI for {branch}")
        await super().start(branch, start_commit, task)


async def test_a_branch_the_judge_opens_that_fails_to_start_does_not_have_the_judge_asked_early(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([_reply({"open": 1})], fallback=_reply({"open": 1}))
    stage.runtime_type = NoRefillRuntime
    stage.settings = replace(SETTINGS, max_hours=1.0)

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    failed = [
        row for row in _decisions(stage) if row["kind"] == "stop" and row["branch"] not in ("b01", INTEGRATION_BRANCH)
    ]
    assert failed and all(str(row["reason"]).startswith(StopCause.START_FAILED.value) for row in failed)
    times = _judge_times(stage)
    assert len(times) > 2
    assert all(later - earlier >= SETTINGS.judge_interval_seconds for earlier, later in zip(times, times[1:]))


async def test_a_repair_admitted_while_a_refill_starts_cuts_the_refill_and_the_session_goes_on(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([_reply({"open": 1}), _reply({"open": 2})])

    class CrowdingRefill(RefusingRuntime):
        async def start(self, branch: str, start_commit: str, task: BranchTask = BranchTask()) -> None:
            await super().start(branch, start_commit, task)
            if branch == "b02":
                stage.forks.append(await stage.setup.desk.admit_repair(_repair("b01", start_commit)))

    stage.runtime_type = CrowdingRefill

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error) == (SessionStopReason.TIME_EXHAUSTED, "")
    assert [(admission.status, admission.branch) for admission in stage.forks] == [(AdmissionStatus.ACCEPTED, "b03")]
    assert _origins(stage) == {
        "b01": BranchOrigin.FIRST_BATCH.value,
        "b02": BranchOrigin.REFILL.value,
        "b03": BranchOrigin.REPAIR.value,
    }
    [cut] = [row for row in _decisions(stage) if row["kind"] == "open" and not row["branch"]]
    assert (cut["reason"], cut["detail"]) == (OPEN_CUT_REASON, {"requested": 2, "opened": 1, "open_cut": 1})


async def test_a_whole_node_quarantine_is_shown_to_the_judge_at_once_and_explore_goes_on(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([_reply({"open": 1, "branches": [], "stop": []})])

    async def quarantine(branch: str) -> None:
        event = QuarantineEvent("node-0", QuarantineAction.QUARANTINE, "the sandbox was lost")
        await stage.handle(Role.TOOLBOX).record_quarantine(event)

    stage.steps = {"b01": quarantine}

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    assert outcome.final is not None
    [event] = [row for row in _decisions(stage) if row["kind"] == "event"]
    assert (event["reason"], event["detail"]["event"]) == ("node node-0 quarantined", "environment_changed")
    at = float(str(event["at"]))
    _judged_within_a_tick_of(stage, at)
    calls = [json.loads(line) for line in stage.layout.judge.read_text().splitlines()]
    [shown] = [call for call in calls if '"event": "environment_changed"' in call["prompt"]]
    assert at <= shown["at"] <= at + EXPLORE_TICK_SECONDS and '"node": "node-0"' in shown["prompt"]


class CardsLostToolbox(LeasingToolbox):
    """The recording toolbox that, once ``lost`` is set, reports every card of its pool lost for good; a double for a
    backend whose cards' processes were not provably gone, not a placeholder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lost = False

    async def measurement_capacity(self) -> MeasurementCapacity:
        capacity = await super().measurement_capacity()
        return replace(capacity, usable_cards=0) if self.lost else capacity


async def test_a_toolbox_without_cards_for_a_test_ends_explore_as_a_failed_session_that_skips_final(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([_reply({"open": 1, "branches": [], "stop": []})])
    stage.toolbox_type = CardsLostToolbox

    async def lose_every_card(branch: str) -> None:
        assert isinstance(stage.toolbox, CardsLostToolbox)
        stage.toolbox.lost = True

    stage.steps = {"b01": lose_every_card}

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (
        SessionStopReason.NO_MEASUREMENT_CARDS,
        NO_CARDS_LEFT,
        1,
    )
    assert outcome.final is None and outcome.close is not None and outcome.close.stop.clean
    phases = [row["reason"] for row in _decisions(stage) if row["kind"] == "phase"]
    assert "final" not in phases
    assert not [request for request in stage.toolbox.requests.values() if request.purpose is Purpose.FINAL]
    [event] = [row for row in _decisions(stage) if row["kind"] == "event"]
    assert event["detail"] == {
        "event": "no_measurement_cards",
        "usable_cards": 0,
        "total_cards": 8,
        "cards_per_test": WORKLOAD.tp,
    }
    assert event["reason"] == f"the toolbox can measure on 0 of its 8 cards, a test needs {WORKLOAD.tp}"
    assert stage.sessions.stopped == ["b01"]
    report = json.loads(stage.layout.report.read_text())
    assert report["stop_reason"] == SessionStopReason.NO_MEASUREMENT_CARDS.value


class AnalysingToolbox(LeasingToolbox):
    """The recording toolbox with an analysis route whose TraceLens job never ends, noting every test taken back; a
    double for a backend that profiles, not a placeholder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.analysed: list[str] = []
        self.taken_back: list[str] = []

    async def submit_analysis(
        self, branch: str, tool: str, params: Mapping[str, Any], profile_request_id: str = ""
    ) -> str:
        self.analysed.append(profile_request_id)
        return "tracelens-1"

    def tool_status(self, job_id: str) -> JobStatus:
        return JobStatus.RUNNING

    async def cancel(self, request_id: str) -> CancelOutcome:
        self.taken_back.append(request_id)
        return await super().cancel(request_id)


def _profiles(toolbox: RecordingToolbox) -> list[tuple[str, TestRequest]]:
    return [
        (request_id, request) for request_id, request in toolbox.requests.items() if request.kind is TestKind.PROFILE
    ]


async def test_a_large_merge_profiles_the_integration_head_without_holding_the_judge_up(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    stage.judge = ScriptedCompletion([_reply({"open": 1, "branches": [], "stop": []})])
    stage.scripts = {"b01": (tagged("c1"),)}
    stage.toolbox_type = AnalysingToolbox

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    head = await stage.repo.head(INTEGRATION_BRANCH)
    [(request_id, profile)] = _profiles(stage.toolbox)
    assert (profile.purpose, profile.branch_id, profile.parent_commit) == (Purpose.PROFILE, INTEGRATION_BRANCH, head)
    assert isinstance(stage.toolbox, AnalysingToolbox) and stage.toolbox.analysed == [request_id]
    [moved] = [row for row in _decisions(stage) if row["kind"] == "event"]
    assert moved["detail"]["event"] == "integration_moved"
    calls = [json.loads(line) for line in stage.layout.judge.read_text().splitlines()]
    running = [call["at"] for call in calls if '"profile_running": true' in call["prompt"]]
    assert len(running) >= 2 and running[1] - running[0] == SETTINGS.judge_interval_seconds
    assert request_id in stage.toolbox.taken_back
    assert not [row for row in _decisions(stage) if row["kind"] == "profile"]


async def test_a_backend_without_tool_jobs_is_asked_for_one_profile_and_noted_as_unable(tmp_path: Path) -> None:
    stage = _stage(tmp_path, child=child_with_patch(tmp_path))
    _script(stage)

    await stage.run()

    [(_, profile)] = _profiles(stage.toolbox)
    assert profile.parent_commit == await stage.repo.head(INTEGRATION_BRANCH)
    [noted] = [row for row in _decisions(stage) if row["kind"] == "profile"]
    assert (noted["reason"], noted["detail"]["code"]) == (PROFILE_UNAVAILABLE, "no_tool_executor")


async def test_an_agentx_large_merge_leaves_the_head_unprofiled_until_the_judge_opens_a_branch_on_it(
    tmp_path: Path,
) -> None:
    child = child_with_patch(tmp_path)
    with_agentx_baseline_run(child, DEFAULT_BASELINE, runtime_seconds=AGENTX_CHILD_RUN_SECONDS)
    stage = _stage(tmp_path, child=child)
    stage.settings = replace(SETTINGS, workload=AGENTX)
    opening = _reply({"open": 1, "branches": [], "stop": []})
    stage.judge = ScriptedCompletion([opening, opening])
    stage.scripts = {"b01": (tagged("c1"),)}
    stage.measured = {**CURVES, frozenset({"c1"}): {MAIN_POINT: 1500.0, CONC: 1500.0}}
    stage.toolbox_type = AnalysingToolbox

    outcome = await stage.run()

    assert (outcome.stop_reason, outcome.error, outcome.exit_code) == (SessionStopReason.TIME_EXHAUSTED, "", 0)
    head = await stage.repo.head(INTEGRATION_BRANCH)
    assert stage.sessions.started["b02"] == head
    [(_, profile)] = _profiles(stage.toolbox)
    assert (profile.branch_id, profile.parent_commit) == (INTEGRATION_BRANCH, head)
    [moved] = [row for row in _decisions(stage) if row["kind"] == "event"]
    calls = [json.loads(line) for line in stage.layout.judge.read_text().splitlines()]
    running = [(call["at"], '"profile_running": true' in call["prompt"]) for call in calls]
    assert running[:3] == [(START, False), (moved["at"], False), (moved["at"] + SETTINGS.judge_interval_seconds, True)]
