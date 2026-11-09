# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session on one machine: the git experiment repository, the toolbox, the rule gate's session
service, the specialist runtime on them and the MCP service that serves the specialists their tools.

:func:`local_peers` builds them inside the session directory: the repository at ``MB-repo``, the toolbox's
sandboxes, archive, submit log and measurement points under ``MB-toolbox``, and the patch and kernel files its
environment compiler reads, like the repository, from the session's blob store at ``MB-blobs``. The MCP service listens on loopback for the whole session and hands every toolbox call to the loop the peers were built
on; its learnings tool recalls this session's learnings and past sessions' lessons from the RSI lesson pool, and none
from the Experience KB, as the multibranch optimizer uses none (:func:`specialist_recall`). Every test is measured on a
:class:`MeasurementHost`: the :class:`ToolboxBackend` its tests run on -- this machine's cards
(:class:`LocalCards`), or the host's sandbox service for a hosted session
(:class:`~hyperloom_multibranch.scheduling.hosted_peers.HostedSandboxes`) --, the image the launcher names in
``HYPERLOOM_ACTIVE_IMAGE_REF`` and ``HYPERLOOM_ACTIVE_IMAGE_DIGEST``, and the InferenceX client inference_suite
staged at its pin (:class:`InferenceXCheckout`), which every round of either workload runs; the canonical
measurement is rendered with the workload's model and GPU type and that client, from the shipped baseline of an
InferenceX workload and from the AgentX benchmark the session writes for an AgentX one. The gate's session service and the
specialist runtime judge on one rule, :func:`workload_rule`; the runtime settles through a rule gate that asks the
scheduler's fork desk for branches, and the MCP service's ``request_repair`` and ``decline_repair`` conclude a blocked
candidate through the same gate, which the scheduler has land each fork it admits and record each fork and repair it
voids. A specialist declares a correctness fix with the MCP service's ``fix_case``; the runtime proves it with the repro
runner the regression set uses and has the setup's critic review it. The runtime's specialist sessions come from a
:data:`SpecialistSessions`: :func:`model_backed_sessions` runs each branch's specialist as an agent CLI in a worktree of
its own, briefed by the session's :class:`SessionBriefs`, whose warm-start leads the session binds once PRELUDE has read
them and which reads the repository for a replay's branch head. :func:`load_peers` reads the measurement host, checks
the workload's model is a directory here, that the image holds the framework package those worktrees branch off and
resolves that CLI first, and refuses when any cannot be had, so the command line fails before it creates anything.

On either backend a test of an AgentX workload may run for its trace window and the boot and warmup around it
(:func:`attempt_timeout_sec`).

The toolbox, its measurement points and the agent sessions load the main flow's runtime, so they are imported inside
the functions that use them: importing this module, the command line or the session loads none of it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import cache, partial
from typing import TYPE_CHECKING, ClassVar, Protocol

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from inference_suite.contracts import AGENTX_DEFAULT_DURATION_SEC
from hyperloom_common.llm.llm_config import preferred_agent_backend, provider_model_defaults, resolve_forge_llm_model
from hyperloom.common.model_paths import resolve_serving_model_path
from hyperloom_multibranch.contracts.llm_prices import price_of
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.interfaces import BranchLogStore, CommitQuery
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.sources import NativeTree, SourceRoot
from hyperloom_multibranch.contracts.toolbox import SpecialistTokens, WorkloadKind
from hyperloom_multibranch.blackboard import BriefTriggers, LocalBlackboard
from hyperloom_multibranch.blackboard.learning_search import LEARNINGS_LOG, RetrievalLog
from hyperloom_multibranch.blackboard.rsi.briefing import SessionBriefs
from hyperloom_multibranch.blackboard.rsi.cases import CasePool, session_cases
from hyperloom_multibranch.blackboard.rsi.lesson_recall import LessonRecall
from hyperloom_multibranch.blackboard.rsi.lessons import LessonPool
from hyperloom_multibranch.blackboard.rsi.session import session_menus
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.regression import RegressionSet, ReproRunner
from hyperloom_multibranch.rulegate.rules import AgentXRule, InferenceXRule, Rule
from hyperloom_multibranch.rulegate.service import GateService
from hyperloom_multibranch.scheduling.launch_isolation import announce_launch_isolation, launch_isolation
from hyperloom_multibranch.scheduling.peers import PeerFactory, Peers, PeerSetup, PeersUnavailable
from hyperloom_multibranch.scheduling.session_base import freeze_session_base, served_launch
from hyperloom_multibranch.scheduling.settings import AGENT_PORT, Workload
from hyperloom_multibranch.specialist.fix_check import FixChecking
from hyperloom_multibranch.specialist.fix_tools import register_fix_tools
from hyperloom_multibranch.specialist.judging import Judging
from hyperloom_multibranch.specialist.runtime import RuntimePacing, SpecialistRuntime

if TYPE_CHECKING:
    from hyperloom_multibranch.specialist.agent_onboarding import AgentSpec
    from hyperloom_multibranch.specialist.gate_client import GateClient
    from pathlib import Path

    from hyperloom.hosted.mcp_server import McpServer
    from hyperloom.measure.hosted_spec import SpecContext
    from hyperloom.measure.pins import ActiveImage
    from hyperloom.measure.points import CanonicalMeasurement
    from hyperloom.orchestrator.state.base_launch import ServerBase
    from hyperloom_multibranch.specialist.agent_session import AgentCommand
    from hyperloom_multibranch.specialist.agent_sessions import AgentSandboxes
    from hyperloom_multibranch.specialist.dev_sandbox import DevLeases, SandboxFor
    from hyperloom_multibranch.specialist.model_sessions import ModelSessions
    from hyperloom_multibranch.specialist.tools import WorktreeFor
    from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup, RealToolbox

logger = logging.getLogger(__name__)

TOOLBOX_WRITER = "toolbox"
GATE_WRITER = "rule-gate"
BRIEFS_READER = "briefs"
CANONICAL_DIR = "canonical"
#: The file :func:`hyperloom.measure.points.default_canonical` renders the canonical configuration to, in the
#: directory it is given; that module exports no name for it.
RENDERED_CANONICAL = "canonical.yaml"
POINTS_DIR = "points"
#: The AgentX benchmark of an AgentX session, under ``MB-toolbox``.
BENCHMARK_FILE = "benchmark.yaml"
#: A claim's lease on the specialist tools: the runtime renews it every 10 minutes.
CLAIM_TTL_SECONDS = 1800.0
#: Where a machine's render nodes are, one ``renderD*`` for every card a process there can open.
RENDER_NODE_DIR = "/dev/dri"
#: The switch the main flow and the measurement renderer read an AgentX workload from
#: (``hyperloom.common.perf_metric.agentx_enabled``).
AGENTX_ENV = "HYPERLOOM_AGENTX"
#: How long a canonical AgentX round replays its trace under measurement.
AGENTX_TRACE_SEC = AGENTX_DEFAULT_DURATION_SEC
#: What an AgentX test spends outside its trace window: the server's boot, which takes minutes for a large model at
#: low tensor parallelism, the replay client's cold-cache warmup before the window opens, and the drain after it. An
#: hour leaves room for both at their slowest; a hung server ends on the toolbox's boot and silence limits.
AGENTX_OUTSIDE_TRACE_SEC = 3600


@dataclass(frozen=True)
class SessionWiring:
    """What a session's specialist sessions are built on: the peer setup, the repository the branches use, the
    tokens that grant a branch's specialist the MCP service's tools (a grant names the service's URL, so it is issued
    when a specialist starts), each branch's log, the session's briefs, its dev sandboxes, ``None`` when it lends
    none, the canonical configuration the toolbox rendered, which the gate's benchmark client is read from, the
    native trees the branches edit (:func:`native_trees`), the sandboxes autonomous agents run in and the server half
    of the session's frozen base launch, which every test starts from."""

    setup: PeerSetup
    repo: ExperimentRepo
    tokens: SpecialistTokens
    blackboards: Callable[[str], BranchLogStore]
    briefs: SessionBriefs
    toolbox: RealToolbox
    dev_leases: DevLeases | None
    canonical: Path
    native: Mapping[str, NativeTree]
    agent_sandboxes: AgentSandboxes
    base: ServerBase


#: Builds the specialist runtime's session factory, and the worktree of each branch the worktree tools serve, on a
#: session's wiring.
SpecialistSessions = Callable[[SessionWiring], "ModelSessions"]


@dataclass(frozen=True)
class BranchSandboxes:
    """Where a session's dev sandboxes and autonomous agents run: ``dev``, the sandbox of each dev lease, which takes
    ``dev_cards`` unless the specialist asks for others, and ``agents``."""

    dev: SandboxFor
    dev_cards: int
    agents: AgentSandboxes


def branch_sandboxes(backend: ToolboxBackend, setup: PeerSetup, worktrees: WorktreeFor) -> BranchSandboxes:
    """This machine's sandboxes when ``backend`` measures on its cards; else the ``dev`` sandboxes of the hosted
    backend's leases, which have the session's TP cards and see the session directory at the path it has here."""

    from hyperloom_multibranch.specialist.agent_sessions import LocalAgentSandboxes, RemoteAgentSandboxes
    from hyperloom_multibranch.specialist.dev_sandbox import hosted_dev_sandboxes, local_dev_sandboxes

    if backend.on_this_machine:
        return BranchSandboxes(local_dev_sandboxes(worktrees, setup.processes), 1, LocalAgentSandboxes(setup.processes))
    dev = hosted_dev_sandboxes(worktrees, setup.layout.branch_dir)
    return BranchSandboxes(dev, setup.workload.tp, RemoteAgentSandboxes())


@dataclass(frozen=True)
class InferenceXCheckout:
    """The InferenceX client every measurement point names: the commit inference_suite pins it at and the root it is
    staged under, the same for either workload."""

    commit: str
    path: str


class ToolboxBackend(Protocol):
    """Where a session's tests run: it builds the session's toolbox on the blackboard handle, the acceptance setup and
    the instruction sheet's context that every backend shares; ``on_this_machine`` says whether they run on this
    machine's cards."""

    on_this_machine: ClassVar[bool]

    def toolbox(
        self, setup: PeerSetup, blackboard: LocalBlackboard, acceptance: AcceptanceSetup, spec: SpecContext
    ) -> RealToolbox: ...


@dataclass(frozen=True)
class LocalCards:
    """Tests run in local sandboxes on these cards of this machine, their sandboxes, archive and submit log under
    ``MB-toolbox``. The runners read the inference_suite staging from this host's runner cache, where it is published
    before the toolbox is built; one already published is verified and kept."""

    cards: tuple[int, ...]
    on_this_machine: ClassVar[bool] = True

    def toolbox(
        self, setup: PeerSetup, blackboard: LocalBlackboard, acceptance: AcceptanceSetup, spec: SpecContext
    ) -> RealToolbox:
        from hyperloom.measure.framework.base import image_configs
        from hyperloom.measure.runner import suite_publish
        from hyperloom_multibranch.toolbox.local_hosted import (
            LocalOptions,
            LocalSession,
            build_local_toolbox,
            runner_cache_dir,
        )

        try:
            suite_publish.publish_own_inference_suite(runner_cache_dir(os.environ))
        except suite_publish.SuitePublishError as exc:
            raise PeersUnavailable(f"the inference_suite staging the runners read cannot be published: {exc}") from exc
        layout = setup.layout
        session = LocalSession(layout.toolbox, layout.session_id, self.cards, setup.workload.tp)
        # The tests read private copies of the tables of the aiter this machine's servers import (a vLLM image's
        # site-packages, not an SGLang image's workspace), or a keep that adds rows to a default table is refused.
        options = LocalOptions(timeout_sec=attempt_timeout_sec(setup.workload), framework_configs=str(image_configs()))
        return build_local_toolbox(session, blackboard, acceptance, spec, options=options)


@dataclass(frozen=True)
class MeasurementHost:
    """What the toolbox measures on: the backend its tests run on, the image they run in (its digest is also the
    base image every recipe must carry, the root recipe's first) and the staged InferenceX client."""

    backend: ToolboxBackend
    image: ActiveImage
    inferencex: InferenceXCheckout


def workload_environment(workload: Workload) -> Mapping[str, str]:
    """The workload as the process environment carries it to the main flow's config renderer, which the toolbox
    renders the canonical measurement and every test's points with, and to the kernel tuning the toolbox runs, which
    takes its backend from ``KERNEL_OPT_BACKEND_ORDER``; ``hyperloom optimize`` exports the same, the model resolved to
    the directory a server loads it from. ``GPU_TYPE`` and ``TARGET_GPU_TYPE`` are both the board, as ``hyperloom optimize``
    sets them. An InferenceX workload carries its sequence lengths; an AgentX one carries
    :data:`AGENTX_ENV` instead."""

    shape = workload.inferencex
    workload_specific = {AGENTX_ENV: "1"} if shape is None else {"ISL": str(shape.isl), "OSL": str(shape.osl)}
    return {
        "FRAMEWORK": workload.framework,
        "MODEL_PATH": resolve_serving_model_path(workload.model),
        "TP": str(workload.tp),
        **workload_specific,
        "PRECISION": workload.precision,
        "GPU_TYPE": workload.gpu_type,
        "TARGET_GPU_TYPE": workload.gpu_type,
        "KERNEL_OPT_BACKEND_ORDER": workload.kernel_optimizer.value,
    }


def attempt_timeout_sec(workload: Workload) -> int:
    """How long one attempt of a test of ``workload`` may run on either backend: the toolbox's default for an
    InferenceX workload, and for an AgentX one its trace window (the workload's, else :data:`AGENTX_TRACE_SEC`) plus
    what the test spends outside it (:data:`AGENTX_OUTSIDE_TRACE_SEC`)."""

    from hyperloom_multibranch.toolbox.hosted_executor import DEFAULT_TEST_TIMEOUT_SEC

    if workload.inferencex is not None:
        return DEFAULT_TEST_TIMEOUT_SEC
    return (workload.agentx_duration_sec or AGENTX_TRACE_SEC) + AGENTX_OUTSIDE_TRACE_SEC


def _export_workload(workload: Workload, toolbox_dir: Path, image_ref: str) -> None:
    """Export :func:`workload_environment` into this process. For an AgentX workload, also write the AgentX benchmark
    ``hyperloom optimize`` generates for it to :data:`BENCHMARK_FILE` in ``toolbox_dir`` and name it in
    ``HYPERLOOM_BENCHMARK_CONFIG``: the canonical measurement is rendered from it. The benchmark serves at the
    workload's EP and names ``image_ref``, the image the session runs on, as its image. It pins the model's directory
    when the model is one, which the command line names by its resolved path."""

    os.environ.update(workload_environment(workload))
    if workload.inferencex is not None:
        return
    from hyperloom.inference_optimizer.cli.benchmark_config import AgentXWorkload, write_agentx_benchmark

    path = toolbox_dir / BENCHMARK_FILE
    benchmark = AgentXWorkload(
        framework=workload.framework,
        model=workload.model,
        precision=workload.precision,
        tp=workload.tp,
        ep=workload.ep,
        conc=workload.conc,
        model_path=workload.model,
        docker_image=image_ref,
        duration_sec=workload.agentx_duration_sec,
    )
    write_agentx_benchmark(path, benchmark)
    os.environ[BENCHMARK_CONFIG_ENV] = str(path.resolve())


def workload_rule(workload: Workload, blackboard: CommitQuery) -> Rule:
    """The rule a session of ``workload`` is judged on: InferenceX throughput at the workload's conc against the
    PRELUDE bar ``blackboard`` holds, or the AgentX objective with its guards."""

    if workload.inferencex is None:
        return AgentXRule(workload.agentx_objective)
    return InferenceXRule(blackboard, workload.conc)


def _canonical(out_dir: Path, workload: Workload, host: MeasurementHost, base: ServerBase) -> CanonicalMeasurement:
    """The canonical measurement every point of the session is rendered from. The renderer takes the model and the
    runner type only as arguments, so a point never falls back to the shipped YAML's model, and the server half of the
    session's frozen ``base`` too: this process is no Hyperloom session, so an AgentX point is seeded with no base it
    would find. Raises :class:`PeersUnavailable` when it cannot be rendered."""

    from hyperloom.measure.points import PointError, default_canonical

    try:
        return default_canonical(
            out_dir,
            primary_concurrency=workload.conc,
            model_path=resolve_serving_model_path(workload.model),
            gpu_type=workload.gpu_type,
            inferencex_path=host.inferencex.path,
            server_base=base,
        )
    except PointError as exc:
        raise PeersUnavailable(f"the canonical measurement cannot be taken: {exc}") from exc


def specialist_recall(reports: Path) -> LessonRecall:
    """The specialists' learnings recall: this session's learnings, then past sessions' lessons from the RSI lesson
    pool beside the environment's archive, each retrieval logged in ``reports``; no Experience KB, as the multibranch
    optimizer uses none."""

    return LessonRecall(pool=LessonPool.from_env(), log=RetrievalLog(reports / LEARNINGS_LOG))


def local_peers(host: MeasurementHost, sessions: SpecialistSessions, setup: PeerSetup) -> Peers:
    """The session's peers on this machine; exports the workload into this process (:func:`_export_workload`) and
    freezes the base launch every test runs on in ``MB-toolbox`` unless the session froze one
    (:func:`~hyperloom_multibranch.scheduling.session_base.freeze_session_base`), which its Arbor agents are served
    when the operator gave it as a reference script. Raises :class:`PeersUnavailable`
    when no base launch can be frozen or the session's canonical measurement cannot be taken.

    Called on the event loop the session runs on. The toolbox is built, not started: the session starts it, and its
    close releases the backend and the sandboxes. The MCP service is started; the session stop stops it. How the
    specialists, sandboxes and tests it launches are isolated is said once here, at WARNING when some are not
    (:func:`~hyperloom_multibranch.scheduling.launch_isolation.announce_launch_isolation`).
    """

    from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
    from hyperloom.orchestrator.state.base_launch import server_base
    from hyperloom_multibranch.toolbox.mcp_tools import McpSpecialistTokens
    from hyperloom_multibranch.specialist.blocked import BlockedCandidates, register_blocked_tools
    from hyperloom_multibranch.specialist.tools import register_worktree_tools

    layout, workload = setup.layout, setup.workload
    _export_workload(workload, layout.toolbox, host.image.ref)
    frozen = freeze_session_base(layout.toolbox, workload, setup.reference_script, setup.inferencex_commit)
    served = served_launch(frozen, setup.reference_script, setup.agent_launch)
    base, setup = server_base(frozen.launch), replace(setup, agent_launch=served)
    native = native_trees(os.environ)
    repo = GitExperimentRepo(layout.repo, setup.blobs, native=native)
    canonical_dir = layout.toolbox / CANONICAL_DIR
    toolbox = _toolbox(host, setup, repo, _canonical(canonical_dir, workload, host, base))
    gate_blackboard = LocalBlackboard(layout.blackboard, Writer(Role.GATE, GATE_WRITER))
    rule = workload_rule(workload, gate_blackboard)

    @cache
    def runtime_blackboard(branch: str) -> LocalBlackboard:
        """The branch's one handle for the session: a handle is a writer of its own, and every inbox cursor holds one
        sequence number per writer, so a handle per MCP call made ``poll_inbox`` answer 91,000 characters after 12 h."""
        return LocalBlackboard(
            layout.blackboard, Writer(Role.SPECIALIST_RUNTIME, branch, branch), briefs=BriefTriggers()
        )

    mcp = McpServer(McpServerConfig(host_token=secrets.token_hex(32)), campaign=layout.session_id)
    briefs = _specialist_briefs(setup, repo)
    _serve_specialist_tools(mcp, toolbox, runtime_blackboard, briefs)
    sandboxes = branch_sandboxes(host.backend, setup, lambda branch: built.worktrees(branch))
    isolation = launch_isolation(local=host.backend.on_this_machine)
    announce_launch_isolation(isolation)
    leases = _lend_dev_sandboxes(mcp, setup, toolbox, runtime_blackboard, sandboxes)
    tokens = McpSpecialistTokens(mcp)
    rendered = canonical_dir / RENDERED_CANONICAL
    wiring = SessionWiring(
        setup, repo, tokens, runtime_blackboard, briefs, toolbox, leases, rendered, native, sandboxes.agents, base
    )
    built = sessions(wiring)
    register_worktree_tools(mcp, repo=repo, worktrees=built.worktrees)
    gate = RuleGate(gate_blackboard, repo, setup.desk, toolbox)
    register_blocked_tools(mcp, blocked=BlockedCandidates(gate_blackboard, gate), home=asyncio.get_running_loop())
    register_fix_tools(mcp, attachments=toolbox)
    runner = ReproRunner(toolbox, gate_blackboard, toolbox)
    fixes = FixChecking(runner, setup.critic, setup.blobs, _specialist_role(workload))
    judging = Judging(gate, toolbox, rule, setup.blobs, fixes)
    runtime = SpecialistRuntime(
        runtime_blackboard, judging, built.factory, pacing=RuntimePacing(claim_ttl_seconds=CLAIM_TTL_SECONDS)
    )
    mcp.start()
    gate_service = GateService(gate_blackboard, toolbox, rule, toolbox)
    regression = RegressionSet(gate_blackboard, runner, repo, pool=session_cases(host.image.digest, gate_blackboard))
    return Peers(
        runtime,
        toolbox,
        repo,
        gate_service,
        mcp,
        briefs,
        host.image.digest,
        gate,
        regression,
        isolation,
        built.dossier_pass,
    )


def _toolbox(
    host: MeasurementHost, setup: PeerSetup, repo: ExperimentRepo, canonical: CanonicalMeasurement
) -> RealToolbox:
    """The session's toolbox on ``host``'s backend, every test's points rendered from ``canonical``, the session's
    canonical measurement, and compiled from the session's blob store. A plain explore runs no accuracy evaluation:
    the rule gate judges it on throughput, and the confirm that follows a candidate runs the evaluation."""

    from hyperloom.measure.compile import EnvironmentCompiler
    from hyperloom.measure.hosted_spec import SpecContext
    from hyperloom_multibranch.toolbox.agentx_checkpoint import CheckpointReferences
    from hyperloom_multibranch.toolbox.planner import HostedPlanner
    from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

    layout, workload, image = setup.layout, setup.workload, host.image
    compiler = EnvironmentCompiler(setup.blobs, image.digest)
    spec = SpecContext(
        image_ref=image.ref,
        image_digest=image.digest,
        gpu_type=workload.gpu_type,
        blobs=setup.blobs,
        blob_root=str(setup.blobs.root),
    )
    planner = HostedPlanner(compiler, canonical, layout.toolbox / POINTS_DIR)
    blackboard = LocalBlackboard(layout.blackboard, Writer(Role.TOOLBOX, TOOLBOX_WRITER))
    checkpoint = None
    if workload.kind is WorkloadKind.AGENTX:
        checkpoint = CheckpointReferences(setup.agentx_early_stop, workload.agentx_objective, blackboard)
    acceptance = AcceptanceSetup(
        repo, planner, image.digest, workload_kind=workload.kind, explore_accuracy=False, checkpoint=checkpoint
    )
    return host.backend.toolbox(setup, blackboard, acceptance, spec)


def _specialist_briefs(setup: PeerSetup, repo: GitExperimentRepo) -> SessionBriefs:
    """The briefs the session's specialists are given, read as an observer of the blackboard, with their learnings
    recall (:func:`specialist_recall`)."""

    learnings = specialist_recall(setup.layout.reports)
    return SessionBriefs(
        LocalBlackboard(setup.layout.blackboard, Writer(Role.OBSERVER, BRIEFS_READER)),
        metric=setup.workload.metric,
        learnings=learnings,
        repo=repo,
        stack_slot=repo.stack_kernel_slot,
        menus=session_menus(setup.workload.metric, setup.layout.reports),
        cases=CasePool.from_env(),
    )


def _serve_specialist_tools(
    mcp: McpServer, toolbox: RealToolbox, blackboards: Callable[[str], LocalBlackboard], briefs: SessionBriefs
) -> None:
    """Register the specialists' toolbox and blackboard tools on ``mcp``, calling the toolbox on the session's loop and
    briefing each branch from ``briefs``."""

    from hyperloom_multibranch.toolbox.mcp_tools import SpecialistToolOptions, register_specialist_tools

    options = SpecialistToolOptions(
        claim_ttl_seconds=CLAIM_TTL_SECONDS,
        toolbox_loop=asyncio.get_running_loop(),
        learnings=briefs.learnings,
        briefing_for=briefs.for_branch,
        analysis=toolbox,
    )
    register_specialist_tools(mcp, toolbox=toolbox, blackboards=blackboards, options=options)


def _lend_dev_sandboxes(
    mcp: McpServer,
    setup: PeerSetup,
    toolbox: RealToolbox,
    blackboards: Callable[[str], LocalBlackboard],
    sandboxes: BranchSandboxes,
) -> DevLeases | None:
    """The session's dev sandboxes on ``toolbox``'s cards, their tools registered on ``mcp`` for the session's loop;
    ``None``, and no tools, when ``setup`` lends none. The branch's worktree ``sandboxes`` import is read only when a
    lease is granted, after the MCP service starts, so it may be of a session built after this call."""

    if setup.dev_leases < 1:
        return None
    from hyperloom_multibranch.specialist.dev_sandbox import DevLeases, DevLimits, register_dev_sandbox_tools

    limits = DevLimits(cap=setup.dev_leases, cards=sandboxes.dev_cards)
    leases = DevLeases(toolbox, sandboxes.dev, blackboards, limits=limits)
    register_dev_sandbox_tools(mcp, leases=leases, home=asyncio.get_running_loop())
    return leases


def render_nodes() -> int:
    """How many render nodes, ``renderD*`` in :data:`RENDER_NODE_DIR`, this machine holds: one for every card a
    process here can open, so a container a device plugin gave some of a node's cards holds theirs alone."""

    from pathlib import Path

    try:
        return sum(1 for node in Path(RENDER_NODE_DIR).iterdir() if node.name.startswith("renderD"))
    except OSError:
        return 0


def machine_cards() -> tuple[int, ...]:
    """The cards of this machine the session may use: the absolute ids the first visible-device mask set names, or,
    when none is set, as many as ``rocm-smi`` shows from card 0; none when neither shows any. Raises
    :class:`PeersUnavailable` when ``rocm-smi`` shows more cards than this machine holds render nodes
    (:func:`render_nodes`): in a container given some of a node's cards it shows them all, and a test leased one it
    was not given could not open it.

    The ids are absolute because every sandbox's ``ROCR_VISIBLE_DEVICES`` is set to its leased cards in place of the
    mask this process runs under; a mask of device UUIDs names no id, so it leaves no card. The main flow resolves its
    whole-machine pool by the same rule (``hyperloom.orchestrator.bus.gpu_pool``), which the scheduler does not load.
    """

    from hyperloom.common.visible_devices import COUNTING_VISIBLE_DEVICE_VARS, detect_gpu_count, parse_device_list

    for name in COUNTING_VISIBLE_DEVICE_VARS:
        mask = os.environ.get(name)
        if mask is not None:
            return tuple(parse_device_list(mask))
    shown, nodes = detect_gpu_count(), render_nodes()
    if shown > nodes:
        raise PeersUnavailable(
            f"rocm-smi shows {shown} cards, more than the {nodes} this machine can open (its render nodes,"
            f" {RENDER_NODE_DIR}/renderD*): name the cards it was given in ROCR_VISIBLE_DEVICES"
        )
    return tuple(range(shown))


def measurement_host(env: Mapping[str, str], backend: ToolboxBackend) -> MeasurementHost:
    """``backend``, the image ``env`` names and the staged InferenceX client (:func:`staged_client`); raises
    :class:`PeersUnavailable` when the image cannot be pinned, an unnamed image included, or the client is not
    staged."""

    from hyperloom.measure.pins import ImagePinError, active_image

    try:
        image = active_image(env)
    except ImagePinError as exc:
        raise PeersUnavailable(f"the measurement image cannot be pinned: {exc}") from exc
    return MeasurementHost(backend, image, staged_client())


def staged_client() -> InferenceXCheckout:
    """The InferenceX client inference_suite staged in this application's tool cache at its pin, which every round of
    either workload runs; raises :class:`PeersUnavailable` when the cache holds no staging that verifies."""

    from hyperloom.orchestrator.actions.executors import staged_tool_cache
    from inference_suite import INFERENCEX_CLIENT_COMMIT, inferencex_root, verified_manifest_sha256

    tool_cache = staged_tool_cache()
    if verified_manifest_sha256(tool_cache) is None:
        raise PeersUnavailable(
            f"no inference_suite staging that verifies is in {tool_cache}: "
            f"run python -m inference_suite stage --tool-cache {tool_cache}"
        )
    return InferenceXCheckout(INFERENCEX_CLIENT_COMMIT, str(inferencex_root(tool_cache)))


def machine_parallel_tests(tp: int) -> int:
    """How many tests of ``tp`` cards the local toolbox runs at once on this machine's cards; raises ``ValueError``
    when a test of ``tp`` cards does not fit on them, or its cards cannot be told (:func:`machine_cards`)."""

    from hyperloom_multibranch.toolbox.local_hosted import max_parallel_tests

    try:
        cards = machine_cards()
    except PeersUnavailable as exc:
        raise ValueError(str(exc)) from exc
    return max_parallel_tests(len(cards), tp)


def specialist_model(env: Mapping[str, str]) -> str:
    """The model every specialist of a session runs on: the one ``env`` names for the backend it is credentialed
    for."""

    return resolve_forge_llm_model(preferred_agent_backend(env), env={**env, **provider_model_defaults(env)})


def specialist_command(env: Mapping[str, str]) -> AgentCommand:
    """The agent CLI every specialist of a session runs: the backend ``env`` is credentialed for, found on its
    ``PATH``, on :func:`specialist_model`; raises :class:`PeersUnavailable` when that CLI is not installed. A model the
    price sheet does not list records its tokens only, which the command line accepts only without a dollar cap, and
    says so here."""

    from hyperloom_multibranch.specialist.agent_session import AgentCommand

    backend = preferred_agent_backend(env)
    executable = shutil.which(backend, path=env.get("PATH"))
    if executable is None:
        raise PeersUnavailable(f"the {backend} CLI the specialists run is not on PATH")
    model = specialist_model(env)
    if price_of(model) is None:
        logger.warning("the specialists run on %s, which the price sheet does not list: they record tokens only", model)
    return AgentCommand(backend, executable, model)


def require_framework_package(framework: str) -> None:
    """Check that the image holds the installed package the specialists' worktrees of ``framework`` branch off;
    raises :class:`PeersUnavailable` when ``framework`` has no package layout or its package cannot be a base."""

    from hyperloom.measure.compile import COMPONENT_LAYOUTS, UnsupportedComponentError, locate_installed_package

    layout = COMPONENT_LAYOUTS.get(framework)
    if layout is None:
        raise PeersUnavailable(f"the framework {framework} has no package layout the specialists' worktrees can use")
    try:
        locate_installed_package(layout)
    except UnsupportedComponentError as exc:
        raise PeersUnavailable(f"the {framework} source the specialists change cannot be laid out: {exc}") from exc


def _specialist_role(workload: Workload) -> str:
    """The workload as the specialists' role prompt names it."""

    shape = workload.inferencex
    requests = (
        f"AgentX trace replay graded on {workload.agentx_objective.title} ({workload.metric})"
        if shape is None
        else f"ISL {shape.isl}, OSL {shape.osl}"
    )
    return (
        f"{workload.model_basename} on {workload.framework}, {workload.precision}, TP {workload.tp}, "
        f"{requests}, concurrency {workload.conc}, {workload.gpu_type}"
    )


#: The component every session's source base lays out beside the framework when the image has a develop install of it.
AITER = "aiter"


def model_backed_sessions(command: AgentCommand) -> SpecialistSessions:
    """Sessions that run ``command`` for every branch's specialist, in a worktree of the workload's framework, and
    each branch's third-party agent in a sandbox leased from the session's toolbox (``wiring.agent_sandboxes``), given
    the gate's benchmark client of the session's canonical configuration; a session that runs no agent reads no
    canonical.

    The CLI runs on the main flow's allowlisted specialist environment, through a local AI work socket whose resource
    log is under ``MB-logs``; its worktrees branch off one source base at ``MB-source``, and each launch rebuilds the
    branch's brief at the head it starts from.
    """

    def build(wiring: SessionWiring) -> ModelSessions:
        from hyperloom.orchestrator.specialists.agent_runner import Liveness, LocalAgentRunner, build_specialist_env
        from hyperloom_multibranch.specialist.agent_session import SPECIALIST_LOST_AFTER_S, AgentSetup
        from hyperloom_multibranch.specialist.agent_sessions import AgentSessions, with_agents
        from hyperloom_multibranch.specialist.model_sessions import model_sessions
        from hyperloom_multibranch.specialist.prompts import RoleWorkload, role_prompt
        from hyperloom_multibranch.specialist.worktree import SourceBase

        layout, workload = wiring.setup.layout, wiring.setup.workload
        agent = AgentSetup(
            command=command,
            role=role_prompt(
                RoleWorkload(
                    _specialist_role(workload), workload.agentx_objective if workload.inferencex is None else None
                ),
                references=wiring.setup.references,
                base=wiring.base,
                dev_leases=wiring.dev_leases.limits.cap if wiring.dev_leases else 0,
                native=wiring.native,
            ),
            env=build_specialist_env(),
            runner=LocalAgentRunner(
                session_dir=layout.logs, liveness=Liveness(lost_after_seconds=SPECIALIST_LOST_AFTER_S)
            ),
            tokens=wiring.tokens,
            processes=wiring.setup.processes,
            repo=wiring.repo,
            briefs=wiring.briefs.text,
            dev_leases=wiring.dev_leases,
            readable=specialist_readable(wiring.setup.references, workload),
            lease_area=_lease_area(wiring),
        )
        open_sessions = partial(
            model_sessions,
            agent,
            base=SourceBase(layout.source, native=wiring.native),
            branch_dir=layout.branch_dir,
            blackboard_for=wiring.blackboards,
        )
        native = tuple(component for component in wiring.native if component != workload.framework)
        sessions = _sessions_over_source(open_sessions, layout.source, workload.framework, native)
        setup = wiring.setup
        agents = AgentSessions(
            _agent_spec(workload, setup.agent_pack, setup.agent_launch, setup.agent_port, setup.agent_review),
            command,
            agent.env,
            wiring.toolbox,
            wiring.repo,
            wiring.agent_sandboxes,
            client=_gate_client(wiring) if wiring.setup.agents else None,
            dossier=wiring.setup.agent_dossier or None,
            maps=wiring.setup.agent_maps,
            least_run_seconds=wiring.setup.agent_share_minutes * 60.0,
        )
        return with_agents(sessions, agents, branch_dir=layout.branch_dir, blackboard_for=wiring.blackboards)

    return build


def _gate_client(wiring: SessionWiring) -> GateClient:
    """The gate's client of the session's rendered canonical configuration, an AgentX replay bounded as one attempt of
    a test is; raises :class:`PeersUnavailable` when the session's base or staged tools cannot run one."""

    from hyperloom.orchestrator.actions.executors import SuiteRoundRefused
    from hyperloom_multibranch.specialist.gate_client import read_gate_client

    try:
        return read_gate_client(
            wiring.canonical, wiring.setup.layout.toolbox, attempt_timeout_sec(wiring.setup.workload)
        )
    except SuiteRoundRefused as exc:
        raise PeersUnavailable(f"the gate's client cannot be given to the session's agents: {exc}") from exc


def specialist_readable(references: Mapping[str, Path], workload: Workload) -> tuple[Path, ...]:
    """What every specialist of a session reads besides its own trees: the reference directories the session copied
    in, and the directory its workload's model is served from when that is one here."""

    from pathlib import Path

    model = resolve_serving_model_path(workload.model)
    served = (Path(os.path.abspath(model)),) if model and os.path.isdir(model) else ()
    return (*references.values(), *served)


def _lease_area(wiring: SessionWiring) -> Callable[[str], Path] | None:
    """Where the roots of a branch's dev leases are made, for its specialist to read: its holder's area of this
    machine's leases, or its directory's ``dev`` on the shared storage a hosted lease's sandbox roots in; ``None``
    when the session lends none."""

    from hyperloom_common.shared_storage import make_shared_dirs
    from hyperloom_multibranch.specialist.dev_sandbox import DEV_DIR
    from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox

    if wiring.dev_leases is None:
        return None
    toolbox = wiring.toolbox
    if isinstance(toolbox, LocalHostedToolbox):
        return toolbox.leases.holder_area
    branch_dir = wiring.setup.layout.branch_dir
    return lambda branch: make_shared_dirs(branch_dir(branch) / DEV_DIR)


def _sessions_over_source(
    open_sessions: Callable[..., ModelSessions], source: Path, framework: str, native: tuple[str, ...]
) -> ModelSessions:
    """The sessions ``open_sessions`` opens over a source base laid out at ``source`` from ``framework``, aiter and the
    ``native`` components, or without aiter when the base refuses it: not installed in the image, or installed from a
    wheel, whose version names only an abbreviated commit. :func:`peers_on` checked the framework, and the toolbox's
    build preflight the native checkouts, so a refusal is aiter's. The native components come last: one may lie
    inside the framework's tree."""

    from hyperloom.measure.compile import UnsupportedComponentError

    try:
        return open_sessions(components=(framework, AITER, *native))
    except UnsupportedComponentError as refused:
        logger.warning("the specialists' source base leaves aiter out, so they cannot change it: %s", refused)
        shutil.rmtree(source)
        return open_sessions(components=(framework, *native))


def native_trees(environ: Mapping[str, str]) -> dict[str, NativeTree]:
    """The tree each native component's build compiles, in the build sources' checkout of the commit the image
    installs, by component; none on a host that builds no native code (``HYPERLOOM_BUILD_STORE`` unset).

    The toolbox refuses to start on a host whose checkouts are not in place, so they are read as they are. A recipe
    that builds from the repository root (vLLM's) has no tree to lay out beside the framework's and is left out.
    """

    from pathlib import Path

    from hyperloom.measure.jit.native.config import BUILD_STORE_ENV
    from hyperloom.measure.runner.native.recipes import native_recipes
    from hyperloom_multibranch.toolbox.build_preflight import image_upstreams
    from hyperloom_multibranch.toolbox.native_builds import BUILD_IMAGE_ROOT_ENV, BUILD_SOURCES_ENV

    if not environ.get(BUILD_STORE_ENV, "").strip():
        return {}
    sources, recipes = Path(environ[BUILD_SOURCES_ENV]), native_recipes()
    trees = {}
    for upstream in image_upstreams(Path(environ[BUILD_IMAGE_ROOT_ENV])):
        checkout, source_dir = sources / upstream.component / upstream.ref, recipes[upstream.component].source_dir
        if not source_dir:
            logger.warning(
                "%s builds from its repository root, so its native sources are not laid out", upstream.component
            )
            continue
        origin = subprocess.run(
            ["git", "-C", str(upstream.checkout or checkout), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        trees[upstream.component] = NativeTree(SourceRoot(source_dir, checkout / source_dir), origin, upstream.ref)
    return trees


def _agent_spec(
    workload: Workload, pack: str = "", launch: str = "", port: int = AGENT_PORT, review: str = ""
) -> AgentSpec:
    """What every agent of a session of ``workload`` is asked to optimize, an Arbor on ``pack`` when one is named,
    served the launch in the file ``launch`` with the root's added when one is given, its own review gate handed over
    as ``review`` says; the root's launch, the workspace, the cards and a port block from ``port`` on are each
    branch's own."""

    from hyperloom_multibranch.specialist.agent_onboarding import AgentSpec

    shape = workload.inferencex
    requests = {"kind": workload.kind.value, "conc": workload.conc}
    if shape is not None:
        requests.update(isl=shape.isl, osl=shape.osl)
    return AgentSpec(
        model=resolve_serving_model_path(workload.model),
        framework=workload.framework,
        tp=workload.tp,
        precision=workload.precision,
        gpu_type=workload.gpu_type,
        workload=requests,
        metric=workload.metric,
        server_args="",
        envs={},
        workspace="",
        cards=(),
        seconds=0.0,
        port=port,
        ep=workload.ep,
        pack=pack or None,
        launch=launch or None,
        review=review,
    )


def load_peers(workload: Workload) -> PeerFactory:
    """The factory of the real peers of a session of ``workload`` whose tests run on this machine's cards; raises
    :class:`PeersUnavailable` when this machine cannot run them.

    It creates nothing, so the command line calls it before the session directory exists.
    """

    require_local_model(workload)
    return peers_on(measurement_host(os.environ, LocalCards(machine_cards())), workload)


def require_local_model(workload: Workload) -> None:
    """Check that ``workload``'s model resolves to a directory on this machine (:func:`resolve_serving_model_path`);
    raises :class:`PeersUnavailable` naming where it was looked for otherwise. A local test's server runs with its
    sandbox directory as its home, so a hub id finds none of this host's Hugging Face cache and is fetched from the
    hub."""

    served = resolve_serving_model_path(workload.model)
    if not os.path.isdir(served):
        base = os.environ.get("HL_MODEL_BASE", "")
        raise PeersUnavailable(
            f"the model {workload.model} is no directory on this machine (HL_MODEL_BASE={base!r}, nor in the "
            "Hugging Face cache): pass --model the checkpoint's directory, or set HL_MODEL_BASE to the directory "
            "holding it under its repository name"
        )


def peers_on(host: MeasurementHost, workload: Workload) -> PeerFactory:
    """The factory of the real peers of a session of ``workload`` measured on ``host``; raises
    :class:`PeersUnavailable` when the image lacks the framework's package or the agent CLI is not installed. Creates
    nothing."""

    require_framework_package(workload.framework)
    command = specialist_command(os.environ)
    return partial(local_peers, host, model_backed_sessions(command))


__all__ = [
    "AGENTX_ENV",
    "CLAIM_TTL_SECONDS",
    "GATE_WRITER",
    "InferenceXCheckout",
    "LocalCards",
    "MeasurementHost",
    "RENDER_NODE_DIR",
    "SessionWiring",
    "SpecialistSessions",
    "ToolboxBackend",
    "load_peers",
    "require_local_model",
    "local_peers",
    "machine_cards",
    "machine_parallel_tests",
    "measurement_host",
    "model_backed_sessions",
    "peers_on",
    "render_nodes",
    "require_framework_package",
    "specialist_command",
    "specialist_model",
    "specialist_recall",
    "workload_environment",
    "workload_rule",
]
