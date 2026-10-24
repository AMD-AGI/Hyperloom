# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session on one machine: the git experiment repository, the local hosted toolbox, the rule
gate's session service, the specialist runtime on them and the MCP service that serves the specialists their tools.

:func:`local_peers` builds them inside the session directory: the repository at ``MB-repo``, the toolbox's
sandboxes, archive, submit log and measurement points under ``MB-toolbox``, and the patch and kernel files its
environment compiler reads, like the repository, from the session's blob store at ``MB-blobs``. The MCP service listens on loopback for the whole session and hands every toolbox call to the loop the peers were built
on; its learnings tool recalls this session's learnings and past sessions' lessons from the RSI lesson pool, and none
from the Experience KB, as MB1 runs on none (:func:`specialist_recall`). Every test is measured on a
:class:`MeasurementHost`: this machine's cards, the image the launcher names in ``HYPERLOOM_ACTIVE_IMAGE_REF`` and
``HYPERLOOM_ACTIVE_IMAGE_DIGEST``, the Magpie commit Hyperloom pins and an InferenceX checkout at the commit the
workload's benchmark pins (:class:`InferenceXCheckout`); the canonical measurement is rendered with the workload's model
and GPU type and that checkout, from the shipped baseline of an InferenceX workload and from the Magpie AgentX benchmark
the session writes for an AgentX one, which the PRELUDE child is bound to as well. The gate's session service and the
specialist runtime judge on one rule, :func:`workload_rule`; the runtime settles through a rule gate that asks the
scheduler's fork desk for branches, and the MCP service's ``request_repair`` and ``decline_repair`` conclude a blocked
candidate through the same gate, which the scheduler has land each fork it admits and record each fork and repair it
voids. A specialist declares a correctness fix with the MCP service's ``fix_case``; the runtime proves it with the repro
runner the regression set uses and has the setup's critic review it. The runtime's specialist sessions come from a
:data:`SpecialistSessions`: :func:`model_backed_sessions` runs each branch's specialist as an agent CLI in a worktree of
its own, briefed by the session's :class:`SessionBriefs`, whose warm-start leads the session binds once PRELUDE has read
them and which reads the repository for a replay's branch head. :func:`load_peers` reads the measurement host, checks
the image holds the framework package those worktrees branch off and resolves that CLI first, and refuses when either
cannot be had, so the command line fails before it creates anything.

The toolbox, its measurement points and the agent sessions load the main flow's runtime, so they are imported inside
the functions that use them: importing this module, the command line or the session loads none of it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, partial
from types import ModuleType
from typing import TYPE_CHECKING

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from hyperloom.common.llm_config import preferred_agent_backend, provider_model_defaults, resolve_forge_llm_model
from hyperloom.common.model_paths import resolve_serving_model_path
from hyperloom.common.workload_defaults import DEFAULT_EP
from hyperloom_multibranch.contracts.llm_prices import price_of
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.interfaces import BranchLogStore, CommitQuery
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import SpecialistTokens
from hyperloom_multibranch.notebook import BriefTriggers, LocalNotebook
from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG, RetrievalLog
from hyperloom_multibranch.notebook.rsi.briefing import SessionBriefs
from hyperloom_multibranch.notebook.rsi.lesson_recall import LessonRecall
from hyperloom_multibranch.notebook.rsi.lessons import LessonPool
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.regression import RegressionSet, ReproRunner
from hyperloom_multibranch.rulegate.rules import AgentXRule, InferenceXRule, Rule
from hyperloom_multibranch.rulegate.service import GateService
from hyperloom_multibranch.scheduling.peers import PeerFactory, Peers, PeerSetup, PeersUnavailable
from hyperloom_multibranch.scheduling.settings import Workload
from hyperloom_multibranch.specialist.fix_check import FixChecking
from hyperloom_multibranch.specialist.fix_tools import register_fix_tools
from hyperloom_multibranch.specialist.judging import Judging
from hyperloom_multibranch.specialist.runtime import RuntimePacing, SpecialistRuntime

if TYPE_CHECKING:
    from hyperloom_multibranch.specialist.agent_onboarding import AgentSpec
    from pathlib import Path

    from hyperloom.hosted.mcp_server import McpServer
    from hyperloom.measure.pins import ActiveImage, ToolCommits
    from hyperloom.measure.points import CanonicalMeasurement
    from hyperloom_multibranch.specialist.agent_session import AgentCommand
    from hyperloom_multibranch.specialist.dev_sandbox import DevLeases
    from hyperloom_multibranch.specialist.model_sessions import ModelSessions
    from hyperloom_multibranch.specialist.tools import WorktreeFor
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

logger = logging.getLogger(__name__)

TOOLBOX_WRITER = "toolbox"
GATE_WRITER = "rule-gate"
BRIEFS_READER = "briefs"
CANONICAL_DIR = "canonical"
POINTS_DIR = "points"
#: The Magpie AgentX benchmark of an AgentX session, under ``MB-toolbox``.
BENCHMARK_FILE = "benchmark.yaml"
#: A claim's lease on the specialist tools: the runtime renews it every 10 minutes (board #26).
CLAIM_TTL_SECONDS = 1800.0
#: The switch the main flow and the measurement renderer read an AgentX workload from
#: (``hyperloom.common.perf_metric.agentx_enabled``).
AGENTX_ENV = "HYPERLOOM_AGENTX"


@dataclass(frozen=True)
class SessionWiring:
    """What a session's specialist sessions are built on: the peer setup, the repository the branches use, the
    tokens that grant a branch's specialist the MCP service's tools (a grant names the service's URL, so it is issued
    when a specialist starts), each branch's log, the session's briefs and its dev sandboxes, ``None`` when it lends
    none."""

    setup: PeerSetup
    repo: ExperimentRepo
    tokens: SpecialistTokens
    notebooks: Callable[[str], BranchLogStore]
    briefs: SessionBriefs
    toolbox: RealToolbox
    dev_leases: DevLeases | None = None


#: Builds the specialist runtime's session factory, and the worktree of each branch the worktree tools serve, on a
#: session's wiring.
SpecialistSessions = Callable[[SessionWiring], "ModelSessions"]


@dataclass(frozen=True)
class InferenceXCheckout:
    """The InferenceX commit a session's measurements are signed with and this machine's checkout at it, which every
    measurement point names: Hyperloom's synthetic pin for an InferenceX workload, the one the installed Magpie pins
    for an AgentX one (Magpie's native runs need its own checkout)."""

    commit: str
    path: str


@dataclass(frozen=True)
class MeasurementHost:
    """What the toolbox measures on: the cards of the machine, the image the tests run in (its digest is also the
    base image every recipe must carry, the root recipe's first), the Magpie commit the runner checks out and the
    InferenceX checkout of the session's workload."""

    cards: tuple[int, ...]
    image: ActiveImage
    magpie_commit: str
    inferencex: InferenceXCheckout


def workload_environment(workload: Workload) -> Mapping[str, str]:
    """The workload as the process environment carries it to the main flow's config renderer, which the toolbox
    renders the canonical measurement and every test's points with, and to the kernel tuning the toolbox runs, which
    takes its backend from ``KERNEL_OPT_BACKEND_ORDER``; ``hyperloom optimize`` exports the same, the model resolved to
    the directory a server loads it from. The command line accepts only GPU types that are their own Magpie runner
    type, so ``GPU_TYPE`` and ``TARGET_GPU_TYPE`` are the same. An InferenceX workload carries its sequence lengths;
    an AgentX one carries :data:`AGENTX_ENV` instead, which the PRELUDE child inherits."""

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


def _export_workload(workload: Workload, toolbox_dir: Path) -> None:
    """Export :func:`workload_environment` into this process. For an AgentX workload, also write the Magpie AgentX
    benchmark ``hyperloom optimize`` generates for it to :data:`BENCHMARK_FILE` in ``toolbox_dir`` and name it in
    ``HYPERLOOM_BENCHMARK_CONFIG``: the canonical measurement is rendered from it, and the PRELUDE child binds its
    command line to it. The file pins no model directory, so the child's ``--model`` stays the workload's model."""

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
        ep=DEFAULT_EP,
        conc=workload.conc,
        gpu_type=workload.gpu_type,
    )
    write_agentx_benchmark(path, benchmark)
    os.environ[BENCHMARK_CONFIG_ENV] = str(path.resolve())


def workload_rule(workload: Workload, notebook: CommitQuery) -> Rule:
    """The rule a session of ``workload`` is judged on: InferenceX throughput at the workload's conc against the
    PRELUDE bar ``notebook`` holds, or AgentX P50 interactivity with its guards."""

    if workload.inferencex is None:
        return AgentXRule()
    return InferenceXRule(notebook, workload.conc)


def _canonical(out_dir: Path, workload: Workload, host: MeasurementHost) -> CanonicalMeasurement:
    """The canonical measurement every point of the session is rendered from. The renderer takes the model and the
    runner type only as arguments, so a point never falls back to the shipped YAML's model."""

    from hyperloom.measure.points import default_canonical

    return default_canonical(
        out_dir,
        primary_concurrency=workload.conc,
        model_path=resolve_serving_model_path(workload.model),
        gpu_type=workload.gpu_type,
        inferencex_path=host.inferencex.path,
    )


def specialist_recall(reports: Path) -> LessonRecall:
    """The specialists' learnings recall: this session's learnings, then past sessions' lessons from the RSI lesson
    pool beside the environment's archive, each retrieval logged in ``reports``; no Experience KB, as MB1 runs on
    none (board #22)."""

    return LessonRecall(pool=LessonPool.from_env(), log=RetrievalLog(reports / LEARNINGS_LOG))


def local_peers(host: MeasurementHost, sessions: SpecialistSessions, setup: PeerSetup) -> Peers:
    """The session's peers on this machine; exports the workload into this process (:func:`_export_workload`).

    Called on the event loop the session runs on. The toolbox is built, not started: the session starts it, and its
    close releases the backend and the sandboxes. The MCP service is started; the session stop stops it.
    """

    from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
    from hyperloom.measure.compile import EnvironmentCompiler
    from hyperloom.measure.hosted_spec import SpecContext
    from hyperloom_multibranch.toolbox.local_hosted import LocalSession, build_local_toolbox
    from hyperloom_multibranch.toolbox.mcp_tools import (
        McpSpecialistTokens,
        SpecialistToolOptions,
        register_specialist_tools,
    )
    from hyperloom_multibranch.specialist.blocked import BlockedCandidates, register_blocked_tools
    from hyperloom_multibranch.specialist.tools import register_worktree_tools
    from hyperloom_multibranch.toolbox.planner import HostedPlanner
    from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

    layout, workload = setup.layout, setup.workload
    _export_workload(workload, layout.toolbox)
    repo = GitExperimentRepo(layout.repo, setup.blobs)
    canonical = _canonical(layout.toolbox / CANONICAL_DIR, workload, host)
    image = host.image
    compiler = EnvironmentCompiler(setup.blobs, image.digest)
    spec = SpecContext(
        image_ref=image.ref,
        image_digest=image.digest,
        gpu_type=workload.gpu_type,
        magpie_commit=host.magpie_commit,
        inferencex_commit=host.inferencex.commit,
        blobs=setup.blobs,
        blob_root=str(setup.blobs.root),
    )
    planner = HostedPlanner(compiler, canonical, layout.toolbox / POINTS_DIR)
    toolbox = build_local_toolbox(
        LocalSession(layout.toolbox, layout.session_id, host.cards, workload.tp),
        LocalNotebook(layout.notebook, Writer(Role.TOOLBOX, TOOLBOX_WRITER)),
        AcceptanceSetup(repo, planner, image.digest, workload_kind=workload.kind),
        spec,
    )
    gate_notebook = LocalNotebook(layout.notebook, Writer(Role.GATE, GATE_WRITER))
    rule = workload_rule(workload, gate_notebook)

    @cache
    def runtime_notebook(branch: str) -> LocalNotebook:
        """The branch's one handle for the session: a handle is a writer of its own, and every inbox cursor holds one
        sequence number per writer, so a handle per MCP call made ``poll_inbox`` answer 91,000 characters after 12 h."""
        return LocalNotebook(layout.notebook, Writer(Role.SPECIALIST_RUNTIME, branch, branch), briefs=BriefTriggers())

    mcp = McpServer(McpServerConfig(host_token=secrets.token_hex(32)), campaign=layout.session_id)
    learnings = specialist_recall(layout.reports)
    briefs = SessionBriefs(
        LocalNotebook(layout.notebook, Writer(Role.OBSERVER, BRIEFS_READER)),
        metric=workload.metric,
        learnings=learnings,
        repo=repo,
        stack_slot=repo.stack_kernel_slot,
    )
    tool_options = SpecialistToolOptions(
        claim_ttl_seconds=CLAIM_TTL_SECONDS,
        toolbox_loop=asyncio.get_running_loop(),
        learnings=learnings,
        briefing_for=briefs.for_branch,
        analysis=toolbox,
    )
    register_specialist_tools(mcp, toolbox=toolbox, notebooks=runtime_notebook, options=tool_options)
    leases = _lend_dev_sandboxes(mcp, setup, toolbox, runtime_notebook, lambda branch: built.worktrees(branch))
    built = sessions(SessionWiring(setup, repo, McpSpecialistTokens(mcp), runtime_notebook, briefs, toolbox, leases))
    register_worktree_tools(mcp, repo=repo, worktrees=built.worktrees)
    gate = RuleGate(gate_notebook, repo, setup.desk, toolbox)
    register_blocked_tools(mcp, blocked=BlockedCandidates(gate_notebook, gate), home=tool_options.toolbox_loop)
    register_fix_tools(mcp, attachments=toolbox)
    runner = ReproRunner(toolbox, gate_notebook, toolbox)
    judging = Judging(gate, toolbox, rule, FixChecking(runner, setup.critic, setup.blobs, _specialist_role(workload)))
    runtime = SpecialistRuntime(
        runtime_notebook, judging, built.factory, pacing=RuntimePacing(claim_ttl_seconds=CLAIM_TTL_SECONDS)
    )
    mcp.start()
    gate_service = GateService(gate_notebook, toolbox, rule, toolbox)
    regression = RegressionSet(gate_notebook, runner, repo)
    return Peers(runtime, toolbox, repo, gate_service, mcp, briefs, image.digest, gate, regression)


def _lend_dev_sandboxes(
    mcp: McpServer,
    setup: PeerSetup,
    toolbox: RealToolbox,
    notebooks: Callable[[str], LocalNotebook],
    worktrees: WorktreeFor,
) -> DevLeases | None:
    """The session's dev sandboxes on ``toolbox``'s cards, their tools registered on ``mcp`` for the session's loop;
    ``None``, and no tools, when ``setup`` lends none. ``worktrees`` is read only when a lease is granted, after the
    MCP service starts, so it may name sessions built after this call."""

    from hyperloom_multibranch.specialist.dev_sandbox import (
        DevLeases,
        DevLimits,
        local_dev_sandboxes,
        register_dev_sandbox_tools,
    )

    if setup.dev_leases < 1:
        return None
    limits = DevLimits(cap=setup.dev_leases)
    leases = DevLeases(toolbox, local_dev_sandboxes(worktrees, setup.processes, limits), notebooks, limits=limits)
    register_dev_sandbox_tools(mcp, leases=leases, home=asyncio.get_running_loop())
    return leases


def machine_cards() -> tuple[int, ...]:
    """The cards of this machine the session may use: the absolute ids the first visible-device mask set names, or,
    when none is set, as many as ``rocm-smi`` shows from card 0; none when neither shows any.

    The ids are absolute because every sandbox's ``ROCR_VISIBLE_DEVICES`` is set to its leased cards in place of the
    mask this process runs under; a mask of device UUIDs names no id, so it leaves no card. The main flow resolves its
    whole-machine pool by the same rule (``hyperloom.orchestrator.bus.gpu_pool``), which the scheduler does not load.
    """

    from hyperloom.common.visible_devices import COUNTING_VISIBLE_DEVICE_VARS, detect_gpu_count, parse_device_list

    for name in COUNTING_VISIBLE_DEVICE_VARS:
        mask = os.environ.get(name)
        if mask is not None:
            return tuple(parse_device_list(mask))
    return tuple(range(detect_gpu_count()))


def measurement_host(env: Mapping[str, str], workload: Workload) -> MeasurementHost:
    """This machine's cards, the image ``env`` names, the Magpie commit Hyperloom pins, ``env`` overriding as
    ``install.sh`` lets it, and the InferenceX checkout of ``workload`` (:func:`_synthetic_checkout`,
    :func:`_agentx_checkout`); raises :class:`PeersUnavailable` when the image or a commit cannot be pinned, an unnamed
    image included, or no writable checkout at the workload's InferenceX commit is found."""

    from hyperloom.measure.pins import ImagePinError, ToolPinError, active_image, pinned_tool_commits

    try:
        image, tools = active_image(env), pinned_tool_commits(env)
    except (ImagePinError, ToolPinError) as exc:
        raise PeersUnavailable(f"the measurement pins cannot be read: {exc}") from exc
    checkout = _synthetic_checkout(env, tools) if workload.inferencex is not None else _agentx_checkout(env, tools)
    return MeasurementHost(machine_cards(), image, tools.magpie, checkout)


def _synthetic_checkout(env: Mapping[str, str], tools: ToolCommits) -> InferenceXCheckout:
    """The InferenceX checkout ``INFERENCEX_PATH`` names or ``install.sh`` left, at the commit Hyperloom pins."""

    from hyperloom.inference_optimizer import inferencex_checkout

    checkout = env.get("INFERENCEX_PATH", "").strip() or inferencex_checkout.detect()
    if not (
        checkout and inferencex_checkout.checkout_ok(checkout, ref=tools.inferencex) and os.access(checkout, os.W_OK)
    ):
        raise PeersUnavailable(
            f"no InferenceX checkout at {tools.inferencex[:12]} is writable at {checkout or 'any detected path'}: "
            "run install.sh, or set INFERENCEX_PATH to one"
        )
    return InferenceXCheckout(tools.inferencex, checkout)


def _magpie_inferencex() -> ModuleType:
    """The installed Magpie's InferenceX module, which owns the native AgentX checkout: its pin, its default clone
    location and its project-root resolution."""

    from Magpie.modes.benchmark import inferencex

    return inferencex


def _agentx_checkout(env: Mapping[str, str], tools: ToolCommits) -> InferenceXCheckout:
    """The InferenceX checkout Magpie's native AgentX runs on: the commit the installed Magpie pins, at
    ``MAGPIE_INFERENCEX_PATH`` or Magpie's default clone location, resolved to its project root as Magpie resolves it
    (the repository root or its ``inferencex-e2e`` directory). A Magpie that pins no commit is older than the
    ``MAGPIE_REF`` Hyperloom pins and is refused."""

    from hyperloom.inference_optimizer import inferencex_checkout

    try:
        magpie = _magpie_inferencex()
    except ImportError as exc:
        raise PeersUnavailable(f"Magpie, which runs an AgentX workload, cannot be imported: {exc}") from exc
    commit = getattr(magpie, "INFERENCEX_DEFAULT_REF", None)
    if not commit:
        raise PeersUnavailable(
            "the installed Magpie pins no InferenceX commit for AgentX (no INFERENCEX_DEFAULT_REF): "
            f"install Magpie at MAGPIE_REF {tools.magpie}"
        )
    path = env.get("MAGPIE_INFERENCEX_PATH", "").strip() or magpie.InferenceXManager().default_dir
    try:
        root = magpie.resolve_inferencex_root(path)
    except (OSError, RuntimeError):
        root = ""
    if not (root and inferencex_checkout.checkout_ok(root, ref=commit) and os.access(root, os.W_OK)):
        raise PeersUnavailable(
            f"no InferenceX checkout at {commit}, the commit Magpie pins for AgentX, is writable at {path}: "
            "set MAGPIE_INFERENCEX_PATH to one"
        )
    return InferenceXCheckout(commit, root)


def machine_parallel_tests(tp: int) -> int:
    """How many tests of ``tp`` cards the local toolbox runs at once on this machine's cards; raises ``ValueError``
    when a test of ``tp`` cards does not fit on them."""

    from hyperloom_multibranch.toolbox.local_hosted import max_parallel_tests

    return max_parallel_tests(len(machine_cards()), tp)


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
    requests = "AgentX trace replay" if shape is None else f"ISL {shape.isl}, OSL {shape.osl}"
    return (
        f"{workload.model_basename} on {workload.framework}, {workload.precision}, TP {workload.tp}, "
        f"{requests}, concurrency {workload.conc}, {workload.gpu_type}"
    )


AGENT_PORT = 31100


def model_backed_sessions(command: AgentCommand) -> SpecialistSessions:
    """Sessions that run ``command`` for every branch's specialist, in a worktree of the workload's framework, and
    each branch's third-party agent in a sandbox leased from the session's toolbox.

    The CLI runs on the main flow's allowlisted specialist environment, through a local AI work socket whose resource
    log is under ``MB-logs``; its worktrees branch off one source base at ``MB-source``, and each launch rebuilds the
    branch's brief at the head it starts from.
    """

    def build(wiring: SessionWiring) -> ModelSessions:
        from hyperloom.orchestrator.specialists.agent_runner import Liveness, LocalAgentRunner, build_specialist_env
        from hyperloom_multibranch.specialist.agent_session import SPECIALIST_LOST_AFTER_S, AgentSetup
        from hyperloom_multibranch.specialist.agent_sessions import AgentSessions, with_agents
        from hyperloom_multibranch.specialist.model_sessions import model_sessions
        from hyperloom_multibranch.specialist.prompts import role_prompt
        from hyperloom_multibranch.specialist.worktree import SourceBase

        layout, workload = wiring.setup.layout, wiring.setup.workload
        agent = AgentSetup(
            command=command,
            role=role_prompt(
                _specialist_role(workload), references=wiring.setup.references, dev_leases=wiring.setup.dev_leases
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
        )
        sessions = model_sessions(
            agent,
            base=SourceBase(layout.source),
            components=(workload.framework,),
            branch_dir=layout.branch_dir,
            notebook_for=wiring.notebooks,
        )
        agents = AgentSessions(
            _agent_spec(workload),
            command,
            agent.env,
            wiring.toolbox,
            wiring.repo,
            wiring.setup.processes,
        )
        return with_agents(sessions, agents, branch_dir=layout.branch_dir, notebook_for=wiring.notebooks)

    return build


def _agent_spec(workload: Workload) -> AgentSpec:
    """What every agent of a session of ``workload`` is asked to optimize; the root's launch, the workspace, the
    cards and the port are each branch's own."""

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
        port=AGENT_PORT,
    )


def load_peers(workload: Workload) -> PeerFactory:
    """The factory of the real peers of a session of ``workload``; raises :class:`PeersUnavailable` when this machine
    cannot run them.

    It creates nothing, so the command line calls it before the session directory exists.
    """

    host = measurement_host(os.environ, workload)
    require_framework_package(workload.framework)
    command = specialist_command(os.environ)
    return partial(local_peers, host, model_backed_sessions(command))


__all__ = [
    "AGENTX_ENV",
    "CLAIM_TTL_SECONDS",
    "GATE_WRITER",
    "InferenceXCheckout",
    "MeasurementHost",
    "SessionWiring",
    "SpecialistSessions",
    "load_peers",
    "local_peers",
    "machine_cards",
    "machine_parallel_tests",
    "measurement_host",
    "model_backed_sessions",
    "require_framework_package",
    "specialist_command",
    "specialist_model",
    "specialist_recall",
    "workload_environment",
    "workload_rule",
]
