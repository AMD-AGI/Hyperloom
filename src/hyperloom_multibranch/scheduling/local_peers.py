# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session on one machine: the git experiment repository, the local hosted toolbox, the rule
gate's session service, the specialist runtime on them and the MCP service that serves the specialists their tools.

:func:`local_peers` builds them inside the session directory: the repository at ``MB-repo``, the toolbox's
sandboxes, archive, submit log and measurement points under ``MB-toolbox``, and the patch and kernel files its
environment compiler reads, like the repository, from the session's blob store at ``MB-blobs``. The MCP service listens on loopback for the whole session and hands every toolbox call to the loop the peers were built
on; its learnings tool recalls this session's learnings only, as MB1 runs on no Experience KB
(:func:`~hyperloom_multibranch.scheduling.mb_fakes.mb_fake_learning_recall`). Every test is measured on a
:class:`MeasurementHost`: this machine's cards, the image the launcher names in ``HYPERLOOM_ACTIVE_IMAGE_REF`` and
``HYPERLOOM_ACTIVE_IMAGE_DIGEST`` and the tool commits Hyperloom pins. The gate's session service and the specialist
runtime judge on one rule, :func:`workload_rule`; the runtime settles through a rule gate that asks the scheduler's fork
desk for branches, and the MCP service's ``request_repair`` and ``decline_repair`` conclude a blocked candidate through
the same gate, which the scheduler has land each fork it admits and record each fork and repair it voids. A specialist
declares a correctness fix with the MCP service's ``fix_case``; the runtime proves it with the repro runner the
regression set uses and has the setup's critic review it. The runtime's specialist sessions come from a
:data:`SpecialistSessions`: :func:`model_backed_sessions` runs each branch's specialist as an agent CLI in a worktree of
its own, briefed by the session's :class:`SessionBriefs`, whose warm-start leads the session binds once PRELUDE has read
them and which reads the repository for a replay's branch head. :func:`load_peers` reads the measurement host, checks
the image holds the framework package those worktrees branch off and resolves that CLI and the price of its model first,
and refuses when any cannot be had, so the command line fails before it creates anything.

The toolbox, its measurement points and the agent sessions load the main flow's runtime, so they are imported inside
the functions that use them: importing this module, the command line or the session loads none of it.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from hyperloom.common.llm_config import preferred_agent_backend, provider_model_defaults, resolve_forge_llm_model
from hyperloom.common.model_paths import resolve_serving_model_path
from hyperloom_multibranch.contracts.llm_prices import UnpricedModel, price_of
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.interfaces import BranchLogStore, CommitQuery
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import SpecialistTokens
from hyperloom_multibranch.notebook import BriefTriggers, LocalNotebook
from hyperloom_multibranch.notebook.rsi.briefing import SessionBriefs
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.regression import RegressionSet, ReproRunner
from hyperloom_multibranch.rulegate.rules import AgentXRule, InferenceXRule, Rule
from hyperloom_multibranch.rulegate.service import GateService
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_learning_recall
from hyperloom_multibranch.scheduling.peers import PeerFactory, Peers, PeerSetup, PeersUnavailable
from hyperloom_multibranch.scheduling.settings import Workload
from hyperloom_multibranch.specialist.fix_check import FixChecking
from hyperloom_multibranch.specialist.fix_tools import register_fix_tools
from hyperloom_multibranch.specialist.judging import Judging
from hyperloom_multibranch.specialist.runtime import RuntimePacing, SpecialistRuntime

if TYPE_CHECKING:
    from hyperloom.measure.pins import ActiveImage, ToolCommits
    from hyperloom_multibranch.specialist.agent_session import AgentCommand
    from hyperloom_multibranch.specialist.model_sessions import ModelSessions

TOOLBOX_WRITER = "toolbox"
GATE_WRITER = "rule-gate"
BRIEFS_READER = "briefs"
CANONICAL_DIR = "canonical"
POINTS_DIR = "points"
#: A claim's lease on the specialist tools: the runtime renews it every 10 minutes (board #26).
CLAIM_TTL_SECONDS = 1800.0
#: The switch the main flow and the measurement renderer read an AgentX workload from
#: (``hyperloom.common.perf_metric.agentx_enabled``).
AGENTX_ENV = "HYPERLOOM_AGENTX"


@dataclass(frozen=True)
class SessionWiring:
    """What a session's specialist sessions are built on: the peer setup, the repository the branches use, the
    tokens that grant a branch's specialist the MCP service's tools (a grant names the service's URL, so it is issued
    when a specialist starts), each branch's log and the session's briefs."""

    setup: PeerSetup
    repo: ExperimentRepo
    tokens: SpecialistTokens
    notebooks: Callable[[str], BranchLogStore]
    briefs: SessionBriefs


#: Builds the specialist runtime's session factory, and the worktree of each branch the worktree tools serve, on a
#: session's wiring.
SpecialistSessions = Callable[[SessionWiring], "ModelSessions"]


@dataclass(frozen=True)
class MeasurementHost:
    """What the toolbox measures on and the session's workload does not name: the cards of the machine, the image
    the tests run in (its digest is also the base image every recipe must carry, the root recipe's first) and the
    Magpie and InferenceX commits the runner checks out."""

    cards: tuple[int, ...]
    image: ActiveImage
    tools: ToolCommits


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


def workload_rule(workload: Workload, notebook: CommitQuery) -> Rule:
    """The rule a session of ``workload`` is judged on: InferenceX throughput at the workload's conc against the
    PRELUDE bar ``notebook`` holds, or AgentX P50 interactivity with its guards."""

    if workload.inferencex is None:
        return AgentXRule()
    return InferenceXRule(notebook, workload.conc)


def local_peers(host: MeasurementHost, sessions: SpecialistSessions, setup: PeerSetup) -> Peers:
    """The session's peers on this machine; exports :func:`workload_environment` into this process.

    Called on the event loop the session runs on. The toolbox is built, not started: the session starts it, and its
    close releases the backend and the sandboxes. The MCP service is started; the session stop stops it.
    """

    from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
    from hyperloom.measure.compile import EnvironmentCompiler
    from hyperloom.measure.hosted_spec import SpecContext
    from hyperloom.measure.points import default_canonical
    from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG, RetrievalLog
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
    os.environ.update(workload_environment(workload))
    repo = GitExperimentRepo(layout.repo, setup.blobs)
    canonical = default_canonical(layout.toolbox / CANONICAL_DIR, primary_concurrency=workload.conc)
    image = host.image
    compiler = EnvironmentCompiler(setup.blobs, image.digest)
    spec = SpecContext(
        image_ref=image.ref,
        image_digest=image.digest,
        gpu_type=workload.gpu_type,
        magpie_commit=host.tools.magpie,
        inferencex_commit=host.tools.inferencex,
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

    def runtime_notebook(branch: str) -> LocalNotebook:
        return LocalNotebook(layout.notebook, Writer(Role.SPECIALIST_RUNTIME, branch, branch), briefs=BriefTriggers())

    mcp = McpServer(McpServerConfig(host_token=secrets.token_hex(32)), campaign=layout.session_id)
    learnings = mb_fake_learning_recall(RetrievalLog(layout.reports / LEARNINGS_LOG))
    briefs = SessionBriefs(
        LocalNotebook(layout.notebook, Writer(Role.OBSERVER, BRIEFS_READER)),
        metric=workload.metric,
        learnings=learnings,
        repo=repo,
    )
    tool_options = SpecialistToolOptions(
        claim_ttl_seconds=CLAIM_TTL_SECONDS,
        toolbox_loop=asyncio.get_running_loop(),
        learnings=learnings,
        briefing_for=briefs.for_branch,
        analysis=toolbox,
    )
    register_specialist_tools(mcp, toolbox=toolbox, notebooks=runtime_notebook, options=tool_options)
    built = sessions(SessionWiring(setup, repo, McpSpecialistTokens(mcp), runtime_notebook, briefs))
    register_worktree_tools(mcp, repo=repo, worktrees=built.worktrees)
    gate = RuleGate(gate_notebook, repo, setup.desk, toolbox)
    blocked = BlockedCandidates(gate_notebook, gate)
    register_blocked_tools(mcp, blocked=blocked, home=tool_options.toolbox_loop)
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


def measurement_host(env: Mapping[str, str]) -> MeasurementHost:
    """This machine's cards, the image ``env`` names and the tool commits Hyperloom pins, ``env`` overriding as
    ``install.sh`` lets it; raises :class:`PeersUnavailable` when the image or a commit cannot be pinned, an
    unnamed image included."""

    from hyperloom.measure.pins import ImagePinError, ToolPinError, active_image, pinned_tool_commits

    try:
        return MeasurementHost(machine_cards(), active_image(env), pinned_tool_commits(env))
    except (ImagePinError, ToolPinError) as exc:
        raise PeersUnavailable(f"the measurement pins cannot be read: {exc}") from exc


def machine_parallel_tests(tp: int) -> int:
    """How many tests of ``tp`` cards the local toolbox runs at once on this machine's cards; raises ``ValueError``
    when a test of ``tp`` cards does not fit on them."""

    from hyperloom_multibranch.toolbox.local_hosted import max_parallel_tests

    return max_parallel_tests(len(machine_cards()), tp)


def specialist_command(env: Mapping[str, str]) -> AgentCommand:
    """The agent CLI every specialist of a session runs: the backend ``env`` is credentialed for, found on its
    ``PATH``, on the model ``env`` names for it; raises :class:`PeersUnavailable` when that CLI is not installed or
    the price sheet has no rate for that model, whose spend could then not be recorded."""

    from hyperloom_multibranch.specialist.agent_session import AgentCommand

    backend = preferred_agent_backend(env)
    executable = shutil.which(backend, path=env.get("PATH"))
    if executable is None:
        raise PeersUnavailable(f"the {backend} CLI the specialists run is not on PATH")
    model = resolve_forge_llm_model(backend, env={**env, **provider_model_defaults(env)})
    try:
        price_of(model)
    except UnpricedModel as exc:
        raise PeersUnavailable(f"the specialists' model cannot be priced: {exc}") from exc
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


def model_backed_sessions(command: AgentCommand) -> SpecialistSessions:
    """Sessions that run ``command`` for every branch's specialist, in a worktree of the workload's framework.

    The CLI runs on the main flow's allowlisted specialist environment, through a local AI work socket whose resource
    log is under ``MB-logs``; its worktrees branch off one source base at ``MB-source``, and each launch rebuilds the
    branch's brief at the head it starts from.
    """

    def build(wiring: SessionWiring) -> ModelSessions:
        from hyperloom.orchestrator.specialists.agent_runner import LocalAgentRunner, build_specialist_env
        from hyperloom_multibranch.specialist.agent_session import AgentSetup
        from hyperloom_multibranch.specialist.model_sessions import model_sessions
        from hyperloom_multibranch.specialist.prompts import role_prompt
        from hyperloom_multibranch.specialist.worktree import SourceBase

        layout, workload = wiring.setup.layout, wiring.setup.workload
        agent = AgentSetup(
            command=command,
            role=role_prompt(_specialist_role(workload)),
            env=build_specialist_env(),
            runner=LocalAgentRunner(session_dir=layout.logs),
            tokens=wiring.tokens,
            processes=wiring.setup.processes,
            repo=wiring.repo,
            briefs=wiring.briefs.text,
        )
        return model_sessions(
            agent,
            base=SourceBase(layout.source),
            components=(workload.framework,),
            branch_dir=layout.branch_dir,
            notebook_for=wiring.notebooks,
        )

    return build


def load_peers(workload: Workload) -> PeerFactory:
    """The factory of the real peers of a session of ``workload``; raises :class:`PeersUnavailable` when this machine
    cannot run them.

    It creates nothing, so the command line calls it before the session directory exists.
    """

    host = measurement_host(os.environ)
    require_framework_package(workload.framework)
    return partial(local_peers, host, model_backed_sessions(specialist_command(os.environ)))


__all__ = [
    "AGENTX_ENV",
    "CLAIM_TTL_SECONDS",
    "GATE_WRITER",
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
    "workload_environment",
    "workload_rule",
]
