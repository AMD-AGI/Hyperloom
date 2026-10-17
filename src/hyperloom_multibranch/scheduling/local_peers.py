# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session on one machine: the git experiment repository, the local hosted toolbox, the rule
gate's session service, the specialist runtime on them and the MCP service that serves the specialists their tools.

:func:`local_peers` builds them inside the session directory: the repository at ``MB-repo``, the toolbox's
sandboxes, archive, submit log, measurement points and measurement config under ``MB-toolbox``, and the patch and
kernel files its environment compiler reads, like the repository, from the session's blob store at ``MB-blobs``. The
MCP service listens on loopback for the whole session and hands every toolbox call to the loop the peers were built
on; its learnings tool recalls past sessions' from the Experience KB the environment names. The gate's session
service and the specialist runtime judge on one InferenceX rule at the workload's conc; the runtime settles through a
rule gate that asks the scheduler's fork desk for branches. The runtime's specialist sessions come from
a :data:`SpecialistSessions`: :func:`model_backed_sessions` runs each branch's specialist as an agent CLI in a
worktree of its own, briefed by the session's :class:`SessionBriefs`, whose warm-start leads the session binds once
PRELUDE has read them. :func:`load_peers` resolves that CLI first and refuses when this machine has none, so the
command line fails before it creates anything.

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

from hyperloom.common.io import atomic_write_json
from hyperloom.common.llm_config import preferred_agent_backend, provider_model_defaults, resolve_forge_llm_model
from hyperloom.common.model_paths import resolve_serving_model_path
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.interfaces import BranchLogStore
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import SpecialistTokens
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.rsi.briefing import SessionBriefs
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.rules import InferenceXRule
from hyperloom_multibranch.rulegate.service import GateService
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_BASE_IMAGE_DIGEST
from hyperloom_multibranch.scheduling.peers import PeerFactory, Peers, PeerSetup, PeersUnavailable
from hyperloom_multibranch.scheduling.settings import PRIMARY_METRIC, Workload
from hyperloom_multibranch.specialist.judging import Judging
from hyperloom_multibranch.specialist.runtime import RuntimePacing, SpecialistRuntime

if TYPE_CHECKING:
    from hyperloom.measure.points import CanonicalMeasurement
    from hyperloom_multibranch.specialist.agent_session import AgentCommand
    from hyperloom_multibranch.specialist.model_sessions import ModelSessions

TOOLBOX_WRITER = "toolbox"
GATE_WRITER = "rule-gate"
BRIEFS_READER = "briefs"
CANONICAL_DIR = "canonical"
POINTS_DIR = "points"
#: A claim's lease on the specialist tools: the runtime renews it every 10 minutes (board #26).
CLAIM_TTL_SECONDS = 1800.0

#: Stands in for the image the local sandboxes run in: they run on this machine, in no image.
MB_FAKE_IMAGE_REF = "local"
#: Stands in for the Magpie and InferenceX commits the runner checks out; the instruction sheet refuses no commit
#: of this shape, but a runner cannot fetch it. Replaced when enablement or the command line name the real commits.
MB_FAKE_TOOL_COMMIT = "0" * 40


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
    the tests run in (``image_digest`` is also the base image every recipe must carry) and the Magpie and InferenceX
    commits the runner checks out."""

    cards: tuple[int, ...]
    image_ref: str
    image_digest: str
    magpie_commit: str
    inferencex_commit: str


def workload_environment(workload: Workload) -> Mapping[str, str]:
    """The workload as the process environment carries it to the main flow's config renderer, which the toolbox
    renders the canonical measurement and every test's points with, and to the kernel tuning the toolbox runs, which
    takes its backend from ``KERNEL_OPT_BACKEND_ORDER``; ``hyperloom optimize`` exports the same, the model resolved to
    the directory a server loads it from. The command line accepts only GPU types that are their own Magpie runner
    type, so ``GPU_TYPE`` and ``TARGET_GPU_TYPE`` are the same."""

    return {
        "FRAMEWORK": workload.framework,
        "MODEL_PATH": resolve_serving_model_path(workload.model),
        "TP": str(workload.tp),
        "ISL": str(workload.isl),
        "OSL": str(workload.osl),
        "PRECISION": workload.precision,
        "GPU_TYPE": workload.gpu_type,
        "TARGET_GPU_TYPE": workload.gpu_type,
        "KERNEL_OPT_BACKEND_ORDER": workload.kernel_optimizer.value,
    }


def measurement_config(workload: Workload, host: MeasurementHost, canonical: CanonicalMeasurement) -> dict[str, object]:
    """The toolbox's measurement config document (board #8): the base image every recipe must carry, and the workload,
    canonical surface, image and tool commits every test is measured on."""

    return {
        "base_image_digest": host.image_digest,
        "config": {
            "workload": encode(workload),
            "primary_concurrency": canonical.primary_concurrency,
            "surface": dict(canonical.surface.fields),
            "surface_sha256": canonical.surface.sha256,
            "image_ref": host.image_ref,
            "magpie_commit": host.magpie_commit,
            "inferencex_commit": host.inferencex_commit,
            "cards": list(host.cards),
        },
    }


def local_peers(host: MeasurementHost, sessions: SpecialistSessions, setup: PeerSetup) -> Peers:
    """The session's peers on this machine; exports :func:`workload_environment` into this process.

    Called on the event loop the session runs on. The toolbox is built, not started: the session starts it, and its
    close releases the backend and the sandboxes. The MCP service is started; the session stop stops it.
    """

    from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
    from hyperloom.measure.compile import EnvironmentCompiler
    from hyperloom.measure.hosted_spec import SpecContext
    from hyperloom.measure.points import default_canonical
    from hyperloom_multibranch.notebook.knowledge.learning_recall import LearningRecall
    from hyperloom_multibranch.notebook.learning_search import LEARNINGS_LOG, RetrievalLog
    from hyperloom_multibranch.toolbox.local_hosted import LocalSession, build_local_toolbox
    from hyperloom_multibranch.toolbox.mcp_tools import (
        McpSpecialistTokens,
        SpecialistToolOptions,
        register_specialist_tools,
    )
    from hyperloom_multibranch.specialist.tools import register_worktree_tools
    from hyperloom_multibranch.toolbox.planner import HostedPlanner
    from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

    layout, workload = setup.layout, setup.workload
    os.environ.update(workload_environment(workload))
    repo = GitExperimentRepo(layout.repo, setup.blobs)
    canonical = default_canonical(layout.toolbox / CANONICAL_DIR, primary_concurrency=workload.conc)
    atomic_write_json(layout.measurement_config, measurement_config(workload, host, canonical), indent=2)
    compiler = EnvironmentCompiler(setup.blobs, host.image_digest)
    spec = SpecContext(
        image_ref=host.image_ref,
        image_digest=host.image_digest,
        gpu_type=workload.gpu_type,
        magpie_commit=host.magpie_commit,
        inferencex_commit=host.inferencex_commit,
        blobs=setup.blobs,
        blob_root=str(setup.blobs.root),
    )
    toolbox = build_local_toolbox(
        LocalSession(layout.toolbox, layout.session_id, host.cards, workload.tp),
        LocalNotebook(layout.notebook, Writer(Role.TOOLBOX, TOOLBOX_WRITER)),
        AcceptanceSetup(repo, HostedPlanner(compiler, canonical, layout.toolbox / POINTS_DIR), host.image_digest),
        spec,
    )
    gate_notebook = LocalNotebook(layout.notebook, Writer(Role.GATE, GATE_WRITER))
    rule = InferenceXRule(gate_notebook, workload.conc)

    def runtime_notebook(branch: str) -> LocalNotebook:
        return LocalNotebook(layout.notebook, Writer(Role.SPECIALIST_RUNTIME, branch, branch))

    mcp = McpServer(McpServerConfig(host_token=secrets.token_hex(32)), campaign=layout.session_id)
    learnings = LearningRecall.from_env(log=RetrievalLog(layout.reports / LEARNINGS_LOG))
    briefs = SessionBriefs(
        LocalNotebook(layout.notebook, Writer(Role.OBSERVER, BRIEFS_READER)), metric=PRIMARY_METRIC, learnings=learnings
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
    judging = Judging(RuleGate(gate_notebook, repo, setup.desk, toolbox), toolbox, rule)
    runtime = SpecialistRuntime(
        runtime_notebook, judging, built.factory, pacing=RuntimePacing(claim_ttl_seconds=CLAIM_TTL_SECONDS)
    )
    mcp.start()
    return Peers(runtime, toolbox, repo, GateService(gate_notebook, toolbox, rule, toolbox), mcp, briefs)


def mb_fake_measurement_host() -> MeasurementHost:
    """The cards of one node as the toolbox counts them, the root recipe's base image and :data:`MB_FAKE_TOOL_COMMIT`
    for both tools.

    Stands in for the measurement host enablement or the command line will name; replaced once one of them does.
    """

    from hyperloom_multibranch.toolbox.local_hosted import MAX_CARDS_PER_NODE

    cards = tuple(range(MAX_CARDS_PER_NODE))
    return MeasurementHost(
        cards, MB_FAKE_IMAGE_REF, MB_FAKE_BASE_IMAGE_DIGEST, MB_FAKE_TOOL_COMMIT, MB_FAKE_TOOL_COMMIT
    )


def machine_parallel_tests(tp: int) -> int:
    """How many tests of ``tp`` cards the local toolbox runs at once on this machine's cards; raises ``ValueError``
    when a test of ``tp`` cards does not fit on them."""

    from hyperloom_multibranch.toolbox.local_hosted import max_parallel_tests

    return max_parallel_tests(len(mb_fake_measurement_host().cards), tp)


def specialist_command(env: Mapping[str, str]) -> AgentCommand:
    """The agent CLI every specialist of a session runs: the backend ``env`` is credentialed for, found on its
    ``PATH``, on the model ``env`` names for it; raises :class:`PeersUnavailable` when that CLI is not installed."""

    from hyperloom_multibranch.specialist.agent_session import AgentCommand

    backend = preferred_agent_backend(env)
    executable = shutil.which(backend, path=env.get("PATH"))
    if executable is None:
        raise PeersUnavailable(f"the {backend} CLI the specialists run is not on PATH")
    model = resolve_forge_llm_model(backend, env={**env, **provider_model_defaults(env)})
    return AgentCommand(backend, executable, model)


def _specialist_role(workload: Workload) -> str:
    """The workload as the specialists' role prompt names it."""

    return (
        f"{workload.model_basename} on {workload.framework}, {workload.precision}, TP {workload.tp}, "
        f"ISL {workload.isl}, OSL {workload.osl}, concurrency {workload.conc}, {workload.gpu_type}"
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


def load_peers() -> PeerFactory:
    """The factory of a session's real peers; raises :class:`PeersUnavailable` when this machine cannot run them.

    It creates nothing, so the command line calls it before the session directory exists.
    """

    return partial(local_peers, mb_fake_measurement_host(), model_backed_sessions(specialist_command(os.environ)))


__all__ = [
    "CLAIM_TTL_SECONDS",
    "GATE_WRITER",
    "MeasurementHost",
    "SessionWiring",
    "SpecialistSessions",
    "load_peers",
    "local_peers",
    "machine_parallel_tests",
    "mb_fake_measurement_host",
    "measurement_config",
    "model_backed_sessions",
    "specialist_command",
    "workload_environment",
]
