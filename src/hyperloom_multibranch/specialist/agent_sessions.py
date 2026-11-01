# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The autonomous branches of a session: what each one's run is built from.

Every agent's branch gets ports of its own: the branch first built takes the first port block, the next the following
one, and a branch keeps its block for the session, a restart included. Its sandbox runs on the cards its preflight
leases from the toolbox: on this host (:class:`LocalAgentSandboxes`), in the lease's directory with the lease's
environment on top of the session's, or, on the hosted backend, in the leased ``dev`` sandbox
(:class:`RemoteAgentSandboxes`), rooted in the branch's session directory. Its onboarding model is the specialists' CLI
(:mod:`agent_onboard`) with its spend recorded in the branch's blackboard, where the agent's own spend goes too, read
from the transcripts of the Claude CLI's directory the sandbox gives the agent (:mod:`agent_spend`). Given the gate's
benchmark client (:mod:`gate_client`), each agent's spec names it aimed at the branch's first port.

The session's dossier pass (:mod:`dossier_pass`) runs on the same: a port block of its own, a sandbox of a lease, a
worktree and a directory beside the branches', under :data:`~.dossier_pass.DOSSIER_HOLDER`. The dossier it writes is
given to every agent branch built after it, which copies it into its checkout; a run given a reviewed map and its
dossier runs no pass, and gives every branch that dossier.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

from hyperloom.measure.compile import COMPONENT_LAYOUTS
from hyperloom.sandbox import Sandbox, SandboxGone
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import BranchLease
from hyperloom_multibranch.specialist.agent_onboard import cli_onboard
from hyperloom_multibranch.specialist.agent_onboarding import AgentSandbox, AgentSpec, OnboardingSetup
from hyperloom_multibranch.specialist.agent_sandbox import LocalAgentSandbox
from hyperloom_multibranch.specialist.agent_session import AgentCommand, BranchPlace, Processes, SessionBlackboard
from hyperloom_multibranch.specialist.agent_spend import AgentSpend
from hyperloom_multibranch.specialist.autonomous_session import AGENT_DIR, AutonomousSession, AutonomousSetup
from hyperloom_multibranch.specialist.dossier_pass import (
    DOSSIER_HOLDER,
    DossierOutcome,
    DossierSetup,
    DossierStart,
    run_dossier_pass,
)
from hyperloom_multibranch.specialist.gate_client import GateClient
from hyperloom_multibranch.specialist.model_sessions import ModelSessions, branch_place
from hyperloom_multibranch.specialist.remote_sandbox import RemoteAgentSandbox, claude_config_dir, forwarded_env
from hyperloom_multibranch.specialist.runtime import Session, SessionStart
from hyperloom_multibranch.specialist.worktree import WorktreeError
from hyperloom_multibranch.toolbox.hosted_leases import HostedBranchLease

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

#: Where a Claude CLI keeps its state, transcripts included, unless the environment names another directory.
CLAUDE_CONFIG_ENV = "CLAUDE_CONFIG_DIR"
DEFAULT_CLAUDE_CONFIG = ".claude"
#: Ports a branch's agent may give its servers, from its spec's ``port`` on.
PORTS_PER_AGENT = 10
#: Where, in a branch's agent directory, the gate's client writes the results of the agent's measurements.
BENCH_DIR = "mb-bench"
#: Where, in a branch's session directory, the root of its agent's hosted sandbox is.
SANDBOX_DIR = "sandbox"


class AgentSandboxes(Protocol):
    """The sandboxes a session's agents run in."""

    def sandbox(self, branch: str, place: BranchPlace, source: Path, lease: BranchLease) -> AgentSandbox:
        """The sandbox of ``lease`` for the agent of ``branch`` in ``place``, importing ``source`` first."""

    def claude_config(self, place: BranchPlace) -> Path:
        """Where the Claude CLI of the agent in ``place`` keeps its transcripts."""


@dataclass(frozen=True)
class LocalAgentSandboxes:
    """Sandboxes on this host, every process joining ``processes``."""

    processes: Processes

    def sandbox(self, branch: str, place: BranchPlace, source: Path, lease: BranchLease) -> AgentSandbox:
        base_env = {**os.environ, **lease.env}
        return LocalAgentSandbox(lease.gpu_ids, (source,), lease.root, self.processes, branch, base_env)

    def claude_config(self, place: BranchPlace) -> Path:
        return Path(os.environ.get(CLAUDE_CONFIG_ENV) or Path.home() / DEFAULT_CLAUDE_CONFIG)


@dataclass
class RemoteAgentSandboxes:
    """The leased ``dev`` sandboxes of the hosted backend. A run after a crash finds its lease's sandbox here by the
    lease's id, as the lease file it reads it from holds no handle; a lease this process did not grant has none here,
    and its sandbox counts as ended."""

    _handles: dict[str, Sandbox] = field(default_factory=dict, init=False)

    def sandbox(self, branch: str, place: BranchPlace, source: Path, lease: BranchLease) -> AgentSandbox:
        """Raises :class:`~hyperloom.sandbox.SandboxGone` when no sandbox of ``lease`` is known."""

        if isinstance(lease, HostedBranchLease):
            self._handles[lease.lease_id] = lease.sandbox
        handle = self._handles.get(lease.lease_id)
        if handle is None:
            raise SandboxGone(f"the sandbox of lease {lease.lease_id} is not this process's")
        root = place.workspace / SANDBOX_DIR
        return RemoteAgentSandbox(handle, root, (source,), len(lease.gpu_ids), forwarded_env(os.environ))

    def claude_config(self, place: BranchPlace) -> Path:
        return claude_config_dir(place.workspace / SANDBOX_DIR)


@dataclass
class AgentSessions:
    """What a session's agents run on: ``spec``, the workload every agent is asked to optimize (the cards, the port, the
    start commit's arguments and the workspace are each branch's own); the onboarding ``command`` and its ``env``; the
    ``toolbox`` the sandboxes are leased from and imports go to, and the ``repo`` they are read against; the
    ``sandboxes`` the agents run in; and ``client``, the gate's benchmark client, ``None`` when the session has none to
    give. ``dossier`` is the dossier every agent branch is given: the one the run was given with its reviewed map, or
    the one the session's dossier pass wrote, once it has."""

    spec: AgentSpec
    command: AgentCommand
    env: Mapping[str, str]
    toolbox: RealToolbox
    repo: ExperimentRepo
    sandboxes: AgentSandboxes
    client: GateClient | None = None
    dossier: str | None = None
    _taken: dict[str, int] = field(default_factory=dict, init=False)

    def session(self, start: SessionStart, place: BranchPlace) -> AutonomousSession:
        """The run of the agent of ``start``'s branch, in ``place``, with the branch's share of the session's
        directions when it has one, and the session's dossier when its dossier pass wrote one."""

        branch, component = start.branch, self.spec.framework
        spec = replace(self._placed(branch, place), plan=start.task.plan or None, dossier=self.dossier)
        onboard = cli_onboard(self.command, self.env, branch, place.blackboard)
        sandbox = self._sandbox(branch, place)

        def onboarding(agent_sandbox: AgentSandbox) -> OnboardingSetup:
            return OnboardingSetup(agent_sandbox, onboard)

        def spend(roots: tuple[Path, ...], ledger: Path) -> AgentSpend:
            config = self.sandboxes.claude_config(place)
            return AgentSpend(config, roots, ledger, branch, self.command.model, place.blackboard)

        setup = AutonomousSetup(onboarding, sandbox, spec, self.toolbox, self.repo, component, spend)
        return AutonomousSession(start, setup, place)

    async def dossier_pass(self, start: DossierStart, place: Callable[[], BranchPlace]) -> DossierOutcome:
        """Run the session's dossier pass of ``start`` in the place ``place`` makes when it starts, and keep the
        dossier it wrote for the branches built after it."""

        try:
            here = place()
        except (WorktreeError, OSError) as error:
            return DossierOutcome(None, None, f"its worktree could not be made: {error}")
        spec, sandbox = self._placed(DOSSIER_HOLDER, here), self._sandbox(DOSSIER_HOLDER, here)
        setup = DossierSetup(spec, self.repo, self.toolbox, sandbox, here.worktree, here.workspace / AGENT_DIR)
        outcome = await run_dossier_pass(start, setup)
        if outcome.dossier is not None:
            self.dossier = str(outcome.dossier)
        return outcome

    def _placed(self, holder: str, place: BranchPlace) -> AgentSpec:
        """The session's spec with ``holder``'s port block, which it keeps for the session, and the gate's client
        aimed at its first port."""

        index = self._taken.setdefault(holder, len(self._taken))
        port = self.spec.port + index * PORTS_PER_AGENT
        client = "" if self.client is None else self.client.command(port, place.workspace / AGENT_DIR / BENCH_DIR)
        return replace(self.spec, port=port, client=client)

    def _sandbox(self, holder: str, place: BranchPlace) -> Callable[[BranchLease], AgentSandbox]:
        """The sandbox of a lease for ``holder`` in ``place``, the framework's source in its worktree imported first."""

        layout = COMPONENT_LAYOUTS[self.spec.framework]
        source = place.worktree.path / PurePosixPath(layout.upstream_prefix).parent

        def sandbox(lease: BranchLease) -> AgentSandbox:
            return self.sandboxes.sandbox(holder, place, source, lease)

        return sandbox


def with_agents(
    sessions: ModelSessions,
    agents: AgentSessions,
    *,
    branch_dir: Callable[[str], Path],
    blackboard_for: Callable[[str], SessionBlackboard],
) -> ModelSessions:
    """``sessions``, with a branch whose task names a third-party agent running that agent on ``agents``, in the
    branch's place that ``branch_dir`` and ``blackboard_for`` give, as ``sessions`` places its specialists."""

    def factory(start: SessionStart) -> Session:
        if not start.task.agent:
            return sessions.factory(start)
        return agents.session(start, placed(start.branch))

    def placed(holder: str) -> BranchPlace:
        return branch_place(sessions.worktrees(holder), blackboard_for(holder), branch_dir(holder))

    async def dossier_pass(start: DossierStart) -> DossierOutcome:
        return await agents.dossier_pass(start, partial(placed, DOSSIER_HOLDER))

    return ModelSessions(factory, sessions.worktrees, dossier_pass)


__all__ = [
    "PORTS_PER_AGENT",
    "AgentSandboxes",
    "AgentSessions",
    "LocalAgentSandboxes",
    "RemoteAgentSandboxes",
    "with_agents",
]
