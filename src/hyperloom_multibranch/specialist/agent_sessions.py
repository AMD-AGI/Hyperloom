# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The autonomous branches of a session: what each one's run is built from.

Every agent's branch gets ports of its own: the branch first built takes the first port block, the next the following
one, and a branch keeps its block for the session, a restart included. Its sandbox (:mod:`agent_sandbox`) runs on the
cards and in the directory its preflight leases from the toolbox, with the lease's environment on top of the
session's; its onboarding model is the specialists' CLI (:mod:`agent_onboard`) with its spend recorded in the branch's
notebook, where the agent's own spend goes too, read from the transcripts of the Claude CLI's directory the session's
environment names (:mod:`agent_spend`). Given the gate's benchmark client (:mod:`gate_client`), each agent's spec names
it aimed at the branch's first port.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from hyperloom.measure.compile import COMPONENT_LAYOUTS
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import BranchLease
from hyperloom_multibranch.specialist.agent_onboard import cli_onboard
from hyperloom_multibranch.specialist.agent_onboarding import AgentSandbox, AgentSpec, OnboardingSetup
from hyperloom_multibranch.specialist.agent_sandbox import LocalAgentSandbox
from hyperloom_multibranch.specialist.agent_session import AgentCommand, BranchPlace, Processes, SessionNotebook
from hyperloom_multibranch.specialist.agent_spend import AgentSpend
from hyperloom_multibranch.specialist.autonomous_session import AGENT_DIR, AutonomousSession, AutonomousSetup
from hyperloom_multibranch.specialist.gate_client import GateClient
from hyperloom_multibranch.specialist.model_sessions import ModelSessions, branch_place
from hyperloom_multibranch.specialist.runtime import Session, SessionStart

if TYPE_CHECKING:
    from hyperloom_multibranch.toolbox.toolbox import RealToolbox

#: Where a Claude CLI keeps its state, transcripts included, unless the environment names another directory.
CLAUDE_CONFIG_ENV = "CLAUDE_CONFIG_DIR"
DEFAULT_CLAUDE_CONFIG = ".claude"
#: Ports a branch's agent may give its servers, from its spec's ``port`` on.
PORTS_PER_AGENT = 10
#: Where, in a branch's agent directory, the gate's client writes the results of the agent's measurements.
BENCH_DIR = "mb-bench"


@dataclass
class AgentSessions:
    """What a session's agents run on: ``spec``, the workload every agent is asked to optimize (the cards, the port, the
    start commit's arguments and the workspace are each branch's own); the onboarding ``command`` and its ``env``; the
    ``toolbox`` the sandboxes are leased from and imports go to, and the ``repo`` they are read against; the
    ``processes`` registry every agent process joins; and ``client``, the gate's benchmark client, ``None`` when the
    session has none to give."""

    spec: AgentSpec
    command: AgentCommand
    env: Mapping[str, str]
    toolbox: RealToolbox
    repo: ExperimentRepo
    processes: Processes
    client: GateClient | None = None
    _taken: dict[str, int] = field(default_factory=dict, init=False)

    def session(self, start: SessionStart, place: BranchPlace) -> AutonomousSession:
        """The run of the agent of ``start``'s branch, in ``place``."""

        branch, component = start.branch, self.spec.framework
        index = self._taken.setdefault(branch, len(self._taken))
        port = self.spec.port + index * PORTS_PER_AGENT
        client = "" if self.client is None else self.client.command(port, place.workspace / AGENT_DIR / BENCH_DIR)
        spec = replace(self.spec, port=port, client=client)
        source = place.worktree.path / PurePosixPath(COMPONENT_LAYOUTS[component].upstream_prefix).parent
        onboard = cli_onboard(self.command, self.env, branch, place.notebook)

        def sandbox(lease: BranchLease) -> AgentSandbox:
            base_env = {**os.environ, **lease.env}
            return LocalAgentSandbox(lease.gpu_ids, (source,), lease.root, self.processes, branch, base_env)

        def onboarding(agent_sandbox: AgentSandbox) -> OnboardingSetup:
            return OnboardingSetup(agent_sandbox, onboard)

        def spend(roots: tuple[Path, ...], ledger: Path) -> AgentSpend:
            config = Path(os.environ.get(CLAUDE_CONFIG_ENV) or Path.home() / DEFAULT_CLAUDE_CONFIG)
            return AgentSpend(config, roots, ledger, branch, self.command.model, place.notebook)

        setup = AutonomousSetup(onboarding, sandbox, spec, self.toolbox, self.repo, component, spend)
        return AutonomousSession(start, setup, place)


def with_agents(
    sessions: ModelSessions,
    agents: AgentSessions,
    *,
    branch_dir: Callable[[str], Path],
    notebook_for: Callable[[str], SessionNotebook],
) -> ModelSessions:
    """``sessions``, with a branch whose task names a third-party agent running that agent on ``agents``, in the
    branch's place that ``branch_dir`` and ``notebook_for`` give, as ``sessions`` places its specialists."""

    def factory(start: SessionStart) -> Session:
        if not start.task.agent:
            return sessions.factory(start)
        branch = start.branch
        place = branch_place(sessions.worktrees(branch), notebook_for(branch), branch_dir(branch))
        return agents.session(start, place)

    return ModelSessions(factory, sessions.worktrees)


__all__ = ["PORTS_PER_AGENT", "AgentSessions", "with_agents"]
