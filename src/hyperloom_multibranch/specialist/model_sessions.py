# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist runtime's model-backed sessions, as a session's peers build them.

A session has one source base, laid out from the installed packages of ``components`` when the sessions are built (a
resumed session finds it in place), and each branch a worktree and a workspace under its directory. ``factory`` gives
the runtime an :class:`AgentSession` per run of a branch's specialist; ``worktrees`` is what the worktree tools serve,
so a specialist's tools and its sessions edit the same tree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.measure.compile import PackageLocator, locate_installed_package
from hyperloom.orchestrator.specialists.agent_runner import AgentRunner
from hyperloom_multibranch.contracts.notebook.interfaces import BranchLogStore
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import SpecialistTokens
from hyperloom_multibranch.specialist.agent_session import AgentCommand, AgentSession, Processes
from hyperloom_multibranch.specialist.prompts import role_prompt
from hyperloom_multibranch.specialist.runtime import SessionFactory
from hyperloom_multibranch.specialist.tools import WorktreeFor
from hyperloom_multibranch.specialist.worktree import BranchWorktree, SourceBase

WORKTREE_DIR = "worktree"
SESSION_DIR = "session"


@dataclass(frozen=True)
class ModelSessions:
    factory: SessionFactory
    worktrees: WorktreeFor


def model_sessions(
    *,
    source: Path,
    branch_dir: Callable[[str], Path],
    components: Sequence[str],
    workload: str,
    command: AgentCommand,
    env: Mapping[str, str],
    runner: AgentRunner,
    tokens: SpecialistTokens,
    processes: Processes,
    repo: ExperimentRepo,
    notebook_for: Callable[[str], BranchLogStore],
    locator: PackageLocator = locate_installed_package,
) -> ModelSessions:
    """The sessions of one session's specialists; ``source`` is where the source base goes."""
    base = SourceBase(source, locator=locator)
    if not source.exists():
        base.build(components)
    role = role_prompt(workload)

    def worktrees(branch: str) -> BranchWorktree:
        return base.worktree(branch, branch_dir(branch) / WORKTREE_DIR)

    def factory(branch: str, start_commit: str, k: int, brief: str) -> AgentSession:
        return AgentSession(
            branch,
            start_commit,
            k,
            brief,
            command=command,
            role=role,
            env=env,
            runner=runner,
            tokens=tokens,
            processes=processes,
            repo=repo,
            worktree=worktrees(branch),
            notebook=notebook_for(branch),
            workspace=branch_dir(branch) / SESSION_DIR,
        )

    return ModelSessions(factory, worktrees)


__all__ = ["SESSION_DIR", "WORKTREE_DIR", "ModelSessions", "model_sessions"]
