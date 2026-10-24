# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist runtime's model-backed sessions, as a session's peers build them.

A session has one source base, laid out from the installed packages of ``components`` when the sessions are built (a
resumed session finds it in place), and each branch a worktree and a workspace under its directory. ``factory`` gives
the runtime an :class:`AgentSession` per run of a branch's specialist, and refuses a branch that runs a third-party
agent (:func:`~hyperloom_multibranch.specialist.agent_sessions.with_agents` runs those); ``worktrees`` is what the
worktree tools serve, so a specialist's tools and its sessions edit the same tree.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom_multibranch.specialist.agent_session import AgentSession, AgentSetup, BranchPlace, SessionNotebook
from hyperloom_multibranch.specialist.runtime import SessionFactory, SessionStart
from hyperloom_multibranch.specialist.tools import WorktreeFor
from hyperloom_multibranch.specialist.worktree import BranchWorktree, SourceBase

WORKTREE_DIR = "worktree"
SESSION_DIR = "session"


@dataclass(frozen=True)
class ModelSessions:
    factory: SessionFactory
    worktrees: WorktreeFor


def model_sessions(
    setup: AgentSetup,
    *,
    base: SourceBase,
    components: Sequence[str],
    branch_dir: Callable[[str], Path],
    notebook_for: Callable[[str], SessionNotebook],
) -> ModelSessions:
    """The sessions of one session's specialists; ``setup.role`` is the
    :func:`~hyperloom_multibranch.specialist.prompts.role_prompt` of the session's workload."""
    if not base.path.exists():
        base.build(components)

    def worktrees(branch: str) -> BranchWorktree:
        return base.worktree(branch, branch_dir(branch) / WORKTREE_DIR)

    def factory(start: SessionStart) -> AgentSession:
        if start.task.agent:
            raise ValueError(f"branch {start.branch} runs the agent {start.task.agent}, which no specialist stands for")
        branch = start.branch
        return AgentSession(start, setup, branch_place(worktrees(branch), notebook_for(branch), branch_dir(branch)))

    return ModelSessions(factory, worktrees)


def branch_place(worktree: BranchWorktree, notebook: SessionNotebook, branch_dir: Path) -> BranchPlace:
    """Where a branch's runs work: its worktree, its notebook handle and the workspace in its directory."""
    return BranchPlace(worktree, notebook, branch_dir / SESSION_DIR)


__all__ = ["SESSION_DIR", "WORKTREE_DIR", "ModelSessions", "branch_place", "model_sessions"]
