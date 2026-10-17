# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist backed by a model: agent CLI sessions launched through the AI work socket, one after another for as
long as its branch runs.

Every session starts at the branch head, with the worktree synced to it and a launch prompt built from the brief,
rebuilt at that head when the setup can, and the branch log. A session that ends as a session should, its model's turn done or its deadline reached, is
followed by a fresh one: a branch runs until the scheduler stops it. One that is lost, exits with an error or ends
within ``min_session_seconds`` raises, and the runtime decides whether to restart it. The session's process registry
adopts every process the socket starts, so the session stop kills it even when this stop cannot.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from hyperloom.common.llm_config import AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX
from hyperloom.orchestrator.specialists.agent_runner import (
    LONG_RUNNING_SESSION_SEC,
    STATE_LOST,
    AgentHandle,
    AgentLaunchSpec,
    AgentRunner,
    AgentStatus,
    McpEndpoint,
)
from hyperloom_multibranch.contracts.notebook.interfaces import BranchLogStore
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST, SpecialistGrant, SpecialistTokens
from hyperloom_multibranch.specialist.prompts import launch_prompt
from hyperloom_multibranch.specialist.runtime import SessionStart
from hyperloom_multibranch.specialist.tools import sync_to_head
from hyperloom_multibranch.specialist.worktree import BranchWorktree

logger = logging.getLogger(__name__)

#: The name the session's MCP service has in the CLI's configuration.
MCP_SERVER = "hyperloom"
#: A session ending sooner than this did not work, whatever its exit code: a misconfigured CLI must not relaunch forever.
MIN_SESSION_S = 120.0
PROMPT_FILE, ROLE_FILE, DONE_FILE, HEARTBEAT_FILE = (
    "prompt.md",
    "system_prompt.md",
    "specialist_done.json",
    "heartbeat.json",
)
_RUNNING = "running"


class SessionCrashed(RuntimeError):
    """A session of the specialist ended the way no working session ends."""


class Processes(Protocol):
    def adopt(self, pid: int, *, branch: str, argv: Sequence[str]) -> object: ...


@dataclass(frozen=True)
class AgentCommand:
    """The agent CLI a specialist runs."""

    backend: str
    executable: str
    model: str = ""
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.backend not in (AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX):
            raise ValueError(f"a specialist runs claude or codex, not {self.backend!r}")

    def argv(self, role_file: Path, worktree: Path) -> tuple[str, ...]:
        """The command line; the socket adds the MCP configuration."""
        if self.backend == AGENT_BACKEND_CODEX:
            model = ("-m", self.model) if self.model else ()
            bypass = "--dangerously-bypass-approvals-and-sandbox"
            return (
                self.executable,
                "exec",
                "--json",
                "--skip-git-repo-check",
                bypass,
                "-C",
                str(worktree),
                *model,
                *self.extra_args,
                "-",
            )
        model = ("--model", self.model) if self.model else ()
        role = ("--system-prompt-file", str(role_file))
        return (
            self.executable,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
            *role,
            *model,
            "--add-dir",
            str(worktree),
            *self.extra_args,
        )

    def stdin(self, role: str, prompt: str) -> str:
        """What the CLI reads: codex takes no system prompt file, so its role opens the prompt."""
        return f"{role}\n\n{prompt}" if self.backend == AGENT_BACKEND_CODEX else prompt


@dataclass(frozen=True)
class AgentSetup:
    """What every model-backed session of a session shares: the CLI and its role, and what a session reaches.

    ``env`` is the environment the CLI runs with, the branch's MCP token added; the session decides nothing about it.
    ``briefs(branch, head)`` rebuilds a branch's brief at the head a launch starts at; without it, every launch shows
    the brief the branch was started with.
    """

    command: AgentCommand
    role: str
    env: Mapping[str, str]
    runner: AgentRunner
    tokens: SpecialistTokens
    processes: Processes
    repo: ExperimentRepo
    briefs: Callable[[str, str], Awaitable[str]] | None = None
    session_seconds: float = LONG_RUNNING_SESSION_SEC
    min_session_seconds: float = MIN_SESSION_S
    poll_seconds: float = 5.0
    clock: Callable[[], float] = time.time


@dataclass(frozen=True)
class BranchPlace:
    """Where one branch's sessions work: its worktree, its log, and the directory a session's files go in."""

    worktree: BranchWorktree
    notebook: BranchLogStore
    workspace: Path


class AgentSession:
    """One run of a model-backed specialist: CLI sessions on its branch until it is stopped or one crashes."""

    def __init__(self, start: SessionStart, setup: AgentSetup, place: BranchPlace) -> None:
        self._start, self._setup, self._place = start, setup, place
        self._stopped = asyncio.Event()
        self._handle: AgentHandle | None = None

    async def run(self) -> None:
        grant = self._setup.tokens.issue(self._start.branch, frozenset({SCOPE_SPECIALIST}))
        last_end = ""
        while not self._stopped.is_set():
            last_end = await self._session(grant, last_end)

    async def set_k(self, k: int) -> None:
        self._start = replace(self._start, k=k)

    async def stop(self) -> None:
        self._stopped.set()
        # Taking the handle makes this stop, not the run waiting on it, the one that ends the CLI.
        handle, self._handle = self._handle, None
        if handle is not None:
            await self._finish(handle, cancel=True)
        self._setup.tokens.revoke(self._start.branch)

    async def _session(self, grant: SpecialistGrant, last_end: str) -> str:
        """Run one CLI session to its end and say how it ended; raises :class:`SessionCrashed` when it did not work."""
        branch, worktree, clock = self._start.branch, self._place.worktree, self._setup.clock
        head = await sync_to_head(self._setup.repo, branch, worktree)
        recent = await self._place.notebook.branch_log(branch)
        start = await self._briefed(head)
        prompt = launch_prompt(start, head=head, worktree=str(worktree.path), recent=recent, last_end=last_end)
        spec = await asyncio.to_thread(self._spec, grant, prompt)
        started = clock()
        handle = self._handle = await self._setup.runner.launch(spec)
        self._adopt(handle, spec.argv)
        status = await self._wait(handle)
        if self._handle is not handle:
            return last_end
        self._handle = None
        await self._finish(handle, cancel=status.state != "exited")
        seconds = clock() - started
        if self._stopped.is_set():
            return last_end
        if (
            status.state == STATE_LOST
            or (status.returncode and not status.timed_out)
            or seconds < self._setup.min_session_seconds
        ):
            raise SessionCrashed(_describe(status, seconds))
        return "it reached its deadline" if status.timed_out else "its model finished its turn"

    async def _briefed(self, head: str) -> SessionStart:
        """The start a launch at ``head`` shows: its brief rebuilt there, or the first one when that fails."""
        if self._setup.briefs is None:
            return self._start
        branch = self._start.branch
        try:
            brief = await self._setup.briefs(branch, head)
        except Exception:
            logger.warning(
                "the brief of %s could not be rebuilt at %s; the launch shows the first one",
                branch,
                head,
                exc_info=True,
            )
            return self._start
        return replace(self._start, brief=brief)

    def _spec(self, grant: SpecialistGrant, prompt: str) -> AgentLaunchSpec:
        setup, workspace, worktree = self._setup, self._place.workspace, self._place.worktree.path
        workspace.mkdir(parents=True, exist_ok=True)
        done = workspace / DONE_FILE
        done.unlink(missing_ok=True)
        role_file = workspace / ROLE_FILE
        role_file.write_bytes(setup.role.encode())
        (workspace / PROMPT_FILE).write_bytes(setup.command.stdin(setup.role, prompt).encode())
        return AgentLaunchSpec(
            task_id=self._start.branch,
            workspace=workspace,
            argv=setup.command.argv(role_file, worktree),
            env={**setup.env, grant.token_env: grant.token},
            cwd=worktree,
            gpus=0,
            deadline_unix=setup.clock() + setup.session_seconds,
            done_paths=(done,),
            heartbeat_path=workspace / HEARTBEAT_FILE,
            mcp_servers=(McpEndpoint(MCP_SERVER, grant.url, grant.token_env),),
            backend=setup.command.backend,
            heartbeat_sink=grant.heartbeat_sink,
        )

    def _adopt(self, handle: AgentHandle, argv: Sequence[str]) -> None:
        branch = self._start.branch
        try:
            self._setup.processes.adopt(handle.pid, branch=branch, argv=argv)
        except ValueError:
            logger.warning("the agent CLI of %s ended before the process registry took it over", branch, exc_info=True)

    async def _wait(self, handle: AgentHandle) -> AgentStatus:
        while True:
            status = await self._setup.runner.poll(handle)
            if status.state != _RUNNING or self._stopped.is_set():
                return status
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), self._setup.poll_seconds)

    async def _finish(self, handle: AgentHandle, *, cancel: bool) -> None:
        if cancel:
            await self._setup.runner.cancel(handle)
        await self._setup.runner.collect(handle)


def _describe(status: AgentStatus, seconds: float) -> str:
    if status.state == STATE_LOST:
        return f"the session was lost ({status.error or 'no heartbeat'})"
    if status.returncode and not status.timed_out:
        return f"the agent CLI exited {status.returncode}" + (f" ({status.error})" if status.error else "")
    return f"the session ended after {seconds:.0f}s, too soon to have worked"


__all__ = [
    "MCP_SERVER",
    "MIN_SESSION_S",
    "AgentCommand",
    "AgentSession",
    "AgentSetup",
    "BranchPlace",
    "Processes",
    "SessionCrashed",
]
