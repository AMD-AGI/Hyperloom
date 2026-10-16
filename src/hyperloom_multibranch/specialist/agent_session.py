# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist backed by a model: agent CLI sessions launched through the AI work socket, one after another for as
long as its branch runs.

Every session starts at the branch head, with the worktree synced to it and a launch prompt built from the startup
brief and the branch log. A session that ends as a session should, its model's turn done or its deadline reached, is
followed by a fresh one: a branch runs until the scheduler stops it. One that is lost, exits with an error or ends
within ``min_session_seconds`` raises, and the runtime decides whether to restart it. The session's process registry
adopts every process the socket starts, so the session stop kills it even when this stop cannot.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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


class AgentSession:
    """One run of a model-backed specialist on ``branch``: CLI sessions until it is stopped or one crashes.

    ``env`` is the environment the CLI runs with, the branch's MCP token added; the session decides nothing about it.
    """

    def __init__(
        self,
        branch: str,
        start_commit: str,
        k: int,
        brief: str,
        *,
        command: AgentCommand,
        role: str,
        env: Mapping[str, str],
        runner: AgentRunner,
        tokens: SpecialistTokens,
        processes: Processes,
        repo: ExperimentRepo,
        worktree: BranchWorktree,
        notebook: BranchLogStore,
        workspace: Path,
        session_seconds: float = LONG_RUNNING_SESSION_SEC,
        min_session_seconds: float = MIN_SESSION_S,
        poll_seconds: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._branch, self._start_commit, self._k, self._brief = branch, start_commit, k, brief
        self._command, self._role, self._env, self._runner = command, role, env, runner
        self._tokens, self._processes, self._repo = tokens, processes, repo
        self._worktree, self._notebook, self._workspace = worktree, notebook, workspace
        self._session_seconds, self._min_session_seconds = session_seconds, min_session_seconds
        self._poll_seconds, self._clock = poll_seconds, clock
        self._stopped = asyncio.Event()
        self._handle: AgentHandle | None = None

    async def run(self) -> None:
        grant = self._tokens.issue(self._branch, frozenset({SCOPE_SPECIALIST}))
        last_end = ""
        while not self._stopped.is_set():
            last_end = await self._session(grant, last_end)

    async def set_k(self, k: int) -> None:
        self._k = k

    async def stop(self) -> None:
        self._stopped.set()
        # Taking the handle makes this stop, not the run waiting on it, the one that ends the CLI.
        handle, self._handle = self._handle, None
        if handle is not None:
            await self._finish(handle, cancel=True)
        self._tokens.revoke(self._branch)

    async def _session(self, grant: SpecialistGrant, last_end: str) -> str:
        """Run one CLI session to its end and say how it ended; raises :class:`SessionCrashed` when it did not work."""
        head = await sync_to_head(self._repo, self._branch, self._worktree)
        recent = await self._notebook.branch_log(self._branch)
        prompt = launch_prompt(
            branch=self._branch,
            start_commit=self._start_commit,
            head=head,
            k=self._k,
            worktree=str(self._worktree.path),
            brief=self._brief,
            recent=recent,
            last_end=last_end,
        )
        spec = await asyncio.to_thread(self._spec, grant, prompt)
        started = self._clock()
        handle = self._handle = await self._runner.launch(spec)
        self._adopt(handle, spec.argv)
        status = await self._wait(handle)
        if self._handle is not handle:
            return last_end
        self._handle = None
        await self._finish(handle, cancel=status.state != "exited")
        seconds = self._clock() - started
        if self._stopped.is_set():
            return last_end
        if (
            status.state == STATE_LOST
            or (status.returncode and not status.timed_out)
            or seconds < self._min_session_seconds
        ):
            raise SessionCrashed(_describe(status, seconds))
        return "it reached its deadline" if status.timed_out else "its model finished its turn"

    def _spec(self, grant: SpecialistGrant, prompt: str) -> AgentLaunchSpec:
        self._workspace.mkdir(parents=True, exist_ok=True)
        done = self._workspace / DONE_FILE
        done.unlink(missing_ok=True)
        role_file = self._workspace / ROLE_FILE
        role_file.write_bytes(self._role.encode())
        (self._workspace / PROMPT_FILE).write_bytes(self._command.stdin(self._role, prompt).encode())
        return AgentLaunchSpec(
            task_id=self._branch,
            workspace=self._workspace,
            argv=self._command.argv(role_file, self._worktree.path),
            env={**self._env, grant.token_env: grant.token},
            cwd=self._worktree.path,
            gpus=0,
            deadline_unix=self._clock() + self._session_seconds,
            done_paths=(done,),
            heartbeat_path=self._workspace / HEARTBEAT_FILE,
            mcp_servers=(McpEndpoint(MCP_SERVER, grant.url, grant.token_env),),
            backend=self._command.backend,
            heartbeat_sink=grant.heartbeat_sink,
        )

    def _adopt(self, handle: AgentHandle, argv: Sequence[str]) -> None:
        try:
            self._processes.adopt(handle.pid, branch=self._branch, argv=argv)
        except ValueError:
            logger.warning(
                "the agent CLI of %s ended before the process registry took it over", self._branch, exc_info=True
            )

    async def _wait(self, handle: AgentHandle) -> AgentStatus:
        while True:
            status = await self._runner.poll(handle)
            if status.state != _RUNNING or self._stopped.is_set():
                return status
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), self._poll_seconds)

    async def _finish(self, handle: AgentHandle, *, cancel: bool) -> None:
        if cancel:
            await self._runner.cancel(handle)
        await self._runner.collect(handle)


def _describe(status: AgentStatus, seconds: float) -> str:
    if status.state == STATE_LOST:
        return f"the session was lost ({status.error or 'no heartbeat'})"
    if status.returncode and not status.timed_out:
        return f"the agent CLI exited {status.returncode}" + (f" ({status.error})" if status.error else "")
    return f"the session ended after {seconds:.0f}s, too soon to have worked"


__all__ = ["MCP_SERVER", "MIN_SESSION_S", "AgentCommand", "AgentSession", "Processes", "SessionCrashed"]
