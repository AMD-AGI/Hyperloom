# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist backed by a model: agent CLI sessions launched through the AI work socket, one after another for as
long as its branch runs.

Every session starts at the branch head, with the worktree synced to it and a launch prompt built from the brief,
rebuilt at that head when the setup can, and the branch log. A Claude CLI keeps its stdin open and lives across turns:
when a turn ends, the session waits, keeping its heartbeat, until something arrives for the branch (an inbox item or a
runtime notice) and sends it to the CLI as the next message, so the model goes on with its whole conversation. A turn
that ends with no test of the branch running and nothing arrived is nudged; ``IDLE_TURNS`` such turns in a row raise.
A session that ends as a session should, its CLI exiting or its deadline reached, is followed by a fresh one: a
branch runs until the scheduler stops it. One that ends within ``min_session_seconds`` while its branch has a test in
flight ended to wait for it: the next launch starts once that test has its result. One that is lost, exits with an
error or otherwise ends within ``min_session_seconds`` raises, and the runtime decides whether to restart it. Claude
streams partial messages into its log, so a long reasoning step shows as log growth and is not taken for a lost
session. The session's process registry adopts every process the socket starts, so the session stop kills it even
when this stop cannot. The spend of each turn, its logged tokens at the gateway's rates, is written to the branch's
notebook when the turn ends and when the launch is collected, however it ended.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Protocol

from hyperloom.common.llm_config import AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX
from hyperloom.orchestrator.specialists.agent_runner import (
    LONG_RUNNING_SESSION_SEC,
    PROCESS_LOG,
    STATE_LOST,
    AgentEnded,
    AgentHandle,
    AgentLaunchSpec,
    AgentStatus,
    Beat,
    HeartbeatSink,
    InteractiveAgentRunner,
    McpEndpoint,
)
from hyperloom_multibranch.contracts.llm_prices import TokenCounts, price_of
from hyperloom_multibranch.contracts.notebook.interfaces import BranchLogStore, Inbox, UsageSink
from hyperloom_multibranch.contracts.notebook.records import LlmUsage, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import UsageSource
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST, SpecialistGrant, SpecialistTokens
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.llm_usage import launch_usage
from hyperloom_multibranch.specialist.prompts import launch_prompt, nudge_message, wake_message
from hyperloom_multibranch.specialist.runtime import SessionCrashed, SessionStart
from hyperloom_multibranch.specialist.tools import sync_to_head
from hyperloom_multibranch.specialist.worktree import BranchWorktree

logger = logging.getLogger(__name__)

#: The name the session's MCP service has in the CLI's configuration.
MCP_SERVER = "hyperloom"
#: A session ending sooner than this did not work, whatever its exit code: a misconfigured CLI must not relaunch forever.
MIN_SESSION_S = 120.0
#: Turns in a row a model may end with no test of its branch running and nothing arrived, each nudged, before its
#: session raises: a model that keeps stopping must not be woken forever.
IDLE_TURNS = 3
#: A session silent this long, with no log growth, CPU work or tool call in flight, is hung: a test takes about eight
#: minutes and one reasoning step can take as long, and Claude streams its reasoning into the log.
SPECIALIST_LOST_AFTER_S = 600.0
#: How a launch that ended early to wait for its branch's test ended, as the next launch is told.
WAITED_END = "its model ended its turn to wait for a test, which has its result now"
PROMPT_FILE, ROLE_FILE, DONE_FILE, HEARTBEAT_FILE = (
    "prompt.md",
    "system_prompt.md",
    "specialist_done.json",
    "heartbeat.json",
)
_RUNNING = "running"


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

    @property
    def interactive(self) -> bool:
        """Whether one CLI takes message after message on its stdin; codex exec reads one prompt and ends."""
        return self.backend == AGENT_BACKEND_CLAUDE

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
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--permission-mode",
            "bypassPermissions",
            *role,
            *model,
            "--add-dir",
            str(worktree),
            *self.extra_args,
        )

    def stdin(self, role: str, prompt: str) -> str:
        """What the CLI reads first: codex takes no system prompt file, so its role opens the prompt."""
        return self.message(prompt) if self.interactive else f"{role}\n\n{prompt}"

    @staticmethod
    def message(text: str) -> str:
        """``text`` as one stream-json user message, the line an interactive CLI reads as its next turn."""
        content = [{"type": "text", "text": text}]
        return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


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
    runner: InteractiveAgentRunner
    tokens: SpecialistTokens
    processes: Processes
    repo: ExperimentRepo
    briefs: Callable[[str, str], Awaitable[str]] | None = None
    session_seconds: float = LONG_RUNNING_SESSION_SEC
    min_session_seconds: float = MIN_SESSION_S
    poll_seconds: float = 5.0
    clock: Callable[[], float] = time.time


class SessionNotebook(BranchLogStore, UsageSink, Inbox, Protocol):
    """A branch's runtime handle as its sessions use it: the branch log, the spend of each turn, the inbox that wakes
    a CLI whose turn ended, and the snapshot that says whether the branch has a test in flight."""

    async def snapshot(self) -> NotebookSnapshot: ...


@dataclass(frozen=True)
class BranchPlace:
    """Where one branch's sessions work: its worktree, its log and spend, and the directory a session's files go in."""

    worktree: BranchWorktree
    notebook: SessionNotebook
    workspace: Path


class AgentSession:
    """One run of a model-backed specialist: CLI sessions on its branch until it is stopped or one crashes."""

    def __init__(self, start: SessionStart, setup: AgentSetup, place: BranchPlace) -> None:
        self._start, self._setup, self._place = start, setup, place
        self._stopped = asyncio.Event()
        self._handle: AgentHandle | None = None
        self._beat: Beat | None = None
        self._recorded: Mapping[str, TokenCounts] = {}

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
        """Run one CLI session to its end and say how it ended; raises :class:`SessionCrashed` when it did not work.

        ``last_end`` is how the previous session of this run ended, empty until one ended well."""
        branch, worktree, clock = self._start.branch, self._place.worktree, self._setup.clock
        head = await sync_to_head(self._setup.repo, branch, worktree)
        recent = await self._place.notebook.branch_log(branch)
        start = await self._briefed(head)
        prompt = launch_prompt(start, head=head, worktree=str(worktree.path), recent=recent, last_end=last_end)
        spec = await asyncio.to_thread(self._spec, grant, prompt)
        started = clock()
        self._recorded = {}
        handle = self._handle = await self._setup.runner.launch(spec)
        self._adopt(handle, spec.argv)
        status = await self._converse(handle, len(recent))
        if self._handle is not handle:
            return last_end
        self._handle = None
        await self._finish(handle, cancel=status.state != "exited")
        seconds = clock() - started
        if self._stopped.is_set():
            return last_end
        failed = status.state == STATE_LOST or bool(status.returncode and not status.timed_out)
        if not failed and seconds < self._setup.min_session_seconds and await self._tests_in_flight():
            await self._until_tests_land()
            return WAITED_END
        if failed or seconds < self._setup.min_session_seconds:
            raise SessionCrashed(_describe(status, seconds), lost=status.state == STATE_LOST, worked=bool(last_end))
        return "it reached its deadline" if status.timed_out else "its model finished its turn"

    async def _converse(self, handle: AgentHandle, logged: int) -> AgentStatus:
        """Wait for the CLI to end; an interactive one whose turn ended is sent what arrived since, or a nudge.

        ``logged`` is how many branch-log entries the launch prompt was built from: notices past it are new."""
        turns = TurnLog(self._place.workspace / PROCESS_LOG) if self._setup.command.interactive else None
        seen = _Seen((await self._place.notebook.poll(self._start.branch)).cursor, logged)
        sent = idle = 0
        while True:
            status = await self._wait(handle, turns, sent + 1)
            if status is not None:
                return status
            await self._record_cost()
            message = await self._wake(handle, seen)
            if message is None:
                return await self._setup.runner.poll(handle)
            idle = 0 if message else idle + 1
            if idle > IDLE_TURNS:
                self._handle = None
                await self._finish(handle, cancel=True)
                raise SessionCrashed(f"its model ended {IDLE_TURNS} turns in a row with nothing to test", worked=True)
            try:
                await self._setup.runner.send(handle, self._setup.command.message(message or nudge_message()).encode())
            except AgentEnded:
                return await self._setup.runner.poll(handle)
            sent += 1

    async def _wake(self, handle: AgentHandle, seen: _Seen) -> str | None:
        """Wait, beating the heartbeat, until something arrives for the branch, and move ``seen`` past it. An empty
        message asks for a nudge: nothing arrived and nothing of the branch runs. ``None`` once the session is stopped
        or the CLI ended."""
        notebook, branch = self._place.notebook, self._start.branch
        while not self._stopped.is_set() and (await self._setup.runner.poll(handle)).state == _RUNNING:
            if self._beat is not None:
                self._beat()
            page = await notebook.poll(branch, seen.cursor)
            log = await notebook.branch_log(branch)
            notices = [record for record in log[seen.logged :] if record.entry.content.startswith(NOTICE)]
            seen.cursor, seen.logged = page.cursor, len(log)
            if page.items or notices:
                return wake_message(page.items, notices)
            if not await self._tests_in_flight():
                return ""
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), self._setup.poll_seconds)
        return None

    async def _tests_in_flight(self) -> bool:
        branch = self._start.branch
        records = (await self._place.notebook.snapshot()).records
        return any(record.request.branch == branch and record.in_flight for record in records)

    async def _until_tests_land(self) -> None:
        """Wait, without a CLI running, until the branch has no test in flight or the session is stopped."""
        while not self._stopped.is_set() and await self._tests_in_flight():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), self._setup.poll_seconds)

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
        return replace(self._start, task=replace(self._start.task, brief=brief))

    def _spec(self, grant: SpecialistGrant, prompt: str) -> AgentLaunchSpec:
        setup, workspace, worktree = self._setup, self._place.workspace, self._place.worktree.path
        workspace.mkdir(parents=True, exist_ok=True)
        done = workspace / DONE_FILE
        done.unlink(missing_ok=True)
        (workspace / PROCESS_LOG).unlink(missing_ok=True)
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
            heartbeat_sink=self._sink(grant.heartbeat_sink),
            stdin_open=setup.command.interactive,
        )

    def _sink(self, service: HeartbeatSink) -> HeartbeatSink:
        """Keep the launch's ``beat`` for the idle wait, and hand it on to the MCP service's sink."""

        def take(beat: Beat) -> None:
            self._beat = beat
            service(beat)

        return take

    def _adopt(self, handle: AgentHandle, argv: Sequence[str]) -> None:
        branch = self._start.branch
        try:
            self._setup.processes.adopt(handle.pid, branch=branch, argv=argv)
        except ValueError:
            logger.warning("the agent CLI of %s ended before the process registry took it over", branch, exc_info=True)

    async def _wait(self, handle: AgentHandle, turns: TurnLog | None, turn: int) -> AgentStatus | None:
        """The CLI's status once it ended, was lost or the session stopped; ``None`` once ``turns`` shows its
        ``turn``-th turn ended with the CLI still up."""
        while True:
            status = await self._setup.runner.poll(handle)
            if status.state != _RUNNING or self._stopped.is_set():
                return status
            if turns is not None and await asyncio.to_thread(turns.ended) >= turn:
                return None
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), self._setup.poll_seconds)

    async def _finish(self, handle: AgentHandle, *, cancel: bool) -> None:
        if cancel:
            await self._setup.runner.cancel(handle)
        await self._setup.runner.collect(handle)
        await self._record_cost()

    async def _record_cost(self) -> None:
        """Record what the launch used since its last record."""
        used = await asyncio.to_thread(launch_usage, self._place.workspace / PROCESS_LOG)
        added = {
            model: _less(counts, self._recorded.get(model, TokenCounts())) for model, counts in used.tokens.items()
        }
        added = {model: counts for model, counts in added.items() if counts != TokenCounts()}
        if not added:
            return
        self._recorded = used.tokens
        total = sum(added.values(), TokenCounts())
        usage = LlmUsage(self._start.branch, UsageSource.SPECIALIST, self._cost(added), total, used.session_id)
        await self._place.notebook.record_llm_usage(usage)

    def _cost(self, tokens: Mapping[str, TokenCounts]) -> float | None:
        """What a launch's calls cost, ``None`` when one of their models has no price."""
        prices = [(price_of(model or self._setup.command.model), counts) for model, counts in tokens.items()]
        if any(price is None for price, _ in prices):
            return None
        return sum(price.cost_usd(counts) for price, counts in prices if price is not None)


@dataclass
class _Seen:
    """What a launch has shown its CLI: the inbox cursor it read up to, and how many branch-log entries."""

    cursor: str
    logged: int


class TurnLog:
    """Counts the turns a Claude stream-json log says ended, a ``result`` row each, reading only what was appended."""

    def __init__(self, process_log: Path) -> None:
        self._path, self._offset, self._partial, self._ended = process_log, 0, b"", 0

    def ended(self) -> int:
        try:
            with self._path.open("rb") as handle:
                handle.seek(self._offset)
                appended = handle.read()
        except OSError:
            return self._ended
        self._offset += len(appended)
        *lines, self._partial = (self._partial + appended).split(b"\n")
        for line in lines:
            if b'"result"' not in line:
                continue
            with contextlib.suppress(json.JSONDecodeError):
                row = json.loads(line)
                self._ended += isinstance(row, dict) and row.get("type") == "result"
        return self._ended


def _less(counts: TokenCounts, earlier: TokenCounts) -> TokenCounts:
    return TokenCounts(*(max(0, getattr(counts, item.name) - getattr(earlier, item.name)) for item in fields(counts)))


def _describe(status: AgentStatus, seconds: float) -> str:
    if status.state == STATE_LOST:
        return f"the session was lost ({status.error or 'no heartbeat'})"
    if status.returncode and not status.timed_out:
        return f"the agent CLI exited {status.returncode}" + (f" ({status.error})" if status.error else "")
    return f"the session ended after {seconds:.0f}s, too soon to have worked"


__all__ = [
    "IDLE_TURNS",
    "MCP_SERVER",
    "MIN_SESSION_S",
    "AgentCommand",
    "AgentSession",
    "AgentSetup",
    "BranchPlace",
    "Processes",
    "SessionNotebook",
]
