# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist backed by a model: agent CLI sessions launched through the AI work socket, one after another for as
long as its branch runs.

Every fresh session starts at the branch head, with the worktree synced to it and a launch prompt built from the brief,
rebuilt at that head when the setup can, and the branch log. A Claude CLI keeps its stdin open and lives across turns:
when a turn ends, the session waits, keeping its heartbeat, until something arrives for the branch (an inbox item or a
runtime notice) and sends it to the CLI as the next message, so the model goes on with its whole conversation. A turn
that ends with no test of the branch in flight (one waiting on its native builds is) and no process running in its
long dev lease waits a patience for something to arrive, and is nudged when nothing did;
``IDLE_TURNS`` such turns in a row end the session with :class:`SessionIdle`, which is no crash. A turn that called
tools counts half: a model reading, writing a patch or having a claim refused is working, yet one that never submits
still ends.
A session that ends as a session should, its CLI exiting or its deadline reached, is followed by a fresh one: a
branch runs until the scheduler stops it. One that ends within ``min_session_seconds`` while its branch has a test in
flight ended to wait for it: the next launch starts once that test has its result. One that is lost, exits with an
error or otherwise ends within ``min_session_seconds`` raises, and the runtime decides whether to restart it; a loss
while the branch's long dev lease runs a process says so, as the model was waiting on it in silence. Claude
streams partial messages into its log, so a long reasoning step shows as log growth and is not taken for a lost
session. The session's process registry adopts every process the socket starts, so the session stop kills it even
when this stop cannot. The spend of each turn, its logged tokens at the gateway's rates, is written to the branch's
blackboard when the turn ends and when the launch is collected, however it ended. The CLI streams of earlier launches
stay in the workspace, numbered in launch order beside ``process.log``, which holds the current launch's only.

A Claude session runs under a conversation id it is given at launch, written with what a resume needs to
``cli_session.json`` in the workspace before the CLI starts and updated at each turn's end. A Claude CLI that exits with
an error or is lost after its conversation finished a turn is resumed by the next run: ``--resume`` on that
conversation, told what arrived since and whether its worktree was kept, which it is while the branch head did not move.
A session that ends idle is resumed the same way, told what changed since its last turn, the integration head included,
as a fresh briefing would bring little the conversation lacks; it first waits, at most ``IDLE_RESUME_WAIT_S``, for
something to arrive or the integration head to move, so it has something to act on. Neither resume lists the runtime's
notice of the restart among what arrived. A clean end, a conversation past ``RESUME_CONTEXT_MAX`` or one the model
refused as too long, and a resumed CLI that crashes again before finishing a turn are followed by a fresh session. A
resume the CLI refuses at once is followed by a fresh session in the same run, with a notice and no crash. A
conversation record that cannot be read means a fresh session and a warning.

A launch, and the message that wakes a CLI once the integration head moved, say what the integration head holds and
what of it the branch head lacks (:mod:`~hyperloom_multibranch.specialist.following`). A branch whose specialist asked
to follow the integration head is moved onto the one the blackboard recorded between turns, once no candidate test of
the branch is in flight or waits for its judge, and the blackboard records the move; the next message says what the move
carried.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Protocol

from hyperloom.common.fs_utils import read_text_nofollow, write_text_nofollow
from hyperloom.common.llm_config import AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX
from hyperloom.contracts.measurement import Purpose, TestKind, Validity
from hyperloom.orchestrator.specialists.agent_runner import (
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
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.blackboard.interfaces import (
    AgentRecords,
    BranchLogStore,
    Inbox,
    LearningStore,
    LineageSink,
    UsageSink,
)
from hyperloom_multibranch.contracts.blackboard.records import (
    BranchLogRecord,
    InboxItem,
    LlmUsage,
    BlackboardSnapshot,
    TestRecord,
)
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH, InboxKind, UsageSource
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST, SpecialistGrant, SpecialistTokens
from hyperloom_multibranch.specialist.exhausted import exhausted_refusal
from hyperloom_multibranch.specialist.following import REROOT, Followed, follow, integration_holdings, reroot_requested
from hyperloom_multibranch.specialist.judging import NOTICE, left_unjudged
from hyperloom_multibranch.specialist.llm_usage import launch_cost, launch_usage
from hyperloom_multibranch.specialist.prompts import (
    IntegrationHeads,
    Resume,
    idle_resume_message,
    integration_section,
    launch_prompt,
    lease_wake_message,
    nudge_message,
    resume_message,
    wake_message,
)
from hyperloom_multibranch.specialist.runtime import SessionCrashed, SessionIdle, SessionStart, restart_notice
from hyperloom_multibranch.specialist.tools import sync_to_head
from hyperloom_multibranch.specialist.worktree import BranchWorktree

logger = logging.getLogger(__name__)

#: The name the session's MCP service has in the CLI's configuration.
MCP_SERVER = "hyperloom"
#: A session ending sooner than this did not work, whatever its exit code: a misconfigured CLI must not relaunch forever.
MIN_SESSION_S = 120.0
#: A launch's own deadline, past any session: the scheduler ends a branch that stops gaining, and the liveness check
#: ends a CLI that hangs, so a deadline would only cut a working conversation short.
SPECIALIST_SESSION_S = 14 * 24 * 3600.0
#: How long a turn that ended with no test of its branch running waits for something to arrive before it counts as
#: idle, one patience per idle turn in a row: a result or a notice often lands minutes after a model stops.
NUDGE_WAITS_S = (60.0, 180.0, 300.0)
#: Idle turns in a row that end a session, all but the last nudged: a model that keeps stopping must not be woken
#: forever.
IDLE_TURNS = len(NUDGE_WAITS_S)
#: What an idle turn adds to a session's strain: a silent one twice what one that called tools does.
_SILENT, _WORKED = 2, 1
#: A session silent this long, with no log growth, CPU work or tool call in flight, is hung: a test takes about eight
#: minutes and one reasoning step can take as long, and Claude streams its reasoning into the log.
SPECIALIST_LOST_AFTER_S = 600.0
#: How long a CLI waiting on its long dev lease's process stays unwoken: the reaper reads only the CLI's own output, so
#: a wait in silence past ``SPECIALIST_LOST_AFTER_S`` would be taken for a hung session.
LEASE_WAKE_S = SPECIALIST_LOST_AFTER_S / 2
#: How a launch that ended early to wait for its branch's test ended, as the next launch is told.
WAITED_END = "its model ended its turn to wait for a test, which has its result now"
#: A conversation whose last call read this many tokens is not resumed: about 60% of the 1M-token window of the
#: sessions' Claude model, past the 73k to 231k the branches of a 12-hour session peaked at, so a resume never
#: starts close to the window and rewriting the whole conversation to the cache stays cheaper than a fresh briefing.
RESUME_CONTEXT_MAX = 600_000
#: How long the resume of an idle session waits for something to arrive first: about one test's time, so a result or
#: a merge usually lands within it, while a branch that nothing reaches still goes on before its slot sits idle long.
IDLE_RESUME_WAIT_S = 600.0
#: Where a re-root writes the patches of the branch's change that no longer apply on the integration head.
REROOTED_DIR = "rerooted"
PROMPT_FILE, ROLE_FILE, DONE_FILE, HEARTBEAT_FILE, CLI_SESSION_FILE = (
    "prompt.md",
    "system_prompt.md",
    "specialist_done.json",
    "heartbeat.json",
    "cli_session.json",
)
_RUNNING = "running"
#: The mode these files had when written with ``write_bytes`` under the default umask.
_SHARED_MODE = 0o644


def _keep_process_log(workspace: Path) -> None:
    """Move the previous launch's stream to the next numbered sibling, counting on from those already there."""
    log = workspace / PROCESS_LOG
    if not log.exists():
        return
    stem, suffix = log.stem, log.suffix
    numbers = (path.name[len(stem) + 1 : -len(suffix)] for path in workspace.glob(f"{stem}-*{suffix}"))
    last = max((int(number) for number in numbers if number.isdigit()), default=0)
    log.rename(workspace / f"{stem}-{last + 1}{suffix}")


class Processes(Protocol):
    def adopt(self, pid: int, *, branch: str, argv: Sequence[str]) -> object: ...


class DevLeaseHolder(Protocol):
    """The session's dev sandboxes (:class:`~hyperloom_multibranch.specialist.dev_sandbox.DevLeases`), as a branch's
    sessions give back what their branch holds: a run's end keeps a :meth:`busy` lease, the branch's stop none."""

    def busy(self, branch: str) -> bool: ...

    async def reclaim(self, branch: str, reason: str) -> None: ...

    async def reclaim_unless_busy(self, branch: str, reason: str) -> None: ...


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

    def argv(self, role_file: Path, worktree: Path, *, session_id: str = "", resume_id: str = "") -> tuple[str, ...]:
        """The command line; the socket adds the MCP configuration. A Claude CLI starts the conversation
        ``session_id``, or goes on with ``resume_id``; codex takes neither."""
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
        conversation = ("--resume", resume_id) if resume_id else ("--session-id", session_id) if session_id else ()
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
            *conversation,
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
    the brief the branch was started with. ``nudge_waits_s`` are the patiences of idle turns in a row; there are as
    many idle turns as patiences. ``idle_resume_wait_s`` bounds how long the resume of an idle session waits for
    something to arrive. ``dev_leases`` are the session's dev sandboxes, ``None`` when it lends none: a
    branch's lease is given back when its run ends, however it ended, and when it is stopped, unless it is a long one
    with a process running, which only the branch's stop gives back. A CLI waiting on such a process is woken every
    ``lease_wake_s``.
    """

    command: AgentCommand
    role: str
    env: Mapping[str, str]
    runner: InteractiveAgentRunner
    tokens: SpecialistTokens
    processes: Processes
    repo: ExperimentRepo
    briefs: Callable[[str, str], Awaitable[str]] | None = None
    session_seconds: float = SPECIALIST_SESSION_S
    min_session_seconds: float = MIN_SESSION_S
    poll_seconds: float = 5.0
    nudge_waits_s: tuple[float, ...] = NUDGE_WAITS_S
    idle_resume_wait_s: float = IDLE_RESUME_WAIT_S
    clock: Callable[[], float] = time.time
    dev_leases: DevLeaseHolder | None = None
    lease_wake_s: float = LEASE_WAKE_S


class SessionBlackboard(BranchLogStore, UsageSink, Inbox, LearningStore, AgentRecords, LineageSink, Protocol):
    """A branch's runtime handle as its sessions use it: the branch log, the spend of each turn, the inbox that wakes
    a CLI whose turn ended, the learnings an autonomous agent's findings become, its agent's manifest, the re-root of
    its branch, and the snapshot that says whether the branch has a test in flight."""

    async def snapshot(self) -> BlackboardSnapshot: ...


@dataclass(frozen=True)
class BranchPlace:
    """Where one branch's sessions work: its worktree, its log and spend, and the directory a session's files go in."""

    worktree: BranchWorktree
    blackboard: SessionBlackboard
    workspace: Path


class AgentSession:
    """One run of a model-backed specialist: CLI sessions on its branch until it is stopped or one crashes or
    ends idle."""

    def __init__(self, start: SessionStart, setup: AgentSetup, place: BranchPlace) -> None:
        self._start, self._setup, self._place = start, setup, place
        self._stopped = asyncio.Event()
        self._handle: AgentHandle | None = None
        self._beat: Beat | None = None
        self._recorded: Mapping[str, TokenCounts] = {}
        self._requested: frozenset[str] = frozenset()

    async def preflight(self) -> None:
        """Nothing to prepare: the CLI is the image's and the worktree the session's."""

    async def release(self) -> None:
        if self._setup.dev_leases is not None:
            await self._setup.dev_leases.reclaim(self._start.branch, "its branch stopped")

    async def run(self) -> None:
        try:
            grant = self._setup.tokens.issue(self._start.branch, frozenset({SCOPE_SPECIALIST}))
            self._requested = await self._branch_requests()
            last_end = ""
            while not self._stopped.is_set():
                last_end = await self._session(grant, last_end)
        finally:
            await self._give_back_dev_lease("its specialist's run ended")

    async def set_k(self, k: int) -> None:
        self._start = replace(self._start, k=k)

    async def stop(self) -> None:
        self._stopped.set()
        # Taking the handle makes this stop, not the run waiting on it, the one that ends the CLI.
        handle, self._handle = self._handle, None
        if handle is not None:
            await self._finish(handle, cancel=True)
        self._setup.tokens.revoke(self._start.branch)
        await self._give_back_dev_lease("its specialist was stopped")

    async def _give_back_dev_lease(self, reason: str) -> None:
        if self._setup.dev_leases is not None:
            await self._setup.dev_leases.reclaim_unless_busy(self._start.branch, reason)

    async def _session(self, grant: SpecialistGrant, last_end: str) -> str:
        """Run one CLI session to its end and say how it ended; raises :class:`SessionCrashed` when it did not work and
        :class:`SessionIdle` when its model kept finding nothing to test.

        ``last_end`` is how the previous session of this run ended, empty until one ended well."""
        launch = await self._resumed()
        if self._stopped.is_set():
            return last_end
        if launch is not None:
            ended = await self._run_launch(grant, launch)
            if ended is None:
                return last_end
            if not self._refused(launch, ended):
                return await self._ended(launch, ended, last_end)
            await self._resume_refused(launch, ended)
        launch = await self._fresh(last_end)
        ended = await self._run_launch(grant, launch)
        return last_end if ended is None else await self._ended(launch, ended, last_end)

    async def _fresh(self, last_end: str) -> _Launch:
        branch, worktree = self._start.branch, self._place.worktree
        followed = await self._follow()
        head = await sync_to_head(self._setup.repo, branch, worktree)
        recent = await self._place.blackboard.branch_log(branch)
        start = await self._briefed(head)
        integration = await self._setup.repo.head(INTEGRATION_BRANCH)
        holdings = _joined(followed.detail if followed else "", await self._holdings(head))
        prompt = _joined(
            launch_prompt(start, head=head, worktree=str(worktree.path), recent=recent, last_end=last_end),
            integration_section(holdings),
        )
        seen = _Seen((await self._place.blackboard.poll(branch)).cursor, len(recent), integration=integration)
        if not self._setup.command.interactive:
            return _Launch(prompt, seen)
        cli = _CliState(
            str(uuid.uuid4()), head=head, integration=integration, seen_cursor=seen.cursor, logged=seen.logged
        )
        return _Launch(prompt, seen, cli)

    async def _resumed(self) -> _Launch | None:
        """The launch that goes on with the conversation the last run's CLI crashed or idled in, an idle one once
        something arrived or its wait is over; ``None`` when there is none.

        The worktree is kept as the CLI left it while the branch head did not move, so its uncommitted edits survive;
        the CLI is shown what arrived from where the conversation's last finished turn had seen up to."""
        if not self._setup.command.interactive:
            return None
        cli = await asyncio.to_thread(self._read_cli)
        if cli is None or not cli.resumable:
            return None
        followed = await self._until_news(cli) if cli.idle else None
        followed = followed or await self._follow()
        branch, worktree = self._start.branch, self._place.worktree
        head = await self._setup.repo.head(branch)
        kept = await asyncio.to_thread(worktree.head) == head
        if not kept:
            head = await sync_to_head(self._setup.repo, branch, worktree)
        integration = IntegrationHeads(cli.integration, await self._setup.repo.head(INTEGRATION_BRANCH))
        seen = _Seen(cli.seen_cursor, cli.logged, integration=integration.now)
        items, notices = await self._arrived(seen)
        resume = Resume(cli.reason, head, str(worktree.path), kept=kept)
        if cli.idle:
            prompt = idle_resume_message(resume, integration, items, notices)
        else:
            prompt = resume_message(resume, items, notices)
        if followed or integration.seen != integration.now:
            holdings = _joined(followed.detail if followed else "", await self._holdings(head))
            prompt = _joined(prompt, integration_section(holdings))
        cli = replace(
            cli,
            resumable=False,
            idle=False,
            resumes_in_row=cli.resumes_in_row + 1,
            head=head,
            integration=integration.now,
            seen_cursor=seen.cursor,
            logged=seen.logged,
        )
        return _Launch(prompt, seen, cli, resumed=True)

    async def _until_news(self, cli: _CliState) -> Followed | None:
        """Wait, with no CLI running, until something arrives past what the idle conversation ``cli`` saw, the
        integration head moves from where it saw it or the branch follows it as its specialist asked, at most
        ``idle_resume_wait_s`` and while the session runs: a model resumed with nothing new to act on idles again, and
        its branch's last idle session goes with it. What following the integration head did, when it did."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._setup.idle_resume_wait_s
        while not self._stopped.is_set() and loop.time() < deadline:
            followed = await self._follow()
            if followed is not None:
                return followed
            if any(await self._arrived(_Seen(cli.seen_cursor, cli.logged))):
                return None
            if await self._setup.repo.head(INTEGRATION_BRANCH) != cli.integration:
                return None
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), min(self._setup.poll_seconds, deadline - loop.time()))
        return None

    async def _run_launch(self, grant: SpecialistGrant, launch: _Launch) -> _Ended | None:
        """Run ``launch``'s CLI to its end; ``None`` when the session was stopped meanwhile."""
        if launch.cli is not None:
            await asyncio.to_thread(self._write_cli, launch.cli)
        spec = await asyncio.to_thread(self._spec, grant, launch)
        clock, started = self._setup.clock, self._setup.clock()
        self._recorded = {}
        if self._setup.command.interactive:
            launch.turns = TurnLog(self._place.workspace / PROCESS_LOG)
        handle = self._handle = await self._setup.runner.launch(spec)
        self._adopt(handle, spec.argv)
        status = await self._converse(handle, launch)
        if self._handle is not handle:
            return None
        self._handle = None
        await self._finish(handle, cancel=status.state != "exited")
        if self._stopped.is_set():
            return None
        if launch.turns is not None:
            await asyncio.to_thread(launch.turns.ended)
        return _Ended(status, clock() - started)

    async def _ended(self, launch: _Launch, ended: _Ended, last_end: str) -> str:
        status, seconds = ended.status, ended.seconds
        failed = status.state == STATE_LOST or bool(status.returncode and not status.timed_out)
        if not failed and seconds < self._setup.min_session_seconds and await self._tests_in_flight():
            await self._until_tests_land()
            return WAITED_END
        if failed or seconds < self._setup.min_session_seconds:
            reason = _describe(status, seconds)
            resumable = failed and await self._keep_for_resume(launch, reason)
            lost = status.state == STATE_LOST
            waiting = lost and self._dev_lease_busy()
            raise SessionCrashed(reason, lost=lost, waiting=waiting, worked=bool(last_end), resumable=resumable)
        return "it reached its deadline" if status.timed_out else "its model finished its turn"

    async def _keep_for_resume(self, launch: _Launch, reason: str, *, idle: bool = False) -> bool:
        """Mark the crashed or ``idle`` launch's conversation for the next run to resume, when it is worth going on
        with: it finished a turn since its last resume, and its context has room left."""
        cli, turns = launch.cli, launch.turns
        if cli is None or turns is None:
            return False
        resumable = (
            cli.turns_done >= 1
            and cli.resumes_in_row < 1
            and cli.context_tokens < RESUME_CONTEXT_MAX
            and not turns.context_exceeded
        )
        launch.cli = replace(cli, resumable=resumable, idle=idle and resumable, reason=reason)
        await asyncio.to_thread(self._write_cli, launch.cli)
        return resumable

    def _refused(self, launch: _Launch, ended: _Ended) -> bool:
        """Whether the CLI refused to resume the conversation at once: no conversation by that id, a record it
        cannot load, a version that does not know the flag. It ends within seconds, before or without a turn."""
        turns = launch.turns
        if turns is None or ended.seconds >= self._setup.min_session_seconds:
            return False
        return not turns.inited or turns.refused

    async def _resume_refused(self, launch: _Launch, ended: _Ended) -> None:
        branch = self._start.branch
        session_id = launch.cli.session_id if launch.cli is not None else ""
        why = _describe(ended.status, ended.seconds)
        logger.warning(
            "the CLI of %s could not resume conversation %s (%s); a fresh session starts", branch, session_id, why
        )
        content = f"{NOTICE} resuming the specialist's conversation failed ({why}); a fresh session starts"
        await self._place.blackboard.append_branch_log(BranchLogEntry(branch, BranchLogType.RESULT, content))

    def _read_cli(self) -> _CliState | None:
        """The conversation the workspace's last launch ran; ``None``, with a warning, when its record is missing
        after a launch or cannot be read."""
        workspace = self._place.workspace
        path = workspace / CLI_SESSION_FILE
        try:
            return _CliState.parse(json.loads(read_text_nofollow(path)))
        except FileNotFoundError:
            if (workspace / PROCESS_LOG).exists():
                logger.warning("%s has no conversation record after a launch; a fresh session starts", workspace)
            return None
        except (OSError, ValueError, TypeError):
            logger.warning("the conversation record %s cannot be read; a fresh session starts", path, exc_info=True)
            return None

    def _write_cli(self, cli: _CliState) -> None:
        """Replace the conversation record whole, readable by its owner only: it names a conversation to resume."""
        workspace = self._place.workspace
        workspace.mkdir(parents=True, exist_ok=True)
        partial = write_text_nofollow(workspace, (f".{CLI_SESSION_FILE}.partial",), json.dumps(asdict(cli)))
        partial.replace(workspace / CLI_SESSION_FILE)

    async def _turn_finished(self, launch: _Launch) -> None:
        """Record in the conversation record a turn that finished without an error, and what the CLI had seen."""
        cli, turns = launch.cli, launch.turns
        if cli is None or turns is None or not turns.succeeded:
            return
        launch.cli = replace(
            cli,
            turns_done=cli.turns_done + 1,
            resumes_in_row=0,
            context_tokens=turns.context,
            head=await self._setup.repo.head(self._start.branch),
            integration=launch.seen.integration,
            seen_cursor=launch.seen.cursor,
            logged=launch.seen.logged,
        )
        await asyncio.to_thread(self._write_cli, launch.cli)

    async def _converse(self, handle: AgentHandle, launch: _Launch) -> AgentStatus:
        """Wait for the CLI to end; an interactive one whose turn ended is sent what arrived since, or a nudge, and
        raises :class:`SessionIdle` once its turns ended idle as many times in a row as it has patiences.

        ``launch.seen`` is what the launch's first message showed: what arrives past it is new."""
        turns, seen = launch.turns, launch.seen
        waits = self._setup.nudge_waits_s
        sent = strain = quiet = 0
        while True:
            status = await self._wait(handle, turns, sent + 1)
            if status is not None:
                return status
            await self._record_cost()
            await self._turn_finished(launch)
            message = await self._wake(handle, seen, waits[min(strain // _SILENT, len(waits) - 1)])
            if message is None:
                return await self._setup.runner.poll(handle)
            strain = 0 if message else strain + (_WORKED if turns is not None and turns.worked() else _SILENT)
            quiet = 0 if message else quiet + 1
            if strain >= _SILENT * len(waits):
                raise await self._idle_end(handle, launch, quiet)
            try:
                await self._setup.runner.send(handle, self._setup.command.message(message or nudge_message()).encode())
            except AgentEnded:
                return await self._setup.runner.poll(handle)
            sent += 1

    async def _idle_end(self, handle: AgentHandle, launch: _Launch, quiet: int) -> SessionIdle:
        """End the CLI whose model ended ``quiet`` turns in a row with nothing to test, keeping its conversation for
        the next run when it has room left; what the run raises."""
        self._handle = None
        await self._finish(handle, cancel=True)
        tested = await self._branch_requests() != self._requested
        reason = f"its model ended {quiet} turns in a row with nothing to test"
        resumable = await self._keep_for_resume(launch, reason, idle=True)
        return SessionIdle(reason, tested=tested, resumable=resumable)

    async def _wake(self, handle: AgentHandle, seen: _Seen, patience: float) -> str | None:
        """Wait, beating the heartbeat, until something arrives for the branch, and move ``seen`` past it. An empty
        message asks for a nudge: nothing arrived while nothing of the branch ran for ``patience`` seconds. ``None``
        once the session is stopped or the CLI ended."""
        loop = asyncio.get_running_loop()
        quiet_since: float | None = None
        waited_since = loop.time()
        while not self._stopped.is_set() and (await self._setup.runner.poll(handle)).state == _RUNNING:
            if self._beat is not None:
                self._beat()
            followed = await self._follow()
            items, notices = await self._arrived(seen)
            shared = await self._shared_profiles()
            landed = [
                _landed(record) for record in shared if record.request_id in seen.profiles and not record.in_flight
            ]
            seen.profiles = {record.request_id for record in shared if record.in_flight}
            integration = await self._setup.repo.head(INTEGRATION_BRANCH)
            moved = integration != seen.integration
            if items or notices or landed or moved or followed:
                seen.integration = integration
                holdings = (
                    await self._holdings(await self._setup.repo.head(self._start.branch)) if moved or followed else ""
                )
                return wake_message(
                    (*items, *landed), notices, integration=_joined(followed.detail if followed else "", holdings)
                )
            if self._dev_lease_busy() and loop.time() - waited_since >= self._setup.lease_wake_s:
                return lease_wake_message()
            if seen.profiles or await self._tests_in_flight() or self._dev_lease_busy():
                quiet_since = None
            else:
                quiet_since = loop.time() if quiet_since is None else quiet_since
                if loop.time() - quiet_since >= patience:
                    return ""
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), self._setup.poll_seconds)
        return None

    async def _arrived(self, seen: _Seen) -> tuple[Sequence[InboxItem], list[BranchLogRecord]]:
        """The inbox items and the runtime's notices past ``seen``, which moves past them, among them the refusal of
        each exhausted record past ``seen`` that :func:`exhausted_refusal` refuses; a notice of a session's restart is
        left out, as the resumed or fresh session's first message says why it starts."""
        blackboard, branch = self._place.blackboard, self._start.branch
        page = await blackboard.poll(branch, seen.cursor)
        log = await blackboard.branch_log(branch)
        refusals = [why for record in log[seen.logged :] if (why := exhausted_refusal(record.entry.content))]
        for why in refusals:
            await blackboard.append_branch_log(BranchLogEntry(branch, BranchLogType.RESULT, f"{NOTICE} {why}"))
        if refusals:
            log = await blackboard.branch_log(branch)
        notices = [
            record
            for record in log[seen.logged :]
            if record.entry.content.startswith(NOTICE) and not restart_notice(record.entry.content)
        ]
        seen.cursor, seen.logged = page.cursor, len(log)
        return page.items, notices

    async def _branch_requests(self) -> frozenset[str]:
        """The branch's candidate tests: a profile measures no candidate, so it does not reset an idle count."""
        branch = self._start.branch
        records = (await self._place.blackboard.snapshot()).records
        return frozenset(
            record.request_id
            for record in records
            if record.request.branch == branch and record.request.kind is TestKind.TEST
        )

    def _dev_lease_busy(self) -> bool:
        return self._setup.dev_leases is not None and self._setup.dev_leases.busy(self._start.branch)

    async def _tests_in_flight(self) -> bool:
        branch = self._start.branch
        records = (await self._place.blackboard.snapshot()).records
        return any(record.request.branch == branch and record.in_flight for record in records)

    async def _follow(self) -> Followed | None:
        """Move the branch onto the integration head when its specialist asked to since the runtime last answered,
        once no candidate of the branch is unsettled: one measured on the old head could no longer land. Writes the
        answer to the branch log; ``None`` while there is nothing to do yet or the blackboard cannot record the move yet."""
        blackboard, branch = self._place.blackboard, self._start.branch
        log = [record.entry for record in await blackboard.branch_log(branch)]
        asked = max((index for index, entry in enumerate(log) if reroot_requested(entry.content)), default=-1)
        answered = max((index for index, entry in enumerate(log) if entry.content.startswith(_ANSWER)), default=-1)
        if asked <= answered or await self._candidates_unsettled():
            return None
        notes = self._place.workspace / REROOTED_DIR
        followed = await follow(self._setup.repo, blackboard, await blackboard.snapshot(), log[asked], notes)
        if followed is None:
            return None
        logger.info("%s asked to follow the integration head: %s", branch, followed.notice)
        await blackboard.append_branch_log(BranchLogEntry(branch, BranchLogType.RESULT, f"{_ANSWER} {followed.notice}"))
        return followed

    async def _holdings(self, head: str) -> str:
        return await integration_holdings(self._setup.repo, await self._place.blackboard.snapshot(), head)

    async def _candidates_unsettled(self) -> bool:
        """Whether a candidate test of the branch is in flight, or measured and neither judged nor closed by a notice
        of its judge's that no judgement follows: until then the judge may still retest or commit it on the head it
        was measured against."""
        branch = self._start.branch
        snapshot = await self._place.blackboard.snapshot()
        answered = {
            request_id
            for record in snapshot.branch_logs.get(branch, ())
            if record.entry.content.startswith(NOTICE) and left_unjudged(record.entry.content)
            for request_id in record.entry.request_ids
        }
        return any(
            record.in_flight
            or (
                record.request.purpose is Purpose.EXPLORE
                and record.judgement is None
                and record.result is not None
                and record.result.validity is Validity.SUCCEEDED
                and record.request_id not in answered
            )
            for record in snapshot.records
            if record.request.branch == branch and record.request.kind is TestKind.TEST
        )

    async def _shared_profiles(self) -> list[TestRecord]:
        """Other branches' profiles of this branch's head: a specialist whose own profile was refused as
        profile_exists waits on one of them, and its result reaches only that branch's inbox."""
        branch = self._start.branch
        head = await self._setup.repo.head(branch)
        records = (await self._place.blackboard.snapshot()).records
        return [
            record
            for record in records
            if record.request.kind is TestKind.PROFILE
            and record.request.parent_commit == head
            and record.request.branch != branch
        ]

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

    def _spec(self, grant: SpecialistGrant, launch: _Launch) -> AgentLaunchSpec:
        setup, workspace, worktree = self._setup, self._place.workspace, self._place.worktree.path
        workspace.mkdir(parents=True, exist_ok=True)
        done = workspace / DONE_FILE
        done.unlink(missing_ok=True)
        _keep_process_log(workspace)
        role_file = write_text_nofollow(workspace, (ROLE_FILE,), setup.role, mode=_SHARED_MODE)
        write_text_nofollow(
            workspace, (PROMPT_FILE,), setup.command.stdin(setup.role, launch.prompt), mode=_SHARED_MODE
        )
        cli = launch.cli
        session_id = cli.session_id if cli is not None else ""
        conversation = {"resume_id": session_id} if launch.resumed else {"session_id": session_id}
        return AgentLaunchSpec(
            task_id=self._start.branch,
            workspace=workspace,
            argv=setup.command.argv(role_file, worktree, **conversation),
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
        usage = LlmUsage(
            self._start.branch,
            UsageSource.SPECIALIST,
            launch_cost(added, self._setup.command.model),
            total,
            used.session_id,
        )
        await self._place.blackboard.record_llm_usage(usage)


@dataclass
class _Seen:
    """What a launch has shown its CLI: the inbox cursor it read up to, how many branch-log entries, the other
    branches' profiles of its head that were still running, and the integration head."""

    cursor: str
    logged: int
    profiles: set[str] = field(default_factory=set)
    integration: str = ""


@dataclass(frozen=True)
class _CliState:
    """A Claude CLI's conversation as ``cli_session.json`` keeps it: its id; the turns it finished without an error,
    and the tokens its last call read; the resumes since it last finished one; the branch and integration heads and
    the inbox cursor and branch-log entries it had seen at its last finished turn; and whether the next run resumes
    it, with why it stopped and whether that was an idle end. ``resumable`` is set only when a launch crashed or ended
    idle, so a clean end leaves it unset."""

    session_id: str
    turns_done: int = 0
    context_tokens: int = 0
    resumes_in_row: int = 0
    head: str = ""
    integration: str = ""
    seen_cursor: str = ""
    logged: int = 0
    resumable: bool = False
    idle: bool = False
    reason: str = ""

    @classmethod
    def parse(cls, raw: object) -> _CliState:
        """The record ``raw`` holds; raises :class:`ValueError` when a field is missing or of another type."""
        if not isinstance(raw, dict) or set(raw) != {item.name for item in fields(cls)}:
            raise ValueError(f"a conversation record has the fields of {cls.__name__}, not {raw!r}")
        kinds = {item.name: type(item.default) for item in fields(cls) if item.name != "session_id"}
        wrong = [name for name, kind in kinds.items() if type(raw[name]) is not kind]
        if wrong or not isinstance(raw["session_id"], str) or not raw["session_id"]:
            raise ValueError(f"a conversation record with fields of the wrong type: {raw!r}")
        return cls(**raw)


@dataclass
class _Launch:
    """One CLI launch: its first message, what that message showed, the conversation it runs (``None`` for codex),
    whether it resumes that conversation, and the turns of its log once it runs."""

    prompt: str
    seen: _Seen
    cli: _CliState | None = None
    resumed: bool = False
    turns: TurnLog | None = None


@dataclass(frozen=True)
class _Ended:
    status: AgentStatus
    seconds: float


#: Opens the runtime's answer to a request to follow the integration head, so a later request is told from an answered
#: one.
_ANSWER = f"{NOTICE} {REROOT}:"


def _joined(*sections: str) -> str:
    return "\n\n".join(section.strip("\n") for section in sections if section)


def _landed(record: TestRecord) -> InboxItem:
    request = record.request
    summary = f"the profile of your head by {request.branch} has its result; its analysis brings the op shares"
    return InboxItem(InboxKind.RESULT, request.branch, record.request_id, summary, 0.0)


class TurnLog:
    """Counts the turns a Claude stream-json log says ended, a ``result`` row each, reading only what was appended.

    It also keeps what a resume decides on: whether the CLI said it started (``inited``), whether its last turn ended
    without an error (``succeeded``), whether a turn erred before any model call (``refused``), whether the model
    refused a prompt as too long (``context_exceeded``), and the tokens the last call read (``context``)."""

    def __init__(self, process_log: Path) -> None:
        self._path, self._offset, self._partial, self._ended = process_log, 0, b"", 0
        self._tools = self._last_tools = 0
        self.inited = self.succeeded = self.refused = self.context_exceeded = False
        self.context = 0

    def worked(self) -> bool:
        """Whether the last turn that ended called a tool."""
        return self._last_tools > 0

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
            if not any(key in line for key in (b'"result"', b'"tool_use"', b'"init"', b'"usage"')):
                continue
            with contextlib.suppress(json.JSONDecodeError):
                self._read(json.loads(line))
        return self._ended

    def _read(self, row: object) -> None:
        if not isinstance(row, dict):
            return
        kind = row.get("type")
        if kind == "system" and row.get("subtype") == "init":
            self.inited = True
        elif kind == "result":
            self._ended += 1
            self._last_tools, self._tools = self._tools, 0
            self._result(row)
        elif kind == "assistant" and isinstance(row.get("message"), dict):
            message = row["message"]
            content, usage = message.get("content"), message.get("usage")
            if isinstance(content, list):
                self._tools += sum(isinstance(item, dict) and item.get("type") == "tool_use" for item in content)
            if isinstance(usage, dict):
                read = TokenCounts.from_anthropic_usage(usage)
                self.context = read.input + read.cache_read + read.cache_write + read.cache_write_1h

    def _result(self, row: dict) -> None:
        erred = row.get("is_error") is True or str(row.get("subtype", "")).startswith("error")
        self.succeeded = not erred
        self.refused = self.refused or (erred and row.get("num_turns") == 0)
        too_long = "Prompt is too long" in str(row.get("result", "")) or "context_window" in str(
            row.get("stop_reason", "")
        )
        self.context_exceeded = self.context_exceeded or (erred and too_long)


def _less(counts: TokenCounts, earlier: TokenCounts) -> TokenCounts:
    return TokenCounts(*(max(0, getattr(counts, item.name) - getattr(earlier, item.name)) for item in fields(counts)))


def _describe(status: AgentStatus, seconds: float) -> str:
    if status.state == STATE_LOST:
        return f"the session was lost ({status.error or 'no heartbeat'})"
    if status.returncode and not status.timed_out:
        return f"the agent CLI exited {status.returncode}" + (f" ({status.error})" if status.error else "")
    return f"the session ended after {seconds:.0f}s, too soon to have worked"


__all__ = [
    "CLI_SESSION_FILE",
    "IDLE_RESUME_WAIT_S",
    "IDLE_TURNS",
    "RESUME_CONTEXT_MAX",
    "NUDGE_WAITS_S",
    "SPECIALIST_SESSION_S",
    "MCP_SERVER",
    "MIN_SESSION_S",
    "AgentCommand",
    "AgentSession",
    "AgentSetup",
    "BranchPlace",
    "DevLeaseHolder",
    "Processes",
    "SessionBlackboard",
]
