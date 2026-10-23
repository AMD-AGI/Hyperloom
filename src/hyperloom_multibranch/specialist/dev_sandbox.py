# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist's dev sandbox: a few of the toolbox's cards lent to its branch for a while, to check that a change
runs and to time a kernel before a full test, through five MCP tools.

The session lends at most :attr:`DevLimits.cap` at once, one a branch, and none while a test waits for cards; a cap
of 0, the default, lends none. A lease lasts at most :attr:`DevLimits.max_s`, and the runtime takes it back, never the model:
when it expires, when the branch made no call for :attr:`DevLimits.idle_s` with nothing left running (or for
:attr:`DevLimits.idle_running_s` with something), and once tests have queued for :attr:`DevLimits.queued_s`. A branch
whose lease ended leases again only after :attr:`DevLimits.cooldown_s`. The branch's sessions give its lease back when
a run ends or is stopped (:class:`~hyperloom_multibranch.specialist.agent_session.DevLeaseHolder`); the toolbox's close
releases whatever is left.

Commands run in a :class:`~hyperloom_multibranch.specialist.agent_sandbox.LocalAgentSandbox` on the leased cards,
importing the branch worktree's source, at a lower CPU priority, with every JIT and kernel cache in the lease's root:
a cache beside the package it compiles for would be in the worktree, and its files in the next patch. Nothing a dev
sandbox measures is written as a test, claim or judgement; its only records are branch-log notices.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, TypeVar

from hyperloom.hosted.mcp_server import CallContext, InvalidParamsError, McpServer, ToolError
from hyperloom.measure.compile import COMPONENT_LAYOUTS
from hyperloom.measure.ports import DEV_SANDBOX_PORTS
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.notebook.interfaces import NotebookError
from hyperloom_multibranch.contracts.notebook.records import BranchLogRecord
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST, BranchLease, QueueState, RequestRejected
from hyperloom_multibranch.specialist.agent_onboarding import AgentProcess, AgentSandbox
from hyperloom_multibranch.specialist.agent_sandbox import LocalAgentSandbox
from hyperloom_multibranch.specialist.agent_session import Processes
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.tools import WorktreeFor

logger = logging.getLogger(__name__)
T = TypeVar("T")

#: The cache variables a sandbox command gets, each a directory below the lease root's ``cache``.
CACHE_DIRS = {"TRITON_CACHE_DIR": "triton", "AITER_JIT_DIR": "aiter-jit", "TORCH_EXTENSIONS_DIR": "torch-extensions"}
#: Lines of output a reply carries unless the caller asks for others.
LINES = 100
MINUTE = 60.0


@dataclass(frozen=True)
class DevLimits:
    """How many leases the session lends at once, how long one lasts and when the runtime takes it back; how long a
    call may hold the MCP request, which times out at 60 s; and the ports and CPU priority a lease's commands get."""

    cap: int = 0
    max_s: float = 1800.0
    queued_s: float = 300.0
    cooldown_s: float = 600.0
    idle_s: float = 600.0
    idle_running_s: float = 900.0
    run_s: float = 40.0
    read_s: float = 40.0
    read_poll_s: float = 0.5
    poll_s: float = 5.0
    #: The range the measurement's port picker never draws from.
    port: int = DEV_SANDBOX_PORTS.start
    ports: int = 10
    niceness: int = 10


class LeaseSource(Protocol):
    """The toolbox, as far as its cards are lent."""

    async def queue_state(self) -> QueueState: ...

    async def lease_branch_sandbox(
        self, branch_id: str, cards: int, *, wait_sec: float | None = None
    ) -> BranchLease: ...

    async def release_branch_sandbox(self, lease_id: str) -> None: ...


class DevSandbox(AgentSandbox, Protocol):
    def alive(self) -> bool:
        """Whether any process of the sandbox is left."""

    async def stop(self) -> None:
        """Kill every process of the sandbox."""


class BranchLogAppender(Protocol):
    async def append_branch_log(self, entry: BranchLogEntry) -> BranchLogRecord: ...


#: The sandbox of a branch's lease.
SandboxFor = Callable[[str, BranchLease], DevSandbox]


class DevRefused(ToolError):
    """A dev sandbox call refused; ``code`` names the check, for the specialist to react to."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code = code


@dataclass(frozen=True)
class Granted:
    lease_id: str
    cards: int
    root: str
    ports: tuple[int, int]
    expires_in_s: float


@dataclass(frozen=True)
class Ran:
    returncode: int
    output: str


@dataclass(frozen=True)
class Started:
    process: int


@dataclass(frozen=True)
class Read:
    process: int
    running: bool
    returncode: int | None
    matched: bool
    output: str


@dataclass(frozen=True)
class Released:
    seconds: float


@dataclass
class _Lease:
    lease: BranchLease
    sandbox: DevSandbox
    slot: int
    granted: float
    expires: float
    last_call: float
    queued_since: float | None = None
    processes: dict[int, AgentProcess] = field(default_factory=dict)


class DevLeases:
    """The session's dev sandboxes, shared by its branches; every method runs on the session's loop."""

    def __init__(
        self,
        toolbox: LeaseSource,
        sandboxes: SandboxFor,
        notebooks: Callable[[str], BranchLogAppender],
        *,
        limits: DevLimits = DevLimits(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._toolbox, self._sandboxes, self._notebooks = toolbox, sandboxes, notebooks
        self._limits, self._clock = limits, clock
        self._slots: dict[str, int] = {}
        self._held: dict[str, _Lease] = {}
        self._ended: dict[str, float] = {}
        self._reclaimed: dict[str, str] = {}
        self._watch: asyncio.Task[None] | None = None

    @property
    def limits(self) -> DevLimits:
        return self._limits

    async def lease(self, branch: str, cards: int, minutes: float) -> Granted:
        self._reclaimed.pop(branch, None)
        await self._admit(branch)
        slot = min(set(range(self._limits.cap)) - set(self._slots.values()))
        self._slots[branch] = slot
        try:
            lease = await self._leased(branch, cards)
        except BaseException:
            del self._slots[branch]
            raise
        try:
            sandbox = self._sandboxes(branch, lease)
        except BaseException:
            del self._slots[branch]
            await self._toolbox.release_branch_sandbox(lease.lease_id)
            raise
        now = self._clock()
        seconds = min(minutes * MINUTE, self._limits.max_s)
        self._held[branch] = _Lease(lease, sandbox, slot, now, now + seconds, now)
        self._watching()
        await self._note(branch, f"dev sandbox leased: {cards} card(s) for up to {seconds / MINUTE:g} min")
        return Granted(lease.lease_id, len(lease.gpu_ids), str(lease.root), self._ports(slot), seconds)

    async def run(self, branch: str, command: str, cwd: str | None, timeout_s: float) -> Ran:
        held = self._touch(branch)
        timeout = min(timeout_s, self._limits.run_s)
        done = await held.sandbox.run(command, _cwd(held, cwd), _caches(held.lease.root), timeout)
        return Ran(done.returncode, done.output)

    async def start(self, branch: str, command: str, cwd: str | None) -> Started:
        held = self._touch(branch)
        process = await held.sandbox.start(command, _cwd(held, cwd), _caches(held.lease.root))
        number = len(held.processes) + 1
        held.processes[number] = process
        return Started(number)

    async def read(self, branch: str, number: int, wait_s: float, until: str | None) -> Read:
        held = self._touch(branch)
        process = held.processes.get(number)
        if process is None:
            raise InvalidParamsError(f"your dev sandbox started no process {number}")
        try:
            pattern = re.compile(until) if until else None
        except re.error as error:
            raise InvalidParamsError(f"until is no regular expression: {error}") from error
        deadline = time.monotonic() + min(wait_s, self._limits.read_s)
        while True:
            output = await process.output()
            matched = pattern is not None and pattern.search(output) is not None
            if process.returncode is not None or matched or time.monotonic() >= deadline:
                break
            await asyncio.sleep(self._limits.read_poll_s)
        return Read(number, process.returncode is None, process.returncode, matched, output)

    async def release(self, branch: str) -> Released:
        held = self._touch(branch)
        await self._end(branch, "released by the specialist")
        return Released(self._clock() - held.granted)

    async def reclaim(self, branch: str, reason: str) -> None:
        """Take back ``branch``'s lease, if it holds one."""
        if branch in self._held:
            await self._end(branch, reason)

    async def sweep(self) -> None:
        """Take back every lease that expired, sat idle, or kept tests waiting for cards too long."""
        if not self._held:
            return
        queued = (await self._toolbox.queue_state()).queued
        now = self._clock()
        for branch, held in list(self._held.items()):
            if not queued:
                held.queued_since = None
            elif held.queued_since is None:
                held.queued_since = now
            reason = self._overdue(held, now)
            if reason:
                await self._end(branch, reason)

    def _overdue(self, held: _Lease, now: float) -> str:
        limits, quiet = self._limits, now - held.last_call
        if now >= held.expires:
            return "it expired"
        if held.queued_since is not None and now - held.queued_since >= limits.queued_s:
            return f"tests waited {limits.queued_s / MINUTE:g} min for its cards"
        if quiet >= limits.idle_running_s or quiet >= limits.idle_s and not held.sandbox.alive():
            return f"it sat idle for {quiet / MINUTE:.0f} min"
        return ""

    async def _admit(self, branch: str) -> None:
        """Refuse a lease the session's rules do not allow now."""
        if self._limits.cap < 1:
            raise DevRefused("disabled", "this session lends no dev sandboxes; submit_test is how you measure")
        if branch in self._slots:
            raise DevRefused("lease_exists", "you hold a dev sandbox already; use it or release it")
        since = self._clock() - self._ended.get(branch, -math.inf)
        if since < self._limits.cooldown_s:
            left = (self._limits.cooldown_s - since) / MINUTE
            raise DevRefused(
                "cooldown", f"your last dev sandbox ended {since / MINUTE:.0f} min ago; {left:.0f} min left"
            )
        if (await self._toolbox.queue_state()).queued:
            raise DevRefused("tests_waiting", "tests are waiting for cards; submit your test instead")
        if branch in self._slots:
            raise DevRefused("lease_exists", "you hold a dev sandbox already; use it or release it")
        if len(self._slots) >= self._limits.cap:
            raise DevRefused("cap_reached", f"the session's {self._limits.cap} dev sandbox(es) are all lent out")

    async def _leased(self, branch: str, cards: int) -> BranchLease:
        """A dev sandbox takes only cards free now: waiting would hold cards a test is about to need."""
        try:
            return await self._toolbox.lease_branch_sandbox(branch, cards, wait_sec=0)
        except RequestRejected as error:
            raise DevRefused(error.code, error.reason) from error
        except SandboxUnavailable as error:
            raise DevRefused("cards_busy", f"no {cards} card(s) are free now") from error

    def _touch(self, branch: str) -> _Lease:
        held = self._held.get(branch)
        if held is None:
            why = self._reclaimed.pop(branch, "")
            raise DevRefused("no_lease", "you hold no dev sandbox" + (f": it was taken back, {why}" if why else ""))
        held.last_call = self._clock()
        return held

    async def _end(self, branch: str, reason: str) -> None:
        held = self._held.pop(branch)
        del self._slots[branch]
        self._ended[branch] = self._clock()
        self._reclaimed[branch] = reason
        try:
            await held.sandbox.stop()
        finally:
            await self._toolbox.release_branch_sandbox(held.lease.lease_id)
        minutes = (self._clock() - held.granted) / MINUTE
        await self._note(branch, f"dev sandbox ended after {minutes:.1f} min: {reason}")

    def _watching(self) -> None:
        if self._watch is None or self._watch.done():
            self._watch = asyncio.get_running_loop().create_task(self._watch_leases())
            self._watch.add_done_callback(_report_failure)

    async def _watch_leases(self) -> None:
        while self._held:
            await asyncio.sleep(self._limits.poll_s)
            await self.sweep()

    async def _note(self, branch: str, content: str) -> None:
        entry = BranchLogEntry(branch, BranchLogType.RESULT, f"{NOTICE} {content}")
        try:
            await self._notebooks(branch).append_branch_log(entry)
        except NotebookError:
            logger.warning("the branch log of %s did not take: %s", branch, content, exc_info=True)

    def _ports(self, slot: int) -> tuple[int, int]:
        first = self._limits.port + slot * self._limits.ports
        return first, first + self._limits.ports - 1


def _report_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.error("the dev sandbox watch failed; leases end with their branch", exc_info=task.exception())


def _cwd(held: _Lease, cwd: str | None) -> Path:
    if not cwd:
        return held.lease.root
    if not Path(cwd).is_absolute():
        raise InvalidParamsError(f"cwd must be an absolute path, not {cwd!r}")
    return Path(cwd)


def _caches(root: Path) -> dict[str, str]:
    return {name: str(root / "cache" / directory) for name, directory in CACHE_DIRS.items()}


def local_dev_sandboxes(
    worktrees: WorktreeFor, processes: Processes | None, limits: DevLimits = DevLimits()
) -> SandboxFor:
    """Sandboxes on this host, importing the source of every component the branch's worktree holds."""

    def sandbox(branch: str, lease: BranchLease) -> LocalAgentSandbox:
        worktree = worktrees(branch)
        sources = tuple(
            worktree.path / PurePosixPath(layout.upstream_prefix).parent
            for component, layout in COMPONENT_LAYOUTS.items()
            if worktree.holds(component)
        )
        base_env = {**os.environ, **lease.env}
        return LocalAgentSandbox(
            lease.gpu_ids,
            tuple(dict.fromkeys(sources)),
            lease.root,
            processes,
            branch,
            base_env,
            niceness=limits.niceness,
        )

    return sandbox


# -- the MCP tools --------------------------------------------------------------------------------------------------

_STRING, _INTEGER, _NUMBER = "string", "integer", "number"
_COMMAND = {"type": _STRING, "minLength": 1, "description": "A bash command."}
_CWD = {"type": _STRING, "description": "An absolute directory to run in; the lease's root when omitted."}
_LINES = {"type": _INTEGER, "minimum": 1, "maximum": 2000, "description": f"Lines of output returned ({LINES})."}


def _schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def register_dev_sandbox_tools(
    server: McpServer, *, leases: DevLeases, home: asyncio.AbstractEventLoop | None = None
) -> None:
    """Register the five dev sandbox tools; ``home`` is the session's loop, which ``leases`` live on (the service
    answers each call on a loop of its own). Without it, calls run on the caller's loop."""

    own = {"scopes": frozenset({SCOPE_SPECIALIST})}
    limits = leases.limits
    server.register(
        "dev_sandbox_lease",
        description=(
            "Lease cards of your own for a while, to check a change runs and time a kernel before a full test; refused "
            "while tests wait for cards. Its results never enter the gate."
        ),
        schema=_schema(
            {
                "cards": {"type": _INTEGER, "minimum": 1, "description": "How many cards (1)."},
                "minutes": {
                    "type": _NUMBER,
                    "exclusiveMinimum": 0,
                    "description": f"How long, at most {limits.max_s / MINUTE:g} ({limits.max_s / MINUTE:g}).",
                },
            }
        ),
        handler=_handler(home, lambda branch, a: leases.lease(branch, a.get("cards", 1), a.get("minutes", math.inf))),
        **own,
    )
    server.register(
        "dev_sandbox_run",
        description=f"Run a command in your dev sandbox to its end, at most {limits.run_s:g} s; its output's last lines.",
        schema=_schema(
            {"command": _COMMAND, "cwd": _CWD, "timeout_s": {"type": _NUMBER}, "lines": _LINES}, ("command",)
        ),
        handler=_handler(
            home,
            lambda branch, a: leases.run(branch, a["command"], a.get("cwd"), a.get("timeout_s", limits.run_s)),
        ),
        **own,
    )
    server.register(
        "dev_sandbox_start",
        description="Start a command in your dev sandbox in the background (a server, a longer benchmark).",
        schema=_schema({"command": _COMMAND, "cwd": _CWD}, ("command",)),
        handler=_handler(home, lambda branch, a: leases.start(branch, a["command"], a.get("cwd"))),
        **own,
    )
    server.register(
        "dev_sandbox_read",
        description=(
            f"Wait, at most {limits.read_s:g} s, until a started process exits or its output matches until; "
            "its state and output's last lines."
        ),
        schema=_schema(
            {
                "process": {"type": _INTEGER, "minimum": 1},
                "wait_s": {"type": _NUMBER, "minimum": 0},
                "until": {"type": _STRING, "description": "A regular expression to wait for in the output."},
                "lines": _LINES,
            },
            ("process",),
        ),
        handler=_handler(
            home, lambda branch, a: leases.read(branch, a["process"], a.get("wait_s", 0.0), a.get("until"))
        ),
        **own,
    )
    server.register(
        "dev_sandbox_release",
        description="Kill everything in your dev sandbox and give its cards back.",
        schema=_schema({}),
        handler=_handler(home, lambda branch, a: leases.release(branch)),
        **own,
    )


def _handler(
    home: asyncio.AbstractEventLoop | None, call: Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        done = asdict(await _at_home(home, call(ctx.principal.branch, arguments)))
        if "output" in done:
            done["output"] = "\n".join(done["output"].splitlines()[-arguments.get("lines", LINES) :])
        return done

    return handle


async def _at_home(home: asyncio.AbstractEventLoop | None, call: Coroutine[Any, Any, T]) -> T:
    if home is None or asyncio.get_running_loop() is home:
        return await call
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(call, home))


__all__ = [
    "CACHE_DIRS",
    "DevLeases",
    "DevLimits",
    "DevRefused",
    "DevSandbox",
    "LeaseSource",
    "SandboxFor",
    "local_dev_sandboxes",
    "register_dev_sandbox_tools",
]
