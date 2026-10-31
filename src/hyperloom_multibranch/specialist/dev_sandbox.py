# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist's dev sandbox: a few of the toolbox's cards lent to its branch for a while, to check that a change
runs and to time a kernel before a full test, through five MCP tools.

The session lends at most :attr:`DevLimits.cap` at once, one a branch, and none while a test waits for cards; a cap
of 0, the default, lends none. A short lease lasts at most :attr:`DevLimits.max_s`, and the runtime takes it back, never
the model: when it expires, when the branch made no call for :attr:`DevLimits.idle_s` with nothing left running (or for
:attr:`DevLimits.idle_running_s` with something), and once tests have queued for :attr:`DevLimits.queued_s`. A long lease,
for a tool that runs for hours (Forge), lasts at most :attr:`DevLimits.long_max_s` and is taken back only when it
expires, or when nothing it started runs and it sat idle. A branch
whose lease ended leases again only after :attr:`DevLimits.cooldown_s`. A branch refused for capacity (every lease lent
out, or no card free) joins a line and is told its place; a lease given back is held for the first in line for
:attr:`DevLimits.hold_s`, then passes to the next, and a stopped branch leaves the line. The branch's sessions give its
lease back when a run ends or is stopped, unless it is :meth:`DevLeases.busy` (a long one with a process running,
which outlives the run), and every lease when the branch stops
(:class:`~hyperloom_multibranch.specialist.agent_session.DevLeaseHolder`); the toolbox's close releases whatever is left.

Commands run in a :class:`~hyperloom_multibranch.specialist.agent_sandbox.LocalAgentSandbox` on the leased cards, at a
lower CPU priority, or, on the hosted backend, in the leased ``dev`` sandbox
(:class:`~hyperloom_multibranch.specialist.remote_sandbox.RemoteAgentSandbox`), whose root is on the shared storage
below the branch's directory. Either way they import the branch worktree's source, with every JIT and kernel cache in
the sandbox's root: a cache beside the package it compiles for would be in the worktree, and its files in the next
patch. A lease's sandbox lives as long as the lease asks, unless the toolbox lets it live less, which then bounds the
lease. A lease whose sandbox is not up within :attr:`DevLimits.lease_wait_s` answers that it is starting, so the MCP
call returns, and the specialist's next lease call waits for the same sandbox again; a command meanwhile is refused
(``lease_not_ready``). Nothing a dev sandbox measures is written as a test, claim or judgement; its records are branch-log notices and the LLM spend the
Forge runs under its root keep in their ledgers, charged to the branch every :attr:`DevLimits.spend_s` and when the
lease ends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, TypeVar

from hyperloom.hosted.mcp_server import CallContext, InvalidParamsError, McpServer, ToolError
from hyperloom.measure.compile import COMPONENT_LAYOUTS
from hyperloom.measure.ports import DEV_SANDBOX_PORTS
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.blackboard.interfaces import BlackboardError
from hyperloom_multibranch.contracts.blackboard.records import BranchLogRecord, LlmUsage, LlmUsageRecord
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind, UsageSource
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.toolbox import (
    SCOPE_SPECIALIST,
    BranchLease,
    LeaseCharge,
    QueueState,
    RequestRejected,
)
from hyperloom_multibranch.specialist.agent_onboarding import AgentProcess, AgentSandbox
from hyperloom_multibranch.specialist.agent_sandbox import LocalAgentSandbox
from hyperloom_multibranch.specialist.agent_session import Processes
from hyperloom_multibranch.specialist.judging import NOTICE
from hyperloom_multibranch.specialist.remote_sandbox import RemoteAgentSandbox, forwarded_env
from hyperloom_multibranch.specialist.tools import WorktreeFor
from hyperloom_multibranch.toolbox.hosted_leases import HostedBranchLease

logger = logging.getLogger(__name__)
T = TypeVar("T")

#: The cache variables a sandbox command gets, each a directory below the lease root's ``cache``.
CACHE_DIRS = {"TRITON_CACHE_DIR": "triton", "AITER_JIT_DIR": "aiter-jit", "TORCH_EXTENSIONS_DIR": "torch-extensions"}
#: Lines of output a reply carries unless the caller asks for others.
LINES = 100
MINUTE = 60.0
#: Where, in a branch's directory, the roots of its hosted dev sandboxes are, one per lease.
DEV_DIR = "dev"
#: The refusals that put a branch in line: every lease lent out, or no card free (the hosted backend's quota included).
CAPACITY = frozenset({"cap_reached", "cards_busy"})


class LeaseTerm(str, Enum):
    """How long a lease may last: ``short`` for a check or a kernel timing, ``long`` for a run of hours."""

    SHORT = "short"
    LONG = "long"


@dataclass(frozen=True)
class DevLimits:
    """How many leases the session lends at once, how long one lasts and when the runtime takes it back; how long a
    call may hold the MCP request, which times out at 60 s; and the ports and CPU priority a lease's commands get."""

    cap: int = 0
    #: The cards a lease takes when the specialist names none; the hosted backend leases the session's TP only.
    cards: int = 1
    max_s: float = 1800.0
    #: A long lease, for work that runs for hours (a Forge fusion or rewrite): taken back only when it expires, or once
    #: nothing it started runs and it sat idle.
    long_max_s: float = 6 * 3600.0
    #: How often a held lease's Forge spend is charged, so the session's dollar cap sees it during a long lease.
    spend_s: float = 60.0
    queued_s: float = 300.0
    cooldown_s: float = 600.0
    #: How long a lease given back is held for the first branch in line before it passes to the next.
    hold_s: float = 300.0
    idle_s: float = 600.0
    idle_running_s: float = 900.0
    #: How long a lease call waits for its sandbox before it answers that the sandbox is starting.
    lease_wait_s: float = 45.0
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
        self,
        branch_id: str,
        cards: int,
        *,
        wait_sec: float | None = None,
        charge: LeaseCharge | None = None,
        lifetime_sec: float | None = None,
    ) -> BranchLease: ...

    async def release_branch_sandbox(self, lease_id: str) -> None: ...


class DevSandbox(AgentSandbox, Protocol):
    #: The sandbox's own directory, on this host's file system, where its caches and the specialist's files go.
    root: Path

    @property
    def lifetime_s(self) -> float:
        """How long the sandbox lives from its lease; ``math.inf`` when only a stop ends it."""

    def alive(self) -> bool:
        """Whether any process of the sandbox is left."""

    async def stop(self) -> None:
        """Kill every process of the sandbox."""


class BranchLogAppender(Protocol):
    async def append_branch_log(self, entry: BranchLogEntry) -> BranchLogRecord: ...

    async def record_llm_usage(self, usage: LlmUsage) -> LlmUsageRecord: ...


#: The sandbox of a branch's lease.
SandboxFor = Callable[[str, BranchLease], DevSandbox]


class DevRefused(ToolError):
    """A dev sandbox call refused; ``code`` names the check, for the specialist to react to."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code, self.reason = code, reason


@dataclass(frozen=True)
class Granted:
    lease_id: str
    cards: int
    root: str
    ports: tuple[int, int]
    expires_in_s: float


@dataclass(frozen=True)
class Starting:
    detail: str
    state: str = "starting"


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


#: What one Forge ledger says its runs spent: dollars and tokens.
Spend = tuple[float, TokenCounts]


@dataclass
class _Lease:
    lease: BranchLease
    sandbox: DevSandbox
    slot: int
    granted: float
    expires: float
    last_call: float
    term: LeaseTerm = LeaseTerm.SHORT
    queued_since: float | None = None
    processes: dict[int, AgentProcess] = field(default_factory=dict)
    charged: dict[Path, Spend] = field(default_factory=dict)
    charged_at: float = 0.0
    #: Whether the branch log says the lease outlived a run, since the branch's last call.
    kept_noted: bool = False

    def busy(self) -> bool:
        """A long lease with something running: its own rules keep it, its specialist's runs aside."""
        return self.term is LeaseTerm.LONG and self.sandbox.alive()


class DevLeases:
    """The session's dev sandboxes, shared by its branches; every method runs on the session's loop."""

    def __init__(
        self,
        toolbox: LeaseSource,
        sandboxes: SandboxFor,
        blackboards: Callable[[str], BranchLogAppender],
        *,
        limits: DevLimits = DevLimits(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._toolbox, self._sandboxes, self._blackboards = toolbox, sandboxes, blackboards
        self._limits, self._clock = limits, clock
        self._slots: dict[str, int] = {}
        self._held: dict[str, _Lease] = {}
        self._starting: dict[str, asyncio.Task[Granted]] = {}
        self._ended: dict[str, float] = {}
        self._reclaimed: dict[str, str] = {}
        #: The branches refused for capacity, first come first, each until it is granted or stopped; the first as many
        #: as there are free slots have one held for them, until the time in ``_holds``.
        self._line: list[str] = []
        self._holds: dict[str, float] = {}
        self._watch: asyncio.Task[None] | None = None

    @property
    def limits(self) -> DevLimits:
        return self._limits

    async def lease(
        self, branch: str, cards: int, minutes: float, term: LeaseTerm = LeaseTerm.SHORT
    ) -> Granted | Starting:
        """Lease ``cards`` for ``minutes`` at most; :class:`Starting` when its sandbox is not up within
        :attr:`DevLimits.lease_wait_s`, and the lease the branch's next call waits for again."""

        pending = self._starting.get(branch)
        if pending is None:
            self._reclaimed.pop(branch, None)
            await self._admit(branch)
            self._holds.pop(branch, None)
            slot = min(set(range(self._limits.cap)) - set(self._slots.values()))
            self._slots[branch] = slot
            longest = self._limits.long_max_s if term is LeaseTerm.LONG else self._limits.max_s
            pending = asyncio.get_running_loop().create_task(
                self._granted(branch, cards, min(minutes * MINUTE, longest), term)
            )
            self._starting[branch] = pending
        done, _ = await asyncio.wait({pending}, timeout=self._limits.lease_wait_s)
        if not done:
            return Starting("the sandbox is starting; call dev_sandbox_lease again for it")
        self._starting.pop(branch, None)
        try:
            granted = pending.result()
        except DevRefused as refused:
            if refused.code not in CAPACITY:
                self._leave_line(branch)
                raise
            if branch not in self._line:
                self._line.append(branch)
            raise DevRefused(refused.code, f"{refused.reason}; {self._place(branch)}") from refused
        except BaseException:
            self._leave_line(branch)
            raise
        finally:
            await self._offer()
        self._leave_line(branch)
        return granted

    async def _granted(self, branch: str, cards: int, seconds: float, term: LeaseTerm) -> Granted:
        """Lease ``branch``'s sandbox for ``seconds`` at most, the slot taken; a sandbox the toolbox lets live less
        bounds the lease."""

        slot = self._slots[branch]
        try:
            lease = await self._leased(branch, cards, seconds)
        except BaseException:
            del self._slots[branch]
            raise
        try:
            sandbox = self._sandboxes(branch, lease)
        except BaseException:
            del self._slots[branch]
            await self._toolbox.release_branch_sandbox(lease.lease_id)
            raise
        now, seconds = self._clock(), min(seconds, sandbox.lifetime_s)
        self._held[branch] = _Lease(lease, sandbox, slot, now, now + seconds, now, term)
        self._watching()
        await self._note(
            branch, f"dev sandbox leased ({term.value}): {cards} card(s) for up to {seconds / MINUTE:g} min"
        )
        return Granted(lease.lease_id, len(lease.gpu_ids), str(sandbox.root), self._ports(slot), seconds)

    async def run(self, branch: str, command: str, cwd: str | None, timeout_s: float) -> Ran:
        held = self._touch(branch)
        timeout = min(timeout_s, self._limits.run_s)
        done = await held.sandbox.run(command, _cwd(held, cwd), _caches(held.sandbox.root), timeout)
        return Ran(done.returncode, done.output)

    async def start(self, branch: str, command: str, cwd: str | None) -> Started:
        held = self._touch(branch)
        process = await held.sandbox.start(command, _cwd(held, cwd), _caches(held.sandbox.root))
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

    def busy(self, branch: str) -> bool:
        """Whether ``branch`` holds a long lease with a process running."""
        held = self._held.get(branch)
        return held is not None and held.busy()

    async def reclaim(self, branch: str, reason: str) -> None:
        """Take back ``branch``'s lease, if it holds one or one is starting."""
        await self._cancel_start(branch)
        self._leave_line(branch)
        if branch in self._held:
            await self._end(branch, reason)
        await self._offer()

    async def reclaim_unless_busy(self, branch: str, reason: str) -> None:
        """Take back ``branch``'s lease, if it holds one, unless it is :meth:`busy`: that one stays until its own rules
        end it, and the branch log says so once until the branch calls again, so the next session reads its process."""
        await self._cancel_start(branch)
        held = self._held.get(branch)
        if held is None:
            return
        if not held.busy():
            await self._end(branch, reason)
        elif not held.kept_noted:
            held.kept_noted = True
            running = ", ".join(str(number) for number, process in held.processes.items() if process.returncode is None)
            what = f"process {running}" if running else "what you started"
            await self._note(
                branch,
                f"dev sandbox kept, though {reason}: its long lease still runs {what}; read it with dev_sandbox_read",
            )

    async def _cancel_start(self, branch: str) -> None:
        """Abandon ``branch``'s lease that is starting; the toolbox ends its sandbox once made."""
        pending = self._starting.pop(branch, None)
        if pending is None:
            return
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        if branch in self._held:
            await self._end(branch, "it was abandoned while starting")

    async def sweep(self) -> None:
        """Take back every lease that expired, sat idle, or kept tests waiting for cards too long."""
        await self._offer()
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
            elif now - held.charged_at >= self._limits.spend_s:
                held.charged_at = now
                await self._charge(branch, held)

    def _overdue(self, held: _Lease, now: float) -> str:
        limits, quiet = self._limits, now - held.last_call
        if now >= held.expires:
            return "it expired"
        if held.term is LeaseTerm.LONG:
            return "" if held.busy() or quiet < limits.idle_s else f"it sat idle for {quiet / MINUTE:.0f} min"
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
        await self._offer()
        if branch not in self._holds and len(self._slots) + len(self._holds) >= self._limits.cap:
            if branch not in self._line:
                self._line.append(branch)
            raise DevRefused(
                "cap_reached",
                f"the session's {self._limits.cap} dev sandbox(es) are all lent out; {self._place(branch)}",
            )

    async def _offer(self) -> None:
        """Pass on the holds that lapsed, and hold each free slot for the next branch in line, telling it so."""
        now = self._clock()
        for branch, until in list(self._holds.items()):
            if now >= until:
                self._leave_line(branch)
                await self._note(branch, "the dev sandbox held for you passed to the next in line; ask again to rejoin")
        free = self._limits.cap - len(self._slots)
        for branch in [waiting for waiting in self._line if waiting not in self._slots][:free]:
            if branch not in self._holds:
                self._holds[branch] = now + self._limits.hold_s
                await self._note(
                    branch,
                    f"a dev sandbox is free and held for you for {self._limits.hold_s / MINUTE:g} min: "
                    "call dev_sandbox_lease to take it",
                )

    def _leave_line(self, branch: str) -> None:
        self._holds.pop(branch, None)
        if branch in self._line:
            self._line.remove(branch)

    def _place(self, branch: str) -> str:
        return (
            f"you are {_ordinal(self._line.index(branch) + 1)} in line: keep working and call dev_sandbox_lease "
            f"again later; a lease given back is held for the first in line for {self._limits.hold_s / MINUTE:g} min"
        )

    async def _leased(self, branch: str, cards: int, lifetime_sec: float) -> BranchLease:
        """A dev sandbox takes only cards free now: waiting would hold cards a test is about to need."""
        try:
            return await self._toolbox.lease_branch_sandbox(
                branch, cards, wait_sec=0, charge=LeaseCharge(LeaseKind.DEV, branch), lifetime_sec=lifetime_sec
            )
        except RequestRejected as error:
            raise DevRefused(error.code, error.reason) from error
        except SandboxUnavailable as error:
            raise DevRefused("cards_busy", f"no {cards} card(s) are free now") from error

    def _touch(self, branch: str) -> _Lease:
        held = self._held.get(branch)
        if held is None and branch in self._starting:
            raise DevRefused("lease_not_ready", "your dev sandbox is still starting; call dev_sandbox_lease for it")
        if held is None:
            why = self._reclaimed.pop(branch, "")
            raise DevRefused("no_lease", "you hold no dev sandbox" + (f": it was taken back, {why}" if why else ""))
        held.last_call = self._clock()
        held.kept_noted = False
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
        await self._offer()
        minutes = (self._clock() - held.granted) / MINUTE
        await self._note(branch, f"dev sandbox ended after {minutes:.1f} min: {reason}")
        await self._charge(branch, held)

    async def _charge(self, branch: str, held: _Lease) -> None:
        """Record what the Forge runs under the lease spent since the last charge."""
        spent = await asyncio.to_thread(forge_spend, held.sandbox.root)
        cost, tokens = 0.0, TokenCounts()
        for ledger, now in spent.items():
            before = held.charged.get(ledger, (0.0, TokenCounts()))
            # A ledger whose total fell was restarted by a rerun in the same directory: all it holds is new.
            new = now if _fell(now, before) else (now[0] - before[0], _less(now[1], before[1]))
            cost, tokens = cost + new[0], _plus(tokens, new[1])
        if cost <= 0 and tokens == TokenCounts():
            return
        held.charged.update(spent)
        await self._blackboards(branch).record_llm_usage(
            LlmUsage(branch, UsageSource.SPECIALIST, cost, tokens, f"forge:{held.lease.lease_id}")
        )

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
            await self._blackboards(branch).append_branch_log(entry)
        except BlackboardError:
            logger.warning("the branch log of %s did not take: %s", branch, content, exc_info=True)

    def _ports(self, slot: int) -> tuple[int, int]:
        first = self._limits.port + slot * self._limits.ports
        return first, first + self._limits.ports - 1


#: The ledger Forge's loop and FlyDSL rewrite keep in each experiments directory (``kernelforge.tracker.usage_ledger``):
#: the running total of the LLM calls made from it.
FORGE_LEDGER = "llm_usage.json"


def forge_spend(root: Path) -> dict[Path, Spend]:
    """What each Forge ledger under ``root`` says its runs spent on LLM calls, by ledger path."""
    spent = {}
    for path in sorted(root.rglob(FORGE_LEDGER)):
        ledger = json.loads(path.read_text(encoding="utf-8"))
        tokens = TokenCounts(
            input=int(ledger.get("input_tokens", 0)),
            output=int(ledger.get("output_tokens", 0)),
            cache_read=int(ledger.get("cache_read_input_tokens", 0)),
            cache_write=int(ledger.get("cache_creation_input_tokens", 0)),
        )
        spent[path] = (float(ledger.get("total_cost_usd", 0.0)), tokens)
    return spent


def _ordinal(place: int) -> str:
    digits = f"0{place}"
    return f"{place}{'th' if digits[-2] == '1' else {'1': 'st', '2': 'nd', '3': 'rd'}.get(digits[-1], 'th')}"


def _less(total: TokenCounts, before: TokenCounts) -> TokenCounts:
    return TokenCounts(**{item.name: getattr(total, item.name) - getattr(before, item.name) for item in fields(total)})


def _plus(one: TokenCounts, other: TokenCounts) -> TokenCounts:
    return TokenCounts(**{item.name: getattr(one, item.name) + getattr(other, item.name) for item in fields(one)})


def _fell(now: Spend, before: Spend) -> bool:
    return now[0] < before[0] or any(getattr(now[1], f.name) < getattr(before[1], f.name) for f in fields(now[1]))


def _report_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.error("the dev sandbox watch failed; leases end with their branch", exc_info=task.exception())


def _cwd(held: _Lease, cwd: str | None) -> Path:
    if not cwd:
        return held.sandbox.root
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
        base_env = {**os.environ, **lease.env}
        return LocalAgentSandbox(
            lease.gpu_ids,
            _sources(worktrees, branch),
            lease.root,
            processes,
            branch,
            base_env,
            niceness=limits.niceness,
        )

    return sandbox


def hosted_dev_sandboxes(worktrees: WorktreeFor, branch_dir: Callable[[str], Path]) -> SandboxFor:
    """The leased ``dev`` sandboxes of the hosted backend, importing what :func:`local_dev_sandboxes` imports, each
    rooted on the shared storage at ``dev/<lease id>`` below the branch's directory, ``branch_dir`` gives."""

    def sandbox(branch: str, lease: BranchLease) -> RemoteAgentSandbox:
        if not isinstance(lease, HostedBranchLease):
            raise TypeError(f"lease {lease.lease_id} holds no hosted sandbox")
        return RemoteAgentSandbox(
            lease.sandbox,
            branch_dir(branch) / DEV_DIR / lease.lease_id,
            _sources(worktrees, branch),
            len(lease.gpu_ids),
            forwarded_env(os.environ),
            lifetime_s=lease.lifetime_sec,
        )

    return sandbox


def _sources(worktrees: WorktreeFor, branch: str) -> tuple[Path, ...]:
    """The source directory of every component ``branch``'s worktree holds, each once."""

    worktree = worktrees(branch)
    sources: Iterable[Path] = (
        worktree.path / PurePosixPath(layout.upstream_prefix).parent
        for component, layout in COMPONENT_LAYOUTS.items()
        if worktree.holds(component)
    )
    return tuple(dict.fromkeys(sources))


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

    for build in (_lease_tool, _run_tool, _start_tool, _read_tool, _release_tool):
        tool = build(leases)
        server.register(
            tool.name,
            description=tool.description,
            schema=tool.schema,
            handler=_handler(home, tool.call),
            scopes=frozenset({SCOPE_SPECIALIST}),
        )


#: A dev sandbox tool's call, on the calling branch with the call's arguments.
_DevCall = Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]


@dataclass(frozen=True)
class _DevTool:
    name: str
    description: str
    schema: dict[str, Any]
    call: _DevCall


def _lease_tool(leases: DevLeases) -> _DevTool:
    most, longest = leases.limits.max_s / MINUTE, leases.limits.long_max_s / MINUTE
    return _DevTool(
        "dev_sandbox_lease",
        "Lease cards of your own for a while, to check a change runs and time a kernel before a full test, or, as a "
        "long lease, to run a tool for hours (a Forge fusion or rewrite); refused while tests wait for cards. Its "
        "results never enter the gate. A sandbox may take minutes to start: while this answers starting, call it "
        "again. Refused because all are lent out, you are put in line and told your place: keep working and call it "
        "again; a lease given back is held for the first in line for a few minutes.",
        _schema(
            {
                "cards": {"type": _INTEGER, "minimum": 1, "description": f"How many cards ({leases.limits.cards})."},
                "minutes": {
                    "type": _NUMBER,
                    "exclusiveMinimum": 0,
                    "description": f"How long, at most {most:g} short or {longest:g} long (the most).",
                },
                "term": {
                    "type": "string",
                    "enum": [term.value for term in LeaseTerm],
                    "description": "short (default): taken back when idle or when tests queue for its cards; long: "
                    "kept while a process it started runs, until it expires.",
                },
            }
        ),
        lambda branch, a: leases.lease(
            branch, a.get("cards", leases.limits.cards), a.get("minutes", math.inf), _term(a)
        ),
    )


def _term(arguments: dict[str, Any]) -> LeaseTerm:
    try:
        return LeaseTerm(arguments.get("term", LeaseTerm.SHORT.value))
    except ValueError as error:
        raise InvalidParamsError(f"term is short or long, not {arguments.get('term')!r}") from error


def _run_tool(leases: DevLeases) -> _DevTool:
    run_s = leases.limits.run_s
    return _DevTool(
        "dev_sandbox_run",
        f"Run a command in your dev sandbox to its end, at most {run_s:g} s; its output's last lines.",
        _schema({"command": _COMMAND, "cwd": _CWD, "timeout_s": {"type": _NUMBER}, "lines": _LINES}, ("command",)),
        lambda branch, a: leases.run(branch, a["command"], a.get("cwd"), a.get("timeout_s", run_s)),
    )


def _start_tool(leases: DevLeases) -> _DevTool:
    return _DevTool(
        "dev_sandbox_start",
        "Start a command in your dev sandbox in the background (a server, a longer benchmark).",
        _schema({"command": _COMMAND, "cwd": _CWD}, ("command",)),
        lambda branch, a: leases.start(branch, a["command"], a.get("cwd")),
    )


def _read_tool(leases: DevLeases) -> _DevTool:
    return _DevTool(
        "dev_sandbox_read",
        f"Wait, at most {leases.limits.read_s:g} s, until a started process exits or its output matches until; "
        "its state and output's last lines.",
        _schema(
            {
                "process": {"type": _INTEGER, "minimum": 1},
                "wait_s": {"type": _NUMBER, "minimum": 0},
                "until": {"type": _STRING, "description": "A regular expression to wait for in the output."},
                "lines": _LINES,
            },
            ("process",),
        ),
        lambda branch, a: leases.read(branch, a["process"], a.get("wait_s", 0.0), a.get("until")),
    )


def _release_tool(leases: DevLeases) -> _DevTool:
    return _DevTool(
        "dev_sandbox_release",
        "Kill everything in your dev sandbox and give its cards back.",
        _schema({}),
        lambda branch, a: leases.release(branch),
    )


def _handler(home: asyncio.AbstractEventLoop | None, call: _DevCall) -> Callable[[dict[str, Any], CallContext], Any]:
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
    "Starting",
    "hosted_dev_sandboxes",
    "local_dev_sandboxes",
    "register_dev_sandbox_tools",
]
