# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End what a bring-up left running, and say precisely what that establishes.

A process group can be left at will by an unprivileged child, so a success here
covers only what was enumerable (:data:`CLAIM_REACHABLE`) and never proves the
target holds nothing.

The descendant set is collected BEFORE signalling, while a child that has left
the group is still reachable as a descendant.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from hyperloom.common.proctree import collect_tree, kill_tree

log = logging.getLogger(__name__)

#: Only what could be listed from procfs at signal time was reached: evidence,
#: not proof.
CLAIM_REACHABLE = "reachable"

#: The reap unit, by name.
BACKEND_PROCESS_GROUP = "process_group"

#: Every recorded process was signalled and then observed gone.
REAP_KILLED = "killed"

#: The holder's own task row records the end of its work, and no process was
#: ever recorded for it.
REAP_HOLDER_REPORTED = "holder_reported"

#: Something recorded for the holder survived the kill.
REAP_HOLDER_ALIVE = "holder_alive"

#: Nothing about the holder could be observed.
REAP_UNOBSERVABLE = "unobservable"

#: How long a reap waits for a kill it asked for to take effect.
REAP_CONFIRM_WINDOW_SEC: float = 3.0

#: How often it looks while it waits.
_REAP_POLL_SEC: float = 0.05

__all__ = [
    "BACKEND_PROCESS_GROUP",
    "CLAIM_REACHABLE",
    "REAP_HOLDER_ALIVE",
    "REAP_HOLDER_REPORTED",
    "REAP_KILLED",
    "REAP_UNOBSERVABLE",
    "ProcessGroupReaper",
    "Reap",
    "ReapBackend",
    "ReapTarget",
    "holder_target",
    "pid_target",
    "select_reaper",
]


@dataclass(frozen=True)
class ReapTarget:
    """What is to be ended, named by the processes that were recorded for it.

    Attributes:
        label: What this target is, for the log.
        pids: The recorded processes. Empty means nothing was recorded, which
            is unobservable rather than dead.
    """

    label: str
    pids: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Reap:
    """What a reap attempt established.

    Attributes:
        confirmed_unix: When the target was confirmed gone; ``None`` when
            nothing confirmed it, which is not the same as it being alive.
        outcome: One of the four ``REAP_*`` outcomes.
        backend: The unit that ran.
        claim: What a confirmed reap by the unit establishes.
    """

    confirmed_unix: float | None
    outcome: str
    backend: str = BACKEND_PROCESS_GROUP
    claim: str = CLAIM_REACHABLE


@runtime_checkable
class ReapBackend(Protocol):
    """A unit of reaping: ``name`` says which, ``claim`` what a confirmed reap
    by it establishes.
    """

    name: str
    claim: str

    def available(self) -> str:
        """Return the reason this unit cannot run here, or ``""`` when it can."""
        ...

    async def reap(self, target: ReapTarget, *, now_unix: float) -> Reap:
        """Attempt to end ``target`` and report what was observed."""
        ...


def pid_target(label: str, pids: Any) -> ReapTarget:
    """Build a target from an iterable of pids, dropping this process.

    Non-positive pids are dropped too, and this process is never a member of a
    target however it was recorded.
    """
    usable = {int(raw) for raw in pids if int(raw) > 0}
    usable.discard(os.getpid())
    return ReapTarget(label=label, pids=frozenset(usable))


async def holder_target(db: Any, holder_task_id: str) -> ReapTarget:
    """Build a target from the processes the lease table recorded for a holder.

    Args:
        db: The session database.
        holder_task_id: The task whose processes are to be reaped.

    Returns:
        ReapTarget: Empty when nothing was recorded for the holder.
    """
    rows = await db.fetchall(
        "SELECT DISTINCT pid FROM leases WHERE task_id = ?",
        (holder_task_id,),
    )
    return pid_target(holder_task_id, (r["pid"] for r in rows))


async def _confirm(
    probe: Any,
    *,
    backend: str,
    claim: str,
    now_unix: float,
    window_sec: float,
) -> Reap:
    """Poll ``probe`` until it reports nothing left, or ``window_sec`` runs out.

    A kill is delivered asynchronously, so a single reading taken straight after
    one proves nothing.

    Returns:
        Reap: Confirmed, or :data:`REAP_HOLDER_ALIVE`.
    """
    deadline = time.monotonic() + max(0.0, window_sec)
    while True:
        if not probe():
            return Reap(float(now_unix), REAP_KILLED, backend, claim)
        if time.monotonic() >= deadline:
            return Reap(None, REAP_HOLDER_ALIVE, backend, claim)
        await asyncio.sleep(_REAP_POLL_SEC)


class ProcessGroupReaper:
    """Signal the recorded processes, their pre-collected descendants and their
    groups, then look for what is left. Claim: :data:`CLAIM_REACHABLE`.
    """

    name = BACKEND_PROCESS_GROUP
    claim = CLAIM_REACHABLE

    def __init__(self, *, confirm_window_sec: float = REAP_CONFIRM_WINDOW_SEC):
        """Record how long to wait for a kill to take effect."""
        self.confirm_window_sec = max(0.0, float(confirm_window_sec))

    def available(self) -> str:
        """Return ``""``: signalling a pid needs nothing this host can lack."""
        return ""

    async def reap(self, target: ReapTarget, *, now_unix: float) -> Reap:
        """Kill the target's tree, confirmed only when every process it
        enumerated is gone. ``now_unix`` is dated onto a confirmed kill.
        """
        if not target.pids:
            return Reap(None, REAP_UNOBSERVABLE, self.name, self.claim)
        tree = collect_tree(target.pids)
        # The escalation blocks on its own grace, so it runs off the loop the
        # session's other work is still being driven by.
        gone = await asyncio.to_thread(kill_tree, tree, confirm_sec=self.confirm_window_sec)
        if gone:
            return Reap(float(now_unix), REAP_KILLED, self.name, self.claim)
        return Reap(None, REAP_HOLDER_ALIVE, self.name, self.claim)


def select_reaper() -> ReapBackend:
    """Build the reap unit this session will use.

    Returns:
        ReapBackend: The process-group unit.
    """
    return ProcessGroupReaper()
