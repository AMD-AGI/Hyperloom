# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's processes: the ones the scheduler starts itself and the ones a peer started and handed over, each in
a session of its own and recorded in ``MB-pids.jsonl``.

``proctree.collect_tree`` never signals the caller's own process group, so a child that shares it cannot be killed
cleanly; every child therefore leads a new session, and a terminal's Ctrl-C no longer reaches it, which is why the
scheduler forwards a stop explicitly. Tree kills block, so they run in a worker thread to keep the event loop and its
signal handlers responsive.

A child that exits orphans whatever it started; a descendant leading a session of its own then leaves both the child's
tree and its group. Inside :func:`adopting_orphans` such orphans are reparented to the scheduler instead of to init, so
every process the session started, registered or not, is a child of the scheduler or a descendant of one: the session
kill reaches them and the leftovers name them.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import signal
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from hyperloom.common.io import append_jsonl
from hyperloom.common.owner_scope import local_owner_scope, process_started_at
from hyperloom.common.proctree import collect_tree, descendants, group_members, kill_tree, proc_identity, running
from hyperloom_multibranch.scheduling.clock import Clock

_ZOMBIE_STATE = "Z"
#: ``prctl`` option from ``<linux/prctl.h>``.
_PR_SET_CHILD_SUBREAPER = 36
#: The branch of a leftover no registered process accounts for.
UNREGISTERED_BRANCH = ""


@contextmanager
def adopting_orphans() -> Iterator[None]:
    """Make this process the child subreaper of everything it starts, for the duration (Linux).

    An orphaned descendant is reparented to this process rather than to init. Nothing reaps it here: every adopted
    orphan, killed or exited on its own, stays a zombie until this process exits. A zombie holds only its pid, and
    ``running`` skips it, so the kills and the leftovers are unaffected. Reaping them is not safe: an in-process
    caller, such as the AI work socket, starts a child in a session of its own and registers it only after the launch,
    and until then nothing tells that child from an orphan, so a reap would take the exit status its ``poll`` needs.
    """

    _set_child_subreaper(1)
    try:
        yield
    finally:
        _set_child_subreaper(0)


def _set_child_subreaper(on: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, on, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


@dataclass(frozen=True)
class ProcessRecord:
    """One line of ``MB-pids.jsonl``; ``pgid`` equals ``pid`` because the child leads its own session.

    A leftover no registered process accounts for is described the same way, on :data:`UNREGISTERED_BRANCH`, with its
    own group and the time it started.
    """

    pid: int
    pgid: int
    started_at: float
    branch: str
    argv: tuple[str, ...]
    owner_scope: str


@dataclass(frozen=True)
class RegisteredProcess:
    record: ProcessRecord
    process: asyncio.subprocess.Process


@dataclass(frozen=True)
class _Tracked:
    """A registered process; ``process`` is ``None`` for an adopted one, which this process did not start."""

    record: ProcessRecord
    process: asyncio.subprocess.Process | None


class ProcessRegistry:
    """Starts or adopts, records and kills the session's processes, by branch or all at once."""

    def __init__(self, pids_file: Path, clock: Clock) -> None:
        self._pids_file = pids_file
        self._clock = clock
        self._registered: list[_Tracked] = []

    async def spawn(self, argv: Sequence[str], *, branch: str, log: Path) -> RegisteredProcess:
        """Start ``argv`` with stdout and stderr appended to ``log``, and record it before returning."""

        with log.open("ab") as out:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=out,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        return RegisteredProcess(self._register(process.pid, branch, argv, process), process)

    def adopt(self, pid: int, *, branch: str, argv: Sequence[str]) -> ProcessRecord:
        """Take over ``pid``, which a peer started for ``branch`` as the leader of a session of its own, so the kills
        of ``branch`` and of the session reach its process group; raises ``ValueError`` when it is not running or
        leads no group.

        The AI work socket starts the agent CLI that way; the specialist runtime adopts every process it has it start.
        The adopted process is never reaped here, so its group, not its pid, says whether it still runs.
        """

        try:
            leads_group = os.getpgid(pid) == pid
        except ProcessLookupError as exc:
            raise ValueError(f"process {pid} of {branch} is not running") from exc
        if not leads_group:
            raise ValueError(f"process {pid} of {branch} does not lead its own process group")
        return self._register(pid, branch, argv, None)

    def _register(
        self, pid: int, branch: str, argv: Sequence[str], process: asyncio.subprocess.Process | None
    ) -> ProcessRecord:
        record = ProcessRecord(
            pid=pid,
            pgid=pid,
            started_at=self._clock.now(),
            branch=branch,
            argv=tuple(argv),
            owner_scope=local_owner_scope(),
        )
        append_jsonl(self._pids_file, asdict(record))
        self._registered.append(_Tracked(record, process))
        return record

    async def terminate(self, proc: RegisteredProcess, grace_seconds: float) -> int | None:
        """SIGTERM the root alone and wait up to ``grace_seconds`` for it to exit; ``None`` when it did not.

        The root is left to wind down what it started itself. Its tree is taken before the signal, so a descendant
        that leads a session of its own, as the main flow's inference servers do, is killed afterwards even once the
        root's exit has orphaned it; :meth:`kill_branch` finds only the root's group. A descendant started after the
        signal is not reached here; inside :func:`adopting_orphans`, :meth:`kill_all` reaches it.
        """

        tree = await asyncio.to_thread(collect_tree, _live_roots(_Tracked(proc.record, proc.process)))
        if proc.process.returncode is None:
            proc.process.send_signal(signal.SIGTERM)
        try:
            return await self._clock.wait_for(proc.process.wait(), grace_seconds)
        except asyncio.TimeoutError:
            return None
        finally:
            await asyncio.to_thread(kill_tree, tree)

    async def kill_branch(self, branch: str) -> bool:
        """Kill every tree registered for ``branch``; True when nothing of them is left."""

        return await self._kill([r for r in self._registered if r.record.branch == branch])

    async def kill_all(self) -> bool:
        """Kill every registered tree and every unregistered child's; True when nothing of them is left."""

        children = await asyncio.to_thread(_unregistered_children, tuple(self._registered))
        return await self._kill(self._registered, children)

    async def leftovers(self) -> tuple[ProcessRecord, ...]:
        """The records whose root or any member of their process group is still running, then one per unregistered
        child still running."""

        registered = tuple(self._registered)
        alive = await asyncio.to_thread(_alive, registered)
        return (*(r.record for r in alive), *await asyncio.to_thread(_unregistered_records, registered))

    async def _kill(self, registered: Sequence[_Tracked], extra: Sequence[int] = ()) -> bool:
        gone = await asyncio.to_thread(_kill_trees, tuple(registered), tuple(extra))
        self._registered = list(await asyncio.to_thread(_alive, tuple(self._registered)))
        return gone


def _alive(registered: tuple[_Tracked, ...]) -> tuple[_Tracked, ...]:
    return tuple(r for r in registered if _live_roots(r))


def _kill_trees(registered: tuple[_Tracked, ...], extra: Sequence[int]) -> bool:
    """Kill every live tree of ``registered`` and of ``extra``; True when nothing of them is left. Blocks on ``/proc``
    and signals."""

    roots = [*(pid for r in registered for pid in _live_roots(r)), *extra]
    return kill_tree(collect_tree(roots)) if roots else True


def _unregistered_children(registered: tuple[_Tracked, ...]) -> tuple[int, ...]:
    """This process's running children outside its own group and every registered group. Blocks on ``/proc``.

    The scheduler's own group holds only what it runs inline, such as git; a registered group is reached through its
    record.
    """

    me = os.getpid()
    known = {os.getpgid(0), *(r.record.pgid for r in registered)}
    children = []
    for pid, _ in descendants(me):
        identity = proc_identity(pid)
        if identity is None or identity[0] != me or not running(pid):
            continue
        try:
            if os.getpgid(pid) not in known:
                children.append(pid)
        except ProcessLookupError:
            continue
    return tuple(sorted(children))


def _unregistered_records(registered: tuple[_Tracked, ...]) -> tuple[ProcessRecord, ...]:
    records = []
    for pid in _unregistered_children(registered):
        started = process_started_at(pid)
        try:
            pgid = os.getpgid(pid)
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
        except OSError:
            continue
        if started is not None:
            argv_text = tuple(arg.decode(errors="replace") for arg in argv)
            records.append(
                ProcessRecord(pid, pgid, started.timestamp(), UNREGISTERED_BRANCH, argv_text, local_owner_scope())
            )
    return tuple(records)


def _live_roots(registered: _Tracked) -> tuple[int, ...]:
    """The registered root while it runs, and every running member of its group.

    A member orphaned by its root's exit is no longer the root's descendant, but stays in the group. The asyncio
    child watcher reaps a started root, so its pid is never reused while ``returncode`` is unset; an adopted root
    leads its group, so it is among the members while it runs. A kill forgets a record whose group is empty, so a
    later group reusing the id is never mistaken for it.
    """

    record = registered.record
    members = {pid for pid, _, state in group_members(record.pgid) if state != _ZOMBIE_STATE}
    process = registered.process
    if process is not None and process.returncode is None and running(record.pid):
        members.add(record.pid)
    return tuple(sorted(members))


__all__ = ["UNREGISTERED_BRANCH", "ProcessRecord", "ProcessRegistry", "RegisteredProcess", "adopting_orphans"]
