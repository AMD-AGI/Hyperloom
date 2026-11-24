# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kill whatever an ended agent session left running: everything it started, or everything of ours inside a directory."""

from __future__ import annotations

import asyncio
import ctypes
import functools
import logging
import os
import signal
import subprocess
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


log = logging.getLogger(__name__)

_POLL_SEC = 0.05
# What a driver holding a GPU is given to shut itself down on SIGTERM.
_TERM_GRACE_SEC = 2.0
# SIGKILL cannot be declined, so this covers scheduling and driver teardown rather than a process deciding to linger.
_KILL_CONFIRM_SEC = 1.0

# prctl(2).
_PR_SET_CHILD_SUBREAPER = 36

# Stamped into every child's environment and read back out of ``/proc/<pid>/environ``.
_OWNER_ENV = "FORGE_CAMPAIGN_OWNER"

# Stamped into one agent session's environment and read back the same way. The campaign's tag says a process is this
# campaign's; this one says which session started it, which its working directory and its parent stop saying the
# moment it changes directory or its shell exits.
_SESSION_ENV = "FORGE_SESSION_OWNER"

# An open fd on one of these is the difference between a leftover process that merely exists and one holding the
# device the next measurement needs.
_DEVICE_PREFIXES = ("/dev/kfd", "/dev/dri/", "/dev/nvidia")


@dataclass(frozen=True)
class _Proc:
    """The ``/proc/<pid>/stat`` fields the reaper decides on."""

    pid: int
    state: str
    ppid: int
    pgid: int
    # Ticks since boot.
    starttime: int


@dataclass(frozen=True)
class ReapReport:
    """What is left in the directory once the reaper has done what it can."""

    directory: str = ""
    reaped: tuple[int, ...] = ()
    unkillable: tuple[int, ...] = ()
    foreign: tuple[int, ...] = ()
    holding_device: tuple[int, ...] = ()

    @property
    def blockers(self) -> tuple[int, ...]:
        """The processes that make the directory unsafe to measure in."""
        return tuple(sorted({*self.unkillable, *self.holding_device}))

    @property
    def contended(self) -> bool:
        """Whether the directory is unsafe to measure in."""
        return bool(self.blockers)

    def describe(self) -> str:
        """One line naming what is left, empty when nothing is."""
        parts: list[str] = []
        if self.unkillable:
            parts.append(f"pid(s) {list(self.unkillable)} survived SIGKILL")
        if self.foreign:
            parts.append(f"pid(s) {list(self.foreign)} are not this campaign's and were left alone")
        if self.holding_device:
            parts.append(f"pid(s) {list(self.holding_device)} hold a device node")
        return f"{self.directory}: " + "; ".join(parts) if parts else ""


@dataclass(frozen=True)
class SessionMark:
    """What every process one agent session starts carries in its environment."""

    token: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def environment(self) -> dict[str, str]:
        """The variable to start the session's first process with; whatever that process starts inherits it."""
        return {_SESSION_ENV: self.token}


def _read_proc(pid: int) -> _Proc | None:
    """One process's stat fields, or None if it is gone or unreadable."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    rest = data.rpartition(b") ")[2].split()
    if len(rest) < 20:
        return None
    try:
        return _Proc(
            pid=pid,
            state=rest[0].decode("ascii", "replace"),
            ppid=int(rest[1]),
            pgid=int(rest[2]),
            starttime=int(rest[19]),
        )
    except ValueError:
        return None


def _process_table() -> dict[int, _Proc]:
    """Every process on the host by pid; empty without a ``/proc``."""
    if not os.path.isdir("/proc"):
        return {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return {}
    table: dict[int, _Proc] = {}
    for entry in entries:
        if not entry.isdigit():
            continue
        proc = _read_proc(int(entry))
        if proc is not None:
            table[proc.pid] = proc
    return table


_owner_pid: int | None = None
_owner_tag = ""
_subreaper_armed = False


def _arm_subreaper() -> bool:
    """Ask for ``PR_SET_CHILD_SUBREAPER``; false where it is declined."""
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return False
    prctl.restype = ctypes.c_int
    prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    # prctl reports failure by returning -1, not by raising.
    return prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0


# --- discharging what the flag makes this process responsible for ----------

_reaper_lock = threading.RLock()
# pid -> start time of every child THIS process forked.
_spawned_children: dict[int, int] = {}
# Set whenever a child is forked.
_spawn_event = threading.Event()
# The children that existed when a bare fork() started, so the one it adds can be told from them.
_fork_snapshot: set[int] = set()
_spawns_tracked = False
_reaper_pid: int | None = None
# Self-pipe written by the SIGCHLD handler and read by the reaper thread.
_wake_read = -1
_wake_write = -1
_previous_sigchld: Any = None

# Only the fallback path sleeps, and only while the sole collectable child belongs to somebody else here.
_REAP_BACKOFF_MIN_SEC = 0.01
_REAP_BACKOFF_MAX_SEC = 1.0


def _child_pids() -> set[int]:
    """This process's direct children, zombies included."""
    found: set[int] = set()
    readable = False
    try:
        tids = os.listdir("/proc/self/task")
    except OSError:
        return found
    for tid in tids:
        try:
            with open(f"/proc/self/task/{tid}/children", "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        readable = True
        found.update(int(part) for part in data.split())
    if readable:
        return found
    # A kernel built without CONFIG_PROC_CHILDREN publishes no per-thread list, so fall back to the whole table.
    own = os.getpid()
    return {proc.pid for proc in _process_table().values() if proc.ppid == own}


def _is_spawned_here(pid: int, starttime: int) -> bool:
    """Whether ``pid`` is a child this process forked, and so not ours to reap."""
    recorded = _spawned_children.get(pid)
    # A child recorded without a start time was already gone from ``/proc`` when it was registered; treat it as ours,
    # because being wrong the other way takes an exit status somebody here is waiting for.
    return recorded is not None and (recorded < 0 or recorded == starttime)


def _remember_spawned(pid: int) -> None:
    """Record a child this process just forked. Caller holds the lock."""
    if _reaper_pid != os.getpid():
        return
    proc = _read_proc(pid)
    _spawned_children[pid] = proc.starttime if proc is not None else -1


def _before_fork() -> None:
    """Hold the reaper still across a bare ``fork()`` and note what preceded it."""
    _reaper_lock.acquire()
    _fork_snapshot.clear()
    _fork_snapshot.update(_child_pids())


def _after_fork_in_parent() -> None:
    """Claim whichever child the fork added, then let the reaper run again."""
    try:
        for pid in _child_pids() - _fork_snapshot:
            _remember_spawned(pid)
        _fork_snapshot.clear()
    finally:
        _reaper_lock.release()
    _spawn_event.set()


def _forget_reaper_state() -> None:
    """Reset in a forked child, which inherits neither the thread nor the flag."""
    global _reaper_lock, _reaper_pid, _wake_read, _wake_write
    if _wake_read >= 0:
        try:
            signal.signal(signal.SIGCHLD, _previous_sigchld or signal.SIG_DFL)
        except (ValueError, OSError, TypeError):
            # Forked off a thread that may not set handlers.
            pass
    for fd in (_wake_read, _wake_write):
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
    _wake_read = -1
    _wake_write = -1
    _reaper_lock = threading.RLock()
    _spawned_children.clear()
    _fork_snapshot.clear()
    _reaper_pid = None


def _wrap_spawner(module: Any, name: str) -> None:
    """Record the pid returned by one of CPython's child-creating primitives."""
    original = getattr(module, name, None)
    if original is None:
        return

    @functools.wraps(original)
    def _spawn(*args: Any, **kwargs: Any) -> int:
        with _reaper_lock:
            pid = original(*args, **kwargs)
            _remember_spawned(pid)
        _spawn_event.set()
        return pid

    setattr(module, name, _spawn)


def _track_spawned_children() -> None:
    """Record every child this process forks, so orphans can be told apart."""
    global _spawns_tracked
    if _spawns_tracked:
        return
    original_init = subprocess.Popen.__init__

    @functools.wraps(original_init)
    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        with _reaper_lock:
            # Recorded only once the constructor has succeeded: a Popen that raises has already reaped its own child,
            # so there is nothing left to protect and nothing that should hold a pid against reuse.
            original_init(self, *args, **kwargs)
            _remember_spawned(self.pid)
        _spawn_event.set()

    subprocess.Popen.__init__ = _init  # type: ignore[method-assign]
    try:
        import _posixsubprocess
    except ImportError:  # pragma: no cover - POSIX only
        pass
    else:
        _wrap_spawner(_posixsubprocess, "fork_exec")
    _wrap_spawner(os, "posix_spawn")
    _wrap_spawner(os, "posix_spawnp")
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_in_parent,
        after_in_child=_forget_reaper_state,
    )
    _spawns_tracked = True


def _adopt_existing_children() -> None:
    """Claim every child that already exists as one this process forked."""
    with _reaper_lock:
        for pid in _child_pids():
            if pid not in _spawned_children:
                _remember_spawned(pid)


def _reap_inherited_orphans() -> tuple[int, ...]:
    """Collect the zombies this process inherited, and only those."""
    reaped: list[int] = []
    with _reaper_lock:
        children = _child_pids()
        for pid in [pid for pid in _spawned_children if pid not in children]:
            del _spawned_children[pid]
        for pid in children:
            proc = _read_proc(pid)
            if proc is None or proc.state != "Z":
                continue
            if _is_spawned_here(pid, proc.starttime):
                continue
            try:
                collected, _ = os.waitpid(pid, os.WNOHANG)
            except OSError:
                continue
            if collected == pid:
                reaped.append(pid)
    if reaped:
        log.debug("collected inherited orphan(s) %s", sorted(reaped))
    return tuple(sorted(reaped))


def _on_sigchld(signum: int, frame: Any) -> None:
    """Wake the reaper thread and nothing else."""
    try:
        os.write(_wake_write, b"\0")
    except OSError:
        # The pipe is full, which means a wake-up is already pending, or it is closed, which means the process is
        # going away.
        pass
    if callable(_previous_sigchld):
        _previous_sigchld(signum, frame)


def _arm_sigchld() -> bool:
    """Route child deaths to the reaper thread; false where that is refused."""
    global _wake_read, _wake_write, _previous_sigchld
    try:
        read_fd, write_fd = os.pipe()
    except OSError:
        return False
    os.set_blocking(write_fd, False)
    os.set_inheritable(read_fd, False)
    os.set_inheritable(write_fd, False)
    _wake_read, _wake_write = read_fd, write_fd
    previous = signal.getsignal(signal.SIGCHLD)
    try:
        # Main thread only, which is where both callers install from.
        signal.signal(signal.SIGCHLD, _on_sigchld)
    except (ValueError, OSError):
        os.close(read_fd)
        os.close(write_fd)
        _wake_read = _wake_write = -1
        return False
    # Chained rather than replaced: whoever was handling child deaths before is still entitled to hear about them.
    _previous_sigchld = previous
    return True


def _wait_for_a_collectable_child() -> None:
    """Block until some child of this process can be collected."""
    try:
        os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOWAIT)
    except ChildProcessError:
        # No children at all, so nothing can be inherited until something here forks -- which is precisely what the
        # spawn event reports, so this waits on that rather than on a timer.
        _spawn_event.wait()
        _spawn_event.clear()
    except OSError:
        log.debug("waitid failed; the orphan reaper is stopping", exc_info=True)
        raise


def _reaper_loop() -> None:
    """Collect inherited orphans as they die, without polling for them."""
    backoff = _REAP_BACKOFF_MIN_SEC
    while True:
        if _wake_read >= 0:
            try:
                if not os.read(_wake_read, 4096):
                    return
            except OSError:
                return
            _reap_inherited_orphans()
            continue
        try:
            _wait_for_a_collectable_child()
        except OSError:
            return
        if _reap_inherited_orphans():
            backoff = _REAP_BACKOFF_MIN_SEC
            continue
        # The collectable child belongs to another waiter here, and waitid will keep naming it until that waiter takes
        # it.
        _spawn_event.wait(backoff)
        _spawn_event.clear()
        backoff = min(backoff * 2, _REAP_BACKOFF_MAX_SEC)


def _start_orphan_reaper() -> None:
    """Start collecting what the subreaper flag will send this way."""
    global _reaper_pid
    if _reaper_pid == os.getpid():
        return
    # Order matters.
    _reaper_pid = os.getpid()
    _track_spawned_children()
    _adopt_existing_children()
    _arm_sigchld()
    threading.Thread(target=_reaper_loop, name="forge-orphan-reaper", daemon=True).start()
    # One pass up front: a subreaper installed by a second session inherits whatever the first one left behind.
    _reap_inherited_orphans()


def install_child_subreaper() -> bool:
    """Become the parent this campaign's orphans fall back to, and collect them."""
    global _owner_pid, _owner_tag, _subreaper_armed
    pid = os.getpid()
    if _owner_pid == pid:
        return _subreaper_armed
    own = _read_proc(pid)
    _owner_tag = f"{pid}:{own.starttime if own is not None else 0}"
    os.environ[_OWNER_ENV] = _owner_tag
    _owner_pid = pid
    _subreaper_armed = _arm_subreaper()
    if _subreaper_armed:
        _start_orphan_reaper()
    else:
        log.debug(
            "PR_SET_CHILD_SUBREAPER unavailable; orphaned session processes "
            "will be recognised by their environment tag alone"
        )
    return _subreaper_armed


def _current_owner_tag() -> str:
    """This process's tag, empty until :func:`install_child_subreaper` ran."""
    return _owner_tag if _owner_pid == os.getpid() else ""


def _carries(pid: int, name: str, value: str) -> bool:
    """Whether a process was exec'd with ``name=value`` in its environment; never for an empty value."""
    if not value:
        return False
    try:
        with open(f"/proc/{pid}/environ", "rb") as handle:
            entries = handle.read().split(b"\0")
    except OSError:
        return False
    return f"{name}={value}".encode() in entries


def _cwd_under(pid: int, resolved: str) -> bool:
    """Whether a process is working inside ``resolved``."""
    try:
        cwd = os.path.realpath(f"/proc/{pid}/cwd")
    except OSError:
        return False
    return cwd == resolved or cwd.startswith(resolved + os.sep)


def _holds_device(pid: int) -> bool:
    """Whether a process has a device node open, by its fd table."""
    try:
        names = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return False
    for name in names:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{name}")
        except OSError:
            continue
        if target.removesuffix(" (deleted)").startswith(_DEVICE_PREFIXES):
            return True
    return False


def _children_by_parent(table: dict[int, _Proc]) -> dict[int, list[int]]:
    """The process table inverted into a parent -> children index."""
    kids: dict[int, list[int]] = {}
    for proc in table.values():
        kids.setdefault(proc.ppid, []).append(proc.pid)
    return kids


def _descendants(kids: dict[int, list[int]], root: int) -> set[int]:
    """Every process below ``root``, ``root`` itself excluded."""
    found: set[int] = set()
    stack = list(kids.get(root, ()))
    while stack:
        pid = stack.pop()
        if pid == root or pid in found:
            continue
        found.add(pid)
        stack.extend(kids.get(pid, ()))
    return found


@dataclass(frozen=True)
class _Survey:
    """Who is working in the directory right now, split by ownership."""

    # pid -> start time, this campaign's and therefore ours to signal.
    owned: dict[int, int]
    # In the directory, not ours, never signalled.
    foreign: tuple[int, ...]


def _survey(resolved: str) -> _Survey:
    """Split the processes working under ``resolved`` by who started them."""
    table = _process_table()
    own_pid = os.getpid()
    own = table.get(own_pid)
    if own is None:
        return _Survey({}, ())
    own_pgid = os.getpgrp()
    kids = _children_by_parent(table)
    ours = _descendants(kids, own_pid)
    owner = _current_owner_tag()

    def signalable(proc: _Proc) -> bool:
        # A zombie holds nothing and cannot be signalled -- and being a subreaper produces them, for as long as it
        # takes the thread installed alongside the flag to collect them.
        return proc.pid != own_pid and proc.pgid != own_pgid and proc.state != "Z"

    seeds: dict[int, int] = {}
    foreign: list[int] = []
    for proc in table.values():
        if not signalable(proc) or not _cwd_under(proc.pid, resolved):
            continue
        # A process older than the campaign cannot have descended from it, so no reading of the parent chain makes it
        # ours.
        older = proc.starttime + 1 < own.starttime
        if not older and (proc.pid in ours or _carries(proc.pid, _OWNER_ENV, owner)):
            seeds[proc.pid] = proc.starttime
        else:
            foreign.append(proc.pid)

    targets = dict(seeds)
    seed_pgids = {table[pid].pgid for pid in seeds}
    for pid in seeds:
        for kid in _descendants(kids, pid):
            proc = table.get(kid)
            if proc is not None and signalable(proc):
                targets[kid] = proc.starttime
    for pid in ours:
        proc = table.get(pid)
        if proc is not None and signalable(proc) and proc.pgid in seed_pgids:
            targets[pid] = proc.starttime
    left = tuple(sorted(pid for pid in foreign if pid not in targets))
    return _Survey(targets, left)


def _session_processes(mark: SessionMark) -> dict[int, int]:
    """The live processes one session started, pid -> start time: those carrying its mark and whatever they started.

    Unlike the survey of a directory, this reaches into this process's own group: the mark already says a process is
    the session's, which is what staying out of the group is there to guarantee where only a directory is known.
    """
    table = _process_table()
    own_pid = os.getpid()
    own = table.get(own_pid)
    if own is None:
        return {}
    kids = _children_by_parent(table)
    found: set[int] = set()
    for proc in table.values():
        # A process older than this one cannot carry a mark made after it started.
        if proc.starttime + 1 < own.starttime or not _carries(proc.pid, _SESSION_ENV, mark.token):
            continue
        found.add(proc.pid)
        found |= _descendants(kids, proc.pid)
    found.discard(own_pid)
    return {pid: table[pid].starttime for pid in found if table[pid].state != "Z"}


def _command_line(pid: int) -> str:
    """One process's command line as a single string, empty once it is gone."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def own_session_processes() -> dict[int, str]:
    """The live processes of the agent session this process runs in, pid -> command line.

    This process and its ancestors are left out -- the shell that ran it and the agent CLI above that -- so what
    comes back is what the session started beside them. Empty outside a marked session.
    """
    token = os.environ.get(_SESSION_ENV, "")
    table = _process_table()
    lineage: set[int] = set()
    pid = os.getpid()
    while pid in table and pid not in lineage:
        lineage.add(pid)
        pid = table[pid].ppid
    return {
        proc.pid: _command_line(proc.pid)
        for proc in table.values()
        if proc.pid not in lineage and proc.state != "Z" and _carries(proc.pid, _SESSION_ENV, token)
    }


def _signal(pid: int, starttime: int, sig: signal.Signals) -> None:
    """Signal one process, and only while it is still the one identified."""
    proc = _read_proc(pid)
    if proc is None or proc.starttime != starttime:
        return
    try:
        os.kill(pid, sig)
    except OSError:
        log.debug("could not signal pid %s", pid, exc_info=True)


async def _escalate(find_live: Callable[[], dict[int, int]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """SIGTERM then SIGKILL until ``find_live`` reports nothing left; returns what went and what would not."""
    loop = asyncio.get_running_loop()
    grace_end = loop.time() + _TERM_GRACE_SEC
    kill_end = grace_end + _KILL_CONFIRM_SEC
    signalled: dict[int, int] = {}
    while True:
        live = find_live()
        if not live:
            return tuple(sorted(signalled)), ()
        now = loop.time()
        if now >= kill_end:
            gone = tuple(sorted(set(signalled) - set(live)))
            return gone, tuple(sorted(live))
        term = now < grace_end
        for pid, starttime in live.items():
            if term and pid in signalled:
                continue
            sig = signal.SIGTERM if term else signal.SIGKILL
            _signal(pid, starttime, sig)
        signalled.update(live)
        await asyncio.sleep(_POLL_SEC)


def device_holders(pids: Iterable[int]) -> dict[int, int]:
    """Identify which of ``pids`` have a device node open, pid -> start time."""
    holders: dict[int, int] = {}
    for pid in pids:
        proc = _read_proc(pid)
        if proc is not None and _holds_device(pid):
            holders[pid] = proc.starttime
    return holders


def still_holding_device(holders: Mapping[int, int]) -> tuple[int, ...]:
    """Which of the recorded holders still have the device."""
    return tuple(
        sorted(
            pid
            for pid, starttime in holders.items()
            if (proc := _read_proc(pid)) is not None and proc.starttime == starttime and _holds_device(pid)
        )
    )


def processes_under(directory: str | os.PathLike[str]) -> set[int]:
    """Every process working inside a directory, this campaign's or not."""
    survey = _survey(os.path.realpath(directory))
    return set(survey.owned) | set(survey.foreign)


async def reap_processes_under(directory: str | os.PathLike[str], *, description: str) -> ReapReport:
    """Terminate this campaign's processes working under ``directory``."""
    resolved = os.path.realpath(directory)
    survey = _survey(resolved)
    reaped: tuple[int, ...] = ()
    unkillable: tuple[int, ...] = ()
    if survey.owned:
        log.info("reaping %d process(es) %s", len(survey.owned), description)
        reaped, unkillable = await _escalate(lambda: _survey(resolved).owned)
    foreign = _survey(resolved).foreign
    report = ReapReport(
        directory=str(directory),
        reaped=reaped,
        unkillable=unkillable,
        foreign=foreign,
        holding_device=tuple(sorted(pid for pid in {*unkillable, *foreign} if _holds_device(pid))),
    )
    if report.contended:
        log.warning("%s is still contended: %s", description, report.describe())
    elif foreign:
        log.info("%s", report.describe())
    return report


async def reap_session_processes(mark: SessionMark, *, directory: str, description: str) -> ReapReport:
    """Terminate whatever one agent session started that is still alive, wherever it is working.

    ``directory`` is where the session ran, named in the report; it does not bound what is found.
    """
    if _session_processes(mark):
        log.info("reaping process(es) %s", description)
    reaped, unkillable = await _escalate(lambda: _session_processes(mark))
    report = ReapReport(
        directory=directory,
        reaped=reaped,
        unkillable=unkillable,
        holding_device=tuple(sorted(pid for pid in unkillable if _holds_device(pid))),
    )
    if report.contended:
        log.warning("%s is still contended: %s", description, report.describe())
    return report


__all__ = [
    "ReapReport",
    "SessionMark",
    "device_holders",
    "install_child_subreaper",
    "own_session_processes",
    "processes_under",
    "reap_processes_under",
    "reap_session_processes",
    "still_holding_device",
]
