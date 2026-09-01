# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kill whatever an ended agent session left running inside a directory.

An agent runs its build, test and benchmark commands from its own shell, each
detached into its own session, so a command still running when the session ends
outlives it. That is not only the session's own lost cause: it holds the device
that the canonical validation and benchmark are about to use, which corrupts the
KEEP/REVERT decision for the whole iteration.

What may be signalled is decided by ownership, not by location. A process is
this campaign's if it descends from this one -- ``PR_SET_CHILD_SUBREAPER`` keeps
that true after the shell that started it exits, so an orphaned benchmark
reparents here instead of to init -- or if it carries the environment tag
stamped into every child. The directory only narrows that set: a human's shell,
a parallel campaign, or a leftover from a previous run can be working in the
same workspace, and none of them is ours to kill. Those are reported instead,
and a report that says the directory is still contended is a reason to skip the
measurement rather than take one that cannot be trusted.

Asking for that flag is an obligation as well as a capability. An orphan that
reparents here has no other parent left to collect it, so if this process never
waits on it, it stays a zombie for the life of the campaign -- holding its
process group open, invisible to the scan below (a zombie is deliberately not
signalable), and multiplied by every detached benchmark an 11-hour run starts.
So the flag and the thread that discharges what it inherits are installed
together, and both are described under :func:`install_child_subreaper`.

Two callers need exactly this -- the claude backend when a session outruns its
wall-clock budget, and the lane fan-out when a lane copy is about to be deleted.
This module sits at the ``kernelforge.llm`` layer because that is the lower of the two
and the only place both can import from.
"""

from __future__ import annotations

import asyncio
import ctypes
import functools
import logging
import os
import signal
import subprocess
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


log = logging.getLogger(__name__)

_POLL_SEC = 0.05
# What a driver holding a GPU is given to shut itself down on SIGTERM.
_TERM_GRACE_SEC = 2.0
# SIGKILL cannot be declined, so this covers scheduling and driver teardown
# rather than a process deciding to linger.
_KILL_CONFIRM_SEC = 1.0

# prctl(2). Makes orphaned descendants reparent to this process instead of to
# init, which is what keeps a detached benchmark attributable to the session
# that started it once that session's shell is gone.
_PR_SET_CHILD_SUBREAPER = 36

# Stamped into every child's environment and read back out of
# ``/proc/<pid>/environ``. Carries this process's start time as well as its pid
# so a recycled pid cannot inherit ownership of someone else's processes.
_OWNER_ENV = "FORGE_CAMPAIGN_OWNER"

# An open fd on one of these is the difference between a leftover process that
# merely exists and one holding the device the next measurement needs.
_DEVICE_PREFIXES = ("/dev/kfd", "/dev/dri/", "/dev/nvidia")


@dataclass(frozen=True)
class _Proc:
    """The ``/proc/<pid>/stat`` fields the reaper decides on."""

    pid: int
    state: str
    ppid: int
    pgid: int
    # Ticks since boot. Together with the pid this is an identity that survives
    # pid reuse, which is what every kill here is keyed on.
    starttime: int


@dataclass(frozen=True)
class ReapReport:
    """What is left in the directory once the reaper has done what it can.

    ``unkillable`` are this campaign's own processes that survived SIGKILL;
    ``foreign`` are processes working in the directory that are not this
    campaign's and were therefore never signalled. Neither is fatal on its own
    -- a leftover editor is not a benchmark -- so callers key on ``contended``:
    something is still holding the device, and a measurement taken now would be
    measuring it too.
    """

    directory: str = ""
    reaped: tuple[int, ...] = ()
    unkillable: tuple[int, ...] = ()
    foreign: tuple[int, ...] = ()
    holding_device: tuple[int, ...] = ()

    @property
    def blockers(self) -> tuple[int, ...]:
        """The processes that make the directory unsafe to measure in.

        Named separately from ``contended`` because a caller that refuses a
        measurement usually also has to say what it is waiting on, and the
        answer has to outlive the directory this report is about: a lane copy
        is deleted the moment its round ends, and these pids are all that is
        left to ask about afterwards.
        """
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


def _read_proc(pid: int) -> _Proc | None:
    """One process's stat fields, or None if it is gone or unreadable.

    ``comm`` is chosen by the process and may contain spaces and parentheses,
    so the fields after it are found from the last ``") "`` rather than by
    splitting the whole line.
    """
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
#
# The whole difficulty is that this process has two kinds of child. The ones it
# forked are being waited on by somebody here -- an asyncio subprocess
# transport, a ``Popen.wait()``, the agent SDK's own transport -- and taking one
# of their exit statuses with a blanket ``waitpid(-1)`` corrupts what that
# waiter reports. The ones the kernel reparented here because of the subreaper
# flag are being waited on by nobody, and are exactly the ones that must be
# collected. Telling them apart is therefore the entire design, and the answer
# is to record the first kind as it is created -- at the handful of primitives
# that create it, since no API above them catches them all.

_reaper_lock = threading.RLock()
# pid -> start time of every child THIS process forked. Start times because pids
# are recycled: a dead child's claim must not cover a later arrival that happens
# to land on the same number.
_spawned_children: dict[int, int] = {}
# Set whenever a child is forked. Only the fallback wake-up below waits on it,
# to avoid sleeping on a timer when there is nothing alive to wait for.
_spawn_event = threading.Event()
# The children that existed when a bare fork() started, so the one it adds can
# be told from them. Only ever touched between the two at-fork handlers below,
# which run on the forking thread with the reaper lock held.
_fork_snapshot: set[int] = set()
_spawns_tracked = False
_reaper_pid: int | None = None
# Self-pipe written by the SIGCHLD handler and read by the reaper thread.
_wake_read = -1
_wake_write = -1
_previous_sigchld: Any = None

# Only the fallback path sleeps, and only while the sole collectable child
# belongs to somebody else here. Bounded and growing, because that state is
# resolved by another thread getting round to its own ``waitpid``.
_REAP_BACKOFF_MIN_SEC = 0.01
_REAP_BACKOFF_MAX_SEC = 1.0


def _child_pids() -> set[int]:
    """This process's direct children, zombies included.

    Read per thread because that is how the kernel keeps the list: a child
    belongs to the thread that forked it, and a reparented orphan is attached to
    whichever thread of ours was alive to take it.
    """
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
    # A kernel built without CONFIG_PROC_CHILDREN publishes no per-thread list,
    # so fall back to the whole table. Costlier per wake-up, and never the path
    # taken where the cheap one exists.
    own = os.getpid()
    return {proc.pid for proc in _process_table().values() if proc.ppid == own}


def _is_spawned_here(pid: int, starttime: int) -> bool:
    """Whether ``pid`` is a child this process forked, and so not ours to reap."""
    recorded = _spawned_children.get(pid)
    # A child recorded without a start time was already gone from ``/proc`` when
    # it was registered; treat it as ours, because being wrong the other way
    # takes an exit status somebody here is waiting for.
    return recorded is not None and (recorded < 0 or recorded == starttime)


def _remember_spawned(pid: int) -> None:
    """Record a child this process just forked. Caller holds the lock.

    A no-op where no reaper is running, so that a forked child -- which inherits
    the hooks below but neither the flag nor the thread -- does not accumulate a
    record nothing will ever read or prune.
    """
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
    """Reset in a forked child, which inherits neither the thread nor the flag.

    The lock is replaced rather than released: it was acquired by the forking
    thread, which does not exist here, and the inherited copy is locked. The
    wake-up pipe is closed for the same reason -- the reader is in the parent --
    and the handler that writes to it goes back to whatever this process's
    disposition was before, so a ``multiprocessing`` worker is not left running
    somebody else's signal handler.
    """
    global _reaper_lock, _reaper_pid, _wake_read, _wake_write
    if _wake_read >= 0:
        try:
            signal.signal(signal.SIGCHLD, _previous_sigchld or signal.SIG_DFL)
        except (ValueError, OSError, TypeError):
            # Forked off a thread that may not set handlers. Harmless: the
            # inherited handler finds no pipe to write to and does nothing.
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
    """Record the pid returned by one of CPython's child-creating primitives.

    The lock is held ACROSS the call, not merely around the bookkeeping. A child
    can die between the fork returning and its pid reaching the record, and a
    reaper scanning in that window would see a child of ours that the record
    does not mention and collect it out from under its real waiter.
    """
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
    """Record every child this process forks, so orphans can be told apart.

    There is no single seam that catches them all, so each way of gaining a
    child is hooked where that code path is stable:

    * ``subprocess.Popen.__init__`` -- ``subprocess`` itself, ``asyncio``'s
      subprocess transport, and the agent SDK through ``anyio.open_process``.
      Hooked at the constructor rather than under it because ``subprocess``
      binds its primitive at import time (``from _posixsubprocess import
      fork_exec as _fork_exec``), so patching the module would not be seen, and
      because the constructor covers the ``posix_spawn`` path as well.
    * ``_posixsubprocess.fork_exec`` -- ``multiprocessing``'s spawn and
      forkserver contexts, whose ``util.spawnv_passfds`` calls it by attribute
      and never builds a ``Popen``. Not theoretical: without this, spawned
      workers looked inherited and their statuses were taken from under
      ``Process.join()``.
    * ``os.posix_spawn`` / ``os.posix_spawnp`` -- any direct caller.
    * ``fork()`` itself, via the at-fork handlers -- ``os.fork``, ``os.forkpty``
      and so ``pty.fork`` and ``multiprocessing``'s fork context. CPython runs
      those handlers from ``fork_exec`` only when a ``preexec_fn`` is set and
      never passes them the new pid, which is why the children are diffed
      around the fork instead.

    Completeness here is the safety property, and it rests on other people's
    internals, so it is asserted rather than assumed: there is a test per seam
    in ``test_process_reaping.py``. A seam that moves fails those loudly instead
    of quietly costing somebody an exit status. What remains outside this is an
    extension module calling ``fork(2)`` in C; nothing in this codebase or its
    dependencies does.
    """
    global _spawns_tracked
    if _spawns_tracked:
        return
    original_init = subprocess.Popen.__init__

    @functools.wraps(original_init)
    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        with _reaper_lock:
            # Recorded only once the constructor has succeeded: a Popen that
            # raises has already reaped its own child, so there is nothing left
            # to protect and nothing that should hold a pid against reuse.
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
    """Claim every child that already exists as one this process forked.

    Sound because the flag has only just been armed: nothing can have been
    reparented here while this process was not a subreaper, so whatever is a
    child of ours right now was forked here and is being waited on here. Without
    this, anything started before the first agent session -- and in the test
    suite, anything started by an earlier test -- would look inherited.
    """
    with _reaper_lock:
        for pid in _child_pids():
            if pid not in _spawned_children:
                _remember_spawned(pid)


def _reap_inherited_orphans() -> tuple[int, ...]:
    """Collect the zombies this process inherited, and only those.

    A zombie child that is not in the spawn record cannot be one an asyncio
    transport, a ``Popen.wait()``, a ``Process.join()`` or the agent SDK is
    waiting for, because those are all recorded under this same lock before the
    call that created them returns. What is left is a descendant that reparented
    here when its shell exited: nothing else will ever collect it, and until
    something does it holds its process group open against anyone checking with
    ``killpg``.
    """
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
    """Wake the reaper thread and nothing else.

    Deliberately does no work here. This runs on the main thread between
    bytecodes, so it can interrupt a ``Popen`` that is mid-fork and holding the
    reaper lock; scanning from a thread instead is what keeps that from
    deadlocking.
    """
    try:
        os.write(_wake_write, b"\0")
    except OSError:
        # The pipe is full, which means a wake-up is already pending, or it is
        # closed, which means the process is going away. Neither is worth doing
        # anything about from a signal handler.
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
        # Main thread only, which is where both callers install from. The
        # fallback below covers a caller that is not.
        signal.signal(signal.SIGCHLD, _on_sigchld)
    except (ValueError, OSError):
        os.close(read_fd)
        os.close(write_fd)
        _wake_read = _wake_write = -1
        return False
    # Chained rather than replaced: whoever was handling child deaths before is
    # still entitled to hear about them.
    _previous_sigchld = previous
    return True


def _wait_for_a_collectable_child() -> None:
    """Block until some child of this process can be collected.

    The fallback for a caller that installed off the main thread, where no
    signal handler can be set. ``WNOWAIT`` makes this a notification rather than
    a collection -- the child stays collectable, so its own waiter still gets
    the status -- at the cost of naming the same child again on the next call
    until somebody takes it, which is what the caller's back-off is for.
    """
    try:
        os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOWAIT)
    except ChildProcessError:
        # No children at all, so nothing can be inherited until something here
        # forks -- which is precisely what the spawn event reports, so this
        # waits on that rather than on a timer.
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
        # The collectable child belongs to another waiter here, and waitid will
        # keep naming it until that waiter takes it. Wait for that to happen (or
        # for a new child) instead of spinning on it.
        _spawn_event.wait(backoff)
        _spawn_event.clear()
        backoff = min(backoff * 2, _REAP_BACKOFF_MAX_SEC)


def _start_orphan_reaper() -> None:
    """Start collecting what the subreaper flag will send this way.

    Runs on a thread of its own so it needs no event loop, which matters: both
    callers install from async code but nothing here may assume a running loop,
    and a campaign that is between sessions has none. Idempotent per process.
    """
    global _reaper_pid
    if _reaper_pid == os.getpid():
        return
    # Order matters. Claiming the process first is what opens the record for
    # writing; hooking next means a child forked from another thread while this
    # runs is recorded rather than missed; adopting last covers everything that
    # already existed. The thread starts only once it has something to read.
    _reaper_pid = os.getpid()
    _track_spawned_children()
    _adopt_existing_children()
    _arm_sigchld()
    threading.Thread(target=_reaper_loop, name="forge-orphan-reaper", daemon=True).start()
    # One pass up front: a subreaper installed by a second session inherits
    # whatever the first one left behind.
    _reap_inherited_orphans()


def install_child_subreaper() -> bool:
    """Become the parent this campaign's orphans fall back to, and collect them.

    Called before any agent process is started, because both halves of it have
    to be in place first: the flag decides where an orphan reparents when its
    shell exits, and the environment tag is only inherited by children that are
    exec'd after it is set.

    Where the kernel grants the flag, this also starts the thread that collects
    what it sends here -- the two are installed together because the second is
    the price of the first. An orphan reparented to a process that never waits
    on it is a zombie until the campaign ends, and a zombie holds its process
    group open while holding nothing else, so it is neither reaped nor reported.
    Nothing is started where the flag is refused: without it there is nothing to
    inherit, and the loop's own children stay its own business.

    Idempotent per process -- the flag does not survive ``fork()``, so it is
    re-armed rather than remembered once per interpreter. Answers whether the
    kernel accepted it; the tag is stamped either way, so on a kernel without
    the call ownership degrades to "carries our tag" rather than disappearing.
    """
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
    """This process's tag, empty until :func:`install_child_subreaper` ran.

    Empty matters: the tag is inherited, so a process that never installed
    would otherwise read its parent campaign's tag out of its own environment
    and claim that campaign's processes as its own.
    """
    return _owner_tag if _owner_pid == os.getpid() else ""


def _carries_owner_tag(pid: int, owner: str) -> bool:
    """Whether a process was exec'd carrying this campaign's tag.

    Reads what the process started with, which is the point: this still
    identifies a child that has since been orphaned, re-exec'd, or moved out of
    the descendant tree the campaign can see.
    """
    if not owner:
        return False
    try:
        with open(f"/proc/{pid}/environ", "rb") as handle:
            entries = handle.read().split(b"\0")
    except OSError:
        return False
    return f"{_OWNER_ENV}={owner}".encode() in entries


def _cwd_under(pid: int, resolved: str) -> bool:
    """Whether a process is working inside ``resolved``.

    A process whose cwd cannot be read -- gone, a zombie, or another user's --
    resolves to its own ``/proc`` entry and does not match.
    """
    try:
        cwd = os.path.realpath(f"/proc/{pid}/cwd")
    except OSError:
        return False
    return cwd == resolved or cwd.startswith(resolved + os.sep)


def _holds_device(pid: int) -> bool:
    """Whether a process has a device node open, by its fd table.

    Best effort in one direction only: a process that mapped the device and
    closed the fd still holds it and is not seen here. What is seen is enough
    to refuse a measurement, never enough to promise one is safe.
    """
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
    """Split the processes working under ``resolved`` by who started them.

    Ownership is the descendant tree plus the environment tag; the directory is
    only the scope. Ownership is then grown back out of that scope, because a
    process of ours that chdir'd elsewhere is still holding what it holds: the
    subtree below anything found here, and anything of ours sharing a process
    group with it -- which is how the detached shell above a benchmark is
    reached when only the benchmark itself is in the workspace.
    """
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
        # A zombie holds nothing and cannot be signalled -- and being a
        # subreaper produces them, for as long as it takes the thread installed
        # alongside the flag to collect them. This process's own group is the
        # campaign itself plus whatever it is running attached, which is never a
        # leftover and is not the reaper's business.
        return proc.pid != own_pid and proc.pgid != own_pgid and proc.state != "Z"

    seeds: dict[int, int] = {}
    foreign: list[int] = []
    for proc in table.values():
        if not signalable(proc) or not _cwd_under(proc.pid, resolved):
            continue
        # A process older than the campaign cannot have descended from it, so
        # no reading of the parent chain makes it ours.
        older = proc.starttime + 1 < own.starttime
        if not older and (proc.pid in ours or _carries_owner_tag(proc.pid, owner)):
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


def _signal(pid: int, starttime: int, sig: signal.Signals) -> None:
    """Signal one process, and only while it is still the one identified.

    Pids are recycled, and on a busy host the whole range wraps in well under
    an hour, so the start time read during the scan is rechecked against the
    live process here. What is left is a window no unprivileged process can
    close, and it is orders of magnitude narrower than signalling a whole
    process group on the strength of a scan.
    """
    proc = _read_proc(pid)
    if proc is None or proc.starttime != starttime:
        return
    try:
        os.kill(pid, sig)
    except OSError:
        log.debug("could not signal pid %s", pid, exc_info=True)


async def _escalate(resolved: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """SIGTERM then SIGKILL until nothing of ours is left under ``resolved``.

    Rescans every poll instead of working from the first list: a shell being
    torn down can start its last child after that list was taken, and a process
    that appears inside the grace window is asked politely before it is killed.
    Answers ``(reaped, unkillable)``.
    """
    loop = asyncio.get_running_loop()
    grace_end = loop.time() + _TERM_GRACE_SEC
    kill_end = grace_end + _KILL_CONFIRM_SEC
    signalled: dict[int, int] = {}
    while True:
        live = _survey(resolved).owned
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
    """Identify which of ``pids`` have a device node open, pid -> start time.

    Recorded as identities rather than bare pids because the answer is meant to
    be read back later -- by a following iteration, or by a following process --
    and a pid is only an identity for as long as its process lives. On a busy
    host the pid range wraps in well under an hour, so without the start time a
    re-check would name whatever landed on the number since.
    """
    holders: dict[int, int] = {}
    for pid in pids:
        proc = _read_proc(pid)
        if proc is not None and _holds_device(pid):
            holders[pid] = proc.starttime
    return holders


def still_holding_device(holders: Mapping[int, int]) -> tuple[int, ...]:
    """Which of the recorded holders still have the device.

    The narrow question a refused measurement waits on -- "is the device free
    now" -- rather than the reaper's, which is "is anything of ours still
    running here". What holds the device may be nothing of ours: a parallel
    campaign, a human's shell, a previous run's leftovers. Re-running the reaper
    would neither be entitled to touch those nor answer this.

    Reads the same fd table :func:`device_holders` did and is best effort in the
    same one direction: a process that mapped the device and closed its fd is
    not seen here. That is enough to keep refusing a measurement and never
    enough to promise one is safe, which is why a caller waiting on this needs
    an end of its own rather than waiting for it to say yes.
    """
    return tuple(
        sorted(
            pid
            for pid, starttime in holders.items()
            if (proc := _read_proc(pid)) is not None and proc.starttime == starttime and _holds_device(pid)
        )
    )


def processes_under(directory: str | os.PathLike[str]) -> set[int]:
    """Every process working inside a directory, this campaign's or not.

    Excludes this process and anything sharing its process group, neither of
    which is ever the reaper's business. Empty where ``/proc`` is not mounted.
    """
    survey = _survey(os.path.realpath(directory))
    return set(survey.owned) | set(survey.foreign)


def owned_processes_under(directory: str | os.PathLike[str]) -> set[int]:
    """The processes working inside a directory that this campaign started."""
    return set(_survey(os.path.realpath(directory)).owned)


async def reap_processes_under(directory: str | os.PathLike[str], *, description: str) -> ReapReport:
    """Terminate this campaign's processes working under ``directory``.

    SIGTERM first, because a driver holding a GPU has teardown of its own to
    do, then SIGKILL for whatever ignored it -- per process rather than per
    group, so a process group shared with anything else cannot widen the blast
    radius. Returns once the directory holds nothing of ours: the caller
    measures the device next, so a process that is merely signalled is still
    one that can corrupt the measurement.

    Whatever could not be cleared comes back in the report rather than only in
    the log, because the caller is the one that has to decide not to measure.
    See :class:`ReapReport`.

    ``description`` completes the log line "reaping N process(es) ...".
    """
    resolved = os.path.realpath(directory)
    survey = _survey(resolved)
    reaped: tuple[int, ...] = ()
    unkillable: tuple[int, ...] = ()
    if survey.owned:
        log.info("reaping %d process(es) %s", len(survey.owned), description)
        reaped, unkillable = await _escalate(resolved)
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


__all__ = [
    "ReapReport",
    "device_holders",
    "install_child_subreaper",
    "owned_processes_under",
    "processes_under",
    "reap_processes_under",
    "still_holding_device",
]
