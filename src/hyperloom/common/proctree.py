# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enumerate a process tree from procfs, signal it, and confirm it is gone."""

from __future__ import annotations

import logging
import os
import signal
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Grace between the polite signal and the fatal one, for every tree kill.
#: Long enough for a serving engine to unmap its device memory, short enough
#: that every window derived from it stays inside a round's stop budget.
TERM_GRACE_SEC: float = 5.0

#: How often a wait looks at what it is waiting for.
_POLL_SEC: float = 0.05

_PROC = Path("/proc")

#: Set in a run's env, its value marks the run: whatever inherited it is reaped on every exit path, however it detached.
REAP_MARK_ENV: str = "HYPERLOOM_REAP_MARK"

__all__ = [
    "REAP_MARK_ENV",
    "TERM_GRACE_SEC",
    "ProcessId",
    "boot_unix",
    "Tree",
    "collect_tree",
    "descendants",
    "group_members",
    "kill_tree",
    "leader_exited",
    "marked_pids",
    "proc_identity",
    "process_uids",
    "reap_exited_children",
    "running",
    "signal_group",
    "signal_processes",
]

#: A process, pinned to the incarnation that was observed: ``(pid, start_time)``.
ProcessId = tuple[int, int]

#: The ``/proc/<pid>/stat`` states of a process that has exited: a zombie, and the dead one being released (``x`` on
#: kernels before 3.14).
_EXITED_STATES = frozenset({"Z", "X", "x"})


def _stat_fields(pid: int) -> list[str]:
    """The ``/proc/<pid>/stat`` fields after the comm, from ``state`` on; raises ``OSError`` when unreadable."""
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    # The comm may itself contain spaces and parentheses, so split after its LAST ')'.
    return stat_text.rsplit(")", 1)[-1].split()


def proc_identity(pid: int) -> tuple[int, int] | None:
    """Read ``(ppid, start_time)`` for one pid.

    Args:
        pid: The process to inspect.

    Returns:
        tuple[int, int] | None: The parent pid and the process start time, or
            ``None`` when the process is gone or its stat entry was truncated.
    """
    try:
        fields = _stat_fields(pid)
        return int(fields[1]), int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def boot_unix() -> float:
    """Wall-clock time the host booted: the origin the start times in ``/proc/<pid>/stat`` count clock ticks from.

    Raises:
        OSError: When ``/proc/uptime`` cannot be read.
    """
    return time.time() - float((_PROC / "uptime").read_text(encoding="ascii").split()[0])


def _status_text(pid: int) -> str:
    """``/proc/<pid>/status``; raises ``OSError`` when unreadable."""
    return (_PROC / str(pid) / "status").read_text(encoding="utf-8", errors="replace")


def process_uids(pid: int) -> tuple[int, int] | None:
    """Read ``(real uid, effective uid)`` for one pid from ``/proc/<pid>/status``; ``None`` when it is gone."""
    try:
        status = _status_text(pid)
        uids = next(line.split()[1:3] for line in status.splitlines() if line.startswith("Uid:"))
        return int(uids[0]), int(uids[1])
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def descendants(root_pid: int) -> list[ProcessId]:
    """Return every descendant of ``root_pid``, deepest first.

    A process comes after all of its own descendants, and the children of one parent are visited in pid order, so
    the result is the same however ``/proc`` happens to list its entries.

    Args:
        root_pid: The root of the tree; not itself included.

    Returns:
        list[ProcessId]: The descendants as ``(pid, start_time)`` pairs.

    Raises:
        OSError: When ``/proc`` cannot be listed.
    """
    children: dict[int, list[ProcessId]] = {}
    for entry in _PROC.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        identity = proc_identity(pid)
        if identity is None:
            continue
        parent_pid, start_time = identity
        children.setdefault(parent_pid, []).append((pid, start_time))
    # The listing order is the filesystem's to choose; ordering siblings by pid makes the walk deterministic.
    for siblings in children.values():
        siblings.sort(key=lambda child: child[0])

    # A sandboxed PID view can list a process as its own ancestor, so a pid is visited once; the walk keeps
    # its own stack, so a deep tree cannot exhaust Python's.
    collected: list[ProcessId] = []
    seen = {root_pid}
    stack = [(child, False) for child in reversed(children.get(root_pid, []))]
    while stack:
        (pid, start_time), expanded = stack.pop()
        if expanded:
            collected.append((pid, start_time))
        elif pid not in seen:
            seen.add(pid)
            stack.append(((pid, start_time), True))
            stack.extend((child, False) for child in reversed(children.get(pid, [])))
    return collected


def signal_processes(processes: list[ProcessId], sig: int) -> list[ProcessId]:
    """Signal each enumerated process, skipping any whose identity changed.

    Args:
        processes: ``(pid, start_time)`` pairs, as :func:`descendants` returns.
        sig: The signal number to send.

    Returns:
        list[ProcessId]: The processes the signal was delivered to.
    """
    delivered: list[ProcessId] = []
    for pid, expected_start_time in processes:
        identity = proc_identity(pid)
        if identity is None or identity[1] != expected_start_time:
            continue
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            # Exited between the identity read and the signal, or is not ours to
            # signal; the caller's confirm step observes the survivor.
            continue
        delivered.append((pid, expected_start_time))
    return delivered


def running(pid: int) -> bool:
    """Whether ``pid`` names a process that has not exited: one whose every thread has exited is not running.

    ``/proc/<pid>/stat`` shows the thread-group leader alone, and a leader that has exited stays a zombie while another
    thread of its process still runs, so a process whose leader has exited is looked at again by its threads.

    Args:
        pid: The process to inspect.

    Returns:
        bool: True when the process exists and has not been torn down.
    """
    if not _exists(pid):
        return False
    try:
        fields = _stat_fields(pid)
    except OSError:
        # Reaped since the signal, or hidden from this account (``hidepid``): only the signal can tell the two apart.
        return _exists(pid)
    if not fields:
        return False
    return fields[0] not in _EXITED_STATES or _threads_left(pid)


def leader_exited(pid: int) -> bool:
    """Whether ``/proc/<pid>/stat`` shows ``pid``'s thread-group leader exited, whatever its other threads do.

    :func:`running` still counts a process whose leader has exited while another thread of it runs, but that leader's
    ``/proc/<pid>/cmdline`` reads empty, so a caller that tells a process by its command line asks this as well.

    Args:
        pid: The process to inspect.

    Returns:
        bool: True when the leader's state is exited; False when it is not, and when the pid is gone or its stat cannot
        be read, which :func:`running` answers.
    """
    try:
        fields = _stat_fields(pid)
    except OSError:
        return False
    return bool(fields) and fields[0] in _EXITED_STATES


def _threads_left(pid: int) -> bool:
    """Whether a process whose thread-group leader has exited still has a thread that has not.

    ``Threads:`` in ``/proc/<pid>/status`` counts every thread not yet released, an exited leader included, and ``State:``
    beside it is read in the same go, so a pid taken by another process since the stat read is answered for that one.
    """
    try:
        status = _status_text(pid)
    except OSError:
        # Reaped since the stat read, or hidden from this account: only the signal can tell the two apart.
        return _exists(pid)
    values = {key: value.split() for key, _, value in (line.partition(":") for line in status.splitlines())}
    try:
        state, threads = values["State"][0], int(values["Threads"][0])
    except (KeyError, IndexError, ValueError):
        # No thread count to go by: as for an unreadable stat, the signal stands.
        return _exists(pid)
    return state not in _EXITED_STATES or threads > 1


def _exists(pid: int) -> bool:
    """Whether a signal can find ``pid``; someone else's process exists too, whether or not it may be signalled."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reap_exited_children(processes: Iterable[ProcessId]) -> int:
    """Wait for each of ``processes`` that has exited and is now a child of this process.

    A process orphaned below a child subreaper becomes the subreaper's child, and once it has exited it stays a zombie
    until the subreaper waits for it. Pass only processes this process did not start itself, such as the descendants of
    a child it has killed: one is waited only while its ``/proc`` entry names this process as the parent, the start time
    it was observed with, and an exited state. Nothing else waits for such a zombie, and its pid cannot be reused until it
    is waited, so the wait takes no exit status a ``Popen`` of this process still needs.

    Args:
        processes: ``(pid, start_time)`` pairs, as :func:`descendants` returns.

    Returns:
        int: How many were waited.
    """
    me = os.getpid()
    waited = 0
    for pid, start_time in processes:
        try:
            fields = _stat_fields(pid)
            ours = fields[0] in _EXITED_STATES and int(fields[1]) == me and int(fields[19]) == start_time
        except (OSError, ValueError, IndexError):
            continue
        if not ours:
            continue
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                waited += 1
        except ChildProcessError:
            continue
    return waited


def signal_group(pgid: int, sig: int, *, what: str = "process group") -> bool | None:
    """Signal every member of a process group.

    Args:
        pgid: The process-group id to signal.
        sig: The signal number to send.
        what: Names the group in the warning logged when the signal fails.

    Returns:
        bool | None: True when the signal was delivered, ``None`` when the group
        was already empty, False when the host refused the signal.
    """
    if os.name != "posix":
        return None
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return None
    except OSError as exc:
        log.warning("%s: killpg(%d, %d) failed: %s", what, pgid, sig, exc)
        return False
    return True


def marked_pids(env_key: str, value: str) -> list[int]:
    """Return every process whose environment carries ``env_key=value``.

    A mark set in a child's environment is inherited by everything it starts, however that detaches, so it finds
    what neither the process tree nor a process group still reaches.

    Args:
        env_key: The environment variable that carries the mark.
        value: The mark to match exactly.

    Returns:
        list[int]: The marked pids, empty when ``/proc`` cannot be listed.
    """
    entry_bytes = f"{env_key}={value}".encode()
    pids: list[int] = []
    try:
        entries = list(_PROC.iterdir())
    except OSError:
        return pids
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if entry_bytes in environ.split(b"\0"):
            pids.append(int(entry.name))
    return pids


def group_members(pgid: int) -> list[tuple[int, int, str]]:
    """Return the live members of one process group.

    Args:
        pgid: The process-group id to enumerate.

    Returns:
        list[tuple[int, int, str]]: ``(pid, start_time, state)`` per member,
        empty when ``/proc`` cannot be listed.
    """
    members: list[tuple[int, int, str]] = []
    try:
        entries = list(_PROC.iterdir())
    except OSError:
        return members
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fields = _stat_fields(int(entry.name))
            if int(fields[2]) != pgid:
                continue
            members.append((int(entry.name), int(fields[19]), fields[0]))
        except (OSError, ValueError, IndexError):
            continue
    return members


@dataclass(frozen=True)
class Tree:
    """What one kill will reach, as enumerated before the first signal.

    Attributes:
        members: The roots and their descendants, each pinned to the
            incarnation that was observed.
        groups: The process groups to signal; never the caller's own.
        roots: The pids the tree was collected from.
    """

    members: tuple[ProcessId, ...]
    groups: tuple[int, ...]
    roots: frozenset[int]

    @property
    def watched(self) -> set[int]:
        """set[int]: The pids whose disappearance confirms the kill."""
        return {pid for pid, _ in self.members} | set(self.roots)


def collect_tree(pids: Iterable[int]) -> Tree:
    """Enumerate the processes, descendants and groups a kill will reach.

    Args:
        pids: The roots to collect from. The calling process is never a root,
            and neither is a non-positive pid.

    Returns:
        Tree: The enumeration, taken while a child that is about to re-parent
        is still reachable as a descendant.

    Raises:
        OSError: When ``/proc`` cannot be listed.
    """
    roots = {int(pid) for pid in pids if int(pid) > 0}
    roots.discard(os.getpid())
    posix = os.name == "posix"
    own_group = os.getpgid(0) if posix else -1
    members: list[ProcessId] = []
    groups: set[int] = set()
    for pid in sorted(roots):
        identity = proc_identity(pid)
        if identity is not None:
            members.append((pid, identity[1]))
        members.extend(descendants(pid))
        if not posix:
            continue
        try:
            pgid = os.getpgid(pid)
        except OSError:
            continue
        # Signalling the caller's own group is suicide, and a launch site that
        # forgot start_new_session shares it.
        if pgid != own_group:
            groups.add(pgid)
    return Tree(tuple(members), tuple(sorted(groups)), frozenset(roots))


def signal_tree(tree: Tree, sig: int) -> None:
    """Send one signal to every enumerated member and every group.

    Args:
        tree: The enumeration to signal.
        sig: The signal number to send.
    """
    signal_processes(list(tree.members), sig)
    for pgid in tree.groups:
        signal_group(pgid, sig, what="reap")


def tree_alive(tree: Tree) -> bool:
    """Whether anything the tree enumerated is still running.

    Args:
        tree: The enumeration to look at.

    Returns:
        bool: True while at least one enumerated process has not been torn down.
    """
    return any(running(pid) for pid in tree.watched)


def kill_tree(
    tree: Tree,
    *,
    grace_sec: float = TERM_GRACE_SEC,
    confirm_sec: float = TERM_GRACE_SEC,
) -> bool:
    """SIGTERM a tree, then SIGKILL whatever outlives the grace.

    Args:
        tree: The enumeration to end.
        grace_sec: How long the tree is given to exit on SIGTERM.
        confirm_sec: How long the kill is given to take effect afterwards.

    Returns:
        bool: True when nothing enumerated is left, False when something
        outlived both signals.
    """
    for sig, window in ((signal.SIGTERM, grace_sec), (signal.SIGKILL, confirm_sec)):
        signal_tree(tree, sig)
        if _wait_gone(tree, window):
            return True
    return False


def _wait_gone(tree: Tree, window_sec: float) -> bool:
    """Poll until nothing enumerated is running, or ``window_sec`` runs out."""
    deadline = time.monotonic() + max(0.0, window_sec)
    while True:
        if not tree_alive(tree):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SEC)
