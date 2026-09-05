# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enumerate a process tree from procfs, signal it, and confirm it is gone.

A tree is collected before anything is signalled: a child that re-parents to
init is a descendant in a pass taken beforehand and an orphan in one taken
afterwards.

Members are remembered as ``(pid, start_time)`` -- field 22 of
``/proc/<pid>/stat`` -- so :func:`signal_processes` can re-read the identity and
skip a pid the kernel has since recycled onto an unrelated process.

:func:`kill_tree` is the one escalation every caller reaps through, so a tree
left by a benchmark, by a bring-up and by the kernel agent all get the same
signals in the same order after the same grace.

Standard library only, and procfs-only.
"""

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

__all__ = [
    "TERM_GRACE_SEC",
    "ProcessId",
    "Tree",
    "collect_tree",
    "descendants",
    "group_alive",
    "group_members",
    "kill_tree",
    "proc_identity",
    "running",
    "signal_group",
    "signal_processes",
]

#: A process, pinned to the incarnation that was observed: ``(pid, start_time)``.
ProcessId = tuple[int, int]


def proc_identity(pid: int) -> tuple[int, int] | None:
    """Read ``(ppid, start_time)`` for one pid.

    Args:
        pid: The process to inspect.

    Returns:
        tuple[int, int] | None: The parent pid and the process start time, or
            ``None`` when the process is gone or its stat entry was truncated.
    """
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # The comm field is parenthesised and may itself contain spaces and
    # parentheses, so the split has to start after its LAST ')'.
    closing_paren = stat_text.rfind(")")
    fields_after_name = stat_text[closing_paren + 2 :].split()
    try:
        return int(fields_after_name[1]), int(fields_after_name[19])
    except (ValueError, IndexError):
        return None


def descendants(root_pid: int) -> list[ProcessId]:
    """Return every descendant of ``root_pid``, deepest first.

    Args:
        root_pid: The root of the tree; not itself included.

    Returns:
        list[ProcessId]: The descendants as ``(pid, start_time)`` pairs.

    Raises:
        OSError: When ``/proc`` cannot be listed.
    """
    children: dict[int, list[ProcessId]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        identity = proc_identity(pid)
        if identity is None:
            continue
        parent_pid, start_time = identity
        children.setdefault(parent_pid, []).append((pid, start_time))

    collected: list[ProcessId] = []

    def _walk(parent_pid: int) -> None:
        for child_pid, start_time in children.get(parent_pid, []):
            _walk(child_pid)
            collected.append((child_pid, start_time))

    _walk(root_pid)
    return collected


def signal_processes(processes: list[ProcessId], sig: int) -> None:
    """Signal each enumerated process, skipping any whose identity changed.

    Args:
        processes: ``(pid, start_time)`` pairs, as :func:`descendants` returns.
        sig: The signal number to send.
    """
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


def running(pid: int) -> bool:
    """Whether ``pid`` names a process that is not a zombie.

    A zombie nobody has waited on still answers signal 0, so the reading is the
    process state rather than the signal.

    Args:
        pid: The process to inspect.

    Returns:
        bool: True when the process answers and has not been torn down.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process: existence is the answer being asked for.
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
            fields = handle.read().rsplit(")", 1)[-1].split()
    except OSError:
        # No procfs reading to refine the signal with; the signal stands.
        return True
    return bool(fields) and fields[0] != "Z"


def group_alive(pgid: int) -> bool:
    """Whether a process group still has at least one member.

    Args:
        pgid: The process-group id to probe.

    Returns:
        bool: True while the group has a member or its liveness cannot be read
        (a sandbox may refuse the probe), False once it is empty or on
        non-POSIX.
    """
    if os.name != "posix":
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # An indeterminate reading is not an empty group.
        return True
    return True


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
        entries = list(Path("/proc").iterdir())
    except OSError:
        return members
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            fields = stat_text[stat_text.rfind(")") + 2 :].split()
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
