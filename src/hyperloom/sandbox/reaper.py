# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End every process a sandbox started, including ones that left the tree."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from hyperloom_common import proctree
from hyperloom.sandbox.card_occupancy import SANDBOX_TAG_ENV

__all__ = ["Leftovers", "kill_launchers", "reap_sandbox_processes", "sweep_tagged", "tag_in_use", "wait_exited"]

_WINDOW_SEC = 5.0
_POLL_SEC = 0.05
#: How many processes a description of leftovers names.
_NAMED = 5


@dataclass(frozen=True)
class Leftovers:
    """What a reap could not confirm gone.

    Attributes:
        running: The processes still running when the reap gave up, each pinned to its incarnation. Each was sent
            SIGKILL before the reap's last look, so none of them can start another process: once all of them are gone,
            nothing the reap was after is left.
        unseen: The reap could not tell what is left: ``/proc`` could not be listed, or processes kept appearing that it
            had not yet signalled.
        launchers: The launchers whose wait timed out: this process's own children, which only their ``Popen`` waits
            for, so that the exit status stays where the ``Popen`` reads it.
    """

    running: tuple[proctree.ProcessId, ...] = ()
    unseen: bool = False
    launchers: tuple[subprocess.Popen[bytes], ...] = field(default=(), repr=False)

    @property
    def clean(self) -> bool:
        """bool: Nothing is left running, as far as the reap could look."""
        return not self.running and not self.unseen

    def __or__(self, other: Leftovers) -> Leftovers:
        """What either reap left, each process once."""
        return Leftovers(
            tuple(dict.fromkeys((*self.running, *other.running))),
            self.unseen or other.unseen,
            tuple(dict.fromkeys((*self.launchers, *other.launchers))),
        )

    def describe(self) -> str:
        """The processes left, by pid, command and state, for a log line."""
        parts = [_described(pid) for pid, _ in self.running[:_NAMED]]
        if len(self.running) > _NAMED:
            parts.append(f"{len(self.running) - _NAMED} more")
        if self.unseen:
            parts.append("what else is left is unknown")
        return ", ".join(parts) or "nothing"


def _described(pid: int) -> str:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"pid {pid} (gone)"
    command = stat_text[stat_text.find("(") + 1 : stat_text.rfind(")")]
    state = stat_text.rsplit(")", 1)[-1].split()[:1] or ["?"]
    return f"pid {pid} ({command}, state {state[0]})"


def _live_descendants(root_pids: Iterable[int]) -> list[proctree.ProcessId]:
    """Every running descendant of the launchers; zombies are already gone for our purposes."""
    found: list[proctree.ProcessId] = []
    for root in root_pids:
        found.extend(d for d in proctree.descendants(root) if proctree.running(d[0]))
    return found


def _signal_until_gone(find: Callable[[], list[proctree.ProcessId]], sig: int, window_sec: float) -> Leftovers:
    """Signal what ``find`` returns until it returns nothing, or the window ends.

    Each pass looks again, so a process forked while we signal is caught on the next pass. When the window ends, the
    passes go on until every process found has been signalled once: after SIGKILL none of them can fork, so what is
    returned is all that is left. If processes the passes have not signalled yet still turn up one more window later,
    what is left is ``unseen``.
    """
    deadline = time.monotonic() + window_sec
    signalled: set[proctree.ProcessId] = set()
    while True:
        alive = find()
        if not alive:
            return Leftovers()
        now = time.monotonic()
        if now >= deadline and signalled.issuperset(alive):
            return Leftovers(tuple(alive))
        if now >= deadline + window_sec:
            return Leftovers(tuple(alive), unseen=True)
        signalled.update(proctree.signal_processes(alive, sig))
        time.sleep(_POLL_SEC)


def _drain(root_pids: list[int], sig: int, window_sec: float) -> Leftovers:
    """Signal descendants of the launchers until none is left, or the window ends.

    The launchers stay alive throughout and are subreapers, so nothing can leave the tree by detaching.
    """
    return _signal_until_gone(lambda: _live_descendants(root_pids), sig, window_sec)


def kill_launchers(procs: list[subprocess.Popen[bytes]], *, grace_sec: float = 2.0) -> Leftovers:
    """Reap everything that runs under the given launchers, then the launchers.

    Order: freeze the launchers, SIGTERM the descendants and give them ``grace_sec``, SIGKILL what is
    left until a fresh scan finds nothing, and only then kill and wait for the launchers.

    A frozen launcher reaps nothing, so the descendants that exited stay zombies below it, and once it is gone they are
    re-parented to its nearest subreaper. When that is this process (a child subreaper) they are waited here, so a
    sandbox leaves no zombie behind.

    A launcher that was already waited is skipped: its pid went back to the system and may name another process.

    Args:
        procs: Launcher processes started by the sandbox.
        grace_sec: How long descendants get to exit on SIGTERM.

    Returns:
        Leftovers: What is still running; clean when no descendant and no launcher is. A launcher whose wait timed out
        is named with its ``Popen``, which alone waits for it later (:func:`wait_exited`).
    """
    procs = [p for p in procs if p.returncode is None]
    roots = [p.pid for p in procs]
    for pid in roots:
        # Freeze each launcher first so it cannot fork a command we have not seen yet.
        # A stopped subreaper still receives orphans; it only stops reaping zombies.
        try:
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            continue
    tree: list[proctree.ProcessId] = []
    try:
        # SIGTERM once, then wait the grace; the SIGKILL passes re-scan until empty.
        proctree.signal_processes(_live_descendants(roots), signal.SIGTERM)
        _drain(roots, 0, grace_sec)
        left = _drain(roots, signal.SIGKILL, _WINDOW_SEC)
        # The trees as the launchers die, the exited included: the ones to wait for once they are this process's.
        tree = [process for root in roots for process in proctree.descendants(root)]
    except OSError:
        left = Leftovers(unseen=True)  # /proc could not be listed; nothing can be confirmed
    for proc in procs:
        launcher = proctree.proc_identity(proc.pid)
        proctree.signal_group(proc.pid, signal.SIGKILL, what="sandbox launcher")
        try:
            proc.kill()
        except ProcessLookupError:
            pass  # the launcher has already ended; the wait below reaps it
        try:
            proc.wait(timeout=_WINDOW_SEC)
        except subprocess.TimeoutExpired:
            stuck = () if launcher is None else ((proc.pid, launcher[1]),)
            left = left | Leftovers(stuck, unseen=launcher is None, launchers=(proc,))
    proctree.reap_exited_children(tree)
    return left


def _tagged(sandbox_id: str) -> list[proctree.ProcessId]:
    """The running processes, other than this one, started with this sandbox's id in their environment."""
    found: list[proctree.ProcessId] = []
    for pid in proctree.marked_pids(SANDBOX_TAG_ENV, sandbox_id):
        identity = proctree.proc_identity(pid)
        if pid != os.getpid() and identity is not None and proctree.running(pid):
            found.append((pid, identity[1]))
    return found


def sweep_tagged(sandbox_id: str) -> Leftovers:
    """SIGKILL every process started with this sandbox's id in its environment.

    A fallback for the case the launchers themselves were lost; it cannot see
    a process that cleared its environment.

    Returns:
        Leftovers: The tagged processes still running when the window ended; ``unseen`` also when ``/proc`` cannot be
        listed, which ``proctree.marked_pids`` answers with no process.
    """
    if not _proc_listed():
        return Leftovers(unseen=True)
    return _signal_until_gone(lambda: _tagged(sandbox_id), signal.SIGKILL, _WINDOW_SEC)


def _proc_listed() -> bool:
    """Whether ``/proc`` can be listed: where it cannot, a look for tagged processes finds none whatever runs."""
    try:
        next(Path("/proc").iterdir(), None)
    except OSError:
        return False
    return True


def reap_sandbox_processes(procs: list[subprocess.Popen[bytes]], sandbox_id: str) -> Leftovers:
    """End all processes of a sandbox and confirm nothing is left.

    What the launchers' reap and the tag sweep named is looked at once more when both are done: a process the launchers'
    reap named may have died during the sweep, and one the sweep named as its window ended may have died since. Only one
    that still runs as the incarnation named is left.

    Args:
        procs: The sandbox's launcher processes.
        sandbox_id: The id the sandbox put in its processes' environment.

    Returns:
        Leftovers: What of the sandbox is still running; clean when nothing is.
    """
    return wait_exited(kill_launchers(procs) | sweep_tagged(sandbox_id))


def wait_exited(left: Leftovers) -> Leftovers:
    """Wait for those of ``left`` that have exited and are now this process's children; return the ones still running.

    Each process in ``left`` was sent SIGKILL before it was named and cannot have started another, so the ones still
    running are all that is left of them. A launcher among them is collected by its ``Popen`` alone; only the others,
    the processes this one adopted, go to ``proctree.reap_exited_children``.
    """
    for launcher in left.launchers:
        launcher.poll()
    own = {launcher.pid for launcher in left.launchers}
    proctree.reap_exited_children(process for process in left.running if process[0] not in own)
    return Leftovers(
        tuple(process for process in left.running if _still_running(process)),
        left.unseen,
        tuple(launcher for launcher in left.launchers if launcher.returncode is None),
    )


def _still_running(process: proctree.ProcessId) -> bool:
    """Whether ``process`` runs: its pid runs and names it, or names a process whose start cannot be read."""
    pid, start_time = process
    if not proctree.running(pid):
        return False
    identity = proctree.proc_identity(pid)
    return identity is None or identity[1] == start_time


def tag_in_use(sandbox_id: str) -> bool:
    """Whether a running process still carries this sandbox's id in its environment; True also when ``/proc`` cannot be
    listed to tell."""
    return not _proc_listed() or bool(_tagged(sandbox_id))
