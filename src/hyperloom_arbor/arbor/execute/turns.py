"""What an agent turn starts, the turn ends.

Each agent turn's CLI runs with ``ARBOR_TURN=<tag>`` in its environment, and every process it starts inherits it: a
``torchrun`` eval sent to the background, a profiled server, a bench client. When the turn ends, Arbor stops every
process of its own user that still carries the tag. The server turn is the one exception: the server it starts is
meant to outlive it, and is stopped with the server (``end_kept``, from ``claude_loop._stop_server``).

Never SIGKILL: SGLang holds KFD memory until its Python process exits cleanly, and a killed server leaves the card
full (``preflight.stop_server``). SIGTERM goes first to the tree's roots, whose servers take their workers down, then
to whatever is left; what survives both is reported, and the free-memory check catches it.

Why: under a host, full-model evals and profiled runs outlived their turns and held most of every leased card's
memory. Every later server start failed for lack of memory, for hours.

A process that clears its environment, or another user's process, is not found here: the free-memory check before
each server start (``claude_loop._lease_held``) is the second line. Linux only; elsewhere nothing is found.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

TURN_ENV = "ARBOR_TURN"
#: The turns whose processes outlive them on purpose: the server turn's server.
KEPT = ("server",)
#: The container a standalone campaign's turns reach through ``docker exec`` (set by the loop), or None. The tag does
#: not follow a process into it: ``docker exec`` starts it from the daemon, with the container's environment. So the
#: container's processes are listed when such a turn starts, and what is new when it ends is stopped there.
CONTAINER: str | None = None


def new_tag(label: str) -> str:
    """A tag no other turn shares: this process, the turn's label, the time."""
    return f"{os.getpid()}-{label}-{time.time_ns()}"


def leftovers(tag: str, proc: Path = Path("/proc"), *, prefix: bool = False) -> list[int]:
    """The processes of this user, other than this one, whose environment carries ``ARBOR_TURN=<tag>``; with
    ``prefix``, any tag that starts with ``tag``."""
    want = f"{TURN_ENV}={tag}".encode()
    uid, me, found = os.getuid(), os.getpid(), []
    try:
        entries = list(proc.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            env = (entry / "environ").read_bytes().split(b"\0")
            if any(e.startswith(want) for e in env) if prefix else want in env:
                found.append(int(entry.name))
        except OSError:
            continue
    return sorted(found)


def stop(pids: list[int], grace: float = 30.0) -> tuple[list[int], list[int]]:
    """SIGTERM ``pids``: the roots (whose parent is not among them) first, the rest after up to ``grace`` seconds.
    Never SIGKILL (the module's docstring). The pids that were alive to stop, and those still running after a second
    ``grace``."""
    group = set(pids)
    roots = [pid for pid in pids if _parent(pid) not in group]
    signalled = _term(roots)
    _wait(signalled, grace)
    rest = [pid for pid in pids if pid not in roots and _running(pid)]
    signalled += _term(rest)
    _wait(signalled, grace)
    return sorted(signalled), [pid for pid in sorted(signalled) if _running(pid)]


def _term(pids: list[int]) -> list[int]:
    sent = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            sent.append(pid)
        except (ProcessLookupError, PermissionError):
            continue
    return sent


def _wait(pids: list[int], grace: float) -> None:
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(_running(pid) for pid in pids):
        time.sleep(0.2)


def _parent(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def end_turn(tag: str, label: str, grace: float = 30.0) -> tuple[list[int], list[int]]:
    """Stop what the turn ``tag`` left running, unless it is a kept turn: the pids stopped, and those still running."""
    if label in KEPT:
        return [], []
    pids = leftovers(tag)
    return stop(pids, grace) if pids else ([], [])


def end_kept(label: str = "server", grace: float = 30.0) -> tuple[list[int], list[int]]:
    """Stop what this Arbor's ``label`` turns left running on purpose (the server turn's servers, a server it started
    by hand): with the server, so that a card still held after a stop is another process's. The pids stopped, and
    those still running."""
    pids = leftovers(f"{os.getpid()}-{label}-", prefix=True)
    return stop(pids, grace) if pids else ([], [])


# ── Inside the container (standalone): what a turn's `docker exec` started ──────────────────────────────────────

# pid, parent, state, start time (five fields), command: the start time tells a new process from a reused pid, and a
# zombie (state Z) is already gone.
_PS = ["ps", "-eo", "pid=,ppid=,stat=,lstart=,args="]


def _docker(container: str, argv: list[str], timeout: int = 60):
    import subprocess
    try:
        return subprocess.run(["docker", "exec", container, *argv], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def container_procs(container: str) -> dict[tuple[int, str], tuple[int, str]] | None:
    """``{(pid, start): (parent, command)}`` of every process in ``container``, or None when it cannot be read."""
    r = _docker(container, _PS)
    if r is None or r.returncode != 0:
        return None
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split(None, 8)
        if len(parts) < 8 or not parts[0].isdigit() or not parts[1].isdigit() or parts[2].startswith("Z"):
            continue
        out[(int(parts[0]), " ".join(parts[3:8]))] = (int(parts[1]), parts[8] if len(parts) > 8 else "")
    return out


def _new(before: dict, now: dict) -> dict[int, tuple[int, str]]:
    """The processes in ``now`` that were not in ``before``, but for the listing itself: ``{pid: (parent, command)}``."""
    return {pid: (ppid, cmd) for (pid, start), (ppid, cmd) in now.items()
            if (pid, start) not in before and not cmd.startswith(" ".join(_PS))}


def end_container_turn(before: dict | None, container: str | None, label: str,
                       grace: float = 30.0) -> tuple[list[int], list[int]]:
    """Stop, inside ``container``, what started there during the turn whose start listing is ``before``: SIGTERM to the
    roots (whose parent is not new too) first, then to the rest after up to ``grace`` seconds; never SIGKILL. The
    container's pids stopped, and those still running. Nothing for a kept turn, or one that could not be listed."""
    if before is None or not container or label in KEPT:
        return [], []
    now = container_procs(container)
    if now is None:
        return [], []
    new = _new(before, now)
    if not new:
        return [], []
    roots = sorted(p for p, (ppid, _) in new.items() if ppid not in new)
    rest = sorted(p for p in new if p not in roots)
    _docker(container, ["kill", "-TERM", *map(str, roots)])
    deadline = time.time() + grace
    while time.time() < deadline:
        left = container_procs(container)
        if left is None or not _new(before, left):
            break
        time.sleep(1)
    left = container_procs(container) or {}
    alive = sorted(_new(before, left))
    if alive:
        _docker(container, ["kill", "-TERM", *map(str, alive)])
        time.sleep(min(grace, 5))
        alive = sorted(_new(before, container_procs(container) or {}))
    return roots + rest, alive
