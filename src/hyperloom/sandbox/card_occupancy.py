# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which processes hold a GPU card, read from the kernel's own list of the processes that use the GPUs.

A measurement is only worth anything when nothing else runs on its cards. The kernel lists every process that has a
compute queue or resident memory on a GPU (the rule pack's ``kfd_process_listing``), and that list is what the ROCm tools
print as well, so it is read here directly: no tool whose output changes between releases, and no root.

A card is *held* by a process that has a queue on it or memory resident on it. A process that only opened the interface
(``rocminfo``, a monitor) holds nothing. Cards are numbered the way ``ROCR_VISIBLE_DEVICES`` numbers them, which is the
order of the GPU nodes of the kernel's topology; the card pool hands out the same numbers.

The probe says what it saw or why it cannot say: an answer of "nobody holds it" is only ever given for a list that was
read. It cannot say in a container with a pid namespace of its own, because the kernel names processes by their id in
the host's namespace and those cannot be matched to what the container sees, nor where ``/proc`` hides other accounts'
processes, nor where the kernel's list is not there.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from hyperloom.common.env_safety import redact_secret_values
from hyperloom.common.outside_facts import (
    INITIAL_PID_NAMESPACE,
    KFD_NODE_GPU_ID,
    KFD_NODES,
    KFD_PROCESSES,
    KFD_QUEUE_GPU_ID,
    KFD_QUEUES,
    KFD_ROOT,
    KFD_VRAM_PREFIX,
    PID_NAMESPACE_LINK,
)
from hyperloom.sandbox.launcher import LAUNCHER_SOURCE

__all__ = [
    "CARD_GUARD_ENV",
    "COMMAND_CHARS",
    "KFD_PATH",
    "PROC_PATH",
    "SANDBOX_TAG_ENV",
    "CardGuard",
    "CardReading",
    "Holder",
    "OwnProcesses",
    "card_guard",
    "own_sandbox_processes",
    "read_holders",
]

#: The directory the kernel's list is under, and ``/proc``; tests point both at a tree of their own.
KFD_PATH = Path(KFD_ROOT)
PROC_PATH = Path("/proc")
#: The environment variable every process of a sandbox carries, whatever it did to its parentage since.
SANDBOX_TAG_ENV = "E2B_SANDBOX_ID"
#: Names what the guard does with what the probe finds: the toolbox sets it for itself and for the runners it starts.
CARD_GUARD_ENV = "HYPERLOOM_CARD_GUARD"
#: A command line is kept to this many characters: it names the process, it is not a record of its arguments.
COMMAND_CHARS = 160
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")
#: Text read from another process's files: what is not valid text is replaced, never raised on.
_LOSSY_TEXT = {"encoding": "utf-8", "errors": "replace"}
#: What a process that is gone while the list is read raises: the entry went with it, so it holds nothing.
_GONE = (FileNotFoundError, ProcessLookupError, NotADirectoryError)
_ZOMBIE = "Z"
_ROOT_ACCOUNT = 0
_HIDDEN_OPTION = "hidepid="
_NOT_HIDING = ("0", "off")


class CardGuard(str, Enum):
    """What is done about a process that holds a card.

    ``OBSERVE`` looks and records, and refuses nothing: a result is kept and says what was seen, a card is handed out and
    the log says who holds it. ``ENFORCE`` refuses a result that was measured beside such a process and does not hand
    out a card some process holds.
    """

    OBSERVE = "observe"
    ENFORCE = "enforce"


def card_guard(environ: Mapping[str, str]) -> CardGuard:
    """The guard ``environ`` names (:data:`CARD_GUARD_ENV`); ``OBSERVE`` when it names none.

    Raises:
        ValueError: When it names something that is not a guard: a misspelling must not read as either one.
    """
    named = environ.get(CARD_GUARD_ENV, "").strip()
    if not named:
        return CardGuard.OBSERVE
    try:
        return CardGuard(named)
    except ValueError:
        raise ValueError(f"{CARD_GUARD_ENV} is {named!r}, not one of {[guard.value for guard in CardGuard]}") from None


class _Unreadable(Exception):
    """The kernel's list cannot be read as the probe needs it; the message says why."""


@dataclass(frozen=True)
class Holder:
    """One process holding one card. ``command`` is the process's command line, shortened, with credentials masked and
    no environment; empty when it could not be read."""

    card: int
    pid: int
    command: str


@dataclass(frozen=True)
class CardReading:
    """What the probe found on some cards: the processes that hold them, or why it cannot say (then there are none
    listed, and none were ruled out)."""

    holders: tuple[Holder, ...] = ()
    unavailable: str | None = None

    def held_cards(self) -> frozenset[int]:
        return frozenset(holder.card for holder in self.holders)

    def foreign(self, own: OwnProcesses) -> tuple[Holder, ...]:
        """The holders that are not ``own``'s."""
        return tuple(holder for holder in self.holders if not own.owns(holder.pid))


@dataclass(frozen=True)
class OwnProcesses:
    """The processes that belong to a test: those under ``roots`` however deep, and any process whose environment holds
    ``tag`` (``NAME=value``), which a process that detached from its parent still carries."""

    roots: frozenset[int]
    tag: str = ""
    procfs: Path = PROC_PATH

    def owns(self, pid: int) -> bool:
        return self._descends(pid) or self._tagged(pid)

    def _descends(self, pid: int) -> bool:
        seen: set[int] = set()
        while pid > 1 and pid not in seen:
            if pid in self.roots:
                return True
            seen.add(pid)
            stat = _stat(self.procfs, pid)
            if stat is None:
                return False
            pid = stat[1]
        return False

    def _tagged(self, pid: int) -> bool:
        if not self.tag:
            return False
        try:
            environment = (self.procfs / str(pid) / "environ").read_bytes()
        except OSError:
            return False
        return self.tag.encode() in environment.split(b"\0")


def own_sandbox_processes(
    environ: Mapping[str, str], *, pid: int | None = None, procfs: Path = PROC_PATH
) -> OwnProcesses:
    """The processes of the sandbox that process ``pid`` (this one) runs in: what runs under it, what runs under the
    launcher that started it (a child subreaper, so a process that detached is still under it) and what carries the
    sandbox's tag."""
    me = os.getpid() if pid is None else pid
    roots = {me}
    seen = {me}
    stat = _stat(procfs, me)
    while stat is not None and stat[1] > 1 and stat[1] not in seen:
        parent = stat[1]
        seen.add(parent)
        if _is_launcher(procfs, parent):
            roots.add(parent)
        stat = _stat(procfs, parent)
    sandbox = environ.get(SANDBOX_TAG_ENV, "")
    return OwnProcesses(frozenset(roots), f"{SANDBOX_TAG_ENV}={sandbox}" if sandbox else "", procfs)


def _is_launcher(procfs: Path, pid: int) -> bool:
    try:
        arguments = (procfs / str(pid) / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return LAUNCHER_SOURCE.encode() in arguments


def _stat(procfs: Path, pid: int) -> tuple[str, int] | None:
    """``(state, parent pid)`` of a process, or ``None`` when it is gone."""
    try:
        text = (procfs / str(pid) / "stat").read_text(**_LOSSY_TEXT)
        fields = text.rsplit(")", 1)[-1].split()
        return fields[0], int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def _command(procfs: Path, pid: int) -> str:
    """The command line of a process, cleaned for a record: no control characters, credentials masked, shortened."""
    try:
        raw = (procfs / str(pid) / "cmdline").read_bytes()
        if not raw:
            raw = b"[" + (procfs / str(pid) / "comm").read_bytes().strip() + b"]"
    except OSError:
        return ""
    text = redact_secret_values(raw.decode(**_LOSSY_TEXT).replace("\0", " "))
    return _CONTROL_CHARS.sub(" ", text).strip()[:COMMAND_CHARS]


def _read_int(path: Path) -> int | None:
    """The number in a file of the kernel's list; ``None`` when the entry went away while it was read."""
    try:
        return int(path.read_text(encoding="ascii").strip())
    except _GONE:
        return None
    except PermissionError as exc:
        raise _Unreadable(f"{path} cannot be read: permission denied") from exc
    except (OSError, ValueError) as exc:
        raise _Unreadable(f"{path} does not hold a number ({type(exc).__name__})") from exc


def _hides_processes(procfs: Path) -> bool:
    """Whether ``procfs`` is mounted so that this account does not see the processes of others."""
    if os.geteuid() == _ROOT_ACCOUNT:
        return False
    for line in (procfs / "self" / "mountinfo").read_text(**_LOSSY_TEXT).splitlines():
        kind, _, rest = line.partition(" - ")[2].partition(" ")
        if kind != "proc":
            continue
        for option in rest.rpartition(" ")[2].split(","):
            if option.startswith(_HIDDEN_OPTION) and option[len(_HIDDEN_OPTION) :] not in _NOT_HIDING:
                return True
    return False


def _cannot_match_ids(kfd: Path, procfs: Path) -> str | None:
    """Why the kernel's process ids cannot be matched to the processes this one sees, or ``None`` when they can."""
    if not (kfd / KFD_PROCESSES).is_dir():
        return f"{kfd / KFD_PROCESSES} is not there: no AMD GPU driver is loaded in this machine or container"
    try:
        namespace = os.readlink(procfs / PID_NAMESPACE_LINK)
        hidden = _hides_processes(procfs)
    except OSError as exc:
        return f"the pid namespace or the mount options of {procfs} cannot be read ({type(exc).__name__})"
    if namespace != INITIAL_PID_NAMESPACE:
        return (
            "this process runs in a pid namespace of its own, so the process ids the kernel lists for the GPUs "
            "(the host's) cannot be matched to the processes it can see"
        )
    if hidden:
        return f"{procfs} hides the processes of other accounts from this one, so a process on a card may not be seen"
    return None


def _gpu_ids(kfd: Path) -> list[int]:
    """The kernel's id of each GPU, in the order the runtime numbers them: the nodes that are not a CPU, by node number."""
    try:
        nodes = sorted(int(entry.name) for entry in (kfd / KFD_NODES).iterdir() if entry.name.isdigit())
    except OSError as exc:
        raise _Unreadable(f"the GPU topology of {kfd} cannot be listed ({type(exc).__name__})") from exc
    ids = []
    for node in nodes:
        gpu_id = _read_int(kfd / KFD_NODES / str(node) / KFD_NODE_GPU_ID)
        if gpu_id is None:
            raise _Unreadable(f"node {node} of the GPU topology went away while it was read")
        if gpu_id:
            ids.append(gpu_id)
    return ids


def _cards_held_by(process: Path, card_of: dict[int, int]) -> set[int]:
    """The cards among ``card_of`` (kernel id to card) that the process has memory resident on or a queue on."""
    held: set[int] = set()
    for name in _listing(process):
        suffix = name[len(KFD_VRAM_PREFIX) :]
        if name.startswith(KFD_VRAM_PREFIX) and suffix.isdigit() and int(suffix) in card_of:
            if (_read_int(process / name) or 0) > 0:
                held.add(card_of[int(suffix)])
    for queue in _listing(process / KFD_QUEUES):
        gpu_id = _read_int(process / KFD_QUEUES / queue / KFD_QUEUE_GPU_ID)
        if gpu_id in card_of:
            held.add(card_of[gpu_id])
    return held


def _listing(directory: Path) -> list[str]:
    try:
        return [entry.name for entry in directory.iterdir()]
    except _GONE:
        return []


def _live(procfs: Path, pid: int) -> bool:
    """Whether the process is there and has not died (a dead process holds nothing, however long its entry lingers)."""
    stat = _stat(procfs, pid)
    return stat is not None and stat[0] != _ZOMBIE


def read_holders(cards: Sequence[int], *, kfd: Path = KFD_PATH, procfs: Path = PROC_PATH) -> CardReading:
    """The processes that hold any of ``cards``, or why that cannot be said.

    ``cards`` are numbered the way ``ROCR_VISIBLE_DEVICES`` numbers them. A card the kernel does not list as a GPU makes
    the whole reading unavailable: a card that is not there is not a card nobody holds.
    """
    reason = _cannot_match_ids(kfd, procfs)
    if reason is not None:
        return CardReading(unavailable=reason)
    try:
        gpu_ids = _gpu_ids(kfd)
        missing = [card for card in cards if not 0 <= card < len(gpu_ids)]
        if missing:
            return CardReading(unavailable=f"card(s) {missing} are not among the {len(gpu_ids)} GPUs the kernel lists")
        card_of = {gpu_ids[card]: card for card in cards}
        found = _holders(kfd, procfs, card_of)
    except _Unreadable as exc:
        return CardReading(unavailable=str(exc))
    except OSError as exc:
        return CardReading(unavailable=f"the kernel's list cannot be read ({type(exc).__name__}: {exc.strerror})")
    return CardReading(holders=found)


def _holders(kfd: Path, procfs: Path, card_of: dict[int, int]) -> tuple[Holder, ...]:
    try:
        pids = sorted(int(entry.name) for entry in (kfd / KFD_PROCESSES).iterdir() if entry.name.isdigit())
    except OSError as exc:
        raise _Unreadable(f"the process list of {kfd} cannot be read ({type(exc).__name__})") from exc
    found = []
    for pid in pids:
        for card in sorted(_cards_held_by(kfd / KFD_PROCESSES / str(pid), card_of)):
            if _live(procfs, pid):
                found.append(Holder(card, pid, _command(procfs, pid)))
    return tuple(found)
