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
read. It cannot say where ``/proc`` hides other accounts' processes, nor where the kernel's list is not there.

In a container with a pid namespace of its own the kernel names processes by their id in the host's namespace, and those
cannot be matched to what the container sees. The probe then does not name whose a holder is: it counts. Before a test
starts nothing may hold a card, so any holder is foreign; while it runs a test's server is the one process with memory on
each of its cards, so a second one is foreign, and so is memory on a card that no process the container sees accounts for
(:mod:`hyperloom.sandbox.drm_fdinfo`). Processes that only opened the interface hold nothing and are not counted.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from hyperloom.common.env_safety import redact_secret_values
from hyperloom.common.outside_facts import (
    INITIAL_PID_NAMESPACE,
    KFD_NODE_GPU_ID,
    KFD_NODE_PROPERTIES,
    KFD_NODES,
    KFD_PROCESSES,
    KFD_QUEUE_GPU_ID,
    KFD_QUEUES,
    KFD_RENDER_MINOR_KEY,
    KFD_ROOT,
    KFD_VRAM_PREFIX,
    PID_NAMESPACE_LINK,
)
from hyperloom.sandbox import drm_fdinfo
from hyperloom.sandbox.launcher import LAUNCHER_SOURCE

__all__ = [
    "CARD_GUARD_ENV",
    "COMMAND_CHARS",
    "EXCESS_SETTLE_SEC",
    "KFD_PATH",
    "METHOD_CENSUS",
    "METHOD_CENSUS_FDINFO",
    "METHOD_PID_MATCH",
    "PROC_PATH",
    "SANDBOX_TAG_ENV",
    "UNACCOUNTED_VRAM_TOLERANCE",
    "CardGuard",
    "CardReading",
    "Holder",
    "Intrusion",
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
#: How a reading told whose a holder is: by matching the kernel's process ids to ``/proc``; by counting the holders of a
#: card (a pid namespace of its own); and by counting them and setting the kernel's memory total of a card against what
#: the processes the reader sees hold.
METHOD_PID_MATCH = "pid-match"
METHOD_CENSUS = "census"
METHOD_CENSUS_FDINFO = "census+fdinfo"
#: Memory on a card the kernel lists and no process this reader sees holds, up to which it is not a foreign holder: what a
#: process allocates between the two reads, and what the driver keeps for itself.
UNACCOUNTED_VRAM_TOLERANCE = 512 << 20
#: How long an excess of memory is read again before it counts. A process that ended stays in the kernel's list, its
#: queues gone and its memory still there, while the driver frees that memory: measured on a GPU pod, a server that had
#: exited kept its 237.9 GiB listed for 2 to 4 seconds, which an immediate second read still saw.
EXCESS_SETTLE_SEC = 6.0
_SETTLE_STEP_SEC = 0.5
_GIB = 1 << 30
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
    no environment; empty when it could not be read (and always, for a holder of a reading that could not match ids, as is
    ``pid``: then it is the id the host gives it). ``vram`` is the bytes resident on the card, ``queues`` the compute
    queues on it."""

    card: int
    pid: int
    command: str
    vram: int = 0
    queues: int = 0


@dataclass(frozen=True)
class Intrusion:
    """Something on a card that is not the test's. ``key`` tells one intrusion from another across looks, ``what`` says it
    and ``command`` names the process when the reading could."""

    key: tuple[str | int, ...]
    what: str
    command: str = ""


@dataclass(frozen=True)
class CardReading:
    """What the probe found on some cards: the processes that hold them, or why it cannot say (then there are none
    listed, and none were ruled out).

    ``method`` says how a holder's ownership can be told. ``unaccounted`` maps a card to the bytes of memory the kernel lists
    on it that no process the reader sees holds, and is empty when that could not be read (``accounting_gap`` says why).
    """

    holders: tuple[Holder, ...] = ()
    unavailable: str | None = None
    method: str = METHOD_PID_MATCH
    unaccounted: Mapping[int, int] = field(default_factory=dict)
    accounting_gap: str | None = None

    def held_cards(self) -> frozenset[int]:
        return frozenset(holder.card for holder in self.holders)

    def foreign(self, own: OwnProcesses) -> tuple[Holder, ...]:
        """The holders that are not ``own``'s."""
        return tuple(holder for holder in self.holders if not own.owns(holder.pid))

    def intrusions(
        self, own: OwnProcesses, *, before_test: bool, accounted: frozenset[tuple[int, int]] = frozenset()
    ) -> tuple[Intrusion, ...]:
        """What on the cards is not the test's: holders that are not ``own``'s when ids could be matched; otherwise any
        holder before the test starts, and while it runs a card with more than one process holding memory on it or with
        memory no process this reader sees accounts for, unless every holder of memory on that card is among
        ``accounted`` (``(card, kernel id)`` pairs an earlier look found accounted for, see :meth:`accounted_holders`)."""
        if self.method == METHOD_PID_MATCH:
            return tuple(
                Intrusion(("pid", h.pid, h.card), f"pid {h.pid} on card {h.card}", h.command) for h in self.foreign(own)
            )
        if before_test:
            return tuple(_host_holder(holder) for holder in self.holders)
        return (*self._crowded(), *self._unexplained(accounted))

    def accounted_holders(self) -> frozenset[tuple[int, int]]:
        """``(card, kernel id)`` of the holders of memory on each card whose memory the processes this reader sees account
        for, to within the tolerance: processes of this container. When one of them ends, the kernel lists its memory for
        a while after its descriptors closed; that memory is then unaccounted for, and it is still not a foreign holder's.
        Empty when the memory could not be set against the processes'."""
        return frozenset(
            (holder.card, holder.pid)
            for holder in self.holders
            if holder.vram > 0
            and self.unaccounted.get(holder.card, UNACCOUNTED_VRAM_TOLERANCE + 1) <= UNACCOUNTED_VRAM_TOLERANCE
        )

    def _crowded(self) -> tuple[Intrusion, ...]:
        found = []
        for card in sorted(self.held_cards()):
            resident = [holder for holder in self.holders if holder.card == card and holder.vram > 0]
            if len(resident) > 1:
                named = ", ".join(f"host pid {h.pid} with {h.vram / _GIB:.1f} GiB" for h in resident)
                found.append(
                    Intrusion(
                        ("crowded", card), f"{len(resident)} processes with memory on card {card} at once ({named})"
                    )
                )
        return tuple(found)

    def _unexplained(self, accounted: frozenset[tuple[int, int]]) -> tuple[Intrusion, ...]:
        found = []
        for card, extra in sorted(self.unaccounted.items()):
            resident = {(card, holder.pid) for holder in self.holders if holder.card == card and holder.vram > 0}
            if extra > UNACCOUNTED_VRAM_TOLERANCE and not (resident and resident <= accounted):
                found.append(
                    Intrusion(
                        ("unaccounted", card),
                        f"{extra / _GIB:.1f} GiB on card {card} that no process this container sees accounts for",
                    )
                )
        return tuple(found)


def _host_holder(holder: Holder) -> Intrusion:
    detail = f"{holder.vram / _GIB:.1f} GiB resident, {holder.queues} queues"
    return Intrusion(("host-pid", holder.pid, holder.card), f"host pid {holder.pid} on card {holder.card} ({detail})")


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


class _PidView(Enum):
    """What this reader's ``/proc`` shows of the processes the kernel's list names."""

    HOST = "host"
    OWN_NAMESPACE = "own namespace"
    HIDDEN = "hidden"


def _driver_absent(kfd: Path) -> str | None:
    if (kfd / KFD_PROCESSES).is_dir():
        return None
    return f"{kfd / KFD_PROCESSES} is not there: no AMD GPU driver is loaded in this machine or container"


def _pid_view(procfs: Path) -> _PidView:
    """Whether the kernel's process ids can be matched to ``procfs``: they can in the host's pid namespace unless ``procfs``
    hides other accounts' processes; in a namespace of its own they cannot be matched at all.

    Raises:
        OSError: When the namespace or the mount options cannot be read.
    """
    namespace = os.readlink(procfs / PID_NAMESPACE_LINK)
    if namespace != INITIAL_PID_NAMESPACE:
        return _PidView.OWN_NAMESPACE
    return _PidView.HIDDEN if _hides_processes(procfs) else _PidView.HOST


class _GpuNode(NamedTuple):
    """A GPU of the kernel's topology: its node number and the id the kernel's process list names it by."""

    node: int
    gpu_id: int


def _gpu_nodes(kfd: Path) -> list[_GpuNode]:
    """The GPUs of the topology, in the order the runtime numbers them: the nodes that are not a CPU, by node number."""
    try:
        nodes = sorted(int(entry.name) for entry in (kfd / KFD_NODES).iterdir() if entry.name.isdigit())
    except OSError as exc:
        raise _Unreadable(f"the GPU topology of {kfd} cannot be listed ({type(exc).__name__})") from exc
    found = []
    for node in nodes:
        gpu_id = _read_int(kfd / KFD_NODES / str(node) / KFD_NODE_GPU_ID)
        if gpu_id is None:
            raise _Unreadable(f"node {node} of the GPU topology went away while it was read")
        if gpu_id:
            found.append(_GpuNode(node, gpu_id))
    return found


def _render_minor(kfd: Path, node: int) -> int | None:
    """The minor of the render node of a GPU's topology node, or ``None`` when its properties do not say."""
    try:
        text = (kfd / KFD_NODES / str(node) / KFD_NODE_PROPERTIES).read_text(**_LOSSY_TEXT)
    except OSError:
        return None
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key == KFD_RENDER_MINOR_KEY:
            return int(value) if value.strip().isdigit() and int(value) > 0 else None
    return None


@dataclass(frozen=True)
class _Holding:
    """What one process has on one card: the bytes resident there and the compute queues on it."""

    vram: int
    queues: int


def _holdings(process: Path, card_of: dict[int, int]) -> dict[int, _Holding]:
    """What the process has on the cards among ``card_of`` (kernel id to card); a card it has neither memory nor a queue on
    is not in the answer."""
    vram: dict[int, int] = {}
    queues: dict[int, int] = {}
    for name in _listing(process):
        suffix = name[len(KFD_VRAM_PREFIX) :]
        if name.startswith(KFD_VRAM_PREFIX) and suffix.isdigit() and int(suffix) in card_of:
            vram[card_of[int(suffix)]] = _read_int(process / name) or 0
    for queue in _listing(process / KFD_QUEUES):
        gpu_id = _read_int(process / KFD_QUEUES / queue / KFD_QUEUE_GPU_ID)
        if gpu_id in card_of:
            queues[card_of[gpu_id]] = queues.get(card_of[gpu_id], 0) + 1
    return {
        card: _Holding(vram.get(card, 0), queues.get(card, 0))
        for card in sorted(vram.keys() | queues.keys())
        if vram.get(card, 0) > 0 or queues.get(card, 0) > 0
    }


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
    the whole reading unavailable: a card that is not there is not a card nobody holds. In a pid namespace of its own the
    holders are counted, not named (see the module's description).
    """
    absent = _driver_absent(kfd)
    if absent is not None:
        return CardReading(unavailable=absent)
    try:
        view = _pid_view(procfs)
    except OSError as exc:
        return CardReading(
            unavailable=f"the pid namespace or the mount options of {procfs} cannot be read ({type(exc).__name__})"
        )
    if view is _PidView.HIDDEN:
        return CardReading(
            unavailable=f"{procfs} hides the processes of other accounts from this one, so a process on a card may not be seen"
        )
    try:
        selected = _selected(kfd, cards)
        if view is _PidView.OWN_NAMESPACE:
            return _census(kfd, procfs, selected)
        return CardReading(
            holders=_holders(kfd, procfs, {gpu.gpu_id: card for card, gpu in selected.items()}, named=True)
        )
    except _Unreadable as exc:
        return CardReading(unavailable=str(exc))
    except OSError as exc:
        return CardReading(unavailable=f"the kernel's list cannot be read ({type(exc).__name__}: {exc.strerror})")


def _selected(kfd: Path, cards: Sequence[int]) -> dict[int, _GpuNode]:
    """The GPU of the topology each of ``cards`` is.

    Raises:
        _Unreadable: When a card is not among the GPUs the kernel lists: a card that is not there is not a free one.
    """
    gpus = _gpu_nodes(kfd)
    missing = [card for card in cards if not 0 <= card < len(gpus)]
    if missing:
        raise _Unreadable(f"card(s) {missing} are not among the {len(gpus)} GPUs the kernel lists")
    return {card: gpus[card] for card in cards}


def _holders(kfd: Path, procfs: Path, card_of: dict[int, int], *, named: bool) -> tuple[Holder, ...]:
    """The holders of the cards among ``card_of``; ``named`` when the kernel's process ids are this reader's own, so a
    process that is not alive is left out and the live ones are given their command lines."""
    try:
        pids = sorted(int(entry.name) for entry in (kfd / KFD_PROCESSES).iterdir() if entry.name.isdigit())
    except OSError as exc:
        raise _Unreadable(f"the process list of {kfd} cannot be read ({type(exc).__name__})") from exc
    found = []
    for pid in pids:
        for card, holding in sorted(_holdings(kfd / KFD_PROCESSES / str(pid), card_of).items()):
            if not named:
                found.append(Holder(card, pid, "", holding.vram, holding.queues))
            elif _live(procfs, pid):
                found.append(Holder(card, pid, _command(procfs, pid), holding.vram, holding.queues))
    return tuple(found)


def _census(kfd: Path, procfs: Path, selected: Mapping[int, _GpuNode]) -> CardReading:
    """The holders of ``selected`` (card to GPU) counted, with the memory the kernel lists on each card set against what the
    processes this reader sees hold. A card found to hold more than the tolerance that nothing accounts for is read again
    until the excess is gone or ``EXCESS_SETTLE_SEC`` has passed, and the last reading decides: a process that ended (the
    test's own server, as it stops) keeps its memory listed for a few seconds after it is gone from ``/proc``, and is not an
    intruder."""
    reading = _census_once(kfd, procfs, selected)
    deadline = time.monotonic() + EXCESS_SETTLE_SEC
    while _has_excess(reading) and time.monotonic() < deadline:
        time.sleep(_SETTLE_STEP_SEC)
        reading = _census_once(kfd, procfs, selected)
    return reading


def _has_excess(reading: CardReading) -> bool:
    return any(extra > UNACCOUNTED_VRAM_TOLERANCE for extra in reading.unaccounted.values())


def _census_once(kfd: Path, procfs: Path, selected: Mapping[int, _GpuNode]) -> CardReading:
    """One look. The kernel's list is read before the processes' descriptors, so memory a process takes in between shows
    in the descriptors and not as an excess."""
    holders = _holders(kfd, procfs, {gpu.gpu_id: card for card, gpu in selected.items()}, named=False)
    minors = {card: _render_minor(kfd, gpu.node) for card, gpu in selected.items()}
    if any(minor is None for minor in minors.values()):
        return CardReading(holders, method=METHOD_CENSUS, accounting_gap="a GPU of the topology lists no render node")
    try:
        seen = drm_fdinfo.vram_by_render_minor(procfs, {minor for minor in minors.values() if minor is not None})
    except drm_fdinfo.FdinfoUnreadable as exc:
        return CardReading(holders, method=METHOD_CENSUS, accounting_gap=str(exc))
    listed = {card: sum(h.vram for h in holders if h.card == card) for card in selected}
    unaccounted = {card: max(0, listed[card] - seen.get(minor or 0, 0)) for card, minor in minors.items()}
    return CardReading(holders, method=METHOD_CENSUS_FDINFO, unaccounted=unaccounted)
