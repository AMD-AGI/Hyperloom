# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the GPUs a round starts its server on are visible to the host's SMI and free of other processes.

A round's GPUs are the server's ``ROCR_VISIBLE_DEVICES`` numbers, which ROCm assigns in KFD node order while
``amd-smi process`` indexes cards in PCI order, so each is translated to its amd-smi card before its row is read. A GPU
another process holds would make the server fail, or worse, measure beside a foreign workload, so the round refuses
before it spawns anything. A ROCm process opens every GPU ROCr enumerates whatever its HIP mask (the round's own server
runs under a ROCr mask of its GPUs) and keeps its page tables and runtime context resident on each, so a process holds a
GPU only when it has more than 1 GiB resident on that GPU.

That figure is read from the kernel: ``/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`` holds the bytes process ``pid``
has resident on the GPU whose ``topology/nodes/<n>/gpu_id`` is ``gpu_id``. ROCm numbers the GPUs in the order of
those topology nodes (the nodes whose ``gpu_id`` is not 0, by node number), which is also the order of the ``node_id``
``amd-smi list`` reports per card, so ROCm GPU ``n`` is the ``n``-th such node and amd-smi card
``gpus_in_rocm_order()[n]``. amd-smi's own per-process figure is not used while the kernel's is readable: for a
process that opened several cards (a vLLM worker opens the render nodes of all eight GPUs) ``amd-smi process`` 26.2
reports one number on every card, the tensor-parallel rank 0's resident bytes on all eight, while the kernel lists
each rank's memory on its own GPU only. A container shown a subset of the GPUs is refused the ``gpu_id`` of the other
GPU nodes; those nodes are left out and the GPUs are numbered among the readable ones, as the runtime numbers them
(``hyperloom.sandbox.card_occupancy`` reads the topology the same way). When the kernel's list cannot be read (no
driver interface, the topology or the process list refused, every GPU node refused, a process's file refused, or a
topology whose GPU count differs from amd-smi's) the check logs why and amd-smi's figure decides instead, as it did
before, including its rule that an entry reporting no memory holds the GPU.

A GPU that is only held is read again until its holder exits or the release window passes, since the holder is most
likely the previous round's server still exiting; a GPU no wait can free refuses at once.

A host with no ``amd-smi`` on it is not an AMD host: its cards are ``nvidia-smi``'s, numbered the way it numbers them.
There the same 1 GiB is read from the compute processes' own figure rather than from a card's whole VRAM, so a desktop
client drawing on the card is not mistaken for a holder; a driver that reports no such figure reads as holding nothing.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.amd_smi import AmdSmiError, gpu_rows, gpus_in_rocm_order, metric_value, read_amd_smi_json

from inference_suite.contracts.report import FailedStep
from inference_suite.runtime.watch import POLL_SEC, RoundGate, Stop

log = logging.getLogger(__name__)

#: A process that opened a card but runs nothing on it keeps 36-52 KiB of page tables there, up to 1.7 MiB once ROCr has
#: initialised the card (recorded on MI355X); a server or a compute job holds gigabytes. 1 GiB leaves room for a runtime
#: that maps larger staging buffers on every card it opens.
_OPENED_ONLY_VRAM_BYTES = 1 << 30

#: How long a held GPU is waited on. A torn-down sglang server's ranks stay <defunct> or exiting, with their memory
#: still listed by amd-smi, for ~25 s after the round that ran them ended (recorded on MI355X between back-to-back
#: rounds on the same cards); 90 s covers a slow exit, and a holder that outlasts it is not the last round's server.
RELEASE_WAIT_SEC = 90.0
#: How often a held GPU is read again; each read runs ``amd-smi process``.
RELEASE_STEP_SEC = 5.0

#: The kernel's list of the processes that use the GPU compute interface, with the GPU topology beside it.
_KFD_ROOT = Path("/sys/class/kfd/kfd")
_KFD_PROCESSES = "proc"
_KFD_NODES = "topology/nodes"
_KFD_NODE_GPU_ID = "gpu_id"
_KFD_VRAM_PREFIX = "vram_"
#: A process or node entry that went away while it was read.
_GONE = (FileNotFoundError, ProcessLookupError, NotADirectoryError)


class _KfdUnreadable(Exception):
    """The kernel's list cannot be read in this view, so the amd-smi figure decides; the message says why, and
    ``refused`` whether it is because this view is refused the file (permission denied)."""

    def __init__(self, message: str, *, refused: bool = False) -> None:
        super().__init__(message)
        self.refused = refused


@dataclass(frozen=True)
class _Refusal:
    """Why one GPU, or the whole check, refuses; ``held`` when a holder exiting would clear it."""

    reason: str
    held: bool


def _holder(entry: object) -> str | None:
    """The process one ``process_list`` entry names when amd-smi's figure says it holds the GPU; amd-smi nests it under
    ``process_info`` on some releases, and an idle GPU's lone entry carries a sentence there instead. An entry that does
    not report its resident memory holds the GPU."""
    info = _info(entry)
    if info is None:
        return None
    memory = info.get("memory_usage")
    vram = metric_value(memory.get("vram_mem")) if isinstance(memory, Mapping) else None
    if vram is not None and vram <= _OPENED_ONLY_VRAM_BYTES:
        return None
    return _named(info.get("pid", "?"), info.get("name"))


def _info(entry: object) -> Mapping[str, Any] | None:
    info = entry.get("process_info", entry) if isinstance(entry, Mapping) else None
    return info if isinstance(info, Mapping) else None


def _named(pid: object, name: object) -> str:
    return f"pid {pid} ({name or 'unnamed'})"


def _holders(row: Mapping[str, Any]) -> list[str]:
    return [holder for entry in row.get("process_list") or () if (holder := _holder(entry)) is not None]


def _kfd_holders(row: Mapping[str, Any], pids: Sequence[int]) -> list[str]:
    """``pids``, which the kernel lists with memory on the GPU, named the way amd-smi's row for its card names them."""
    names = {info.get("pid"): info.get("name") for entry in row.get("process_list") or () if (info := _info(entry))}
    return [_named(pid, names.get(pid)) for pid in pids]


def _read_number(path: Path) -> int | None:
    """The number in one file of the kernel's list; ``None`` when its entry went away while it was read.

    Raises:
        _KfdUnreadable: The file is refused to this view or does not hold a number.
    """
    try:
        return int(path.read_text(encoding="ascii").strip())
    except _GONE:
        return None
    except PermissionError as exc:
        raise _KfdUnreadable(f"{path} is refused to this view", refused=True) from exc
    except (OSError, ValueError) as exc:
        raise _KfdUnreadable(f"{path} does not hold a number ({type(exc).__name__})") from exc


def _kfd_gpu_ids(kfd: Path) -> list[int]:
    """The kernel's id of every GPU this view can read, in the order ROCm numbers them: the topology nodes that are not
    a CPU, by node number, leaving out the nodes whose ``gpu_id`` this view is refused (another container's GPUs).

    Raises:
        _KfdUnreadable: There is no driver interface, the topology cannot be listed, a node went away or is unreadable
            for another reason than a refusal, or every GPU node is refused.
    """
    if not (kfd / _KFD_PROCESSES).is_dir():
        raise _KfdUnreadable(f"{kfd / _KFD_PROCESSES} is not there: no AMD GPU driver interface in this view")
    try:
        nodes = sorted(int(entry.name) for entry in (kfd / _KFD_NODES).iterdir() if entry.name.isdigit())
    except OSError as exc:
        raise _KfdUnreadable(f"the topology {kfd / _KFD_NODES} cannot be listed ({type(exc).__name__})") from exc
    gpu_ids = []
    refused = 0
    for node in nodes:
        try:
            gpu_id = _read_number(kfd / _KFD_NODES / str(node) / _KFD_NODE_GPU_ID)
        except _KfdUnreadable as exc:
            if not exc.refused:
                raise
            refused += 1
            continue
        if gpu_id is None:
            raise _KfdUnreadable(f"topology node {node} went away while it was read")
        if gpu_id:
            gpu_ids.append(gpu_id)
    if refused and not gpu_ids:
        raise _KfdUnreadable(f"every one of the {refused} GPU nodes of {kfd / _KFD_NODES} is refused to this view")
    return gpu_ids


def _kfd_resident(gpus: Sequence[int], card_count: int, kfd: Path) -> dict[int, list[int]]:
    """The processes with more than :data:`_OPENED_ONLY_VRAM_BYTES` resident on each of ``gpus`` (ROCm numbers that
    are all visible), from the kernel's per-process, per-GPU figure.

    Raises:
        _KfdUnreadable: The kernel's list cannot be read in this view, or its GPUs are not the ``card_count`` cards
            amd-smi lists.
    """
    gpu_ids = _kfd_gpu_ids(kfd)
    if len(gpu_ids) != card_count:
        raise _KfdUnreadable(
            f"this view reads {len(gpu_ids)} GPUs in the kernel's topology and amd-smi lists {card_count}"
        )
    try:
        pids = sorted(int(entry.name) for entry in (kfd / _KFD_PROCESSES).iterdir() if entry.name.isdigit())
    except OSError as exc:
        raise _KfdUnreadable(
            f"the process list {kfd / _KFD_PROCESSES} cannot be listed ({type(exc).__name__})"
        ) from exc
    resident: dict[int, list[int]] = {gpu: [] for gpu in gpus}
    for pid in pids:
        process = kfd / _KFD_PROCESSES / str(pid)
        for gpu in gpus:
            # A process has no vram file for a GPU it never opened, and none once it is gone: nothing resident.
            if (_read_number(process / f"{_KFD_VRAM_PREFIX}{gpu_ids[gpu]}") or 0) > _OPENED_ONLY_VRAM_BYTES:
                resident[gpu].append(pid)
    return resident


def _refusal(
    gpu: int,
    rocm_order: Sequence[int],
    rows: Mapping[int, Mapping[str, Any]],
    resident: Mapping[int, Sequence[int]] | None,
) -> _Refusal | None:
    if gpu >= len(rocm_order):
        return _Refusal(f"GPU {gpu}: not visible (the host has GPUs 0-{len(rocm_order) - 1})", held=False)
    card = rocm_order[gpu]
    row = rows.get(card)
    if row is None:
        return _Refusal(f"GPU {gpu} (amd-smi card {card}): not listed by amd-smi process", held=False)
    holders = _holders(row) if resident is None else _kfd_holders(row, resident[gpu])
    if holders:
        return _Refusal(f"GPU {gpu} (amd-smi card {card}): held by {', '.join(holders)}", held=True)
    return None


#: The same threshold in the unit ``nvidia-smi`` reports per-process memory in.
_OPENED_ONLY_VRAM_MIB = _OPENED_ONLY_VRAM_BYTES >> 20


def _nvidia_query(columns: str) -> list[list[str]]:
    """One ``nvidia-smi --query-*`` reading, as rows of trimmed fields."""
    done = subprocess.run(
        ["nvidia-smi", f"--query-{columns}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if done.returncode != 0:
        raise OSError(done.stderr.strip() or f"nvidia-smi exited {done.returncode}")
    return [[field.strip() for field in line.split(",")] for line in done.stdout.splitlines() if line.strip()]


def _nvidia_refusals(gpus: Sequence[int]) -> list[_Refusal]:
    """``_refusals`` for a host ``amd-smi`` does not report: an NVIDIA board, where ``nvidia-smi`` names the cards.

    A card is held when the compute processes on it account for more than the same 1 GiB, read from the processes
    themselves rather than from the card's whole VRAM: a graphics client (a desktop compositor, an overlay) draws on
    the card without holding a compute context a server would collide with, and on a desktop it is always there. A
    driver that reports no per-process figure -- Windows' WDDM model, and the unified-memory parts that answer
    ``[N/A]`` -- cannot show a holder, and reads as holding nothing: the alternative is to read every graphics
    client as one and never start a round.
    """
    try:
        cards = _nvidia_query("gpu=index,uuid")
        apps = _nvidia_query("compute-apps=gpu_uuid,used_memory")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return [_Refusal(f"GPUs {', '.join(map(str, gpus))} cannot be checked: {exc}", held=False)]
    uuid_of = {int(row[0]): row[1].lower() for row in cards}
    card_of = {uuid: card for card, uuid in uuid_of.items()}
    used: dict[int, int] = dict.fromkeys(uuid_of, 0)
    for app in apps:
        card = card_of.get(app[0].lower())
        if card is not None and app[1].isdigit():
            used[card] += int(app[1])
    refusals: list[_Refusal] = []
    for gpu in gpus:
        if gpu not in uuid_of:
            refusals.append(_Refusal(f"GPU {gpu}: not visible (the host has GPUs 0-{len(uuid_of) - 1})", held=False))
        elif used[gpu] > _OPENED_ONLY_VRAM_MIB:
            refusals.append(_Refusal(f"GPU {gpu} (nvidia-smi card {gpu}): held by a compute process", held=True))
    return refusals


def _smi_tool() -> str:
    """The tool that reports this host's cards: the platform's own, else the one a non-AMD board ships.

    ``amd-smi`` wins whenever it is on the host, so an AMD machine reads exactly what it read before. A host without
    it is not an AMD host, and ``nvidia-smi`` is read instead when that is there. With neither, the name comes back
    as ``amd-smi`` so the failure below still reads the way it always has.
    """
    if shutil.which("amd-smi") is not None:
        return "amd-smi"
    if shutil.which("nvidia-smi") is not None:
        return "nvidia-smi"
    return "amd-smi"


def _refusals(gpus: Sequence[int], fallbacks: set[str]) -> list[_Refusal]:
    """One reading of why each of ``gpus`` cannot take a new server; empty when all are free.

    On an AMD host the kernel's per-GPU figure decides who holds a GPU; amd-smi's only when the kernel's list cannot be
    read, which is logged once per reason in ``fallbacks`` (the reasons already logged for this check). A host with no
    ``amd-smi`` is read through ``nvidia-smi`` instead, where the processes' own figure is the only one there is."""
    if _smi_tool() == "nvidia-smi":
        return _nvidia_refusals(gpus)
    try:
        rocm_order = gpus_in_rocm_order()
        rows = {row["gpu"]: row for row in gpu_rows(read_amd_smi_json("process"))}
    except AmdSmiError as exc:
        return [_Refusal(f"GPUs {', '.join(map(str, gpus))} cannot be checked: {exc}", held=False)]
    visible = [gpu for gpu in gpus if gpu < len(rocm_order)]
    try:
        resident: dict[int, list[int]] | None = _kfd_resident(visible, len(rocm_order), _KFD_ROOT)
    except _KfdUnreadable as exc:
        resident = None
        if str(exc) not in fallbacks:
            fallbacks.add(str(exc))
            log.warning(
                "GPUs %s: the kernel's per-GPU memory list cannot be read (%s); amd-smi's figure decides who holds them",
                ", ".join(map(str, gpus)),
                exc,
            )
    return [refusal for gpu in gpus if (refusal := _refusal(gpu, rocm_order, rows, resident)) is not None]


def _gated_sleep(seconds: float, gate: RoundGate) -> Stop | None:
    until = time.monotonic() + seconds
    while (now := time.monotonic()) < until:
        if (stop := gate.check(now)) is not None:
            return stop
        time.sleep(min(POLL_SEC, until - now))
    return None


def amd_smi_cards(gpus: Sequence[int]) -> tuple[int, ...]:
    """The amd-smi card behind each of ``gpus`` (``ROCR_VISIBLE_DEVICES`` numbers), in the same order.

    Raises:
        AmdSmiError: The card order cannot be read, or a GPU is not visible.
    """
    rocm_order = gpus_in_rocm_order()
    invisible = [gpu for gpu in gpus if gpu >= len(rocm_order)]
    if invisible:
        raise AmdSmiError(f"GPUs {', '.join(map(str, invisible))} are not visible (the host has {len(rocm_order)})")
    return tuple(rocm_order[gpu] for gpu in gpus)


def await_free_gpus(gpus: Sequence[int], gate: RoundGate) -> Stop | None:
    """``None`` once ``gpus`` (``ROCR_VISIBLE_DEVICES`` numbers) can take a new server; otherwise a
    ``GPUS_UNAVAILABLE`` stop naming each GPU that is held or not visible, or the stop ``gate`` trips while a held
    GPU is waited on."""
    deadline = time.monotonic() + RELEASE_WAIT_SEC
    fallbacks: set[str] = set()
    while refusals := _refusals(gpus, fallbacks):
        if not all(refusal.held for refusal in refusals) or time.monotonic() >= deadline:
            return Stop(FailedStep.GPUS_UNAVAILABLE, "; ".join(refusal.reason for refusal in refusals))
        if (stop := _gated_sleep(RELEASE_STEP_SEC, gate)) is not None:
            return stop
    return None


__all__ = ["RELEASE_STEP_SEC", "RELEASE_WAIT_SEC", "amd_smi_cards", "await_free_gpus"]
