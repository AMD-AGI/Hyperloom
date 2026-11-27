# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the GPUs a round starts its server on are visible to ``amd-smi`` and free of other processes.

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
each rank's memory on its own GPU only. When the kernel's list cannot be read (no driver interface, a GPU node or a
process file refused to this view, or a topology whose GPU count differs from amd-smi's) the amd-smi figure decides
instead, as it did before.

A GPU that is only held is read again until its holder exits or the release window passes, since the holder is most
likely the previous round's server still exiting; a GPU no wait can free refuses at once.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.amd_smi import AmdSmiError, gpu_rows, gpus_in_rocm_order, metric_value, read_amd_smi_json

from inference_suite.contracts.report import FailedStep
from inference_suite.runtime.watch import POLL_SEC, RoundGate, Stop

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
_GONE = (FileNotFoundError, ProcessLookupError)


class _KfdUnreadable(Exception):
    """The kernel's list cannot be read in this view, so the amd-smi figure decides."""


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
    except (OSError, ValueError) as exc:
        raise _KfdUnreadable(f"{path}: {type(exc).__name__}") from exc


def _kfd_gpu_ids(kfd: Path) -> list[int]:
    """The kernel's id of every GPU, in the order ROCm numbers them: the topology nodes that are not a CPU, by number.

    Raises:
        _KfdUnreadable: The topology cannot be listed, or a node is refused to this view or went away.
    """
    try:
        nodes = sorted(int(entry.name) for entry in (kfd / _KFD_NODES).iterdir() if entry.name.isdigit())
    except OSError as exc:
        raise _KfdUnreadable(f"{kfd / _KFD_NODES}: {type(exc).__name__}") from exc
    gpu_ids = []
    for node in nodes:
        gpu_id = _read_number(kfd / _KFD_NODES / str(node) / _KFD_NODE_GPU_ID)
        if gpu_id is None:
            raise _KfdUnreadable(f"topology node {node} went away while it was read")
        if gpu_id:
            gpu_ids.append(gpu_id)
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
        raise _KfdUnreadable(f"the kernel lists {len(gpu_ids)} GPUs and amd-smi {card_count}")
    try:
        pids = sorted(int(entry.name) for entry in (kfd / _KFD_PROCESSES).iterdir() if entry.name.isdigit())
    except OSError as exc:
        raise _KfdUnreadable(f"{kfd / _KFD_PROCESSES}: {type(exc).__name__}") from exc
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


def _refusals(gpus: Sequence[int]) -> list[_Refusal]:
    """One reading of why each of ``gpus`` cannot take a new server; empty when all are free. The kernel's per-GPU
    figure decides who holds a GPU; amd-smi's only when the kernel's list cannot be read."""
    try:
        rocm_order = gpus_in_rocm_order()
        rows = {row["gpu"]: row for row in gpu_rows(read_amd_smi_json("process"))}
    except AmdSmiError as exc:
        return [_Refusal(f"GPUs {', '.join(map(str, gpus))} cannot be checked: {exc}", held=False)]
    visible = [gpu for gpu in gpus if gpu < len(rocm_order)]
    try:
        resident: dict[int, list[int]] | None = _kfd_resident(visible, len(rocm_order), _KFD_ROOT)
    except _KfdUnreadable:
        resident = None
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
    while refusals := _refusals(gpus):
        if not all(refusal.held for refusal in refusals) or time.monotonic() >= deadline:
            return Stop(FailedStep.GPUS_UNAVAILABLE, "; ".join(refusal.reason for refusal in refusals))
        if (stop := _gated_sleep(RELEASE_STEP_SEC, gate)) is not None:
            return stop
    return None


__all__ = ["RELEASE_STEP_SEC", "RELEASE_WAIT_SEC", "amd_smi_cards", "await_free_gpus"]
