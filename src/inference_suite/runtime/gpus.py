# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the GPUs a round starts its server on are visible to ``amd-smi`` and free of other processes.

A round's GPUs are the server's ``HIP_VISIBLE_DEVICES`` numbers, which ROCm assigns in KFD node order while
``amd-smi process`` indexes cards in PCI order, so each is translated to its amd-smi card before its row is read. A GPU
another process holds would make the server fail, or worse, measure beside a foreign workload, so the round refuses
before it spawns anything. amd-smi lists a process on every GPU it opened, and a ROCm process opens every GPU whatever
its device mask, keeping its page tables and runtime context resident on each; so a process holds a GPU only when the
VRAM amd-smi reports for it on that card (read per card from the process's DRM fdinfo) exceeds 1 GiB.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hyperloom_common.amd_smi import AmdSmiError, gpu_rows, gpus_in_rocm_order, metric_value, read_amd_smi_json

#: A process that opened a card but runs nothing on it keeps 36-52 KiB of page tables there, up to 1.7 MiB once ROCr has
#: initialised the card (recorded on MI355X); a server or a compute job holds gigabytes. 1 GiB leaves room for a runtime
#: that maps larger staging buffers on every card it opens.
_OPENED_ONLY_VRAM_BYTES = 1 << 30


def _holder(entry: object) -> str | None:
    """The process one ``process_list`` entry names when it holds the GPU; amd-smi nests it under ``process_info`` on
    some releases, and an idle GPU's lone entry carries a sentence there instead. An entry that does not report its
    resident memory holds the GPU."""
    info = entry.get("process_info", entry) if isinstance(entry, Mapping) else None
    if not isinstance(info, Mapping):
        return None
    memory = info.get("memory_usage")
    vram = metric_value(memory.get("vram_mem")) if isinstance(memory, Mapping) else None
    if vram is not None and vram <= _OPENED_ONLY_VRAM_BYTES:
        return None
    return f"pid {info.get('pid', '?')} ({info.get('name') or 'unnamed'})"


def _holders(row: Mapping[str, Any]) -> list[str]:
    return [holder for entry in row.get("process_list") or () if (holder := _holder(entry)) is not None]


def _refusal(gpu: int, rocm_order: Sequence[int], rows: Mapping[int, Mapping[str, Any]]) -> str | None:
    if gpu >= len(rocm_order):
        return f"GPU {gpu}: not visible (the host has GPUs 0-{len(rocm_order) - 1})"
    card = rocm_order[gpu]
    row = rows.get(card)
    if row is None:
        return f"GPU {gpu} (amd-smi card {card}): not listed by amd-smi process"
    if holders := _holders(row):
        return f"GPU {gpu} (amd-smi card {card}): held by {', '.join(holders)}"
    return None


def unavailable_gpus(gpus: Sequence[int]) -> str | None:
    """Why ``gpus`` (``HIP_VISIBLE_DEVICES`` numbers) cannot take a new server, naming each GPU that is held or not
    visible; ``None`` when all are free."""
    try:
        rocm_order = gpus_in_rocm_order()
        rows = {row["gpu"]: row for row in gpu_rows(read_amd_smi_json("process"))}
    except AmdSmiError as exc:
        return f"GPUs {', '.join(map(str, gpus))} cannot be checked: {exc}"
    return "; ".join(reason for gpu in gpus if (reason := _refusal(gpu, rocm_order, rows)) is not None) or None


__all__ = ["unavailable_gpus"]
