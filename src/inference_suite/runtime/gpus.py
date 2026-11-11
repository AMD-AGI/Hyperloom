# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the GPUs a round starts its server on are visible to ``amd-smi`` and free of other processes.

A round's GPUs are the server's ``ROCR_VISIBLE_DEVICES`` numbers, which ROCm assigns in KFD node order while
``amd-smi process`` indexes cards in PCI order, so each is translated to its amd-smi card before its row is read. A GPU
another process holds would make the server fail, or worse, measure beside a foreign workload, so the round refuses
before it spawns anything. amd-smi lists a process on every GPU it opened, and a ROCm process opens every GPU ROCr
enumerates whatever its HIP mask (the round's own server runs under a ROCr mask of its GPUs), keeping its page tables
and runtime context resident on each; so a process holds a GPU only when the VRAM amd-smi reports for it on that card (read per card from the process's DRM fdinfo) exceeds 1 GiB.

A GPU that is only held is read again until its holder exits or the release window passes, since the holder is most
likely the previous round's server still exiting; a GPU no wait can free refuses at once.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _Refusal:
    """Why one GPU, or the whole check, refuses; ``held`` when a holder exiting would clear it."""

    reason: str
    held: bool


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


def _refusal(gpu: int, rocm_order: Sequence[int], rows: Mapping[int, Mapping[str, Any]]) -> _Refusal | None:
    if gpu >= len(rocm_order):
        return _Refusal(f"GPU {gpu}: not visible (the host has GPUs 0-{len(rocm_order) - 1})", held=False)
    card = rocm_order[gpu]
    row = rows.get(card)
    if row is None:
        return _Refusal(f"GPU {gpu} (amd-smi card {card}): not listed by amd-smi process", held=False)
    if holders := _holders(row):
        return _Refusal(f"GPU {gpu} (amd-smi card {card}): held by {', '.join(holders)}", held=True)
    return None


def _refusals(gpus: Sequence[int]) -> list[_Refusal]:
    """One ``amd-smi`` reading of why each of ``gpus`` cannot take a new server; empty when all are free."""
    try:
        rocm_order = gpus_in_rocm_order()
        rows = {row["gpu"]: row for row in gpu_rows(read_amd_smi_json("process"))}
    except AmdSmiError as exc:
        return [_Refusal(f"GPUs {', '.join(map(str, gpus))} cannot be checked: {exc}", held=False)]
    return [refusal for gpu in gpus if (refusal := _refusal(gpu, rocm_order, rows)) is not None]


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
