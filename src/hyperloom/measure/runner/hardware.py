# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the machine says about its GPUs: model, power cap, partition mode, ECC counts, memory in use (design 4.8.2).

Everything is read through ``amd-smi`` and ``rocm-smi`` and is ``None`` when it cannot be read; a value that could not
be read is never filled in, and a check that needs it is not made (the runner records that it was not).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from hyperloom.common.gpu_identity import (
    AMD_GPU_BOARDS,
    gfx_arch_for_gpu_type,
    gpu_type_for_device_id,
    normalise_device_id,
)
from hyperloom.common.rocm_smi import GpuVram, gpu_vram_usage, vram_busy

__all__ = [
    "CARD_RELEASE_WAIT_SEC",
    "EccCounts",
    "GpuFacts",
    "busy_gpus",
    "check_hardware",
    "ecc_counts",
    "read_gpu_facts",
    "released_gpus",
]

_JSON = "--json"
#: How long a visible card that holds memory before the test is read again. A server that crashed in the test before on
#: the card frees its memory only once its processes are torn down, and a retry that starts at once (8 s after the crash
#: on the GPU check) found the card still held and failed three times over; a holder that lasts longer is not the last
#: test's. The census's settle (``EXCESS_SETTLE_SEC``) is for a server that exited, which frees its memory in seconds.
CARD_RELEASE_WAIT_SEC = 60.0
_RELEASE_STEP_SEC = 1.0
_TIMEOUT_SEC = 30.0
Runner = Callable[[Sequence[str]], str | None]


@dataclass(frozen=True)
class GpuFacts:
    """What one GPU says about itself.

    ``model`` is the market name amd-smi prints; it is kept for the record and is not what identifies the card (a node's
    VBIOS may call an MI355X "AMD Instinct MI355 OAM"). ``device_id`` and ``gfx_arch`` identify it.
    """

    index: int
    model: str | None
    power_cap_w: float | None
    partition_mode: str | None
    device_id: str | None = None
    gfx_arch: str | None = None


@dataclass(frozen=True)
class EccCounts:
    correctable: int
    uncorrectable: int


def _run_tool(tool: str, args: Sequence[str]) -> str | None:
    """The stdout of ``<tool> <args>``, or ``None`` when the tool is missing, fails or times out."""
    if not shutil.which(tool):
        return None
    try:
        done = subprocess.run([tool, *args], capture_output=True, text=True, timeout=_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def run_amd_smi(args: Sequence[str]) -> str | None:
    """The stdout of ``amd-smi <args>``, or ``None`` when it is missing, fails or times out."""
    return _run_tool("amd-smi", args)


def run_rocm_smi(args: Sequence[str]) -> str | None:
    """The stdout of ``rocm-smi <args>``, or ``None`` when it is missing, fails or times out."""
    return _run_tool("rocm-smi", args)


def _gpu_data(args: Sequence[str], run: Runner) -> list[dict[str, Any]] | None:
    raw = run(args)
    if raw is None:
        return None
    try:
        document = json.loads(raw)
    except ValueError:
        return None
    data = document.get("gpu_data") if isinstance(document, dict) else document
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else None


def _watts(limit: Any) -> float | None:
    value = ((limit or {}).get("ppt0") or {}).get("socket_power_limit") if isinstance(limit, dict) else None
    number = value.get("value") if isinstance(value, dict) else None
    return float(number) if isinstance(number, (int, float)) else None


def read_gpu_facts(visible: Sequence[int], run: Runner = run_amd_smi) -> list[GpuFacts] | None:
    """Model, power cap and partition mode of each visible GPU, or ``None`` when ``amd-smi`` cannot say."""
    static = _gpu_data(["static", _JSON], run)
    if static is None:
        return None
    partitions = {
        item.get("gpu"): item.get("partition") for item in _gpu_data(["static", "--partition", _JSON], run) or []
    }
    facts = []
    for item in static:
        index = item.get("gpu")
        if index not in visible:
            continue
        partition = partitions.get(index) or {}
        mode = "/".join(str(partition[k]) for k in ("accelerator_partition", "memory_partition") if partition.get(k))
        asic = item.get("asic") if isinstance(item.get("asic"), dict) else {}
        arch = asic.get("target_graphics_version")
        facts.append(
            GpuFacts(
                index,
                asic.get("market_name"),
                _watts(item.get("limit")),
                mode or None,
                normalise_device_id(asic.get("device_id")),
                arch.strip().lower() if isinstance(arch, str) and arch.strip() else None,
            )
        )
    return facts


def _amd_smi_ecc(visible: Sequence[int], run: Runner) -> dict[int, EccCounts] | None:
    data = _gpu_data(["metric", "--ecc", _JSON], run)
    if data is None:
        return None
    counts: dict[int, EccCounts] = {}
    for item in data:
        ecc = item.get("ecc") or {}
        correctable, uncorrectable = ecc.get("total_correctable_count"), ecc.get("total_uncorrectable_count")
        if item.get("gpu") in visible and isinstance(correctable, int) and isinstance(uncorrectable, int):
            counts[item["gpu"]] = EccCounts(correctable, uncorrectable)
    return counts if counts and set(counts) == set(visible) else None


_RAS_GPU = re.compile(r"^GPU\[(\d+)\]:")
_RAS_BLOCK = re.compile(r"^\s*\w+\s+(?:ENABLED|DISABLED)\s+(\d+)\s+(\d+)\s*$")


def _rocm_smi_ras(visible: Sequence[int], run: Runner) -> dict[int, EccCounts] | None:
    """The counts ``rocm-smi --showrasinfo`` prints per block, summed per GPU; a block without counts adds nothing.

    A GPU whose section has no block with counts says nothing about its errors, so it is not in the result.
    """
    raw = run(["--showrasinfo"])
    if raw is None:
        return None
    totals: dict[int, list[int]] = {}
    gpu: int | None = None
    for line in raw.splitlines():
        header, block = _RAS_GPU.match(line), _RAS_BLOCK.match(line)
        if header:
            gpu = int(header.group(1))
        elif block and gpu is not None:
            sums = totals.setdefault(gpu, [0, 0])
            sums[0] += int(block.group(1))
            sums[1] += int(block.group(2))
    counts = {index: EccCounts(*sums) for index, sums in totals.items() if index in visible}
    return counts if counts and set(counts) == set(visible) else None


def ecc_counts(
    visible: Sequence[int], run: Runner = run_amd_smi, ras: Runner = run_rocm_smi
) -> dict[int, EccCounts] | None:
    """Accumulated correctable and uncorrectable ECC counts per visible GPU, or ``None`` when they cannot be read.

    ``amd-smi metric --ecc`` first; when it is missing or does not give every visible GPU, the RAS information of
    ``rocm-smi``.
    """
    counts = _amd_smi_ecc(visible, run)
    return counts if counts is not None else _rocm_smi_ras(visible, ras)


def busy_gpus(visible: Sequence[int], usage: Callable[[], list[GpuVram] | None] = gpu_vram_usage) -> list[int] | None:
    """The visible GPUs that have memory resident, or ``None`` when the memory use of any of them cannot be read."""
    used = usage()
    if used is None or any(index >= len(used) for index in visible):
        return None
    return [index for index in visible if vram_busy(used[index])]


def released_gpus(visible: Sequence[int]) -> list[int] | None:
    """:func:`busy_gpus`, read again every second while a card holds memory, until it is released or
    :data:`CARD_RELEASE_WAIT_SEC` has passed; the last reading decides."""
    deadline = time.monotonic() + CARD_RELEASE_WAIT_SEC
    busy = busy_gpus(visible)
    while busy and time.monotonic() < deadline:
        time.sleep(_RELEASE_STEP_SEC)
        busy = busy_gpus(visible)
    return busy


def _wrong_card(gpu: GpuFacts, gpu_type: str) -> bool:
    """Whether ``gpu`` is a different card from the requested type, judged by what the silicon says before its name.

    The device id names the board when it belongs to one; failing that, a different architecture is a different card;
    only when neither can say is the market name compared, as a substring.
    """
    wanted = gpu_type.replace(" ", "").lower()
    if wanted in AMD_GPU_BOARDS:
        board = gpu_type_for_device_id(gpu.device_id)
        if board is not None:
            return board != wanted
        arch = gfx_arch_for_gpu_type(wanted)
        if gpu.gfx_arch is not None and arch is not None and gpu.gfx_arch != arch:
            return True
    return gpu.model is not None and wanted not in gpu.model.replace(" ", "").lower()


def check_hardware(
    facts: Sequence[GpuFacts], *, gpu_type: str, gpu_count: int, power_cap_w: float | None, partition_mode: str | None
) -> list[str]:
    """What differs between the machine and the sheet's ``hardware``; a value the machine did not give is not compared."""
    problems = []
    if len(facts) < gpu_count:
        problems.append(f"{len(facts)} GPUs visible, {gpu_count} needed")
    for gpu in facts:
        if _wrong_card(gpu, gpu_type):
            seen = ", ".join(x for x in (gpu.device_id and f"device id {gpu.device_id}", gpu.gfx_arch) if x)
            problems.append(f"GPU {gpu.index} is {gpu.model}{f' ({seen})' if seen else ''}, not {gpu_type}")
        if power_cap_w is not None and gpu.power_cap_w is not None and gpu.power_cap_w != power_cap_w:
            problems.append(f"GPU {gpu.index} power cap {gpu.power_cap_w} W, not {power_cap_w} W")
        if partition_mode is not None and gpu.partition_mode is not None and gpu.partition_mode != partition_mode:
            problems.append(f"GPU {gpu.index} partition {gpu.partition_mode}, not {partition_mode}")
    return problems
