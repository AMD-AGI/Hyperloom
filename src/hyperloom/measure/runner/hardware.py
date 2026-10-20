# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the machine says about its GPUs: model, power cap, partition mode, ECC counts, memory in use (design 4.8.2).

Everything is read through ``amd-smi`` and ``rocm-smi`` and is ``None`` when it cannot be read; a value that could not
be read is never filled in, and a check that needs it is not made.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from hyperloom.common.gpu_identity import (
    AMD_GPU_DISPATCH_IDENTITIES,
    gfx_arch_for_gpu_type,
    gpu_type_for_device_id,
    normalise_device_id,
)
from hyperloom.common.rocm_smi import gpu_vram_usage, vram_busy

__all__ = ["GpuFacts", "EccCounts", "busy_gpus", "check_hardware", "ecc_counts", "read_gpu_facts"]

_JSON = "--json"
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


def run_amd_smi(args: Sequence[str]) -> str | None:
    """The stdout of ``amd-smi <args>``, or ``None`` when it is missing, fails or times out."""
    if not shutil.which("amd-smi"):
        return None
    try:
        done = subprocess.run(["amd-smi", *args], capture_output=True, text=True, timeout=_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


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


def ecc_counts(visible: Sequence[int], run: Runner = run_amd_smi) -> dict[int, EccCounts] | None:
    """Accumulated correctable and uncorrectable ECC counts per visible GPU, or ``None`` when they cannot be read."""
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


def busy_gpus() -> list[int] | None:
    """Indices of GPUs with memory resident, or ``None`` when memory use cannot be read."""
    usage = gpu_vram_usage()
    return None if usage is None else [index for index, gpu in enumerate(usage) if vram_busy(gpu)]


def _wrong_card(gpu: GpuFacts, gpu_type: str) -> bool:
    """Whether ``gpu`` is a different card from the requested type, judged by what the silicon says before its name.

    The device id names the board when it belongs to one; failing that, a different architecture is a different card;
    only when neither can say is the market name compared, as a substring.
    """
    wanted = gpu_type.replace(" ", "").lower()
    if wanted in AMD_GPU_DISPATCH_IDENTITIES:
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
