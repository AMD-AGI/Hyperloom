# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the GPUs a round starts its server on are visible to ``amd-smi`` and free of other processes.

``amd-smi process`` numbers GPUs as the server's ``HIP_VISIBLE_DEVICES`` does. A GPU another process holds would make
the server fail, or worse, measure beside a foreign workload, so the round refuses before it spawns anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hyperloom_common.amd_smi import AmdSmiError, gpu_rows, read_amd_smi_json


def _holder(entry: object) -> str | None:
    """The process one ``process_list`` entry names; amd-smi nests it under ``process_info`` on some releases, and an
    idle GPU's lone entry carries a sentence there instead."""
    info = entry.get("process_info", entry) if isinstance(entry, Mapping) else None
    if not isinstance(info, Mapping):
        return None
    return f"pid {info.get('pid', '?')} ({info.get('name') or 'unnamed'})"


def _holders(row: Mapping[str, Any]) -> list[str]:
    return [holder for entry in row.get("process_list") or () if (holder := _holder(entry)) is not None]


def unavailable_gpus(gpus: Sequence[int]) -> str | None:
    """Why ``gpus`` cannot take a new server, naming each GPU that is held or not visible; ``None`` when all are free."""
    try:
        rows = {row["gpu"]: row for row in gpu_rows(read_amd_smi_json("process"))}
    except AmdSmiError as exc:
        return f"GPUs {', '.join(map(str, gpus))} cannot be checked: {exc}"
    reasons: list[str] = []
    for gpu in gpus:
        row = rows.get(gpu)
        if row is None:
            reasons.append(f"GPU {gpu}: not visible (the host lists GPUs {', '.join(map(str, sorted(rows)))})")
        elif holders := _holders(row):
            reasons.append(f"GPU {gpu}: held by {', '.join(holders)}")
    return "; ".join(reasons) or None


__all__ = ["unavailable_gpus"]
