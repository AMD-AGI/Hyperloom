# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared rocm-smi VRAM reader.

stdlib-only and free of hyperloom imports, so it is loadable from any layer.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, NamedTuple

_TIMEOUT_SEC = 20.0


class GpuVram(NamedTuple):
    """Per-GPU VRAM usage in MiB."""

    used_mib: float
    total_mib: float


def gpu_vram_usage() -> list[GpuVram] | None:
    """Return per-GPU VRAM usage via rocm-smi, or None when it cannot be read.

    Uses ``rocm-smi --showmeminfo vram --json``, which reports byte counts and
    needs no HIP device visibility. Every unreadable case yields ``None`` --
    binary absent, non-zero exit, unparseable JSON, or a card missing either
    figure -- so callers can tell "unknown" apart from "idle".

    Returns:
        One :class:`GpuVram` per card in rocm-smi key order, or ``None``.
    """
    if not shutil.which("rocm-smi"):
        return None
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    result: list[GpuVram] = []
    for fields in data.values():
        if not isinstance(fields, dict):
            continue
        raw: dict[str, Any] = {}
        for key, val in fields.items():
            kl = key.lower()
            if "vram" not in kl:
                continue
            # "VRAM Total Used Memory (B)" matches both, so used must win.
            if "used" in kl:
                raw["used"] = val
            elif "total" in kl:
                raw["total"] = val
        if not raw:
            continue
        try:
            usage = GpuVram(float(raw["used"]) / 1024**2, float(raw["total"]) / 1024**2)
        except (KeyError, TypeError, ValueError):
            return None
        if usage.total_mib <= 0.0:
            return None
        result.append(usage)
    return result or None


__all__ = ["GpuVram", "gpu_vram_usage"]
