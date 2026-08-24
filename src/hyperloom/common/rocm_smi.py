# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Minimal rocm-smi VRAM reader shared across callers that must not carry Ray.

Single source of truth for ``rocm-smi --showmeminfo vram --json`` parsing.
stdlib-only; no hyperloom imports so it can be loaded from any layer.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import NamedTuple


class GpuVram(NamedTuple):
    """Per-GPU VRAM snapshot (MiB)."""

    used_mib: float
    total_mib: float | None


def gpu_vram_usage(*, timeout: float = 20.0) -> list[GpuVram] | None:
    """Return per-GPU VRAM usage via rocm-smi, or None when unavailable.

    Uses ``rocm-smi --showmeminfo vram --json`` (no HIP device visibility
    required). Any exec, parse, or empty-result failure returns ``None`` so
    callers can treat absence as "unknown" rather than "clean".

    Args:
        timeout: Seconds to wait for rocm-smi before treating it as failed.

    Returns:
        One :class:`GpuVram` per card ordered by card key, or ``None`` when
        rocm-smi is missing, exits non-zero, or produces unparseable output.
    """
    if not shutil.which("rocm-smi"):
        return None
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None

    result: list[GpuVram] = []
    for fields in data.values():
        if not isinstance(fields, dict):
            continue
        used_b: float | None = None
        total_b: float | None = None
        for key, val in fields.items():
            kl = key.lower()
            if "vram" not in kl:
                continue
            if "used" in kl:
                try:
                    used_b = float(val)
                except (TypeError, ValueError):
                    pass
            elif "total" in kl:
                try:
                    total_b = float(val)
                except (TypeError, ValueError):
                    pass
        if used_b is not None:
            result.append(
                GpuVram(
                    used_mib=used_b / (1024.0 * 1024.0),
                    total_mib=total_b / (1024.0 * 1024.0) if total_b is not None else None,
                )
            )
    return result or None


__all__ = ["GpuVram", "gpu_vram_usage"]
