# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which AMD GPU a run targets: the one mapping between product names and gfx targets, and the local probe."""

from __future__ import annotations

import re
import subprocess

# Product names fold onto their gfx target so a caller reporting ``MI355X`` and a probe reporting ``gfx950`` agree.
_ARCH_BY_GPU_TYPE = {
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi355x": "gfx950",
}
# The product name each gfx target is recorded under, so one chip has one name in KB identities.
_GPU_TYPE_BY_ARCH = {
    "gfx942": "mi300x",
    "gfx950": "mi355x",
}
# A concrete target only: rocminfo also lists generic ISAs such as ``gfx12-generic`` beside the chip's own.
_GFX_RE = re.compile(r"\bgfx[0-9a-f]+\b(?!-)", re.IGNORECASE)


def canon_arch(value: str) -> str:
    """The canonical lowercase gfx target ``value`` names, product name or gfx alike; ``""`` when it names none."""
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    match = _GFX_RE.search(raw)
    if match:
        return match.group(0).lower()
    for gpu_type, arch in _ARCH_BY_GPU_TYPE.items():
        if gpu_type in raw:
            return arch
    return ""


def canonical_gpu_type(value: str) -> str:
    """The product name a known chip is recorded under; an unknown one keeps its own normalized spelling."""
    normalized = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    return _GPU_TYPE_BY_ARCH.get(canon_arch(normalized), normalized)


def detect_arch(timeout_s: float = 15.0) -> str:
    """The local gfx target rocminfo reports; ``""`` when it cannot run, fails, or reports none or several."""
    try:
        completed = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    targets = {match.lower() for match in _GFX_RE.findall(completed.stdout or "")}
    return targets.pop() if len(targets) == 1 else ""


__all__ = ["canon_arch", "canonical_gpu_type", "detect_arch"]
