# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which AMD GPU a run targets: the one mapping between product names and gfx targets, and the local probe."""

from __future__ import annotations

import re
import subprocess

# Product names fold onto their gfx target so a caller reporting ``MI355X`` and a probe reporting ``gfx950`` agree.
# Keys are spelled lowercase with spaces and dashes as underscores.
_ARCH_BY_GPU_TYPE = {
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi355x": "gfx950",
    "amd_instinct_mi300x": "gfx942",
    "amd_instinct_mi355x": "gfx950",
}
# The product name each gfx target is recorded under, so one chip has one name in KB identities.
_GPU_TYPE_BY_ARCH = {
    "gfx942": "mi300x",
    "gfx950": "mi355x",
}
# A concrete target only: rocminfo also lists generic ISAs such as ``gfx12-generic`` beside the chip's own.
_GFX_RE = re.compile(r"\bgfx[0-9a-f]+\b(?!-)", re.IGNORECASE)


def canon_arch(value: str) -> str:
    """The canonical lowercase gfx target ``value`` names, product name or gfx alike; ``""`` when it names none.

    Lenient: any spelling that contains a gfx target or a known product name folds onto it (``mi300x_vf`` is
    ``gfx942``). Use it to pick a target, not to key stored records; :func:`canonical_gpu_type` does that.
    """
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
    """The product name a known chip is recorded under; any other value keeps its own normalized spelling.

    Exact, because the result keys KB records: only a known product name or gfx target maps, so a spelling that
    merely contains one (``mi300x_vf``, ``gfx942:sramecc+``) keeps the records it was stored under.
    """
    normalized = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    arch = _ARCH_BY_GPU_TYPE.get(normalized, normalized)
    return _GPU_TYPE_BY_ARCH.get(arch, normalized)


def product_arch(value: str) -> str:
    """The gfx target of ``value`` when it is spelled exactly as a known product name; ``""`` otherwise."""
    return _ARCH_BY_GPU_TYPE.get(str(value or "").strip().lower(), "")


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


__all__ = ["canon_arch", "canonical_gpu_type", "detect_arch", "product_arch"]
