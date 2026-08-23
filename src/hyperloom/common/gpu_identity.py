# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical AMD GPU type -> dispatch identity table.

Owned here rather than in ``inference_optimizer`` so provenance can consult it
without importing a higher layer. ``gpu_types.py`` re-exports it, so there is
one table rather than two that drift.

Note the mapping is many-to-one: MI300X, MI308X and MI325X are all ``gfx942``.
An arch therefore identifies the ISA, not the board -- which is why the session
``--gpu-type`` remains the authority for anything that must tell them apart.
"""

from __future__ import annotations

#: gpu_type -> (dispatch gfx arch, compute-unit count).
AMD_GPU_DISPATCH_IDENTITIES: dict[str, tuple[str, int]] = {
    "mi300x": ("gfx942", 304),
    "mi308x": ("gfx942", 304),
    "mi325x": ("gfx942", 304),
    "mi355x": ("gfx950", 256),
    # RDNA4 consumer/workstation. Unlike the Instinct rows this is a wave-32
    # arch with no MFMA, so anything that assumes CDNA matrix instructions must
    # branch on the arch rather than on "is AMD". CU count read from the board
    # (amd-smi NUM_COMPUTE_UNITS on an AMD Radeon AI PRO R9700), not a spec
    # sheet. gfx1200 (Navi 44) is deliberately absent: same ISA family, but no
    # one has measured it here, and a runner label is a claim of support.
    "r9700": ("gfx1201", 64),
}


def gfx_arch_for_gpu_type(gpu_type: str | None) -> str | None:
    """Return the gfx arch for a GPU type, or ``None`` when unrecognised."""
    identity = AMD_GPU_DISPATCH_IDENTITIES.get(str(gpu_type or "").strip().lower())
    return identity[0] if identity else None


__all__ = ["AMD_GPU_DISPATCH_IDENTITIES", "gfx_arch_for_gpu_type"]
