# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical AMD GPU type -> dispatch identity table."""

from __future__ import annotations

#: gpu_type -> (dispatch gfx arch, compute-unit count).
AMD_GPU_DISPATCH_IDENTITIES: dict[str, tuple[str, int]] = {
    "mi300x": ("gfx942", 304),
    "mi308x": ("gfx942", 304),
    "mi325x": ("gfx942", 304),
    "mi355x": ("gfx950", 256),
    "radeon8065s": ("gfx1151", 40),
}


def gfx_arch_for_gpu_type(gpu_type: str | None) -> str | None:
    """Return the gfx arch for a GPU type, or ``None`` when unrecognised."""
    identity = AMD_GPU_DISPATCH_IDENTITIES.get(str(gpu_type or "").strip().lower())
    return identity[0] if identity else None


__all__ = ["AMD_GPU_DISPATCH_IDENTITIES", "gfx_arch_for_gpu_type"]
