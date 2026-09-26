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
}


def gfx_arch_for_gpu_type(gpu_type: str | None) -> str | None:
    """Return the gfx arch for a GPU type, or ``None`` when unrecognised."""
    identity = AMD_GPU_DISPATCH_IDENTITIES.get(str(gpu_type or "").strip().lower())
    return identity[0] if identity else None


def is_gfx_arch(gpu_type: str | None, arch: str) -> bool:
    """Return whether ``gpu_type`` means ``arch``.

    Callers receive a GPU type that may already have been resolved to an arch, so the
    arch names itself as well as every board that dispatches to it.
    """
    key = str(gpu_type or "").strip().lower()
    return bool(key) and (key == arch or gfx_arch_for_gpu_type(key) == arch)


__all__ = ["AMD_GPU_DISPATCH_IDENTITIES", "gfx_arch_for_gpu_type", "is_gfx_arch"]
