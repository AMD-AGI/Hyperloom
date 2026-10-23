# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical AMD GPU type -> dispatch identity table."""

from __future__ import annotations

from collections.abc import Mapping

#: gpu_type -> (dispatch gfx arch, compute-unit count).
AMD_GPU_DISPATCH_IDENTITIES: dict[str, tuple[str, int]] = {
    "mi300x": ("gfx942", 304),
    "mi308x": ("gfx942", 304),
    "mi325x": ("gfx942", 304),
    "mi355x": ("gfx950", 256),
    "r9700": ("gfx1201", 64),
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


def row_dispatches_on(row: Mapping[str, object], identity: tuple[str, int]) -> bool:
    """Whether a tuned-table row's ``gfx`` and ``cu_num`` name ``identity``.

    aiter keys its tuned GEMM tables by ``(gfx, cu_num)`` and builds and dispatches only
    the rows of the GPU it runs on.
    """
    try:
        cu_num = int(float(str(row.get("cu_num") or "").strip()))
    except ValueError:
        return False
    return str(row.get("gfx") or "").strip().lower() == identity[0] and cu_num == identity[1]


__all__ = ["AMD_GPU_DISPATCH_IDENTITIES", "gfx_arch_for_gpu_type", "is_gfx_arch", "row_dispatches_on"]
