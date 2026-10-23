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
    "r9700": ("gfx1201", 64),
}

#: gpu_type -> the PCI device ids that ``amd-smi static --asic`` reports (``asic.device_id``) for that board, lower case,
#: no ``0x``. A board's name is whatever the VBIOS says (an MI355X node answers "AMD Instinct MI355 OAM"); its device id
#: is what the silicon is. Read from the ``amdgpu.ids`` that ships with the driver, with the variants (HF, VF) listed
#: under their board. 0x74a2 and 0x74b6 are left out on purpose: the same id is both an MI308X and, at revision A1, an
#: MI300X, so it names no single board and the caller falls back to the name. So is 0x7551, the R9700 at revisions C0
#: and C1 (R9700S) and the R9600D at C8: the R9700 lists no id and is known by its name alone.
AMD_GPU_DEVICE_IDS: dict[str, frozenset[str]] = {
    "mi300x": frozenset({"74a1", "74a9", "74b5", "74bd"}),
    "mi308x": frozenset({"74a8"}),
    "mi325x": frozenset({"74a5", "74b9"}),
    "mi355x": frozenset({"75a3", "75b3"}),
    "r9700": frozenset(),
}


def normalise_device_id(device_id: object) -> str | None:
    """``"0x75A3"`` (or the integer ``0x75a3``) as ``"75a3"``; ``None`` when it is not a device id."""
    if isinstance(device_id, int) and not isinstance(device_id, bool) and device_id >= 0:
        return f"{device_id:04x}"
    text = str(device_id or "").strip().lower().removeprefix("0x")
    return text if text and all(c in "0123456789abcdef" for c in text) else None


def gpu_type_for_device_id(device_id: object) -> str | None:
    """The one board a PCI device id belongs to, or ``None`` when it is unknown, ambiguous or unreadable."""
    key = normalise_device_id(device_id)
    return next((board for board, ids in AMD_GPU_DEVICE_IDS.items() if key in ids), None)


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


__all__ = [
    "AMD_GPU_DEVICE_IDS",
    "AMD_GPU_DISPATCH_IDENTITIES",
    "gfx_arch_for_gpu_type",
    "gpu_type_for_device_id",
    "is_gfx_arch",
    "normalise_device_id",
]
