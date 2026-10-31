# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware facts for each supported AMD board, and the rule that names a board in probe output."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuBoard:
    """One AMD board as the hardware describes it; Hyperloom's policy for it lives in ``gpu_types``."""

    #: Lowercase product key, also the tag rocm-smi prints in the product name (``MI300X``, ``R9700``).
    product: str
    #: Architecture the board dispatches to.
    gfx: str
    compute_units: int
    #: ``None`` when Hyperloom carries no published figure for the board.
    hbm_gb: float | None = None
    #: Vendor peak memory bandwidth, ``None`` like ``hbm_gb``.
    hbm_bw_gbps: float | None = None
    #: Vendor dense peak by precision (``bf16``, ``fp16``, ``fp8``, ``fp4``, ``fp32``), for the precisions AMD
    #: publishes a peak for on this board.
    vendor_peak_tflops: Mapping[str, float] = field(default_factory=dict)
    #: Other spellings probe output uses for the board (rocm-smi prints ``Radeon 8060S Graphics``).
    aliases: tuple[str, ...] = ()

    def vendor_peaks_by_tag(self, tags: Mapping[str, tuple[str, ...]]) -> dict[str, float]:
        """``vendor_peak_tflops`` under each spelling a caller reads, from ``tags``: precision -> its spellings."""
        return {tag: peak for precision, peak in self.vendor_peak_tflops.items() for tag in tags[precision]}

    def named_in(self, text: str) -> bool:
        """Whether ``text`` names this board as a whole word; a longer name such as ``MI300XL`` is another product."""
        upper = text.upper()
        return any(
            re.search(rf"(?<![A-Z0-9]){re.escape(name.upper())}(?![A-Z0-9])", upper) is not None
            for name in (self.product, *self.aliases)
        )


def _matrix_peaks(*, bf16: float, fp8: float, fp32: float, fp4: float | None = None) -> dict[str, float]:
    """Vendor peaks keyed by precision; AMD rates FP16 and BF16 matrix math the same on every board listed."""
    peaks = {"bf16": bf16, "fp16": bf16, "fp8": fp8, "fp32": fp32}
    if fp4 is not None:
        peaks["fp4"] = fp4
    return peaks


_MI300X_PEAKS = _matrix_peaks(bf16=1307.4, fp8=2614.9, fp32=163.4)

AMD_GPU_BOARDS: dict[str, GpuBoard] = {
    board.product: board
    for board in (
        # product, gfx, CUs, HBM GB, HBM GB/s, vendor peak TFLOPS
        GpuBoard("mi300x", "gfx942", 304, 192.0, 5300.0, _MI300X_PEAKS, aliases=("AMD Instinct MI300X",)),
        GpuBoard("mi308x", "gfx942", 304, 192.0, 5300.0, _MI300X_PEAKS, aliases=("AMD Instinct MI308X",)),
        GpuBoard("mi325x", "gfx942", 304, 256.0, 6000.0, _MI300X_PEAKS, aliases=("AMD Instinct MI325X",)),
        GpuBoard(
            "mi355x",
            "gfx950",
            256,
            288.0,
            8000.0,
            _matrix_peaks(bf16=2516.6, fp8=5033.2, fp4=10066.4, fp32=157.3),
            aliases=("AMD Instinct MI355X",),
        ),
        # https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html
        GpuBoard("r9700", "gfx1201", 64, 32.0, 640.0, _matrix_peaks(bf16=191.0, fp8=383.0, fp32=47.8)),
        # The roofline reads TraceLens's bundled Radeon_8060S.json; Hyperloom carries no peaks for it.
        GpuBoard("radeon8060s", "gfx1151", 40, aliases=("Radeon 8060S",)),
    )
}

#: gpu_type -> the PCI device ids that ``amd-smi static --asic`` reports (``asic.device_id``) for that board, lower case,
#: no ``0x``. A board's name is whatever the VBIOS says (an MI355X node answers "AMD Instinct MI355 OAM"); its device id
#: is what the silicon is. Read from the ``amdgpu.ids`` that ships with the driver, with the variants (HF, VF) listed
#: under their board. 0x74a2 and 0x74b6 are left out on purpose: the same id is both an MI308X and, at revision A1, an
#: MI300X, so it names no single board and the caller falls back to the name. So is 0x7551, the R9700 at revisions C0
#: and C1 (R9700S) and the R9600D at C8: the R9700 lists no id and is known by its name alone. The Radeon 8060S lists
#: none either and is known by its name.
AMD_GPU_DEVICE_IDS: dict[str, frozenset[str]] = {
    "mi300x": frozenset({"74a1", "74a9", "74b5", "74bd"}),
    "mi308x": frozenset({"74a8"}),
    "mi325x": frozenset({"74a5", "74b9"}),
    "mi355x": frozenset({"75a3", "75b3"}),
    "r9700": frozenset(),
    "radeon8060s": frozenset(),
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


def gpu_board(gpu_type: str | None) -> GpuBoard | None:
    """Return the board a GPU type names, or ``None`` when unrecognised."""
    return AMD_GPU_BOARDS.get(str(gpu_type or "").strip().lower())


def board_named_in(text: str | None) -> GpuBoard | None:
    """Return the board that ``text`` (rocm-smi product output, a marketing name) names, or ``None``."""
    return next((board for board in AMD_GPU_BOARDS.values() if board.named_in(str(text or ""))), None)


def gfx_arch_for_gpu_type(gpu_type: str | None) -> str | None:
    """Return the gfx arch for a GPU type, or ``None`` when unrecognised."""
    board = gpu_board(gpu_type)
    return board.gfx if board else None


def is_gfx_arch(gpu_type: str | None, arch: str) -> bool:
    """Return whether ``gpu_type`` means ``arch``.

    Callers receive a GPU type that may already have been resolved to an arch, so the
    arch names itself as well as every board that dispatches to it.
    """
    key = str(gpu_type or "").strip().lower()
    return bool(key) and (key == arch or gfx_arch_for_gpu_type(key) == arch)


__all__ = [
    "AMD_GPU_BOARDS",
    "AMD_GPU_DEVICE_IDS",
    "GpuBoard",
    "board_named_in",
    "gfx_arch_for_gpu_type",
    "gpu_board",
    "gpu_type_for_device_id",
    "is_gfx_arch",
    "normalise_device_id",
]
