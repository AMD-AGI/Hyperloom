# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware facts for each supported AMD board, and the rule that names a board in probe output."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuBoard:
    """One AMD board as the hardware describes it; Hyperloom's policy for it lives in ``gpu_types``."""

    #: Lowercase product key, also the tag rocm-smi prints in the product name (``MI300X``, ``R9700``).
    product: str
    marketing_name: str
    #: Architecture the board dispatches to.
    gfx: str
    compute_units: int
    hbm_gb: float
    #: Vendor peak memory bandwidth.
    hbm_bw_gbps: float
    #: Vendor dense peak by precision (``bf16``, ``fp16``, ``fp8``, ``fp4``, ``fp32``), for the precisions AMD
    #: publishes a peak for on this board.
    vendor_peak_tflops: Mapping[str, float]

    def vendor_peaks_by_tag(self, tags: Mapping[str, tuple[str, ...]]) -> dict[str, float]:
        """``vendor_peak_tflops`` under each spelling a caller reads, from ``tags``: precision -> its spellings."""
        return {tag: peak for precision, peak in self.vendor_peak_tflops.items() for tag in tags[precision]}

    @property
    def identifiers(self) -> frozenset[str]:
        """The exact lowercase names of this board: its product key and its marketing name in snake case."""
        return frozenset({self.product, re.sub(r"[\s-]+", "_", self.marketing_name.lower())})

    def named_in(self, text: str) -> bool:
        """Whether ``text`` names this board as a whole word; a longer name such as ``MI300XL`` is another product."""
        return re.search(rf"(?<![A-Z0-9]){re.escape(self.product.upper())}(?![A-Z0-9])", text.upper()) is not None


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
        # product, marketing name, gfx, CUs, HBM GB, HBM GB/s, vendor peak TFLOPS
        GpuBoard("mi300x", "AMD Instinct MI300X", "gfx942", 304, 192.0, 5300.0, _MI300X_PEAKS),
        GpuBoard("mi308x", "AMD Instinct MI308X", "gfx942", 304, 192.0, 5300.0, _MI300X_PEAKS),
        GpuBoard("mi325x", "AMD Instinct MI325X", "gfx942", 304, 256.0, 6000.0, _MI300X_PEAKS),
        GpuBoard(
            "mi355x",
            "AMD Instinct MI355X",
            "gfx950",
            256,
            288.0,
            8000.0,
            _matrix_peaks(bf16=2516.6, fp8=5033.2, fp4=10066.4, fp32=157.3),
        ),
        # https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html
        GpuBoard(
            "r9700",
            "AMD Radeon AI PRO R9700",
            "gfx1201",
            64,
            32.0,
            640.0,
            _matrix_peaks(bf16=191.0, fp8=383.0, fp32=47.8),
        ),
    )
}


def gpu_board(gpu_type: str | None) -> GpuBoard | None:
    """Return the board a GPU type names, or ``None`` when unrecognised."""
    return AMD_GPU_BOARDS.get(str(gpu_type or "").strip().lower())


_BOARDS_BY_IDENTIFIER: dict[str, GpuBoard] = {
    identifier: board for board in AMD_GPU_BOARDS.values() for identifier in board.identifiers
}


def board_by_identifier(name: str | None) -> GpuBoard | None:
    """Return the board ``name`` is an exact identifier of (``mi300x``, ``amd_instinct_mi300x``), or ``None``."""
    return _BOARDS_BY_IDENTIFIER.get(str(name or "").strip().lower())


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
    "GpuBoard",
    "board_by_identifier",
    "board_named_in",
    "gfx_arch_for_gpu_type",
    "gpu_board",
    "is_gfx_arch",
]
