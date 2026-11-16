# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which AMD GPU a run targets: the one mapping between product names and gfx targets, and the local probe."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from hyperloom_common.gpu_identity import AMD_GPU_BOARDS, board_named_in


def _spelling(value: str) -> str:
    """``value`` lowercase with spaces and dashes as underscores, the one spelling names are compared in."""
    return re.sub(r"[\s-]+", "_", str(value or "").strip().lower())


# Product names fold onto their gfx target so a caller reporting ``MI355X`` and a probe reporting ``gfx950`` agree.
_ARCH_BY_GPU_TYPE = {
    _spelling(name): board.gfx for board in AMD_GPU_BOARDS.values() for name in (board.product, *board.aliases)
}
# The product name each gfx target is recorded under, so one chip has one name in KB identities.
_GPU_TYPE_BY_ARCH = {
    "gfx942": "mi300x",
    "gfx950": "mi355x",
    "gfx1151": "radeon8060s",
    "gfx1201": "r9700",
    "gfx1100": "w7900",
}
# A concrete target only: rocminfo also lists generic ISAs such as ``gfx12-generic`` beside the chip's own.
_GFX_RE = re.compile(r"\bgfx[0-9a-f]+\b(?!-)", re.IGNORECASE)


def canon_arch(value: str) -> str:
    """The canonical lowercase gfx target ``value`` names, product name or gfx alike; ``""`` when it names none.

    Lenient: a spelling that contains a gfx target, or names a board as a whole word the way
    :func:`hyperloom_common.gpu_identity.board_named_in` reads probe output, folds onto it (``mi300x_vf`` is
    ``gfx942``; ``R9700S`` and ``MI300XL`` are other products and name none). Use it to pick a target, not to key
    stored records; :func:`canonical_gpu_type` does that.
    """
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    match = _GFX_RE.search(raw)
    if match:
        return match.group(0).lower()
    board = board_named_in(re.sub(r"[_-]+", " ", raw))
    return board.gfx if board else ""


def canonical_gpu_type(value: str) -> str:
    """The product name a known chip is recorded under; any other value keeps its own normalized spelling.

    Exact, because the result keys KB records: only a known product name or gfx target maps, so a spelling that
    merely contains one (``mi300x_vf``, ``gfx942:sramecc+``) keeps the records it was stored under.
    """
    normalized = _spelling(value)
    arch = _ARCH_BY_GPU_TYPE.get(normalized, normalized)
    return _GPU_TYPE_BY_ARCH.get(arch, normalized)


def product_arch(value: str) -> str:
    """The gfx target of ``value`` when it is spelled exactly as a known product name; ``""`` otherwise."""
    return _ARCH_BY_GPU_TYPE.get(str(value or "").strip().lower(), "")


@dataclass(frozen=True)
class OccupancyRule:
    """The register and LDS levels at which a kernel on one gfx target drops an occupancy step."""

    vgpr_limit: int
    vgpr_within: str
    vgpr_beyond: str
    lds_limit_kib: int
    lds_within: str
    lds_beyond: str


#: CDNA3/CDNA4 (gfx942/gfx950): the 256-VGPR and ~80 KiB-LDS dual-occupancy thresholds hold for both.
CDNA_OCCUPANCY = OccupancyRule(
    vgpr_limit=256,
    vgpr_within="occupancy≥2 possible",
    vgpr_beyond="occupancy=1 ONLY",
    lds_limit_kib=80,
    lds_within="dual-occupancy OK",
    lds_beyond="single-occupancy",
)


@dataclass(frozen=True)
class ArchProfile:
    """What KernelForge does differently on one gfx target.

    A target without a profile keeps the Instinct defaults every consumer had before profiles existed.
    """

    family: str
    #: Knowledge folder, relative to the knowledge root, injected as the ``hardware`` pillar.
    hardware_knowledge: str = "hardware"
    #: Whether rocprof-compute supports the target, which the profiled Analysis tier and self-profiling run on.
    rocprof_compute: bool = True
    #: Kernel backends that can build for the target; ``None`` means every registered backend.
    kernel_backends: frozenset[str] | None = None
    occupancy: OccupancyRule = CDNA_OCCUPANCY
    #: Always-inlined note that re-reads the Instinct vocabulary of the backend prompts for this target.
    prompt_note: str = ""

    def supports_kernel_backend(self, backend: str) -> bool:
        return self.kernel_backends is None or backend in self.kernel_backends


_RDNA3_PROMPT_NOTE = """\
## Target architecture: gfx1100 (RDNA 3, Radeon PRO W7900) — overrides any Instinct fact in this prompt

The backend guidance and the knowledge cards are mostly written for Instinct (CDNA) GPUs. On this target:
- Matrix instructions are **WMMA** (`v_wmma_*`, 16x16x16), not MFMA. Wherever the guidance says MFMA,
  read WMMA; MFMA intrinsics, scaled MFMA and their shape tables do not exist here.
- There are **no AGPRs**. Accumulators live in ordinary VGPRs; ignore AGPR budgets and `v_accvgpr_*` advice.
- HIP compiles gfx1100 for **wave32** unless `-mwavefrontsize64` is passed; check a Triton kernel's wave size in
  its AMDGCN dump. Lane, shuffle, ballot and reduction math follows the wave size the build chose, not CDNA's 64.
- **No FP8/FP4** matrix path and no FNUZ/OCP question: FP16, BF16, INT8 and INT4 are the WMMA input types.
- Memory is **GDDR6 (~864 GB/s)** behind a 96 MB Infinity Cache, not HBM; the roofline ridge is far lower.
- rocprof-compute does not support this target, so "wait/MFMA" ratios and its PMC reports are unavailable. Judge
  bottlenecks with the RDNA 3 cards under `hardware/rdna3/` and with rocprofv3 where it is installed.
- The kernel backends that build here are triton, hip, hipblaslt and ck. HipKittens and the `gl.amd.cdna3`/`cdna4`
  Gluon ops are CDNA only.
Every concrete number for this chip is in `hardware/rdna3/`; read those cards, not the MI350 ones."""

#: gfx1100 in wave32: 1536 VGPRs per SIMD allocated in blocks of 24 and at most 16 waves, so 96 VGPRs is the last
#: count that keeps every wave slot; LDS is 128 KiB per WGP with a 64 KiB cap per work-group (RDNA 3 ISA §3.3).
_RDNA3_OCCUPANCY = OccupancyRule(
    vgpr_limit=96,
    vgpr_within="16 waves/SIMD, full occupancy in wave32",
    vgpr_beyond="fewer than 16 waves/SIMD: 1536 / VGPRs rounded up to 24",
    lds_limit_kib=64,
    lds_within="fits the 64 KiB work-group cap; 128 KiB per WGP",
    lds_beyond="over the 64 KiB work-group cap: will not launch",
)

_ARCH_PROFILES: dict[str, ArchProfile] = {
    "gfx942": ArchProfile(family="CDNA3"),
    "gfx950": ArchProfile(family="CDNA4"),
    "gfx1100": ArchProfile(
        family="RDNA3",
        hardware_knowledge="hardware/rdna3",
        rocprof_compute=False,
        kernel_backends=frozenset({"triton", "hip", "hipblaslt", "ck"}),
        occupancy=_RDNA3_OCCUPANCY,
        prompt_note=_RDNA3_PROMPT_NOTE,
    ),
}
_DEFAULT_PROFILE = ArchProfile(family="")


def arch_profile(gpu_target: str) -> ArchProfile:
    """The profile of the gfx target ``gpu_target`` names; targets without one get the Instinct defaults."""
    return _ARCH_PROFILES.get(canon_arch(gpu_target), _DEFAULT_PROFILE)


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


__all__ = [
    "CDNA_OCCUPANCY",
    "ArchProfile",
    "OccupancyRule",
    "arch_profile",
    "canon_arch",
    "canonical_gpu_type",
    "detect_arch",
    "product_arch",
]
