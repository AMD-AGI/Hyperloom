# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which AMD GPU a run targets: the gfx targets the stored GPU labels stand for, and the local probe."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import Enum

from hyperloom_common.gpu_card import board_label


def _spelling(value: str) -> str:
    """``value`` lowercase with spaces and dashes as underscores, the one spelling names are compared in."""
    return re.sub(r"[\s-]+", "_", str(value or "").strip().lower())


# The labels KB records are stored under; changing one strands the records filed under it. gfx1201 stays an ISA
# because it names more than one board.
_GPU_TYPE_BY_ARCH = {
    "gfx942": "mi300x",
    "gfx950": "mi355x",
    "gfx1100": "w7900",
    "gfx1151": "radeon8060s",
}
_ARCH_BY_GPU_TYPE = {gpu_type: arch for arch, gpu_type in _GPU_TYPE_BY_ARCH.items()}
# A concrete target only: rocminfo also lists generic ISAs such as ``gfx12-generic`` beside the chip's own.
_GFX_RE = re.compile(r"\bgfx[0-9a-f]+\b(?!-)", re.IGNORECASE)


def canon_arch(value: str) -> str:
    """The canonical lowercase gfx target ``value`` names; ``""`` when it names none.

    Lenient: a spelling that contains a gfx target names it (``gfx942:sramecc+`` is ``gfx942``), and a name whose
    :func:`hyperloom_common.gpu_card.board_label` is one of the stored labels names that label's target
    (``AMD Instinct MI300X`` and ``mi300x_vf`` are ``gfx942``). Any other product name names none. Use it to pick a
    target, not to key stored records; :func:`canonical_gpu_type` does that.
    """
    raw = str(value or "").strip().lower()
    match = _GFX_RE.search(raw)
    if match:
        return match.group(0)
    return _ARCH_BY_GPU_TYPE.get(board_label(raw), "")


def canonical_gpu_type(value: str) -> str:
    """The label a stored gfx target's records are filed under; any other value keeps its own normalized spelling.

    Exact, because the result keys KB records: only the gfx target itself maps, so a spelling that merely contains
    one (``gfx942:sramecc+``) or names another board on the same target (``mi325x``) keeps the records it was stored
    under.
    """
    normalized = _spelling(value)
    return _GPU_TYPE_BY_ARCH.get(normalized, normalized)


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


class Profiler(Enum):
    """The hardware profiler a gfx target is profiled with."""

    #: rocprof-compute (Omniperf): the profiled Analysis tier and the implementer's rocpc_profile.py run on it.
    ROCPROF_COMPUTE = "rocprof-compute"
    #: rocprofv3 only: kernel traces and the counters the device lists, which the implementer drives by hand.
    ROCPROFV3 = "rocprofv3"
    NONE = "none"


@dataclass(frozen=True)
class ArchProfile:
    """What KernelForge does differently on one gfx target.

    A target without a profile keeps the Instinct defaults every consumer had before profiles existed.
    """

    family: str
    #: Knowledge folder, relative to the knowledge root, injected as the ``hardware`` pillar.
    hardware_knowledge: str = "hardware"
    profiler: Profiler = Profiler.ROCPROF_COMPUTE
    #: Kernel backends that can build for the target; ``None`` means every registered backend.
    kernel_backends: frozenset[str] | None = None
    occupancy: OccupancyRule = CDNA_OCCUPANCY
    #: Always-inlined note that re-reads the Instinct vocabulary of the backend prompts for this target;
    #: ``{kernel_backends}`` stands for the backends that build for it.
    prompt_note: str = ""

    def supports_kernel_backend(self, backend: str) -> bool:
        return self.kernel_backends is None or backend in self.kernel_backends

    def rendered_prompt_note(self) -> str:
        """``prompt_note`` with the target's kernel backends filled in from ``kernel_backends``."""
        return self.prompt_note.format(kernel_backends=", ".join(sorted(self.kernel_backends or ())))


_RDNA3_PROMPT_NOTE = """\
## Target architecture: gfx1100 (RDNA 3, Radeon PRO W7900 / W7900D) — overrides any Instinct fact in this prompt

The backend guidance and the knowledge cards are mostly written for Instinct (CDNA) GPUs. On this target:
- Matrix instructions are **WMMA** (`v_wmma_*`, 16x16x16), not MFMA. Wherever the guidance says MFMA,
  read WMMA; MFMA intrinsics, scaled MFMA and their shape tables do not exist here.
- There are **no AGPRs**. Accumulators live in ordinary VGPRs; ignore AGPR budgets and `v_accvgpr_*` advice.
- HIP and Triton compile gfx1100 for **wave32** (HIP unless `-mwavefrontsize64` is passed). Lane, shuffle, ballot
  and reduction math follows the wave size the build chose (`.amdhsa_wavefront_size32` in the dump), not CDNA's 64.
- **No FP8/FP4** matrix path and no FNUZ/OCP question: FP16, BF16, INT8 and INT4 are the WMMA input types.
- Memory is **GDDR6 (~864 GB/s)** behind a 96 MB Infinity Cache, not HBM; the roofline ridge is far lower.
- rocprof-compute does not support this target, so "wait/MFMA" ratios and its PMC reports are unavailable and the
  Analysis Agent stays static. Judge bottlenecks with the RDNA 3 cards under `hardware/rdna3/`; the implementer
  can profile with rocprofv3 (its Self-profiling section says how).
- The kernel backends that build here are {kernel_backends}. HipKittens and the `gl.amd.cdna3`/`cdna4` Gluon ops
  are CDNA only.
- Benchmarks are noisy on this board: a gain under ~10% on a kernel of tens of microseconds is usually not
  measurable. Prefer changes with a large expected gain, and benchmark cases long enough to time reliably.
Every concrete number for this chip is in `hardware/rdna3/`. Where another card cites `hardware/mi350_*.md`, read the
RDNA 3 card for that topic instead (the map in `hardware/rdna3/INDEX.md` pairs them); the MI350 numbers are wrong here."""

#: gfx1100/gfx1201 static wave32: 1536 VGPRs per SIMD, allocation granule 24 and at most 16 waves; LLVM's
#: Feature1536VGPRs owns that rule. LDS is 128 KiB per WGP with a 64 KiB work-group cap.
_RDNA_WAVE32_OCCUPANCY = OccupancyRule(
    vgpr_limit=96,
    vgpr_within="16 waves/SIMD, full occupancy in wave32",
    vgpr_beyond="fewer than 16 waves/SIMD: 1536 / VGPRs rounded up to 24",
    lds_limit_kib=64,
    lds_within="fits the 64 KiB work-group cap; 128 KiB per WGP",
    lds_beyond="over the 64 KiB work-group cap: will not launch",
)

_RDNA4_PROMPT_NOTE = """\
## Target architecture: gfx1201 (RDNA 4, Radeon AI PRO R9700) — overrides any Instinct fact in this prompt

Use wave32 and WMMA; inspect the compiled gfx1201 ISA before choosing lane layouts or matrix instructions.
Use VGPRs for accumulators and the GDDR6 memory facts in `hardware/rdna4/`.
For GPT-OSS BF16 activation × packed-MXFP4 weight tuning, read
`languages/triton/skills/optimize/triton_levers/r9700_mxfp4_matmul_ogs.md`.
Use rocprofv3 kernel tracing where installed, retain raw traces,
and collect only small runtime-supported counter groups with finite timeouts; unavailable counters and native
scaled-FP4 ceilings remain unknown. The Analysis Agent stays static-only under the current profiling plan;
the implementer self-profiles with rocprofv3 when profiling is requested."""

_ARCH_PROFILES: dict[str, ArchProfile] = {
    "gfx942": ArchProfile(family="CDNA3"),
    "gfx950": ArchProfile(family="CDNA4"),
    "gfx1100": ArchProfile(
        family="RDNA3",
        hardware_knowledge="hardware/rdna3",
        profiler=Profiler.ROCPROFV3,
        kernel_backends=frozenset({"triton", "hip", "hipblaslt", "ck"}),
        occupancy=_RDNA_WAVE32_OCCUPANCY,
        prompt_note=_RDNA3_PROMPT_NOTE,
    ),
    "gfx1201": ArchProfile(
        family="RDNA4",
        hardware_knowledge="hardware/rdna4",
        profiler=Profiler.ROCPROFV3,
        occupancy=_RDNA_WAVE32_OCCUPANCY,
        prompt_note=_RDNA4_PROMPT_NOTE,
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
    "Profiler",
    "arch_profile",
    "canon_arch",
    "canonical_gpu_type",
    "detect_arch",
]
