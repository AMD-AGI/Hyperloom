# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Datasheet hardware peaks, used only when an empirical roof is unavailable.

These are vendor peaks. No implementation reaches them: the packaged knowledge
base says so in as many words (``hardware/mi350_matrix_core.md``: "Conflating
peak with achievable -- ~45-55% of peak is the practical ceiling";
``hardware/mi350_memory.md``: "Quoting HBM peak as achievable -- sustained is
below 8.0 TB/s"). A ceiling computed from this table is therefore an absolute
lower bound on latency, not an achievable target, and every consumer has to be
able to tell the two apart -- which is what :data:`PEAK_SOURCE_DATASHEET` on the
resolved hardware record is for.

The empirical path (``rocprof-compute --roof-only``) is the default and measures
this box. This table exists so a host without a working profiler still produces
a number, clearly labelled.

Peaks are keyed by *instruction path*, not by dtype, because the two are not the
same question. An A16W4 kernel that unpacks to BF16 MFMA runs at the BF16 rate
however "fp4" its weights are, and keying by dtype would hand it the 10 PF FP4
roof and understate its ceiling by 4x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: ``hardware.peak_source`` value for a peak read off this table.
PEAK_SOURCE_DATASHEET = "datasheet"
#: ``hardware.peak_source`` value for a peak measured by ``--roof-only``.
PEAK_SOURCE_EMPIRICAL = "roof_only_empirical"


@dataclass(frozen=True)
class ArchSpec:
    """Datasheet peaks for one GPU architecture.

    Only HBM is carried. The cache levels have no datasheet figure worth
    quoting -- the knowledge base gives Infinity Cache a latency and no
    bandwidth -- and inventing one would be worse than the analyst knowing it
    has none. ``--roof-only`` measures them; the datasheet path says it cannot.
    """

    arch: str
    hbm_bw_bytes_per_s: float
    #: instruction path -> peak FLOP/s (or OP/s for the integer paths).
    peak_flops: dict[str, float] = field(default_factory=dict)
    source: str = ""

    def bandwidth(self) -> dict[str, float]:
        """Memory level -> bytes/s, in the shape the hardware record carries."""
        return {"hbm": self.hbm_bw_bytes_per_s}


# gfx950 / MI350X / MI355X. Matrix-core rates from
# data/local_knowledge/hardware/mi350_matrix_core.md ("FP16/BF16 2.5 PF, FP8
# 5 PF, FP6/FP4 10 PF", "FP32 matrix (157 TF)", "INT8 MFMA (~5 POPS)"); HBM3E
# 8.0 TB/s from hardware/mi350_memory.md.
#
# The scaled (MXFP) paths share the rate of their widest operand, which is the
# cycle rule the same card states: for ``f8f6f4`` and ``scale_*`` the lower
# cycle count applies only when neither A nor B is FP8. So mxfp8 sits at the FP8
# rate and mxfp6/mxfp4 at the FP6/FP4 rate.
_GFX950 = ArchSpec(
    arch="gfx950",
    hbm_bw_bytes_per_s=8.0e12,
    peak_flops={
        "bf16_mfma": 2.5e15,
        "fp16_mfma": 2.5e15,
        "fp8_mfma": 5.0e15,
        "fp6_mfma": 1.0e16,
        "fp4_mfma": 1.0e16,
        "mxfp8_scaled_mfma": 5.0e15,
        "mxfp6_scaled_mfma": 1.0e16,
        "mxfp4_scaled_mfma": 1.0e16,
        "int8_mfma": 5.0e15,
        "fp32_matrix": 1.57e14,
    },
    source="data/local_knowledge/hardware/mi350_matrix_core.md, mi350_memory.md",
)

# gfx942 / MI300X / MI325X. Rates from the same knowledge family
# (hardware/mi350_memory.md quotes the CDNA3 ridge for comparison) and GEAK's
# perf_knowledge/profiling/roofline_on_mi.md: FP16/BF16 1307 TF, FP8/INT8
# 2615 TF, FP32 163 TF, HBM3 5.325 TB/s.
_GFX942 = ArchSpec(
    arch="gfx942",
    hbm_bw_bytes_per_s=5.325e12,
    peak_flops={
        "bf16_mfma": 1.3074e15,
        "fp16_mfma": 1.3074e15,
        "fp8_mfma": 2.6149e15,
        "int8_mfma": 2.6149e15,
        "fp32_matrix": 1.634e14,
    },
    source="GEAK perf_knowledge/profiling/roofline_on_mi.md",
)

_ARCH_SPECS: dict[str, ArchSpec] = {spec.arch: spec for spec in (_GFX950, _GFX942)}

#: Every instruction path the analyst may name, whether or not this module
#: carries a datasheet peak for it. The vector and SFU paths are here because
#: the guide is explicit that scalar and transcendental work must not be priced
#: at the MFMA rate; ``--roof-only`` measures them, the datasheet table does not,
#: and a run that falls back to the datasheet reports the resulting gap rather
#: than borrowing a neighbouring rate.
#:
#: A path outside this set is never silently mapped onto a neighbour. Guessing
#: is precisely how an A16W4 kernel that unpacks to BF16 ends up measured against
#: the FP4 roof and reports a ceiling four times too low.
CANONICAL_INSTRUCTION_PATHS = (
    "bf16_mfma",
    "fp16_mfma",
    "fp8_mfma",
    "fp6_mfma",
    "fp4_mfma",
    "mxfp8_scaled_mfma",
    "mxfp6_scaled_mfma",
    "mxfp4_scaled_mfma",
    "int8_mfma",
    "fp32_matrix",
    "fp64_matrix",
    "fp16_valu",
    "bf16_valu",
    "fp32_valu",
    "fp64_valu",
)

KNOWN_INSTRUCTION_PATHS = frozenset(CANONICAL_INSTRUCTION_PATHS)


def arch_spec(arch: str) -> ArchSpec | None:
    """Return the datasheet spec for ``arch``, or ``None`` when unknown."""
    return _ARCH_SPECS.get(str(arch or "").strip().lower())


def supported_arches() -> tuple[str, ...]:
    """Architectures this module carries a datasheet peak table for."""
    return tuple(sorted(_ARCH_SPECS))


__all__ = [
    "CANONICAL_INSTRUCTION_PATHS",
    "KNOWN_INSTRUCTION_PATHS",
    "PEAK_SOURCE_DATASHEET",
    "PEAK_SOURCE_EMPIRICAL",
    "ArchSpec",
    "arch_spec",
    "supported_arches",
]
