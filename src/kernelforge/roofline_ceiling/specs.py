# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Datasheet hardware peaks, the last resort behind the shipped device profiles.

These are vendor peaks, transcribed here as literals: nothing at run time reads
the knowledge-base cards cited below, they are recorded only so a figure can be
traced to where it came from.

No implementation reaches them. The packaged knowledge base says so in as many
words (``hardware/mi350_matrix_core.md``: "Conflating peak with achievable --
~45-55% of peak is the practical ceiling"; ``hardware/mi350_memory.md``:
"Quoting HBM peak as achievable -- sustained is below 8.0 TB/s"). A ceiling
computed from this table is therefore an absolute lower bound on latency rather
than an achievable target, and attainment measured against it reads far below
what the kernel deserves -- so a campaign with an attainment target will never
reach it. That is the intended failure: running too long is recoverable, and
stopping early on a roof nobody checked is not.

The ordinary source is a shipped device profile in
:mod:`~kernelforge.roofline_ceiling.device_profile`, measured on a real card of
the configuration it claims. This table is what answers for a machine no
profile covers.

Peaks are keyed by *instruction path*, not by dtype, because the two are not the
same question. An A16W4 kernel that unpacks to BF16 MFMA runs at the BF16 rate
however "fp4" its weights are, and keying by dtype would hand it the 10 PF FP4
roof and understate its ceiling by 4x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: ``hardware.peak_source`` value for a peak the analyst recalled rather than
#: measured. Always the vendor datasheet in practice, whatever the analyst calls
#: it, so it carries the datasheet's warning.
PEAK_SOURCE_DATASHEET = "datasheet"
#: ``hardware.peak_source`` value for a peak the analyst measured on this box
#: during its session, with ``rocprof-compute --roof-only`` or an equivalent
#: saturating benchmark. The ordinary source, and the only one worth a target.
PEAK_SOURCE_MEASURED = "measured_on_this_box"

#: What each source means, in one phrase, for every reader that has to say so.
#: Centralized so no consumer has to re-derive the distinction and get it wrong.
_PEAK_SOURCE_MEANING = {
    PEAK_SOURCE_MEASURED: (
        "measured on this box during the analysis session; the figures the chip actually "
        "sustained, not the ones it is sold with"
    ),
    PEAK_SOURCE_DATASHEET: (
        "vendor datasheet, measured on no card. The gap to a real card is not a fixed discount: "
        "on gfx950 it is 1.2% for FP32 matrix and 50.8% for FP16 matrix, so cases of different "
        "dtypes stop being comparable and no single correction restores them"
    ),
}


def peak_source_meaning(peak_source: str) -> str:
    """One phrase describing where a set of peaks came from."""
    return _PEAK_SOURCE_MEANING.get(str(peak_source or "").strip(), "of unrecorded origin")


@dataclass(frozen=True)
class ArchSpec:
    """Datasheet peaks for one GPU architecture.

    Only HBM is carried. The cache levels have no datasheet figure worth
    quoting -- the knowledge base gives Infinity Cache a latency and no
    bandwidth -- and inventing one would be worse than the analyst knowing it
    has none. A measured device profile carries them; this table says it cannot.
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

#: Instruction-path pairs an architecture is documented to run at one rate, so
#: a committed profile reporting them apart has transcribed a profiler artifact.
#:
#: Listed explicitly rather than inferred from equal datasheet peaks. The
#: datasheet groups by theoretical peak, and measured throughput within a group
#: legitimately differs -- on gfx950 the table rates fp4 and fp6 together at
#: 10 PF while a real card measures 9.77 and 8.34 PF, which is a property of the
#: chip and not a mistake. Only a pair the vendor states as a single rate can
#: carry the inference, and for gfx950 and gfx942 the knowledge base states
#: exactly one: "FP16/BF16 2.5 PF" and "FP16/BF16 1307 TF" are each one entry.
EQUAL_RATE_PATHS: dict[str, tuple[tuple[str, str], ...]] = {
    "gfx950": (("bf16_mfma", "fp16_mfma"),),
    "gfx942": (("bf16_mfma", "fp16_mfma"),),
}


def equal_rate_paths(arch: str) -> tuple[tuple[str, str], ...]:
    """Instruction-path pairs ``arch`` is documented to run at a single rate."""
    return EQUAL_RATE_PATHS.get(str(arch or "").strip().lower(), ())

#: Every instruction path the analyst may name, whether or not this module
#: carries a datasheet peak for it. A measured device profile carries all of
#: them; the datasheet table carries only the matrix paths, and a run that falls
#: back to the datasheet reports the resulting gap rather than borrowing a
#: neighbouring rate.
#:
#: The vector paths are here because the guide is explicit that scalar work must
#: not be priced at the MFMA rate, and the integer ones because they are where
#: MoE routing, expert sorting, index arithmetic and quantization pack/unpack
#: actually run -- pricing those against a float roof is the same error as
#: pricing softmax against the matrix cores, one level down.
#:
#: There is deliberately no transcendental path. rocprofiler-compute's roofline
#: measures none, and no card in the knowledge base states one, so an entry here
#: could only be filled by a guess. The role document tells the analyst to price
#: SFU work against the vector roof instead and to say that it did: the vector
#: roof is an upper bound on what the transcendental unit can retire, so the
#: substitution makes the ceiling loose in the safe direction -- attainment
#: reads lower than it should, and a campaign runs on rather than stopping
#: early.
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
    "fp8_valu",
    "fp32_valu",
    "fp64_valu",
    "int8_valu",
    "int32_valu",
    "int64_valu",
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
    "EQUAL_RATE_PATHS",
    "KNOWN_INSTRUCTION_PATHS",
    "PEAK_SOURCE_DATASHEET",
    "PEAK_SOURCE_MEASURED",
    "ArchSpec",
    "arch_spec",
    "equal_rate_paths",
    "peak_source_meaning",
    "supported_arches",
]
