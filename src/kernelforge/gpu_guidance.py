# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Target-specific authoring and profiler guidance for KernelForge."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetGuidance:
    """Facts that change the authoring prompt and profiler choice for a GPU target."""

    hardware_intro: str
    hardware_pointer: str
    autotune_seed: str
    matrix_instruction: str
    utilization: str
    profiler: str


_R9700 = TargetGuidance(
    hardware_intro=(
        "Triton-on-AMD facts (wave size and matrix instructions, num_warps/num_stages guidance,\n"
        "buffer-load and epilogue knobs, fp8 FNUZ/OCP, LDS/occupancy budgets) live in the\n"
        "<knowledge> maps below. Load the relevant card with the `Read` tool for {gpu_target}\n"
        "instead of relying on a remembered value:"
    ),
    hardware_pointer=(
        "- Wavefront / WMMA / LDS / occupancy hardware facts → `hardware/`\n"
        "- Bottleneck classification & numerics → `common_methodology/`\n"
        "Use wave32 and WMMA on the verified R9700 target. Read `hardware/r9700.md` for\n"
        "hardware facts and `languages/triton/skills/optimize/triton_levers/r9700_mxfp4_matmul_ogs.md`\n"
        "for the BF16 activation × MXFP4 weight path. Use rocprofv3 kernel tracing;\n"
        "treat unavailable counters and native FP4 ceilings as unknown."
    ),
    autotune_seed="target-appropriate",
    matrix_instruction="target matrix instruction",
    utilization="matrix-pipe",
    profiler="rocprofv3",
)


def target_guidance(gpu_target: str | None) -> TargetGuidance | None:
    """Return specialized guidance for targets that differ from the CDNA default."""
    return _R9700 if gpu_target == "gfx1201" else None


def profiler_for_target(gpu_target: str | None) -> str:
    """Choose the supported profiler before installing or invoking it."""
    guidance = target_guidance(gpu_target)
    return guidance.profiler if guidance else "rocprof-compute"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the profiler for a GPU target")
    parser.add_argument("gpu_target", nargs="?")
    print(profiler_for_target(parser.parse_args().gpu_target))
