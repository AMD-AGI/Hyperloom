# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launch banners that report which phases and integrations a run brings."""

from __future__ import annotations

import argparse
import sys


def print_lineup_banners(args: argparse.Namespace, *, no_kernel: bool, no_framework_agent: bool) -> None:
    """Print the phase-toggle and integration banners of the run."""
    # Mirror the kernel banner so both phase toggles surface their state.
    if no_framework_agent:
        print(
            "Optimize phase  : DISABLED (--no-framework-agent); "
            f"{'baseline -> SWEEP' if no_kernel else 'baseline -> KERNEL -> SWEEP'}"
        )
    else:
        print("Optimize phase  : ENABLED")
    if no_framework_agent and no_kernel:
        print(
            "WARNING: --no-framework-agent and --no-kernel are both set; the "
            "run collapses to baseline -> SWEEP over an empty "
            "optimization_stack (no config or source search, no KERNEL "
            "rewrites). SWEEP only re-validates the baseline recipe. "
            "Continuing as requested.",
            file=sys.stderr,
        )
    if bool(getattr(args, "research_scout", True)):
        print(
            "Research scout  : ENABLED at PRELUDE (re-dispatch every "
            f"{max(1, int(getattr(args, 'research_scout_interval', 3) or 3))} "
            "explore rounds)"
        )
    else:
        print("Research scout  : DISABLED (--no-research-scout)")
    if bool(getattr(args, "target_advisory", True)):
        print("Target advisory : ENABLED (External target gap injected into prompts; advisory-only)")
    else:
        print("Target advisory : DISABLED (--no-target-advisory)")
    if bool(getattr(args, "recipe_sediment", True)):
        print("Recipe sediment : ENABLED (KEEP/REVERT provenance written to persistent recipe)")
    else:
        print("Recipe sediment : DISABLED (--no-recipe-sediment)")
