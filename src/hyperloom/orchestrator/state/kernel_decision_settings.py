# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared kernel-decision defaults used by state and kernel write owners."""

from __future__ import annotations

import os


# Independent bounded budget for integration-fault *attempts* (separate from the REVERT ``max_attempts`` quota).
_MAX_INTEGRATE_FAULT_ATTEMPTS = 2

# Bounded quota of non-fault gate verdicts before a patch without a KEEP is rejected.
_MAX_INTEGRATE_E2E_ATTEMPTS = 3

# Minimum GPU share for a reusable hot kernel to stay owed work until a backend produces a patch for it.
_DEFAULT_HOT_KERNEL_MIN_GPU_PCT = 5.0


def resolve_hot_kernel_min_gpu_pct() -> float:
    """Resolve the GPU-share threshold a hot kernel must clear to be dispatched."""
    try:
        return float(
            os.environ.get(
                "HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT",
                _DEFAULT_HOT_KERNEL_MIN_GPU_PCT,
            )
        )
    except (TypeError, ValueError):
        return _DEFAULT_HOT_KERNEL_MIN_GPU_PCT


# Only the top-N reusable hot kernels are enforced.
_DEFAULT_HOT_KERNEL_GATE_TOP_N = 5

# Per-action audit history cap (``<action>_attempts`` lists keep most recent N).
_DEFAULT_ATTEMPTS_HISTORY = 20
