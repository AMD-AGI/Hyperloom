# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared kernel-decision defaults used by state and kernel write owners."""

from __future__ import annotations

import os

from hyperloom.common.timeutil import now_iso


# microseconds + ``+00:00`` (canonical helper; kept importable via shared_state for callers that still use that legacy
# path).
_now_iso = now_iso

# Default partial-attempt cap for run_optimization; override via env in the integrate lane's retirement check (1
# disables second chance).
_DEFAULT_KERNEL_OPT_MAX_PARTIAL = 2

# Backend ladder infra failures can be transient; require two failed ladders before retiring the kernel.
_DEFAULT_KERNEL_OPT_MAX_FAILURES = 2


def resolve_kernel_opt_max_failures() -> int:
    """Resolve the infra-failure retry budget (>=1)."""
    env_f = os.environ.get("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES")
    if env_f:
        try:
            return max(1, int(env_f))
        except (TypeError, ValueError):
            pass
    return _DEFAULT_KERNEL_OPT_MAX_FAILURES


# Independent bounded budget for integration-fault *attempts* (separate from the REVERT ``max_attempts`` quota).
_MAX_INTEGRATE_FAULT_ATTEMPTS = 2

# Minimum GPU share for a reusable hot kernel to still owe a kernel_opt attempt.
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


def effective_hot_kernel_gpu_pct(candidate: dict) -> float:
    """GPU-time share used for the hot-kernel gate."""
    try:
        row_pct = float(candidate.get("gpu_pct") or 0.0)
    except (TypeError, ValueError):
        row_pct = 0.0
    aggregate = candidate.get("vendor_playbook_aggregate_gpu_pct")
    if aggregate is None:
        return row_pct
    try:
        return max(row_pct, float(aggregate))
    except (TypeError, ValueError):
        return row_pct


def effective_hot_kernel_min_gpu_pct(candidate: dict, min_gpu_pct: float) -> float:
    """Threshold ``candidate`` must clear, honoring a playbook's own floor."""
    floor = candidate.get("vendor_playbook_min_gpu_pct_floor")
    if floor is None:
        return min_gpu_pct
    try:
        return max(min_gpu_pct, float(floor))
    except (TypeError, ValueError):
        return min_gpu_pct


# Only the top-N reusable hot kernels are enforced.
_DEFAULT_HOT_KERNEL_GATE_TOP_N = 5

# Per-action audit history cap (``<action>_attempts`` lists keep most recent N).
_DEFAULT_ATTEMPTS_HISTORY = 20
