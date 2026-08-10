# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared kernel-decision defaults used by state and kernel write owners."""

from __future__ import annotations

import os

from hyperloom.common.timeutil import now_iso


# microseconds + ``+00:00`` (canonical helper; kept importable via shared_state
# for callers that still use that legacy path).
_now_iso = now_iso

# Default partial-attempt cap for run_optimization; override via env in
# ``record_kernel_opt`` (1 disables second chance).
_DEFAULT_KERNEL_OPT_MAX_PARTIAL = 2

# Backend ladder infra failures can be transient; require two failed ladders
# before retiring the kernel. Override via
# ``INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES`` (>=1).
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


# Independent bounded budget for integration-fault *attempts* (separate from the
# REVERT ``max_attempts`` quota). NOTE: this counts total fault attempts, not
# retries-after-the-first: a value of 2 means "one initial fault plus one retry,
# then reject".
_MAX_INTEGRATE_FAULT_ATTEMPTS = 2

# Minimum GPU share for a reusable hot kernel to still owe a kernel_opt attempt.
# Holds KERNEL phase-advance open (kernel_work_pending), filters the dispatch
# batch queue, and drives the advisory 'untried hot kernels' report annotation.
# It does NOT block ``report``.
_DEFAULT_HOT_KERNEL_MIN_GPU_PCT = 10.0


def resolve_hot_kernel_min_gpu_pct() -> float:
    """Resolve the GPU-share threshold a hot kernel must clear to be dispatched.

    The dispatch batch filter, the phase-advance gate and the report's
    unattempted-reason breakdown must all name the same number, or the report
    explains a skip the dispatcher never made.

    Returns:
        float: The threshold from ``HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT``, or the
            shipped default when it is unset or unparseable.
    """
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
