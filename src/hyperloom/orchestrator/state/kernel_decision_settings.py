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


def effective_hot_kernel_gpu_pct(candidate: dict) -> float:
    """GPU-time share used for the hot-kernel gate.

    A vendor-playbook group (e.g. mori's dispatch+combine, deliberately
    submitted as one forge-loop session -- see
    ``agents/kernel/tools/_vendor_operator_playbooks.py``) stamps
    ``aggregate_gpu_pct`` as the summed share across the whole group, so a
    split load (7% + 5%) is not silently dropped as below-threshold on
    either member despite the pair clearing it together. Prefer the
    aggregate over the per-row ``gpu_pct`` whenever it is present and larger
    (never smaller, so this can only let a grouped row through a gate it
    would otherwise fail -- it cannot cause an ungrouped row to fail one it
    would otherwise pass).

    Returns:
        float: The larger of ``gpu_pct`` and ``aggregate_gpu_pct`` (when the
            latter is present and parses); ``gpu_pct`` alone otherwise.
    """
    try:
        row_pct = float(candidate.get("gpu_pct") or 0.0)
    except (TypeError, ValueError):
        row_pct = 0.0
    aggregate = candidate.get("aggregate_gpu_pct")
    if aggregate is None:
        return row_pct
    try:
        return max(row_pct, float(aggregate))
    except (TypeError, ValueError):
        return row_pct


def effective_hot_kernel_min_gpu_pct(candidate: dict, min_gpu_pct: float) -> float:
    """Threshold ``candidate`` must clear, honoring a playbook's own floor.

    A vendor playbook may pin a ``min_gpu_pct_floor`` (see
    ``vendor_operator_playbooks.json``) so its group is never dispatched
    below that share even when ``HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT`` is
    loosened for other purposes (e.g. a smaller test fixture) -- the
    playbook's own forge-loop session is a heavier investment than an
    ordinary per-file rewrite attempt, so the floor can only raise the
    effective threshold, never lower it below the caller's own ``min_gpu_pct``.

    Returns:
        float: The larger of ``min_gpu_pct`` and the candidate's stamped
            ``vendor_playbook_min_gpu_pct_floor`` (when present and parses);
            ``min_gpu_pct`` alone otherwise.
    """
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
