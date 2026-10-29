# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a sweep rung reads back when it never ran, or when its run raised."""

from __future__ import annotations

from ..actions.executors._grid_runner import GridVariant, VariantResult
from ..actions.stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)


def _budget_skip_result(variant: GridVariant) -> VariantResult:
    """Synthetic VariantResult for a budget-exhausted variant; ``skipped`` status distinguishes \"out of time\" from \"Magpie crashed\".

    No config was composed for this rung and no round of it ran, so it was granted no
    cap and ``granted_cap_sec`` stays ``None``: the grid runner is the only party that
    grants one, and only once it knows the env the rounds launch under.
    """
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="skipped",
        output_throughput=None,
        request_throughput=None,
        total_token_throughput=None,
        error="conc_sweep total budget exhausted before this variant ran",
        error_class="budget_exhausted",
        note=variant.note,
    )


_SWEEP_BUDGET_STOP = StoppedByTheRun(
    error_class="budget_exhausted",
    interrupted="conc_sweep total budget exhausted while this round was running",
    never_started="conc_sweep total budget exhausted before this round ran",
    ends_the_batch=True,
)


# The orchestrator stopped the sweep's action from outside -- a phase transition or a shutdown. Every rung after the
# one it interrupted is recorded under this cause and never run.
_CANCEL_STOP = STOPPED_BY_THE_RUN[ORCHESTRATOR_CANCELLED_CLASS]


def _deadline_skip_result(variant: GridVariant, stopped: StoppedByTheRun) -> VariantResult:
    """Keep the owner of the deadline on every rung that never ran."""
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="skipped",
        error=stopped.never_started,
        error_class=stopped.error_class,
        note=variant.note,
    )


def _raised_result(variant: GridVariant, stage: str, exc: Exception) -> VariantResult:
    """The failed result of a rung whose run raised ``exc`` at ``stage``."""
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="failed",
        error=f"{stage}: {exc}",
        error_class=type(exc).__name__,
    )
