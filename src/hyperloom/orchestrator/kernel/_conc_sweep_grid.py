# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two arms a conc sweep compares, the CONC ladder each runs, and what one rung is launched and recorded with."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..actions.executors._grid_runner import GridVariant, VariantResult
from ..actions.executors._proposal_identity import controls_of, is_executable, normalize_proposal
from ..actions.stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    STOPPED_BY_THE_RUN,
    StoppedByTheRun,
)
from ..state.shared_state import SharedState


@dataclass(frozen=True)
class _Arm:
    """One side of the comparison: the launch recipe every rung of its ladder runs with."""

    name: str
    args: str = ""
    envs: dict[str, str] = field(default_factory=dict)
    overlay: str = ""
    controls: Mapping[str, Any] = field(default_factory=dict)


def _optimized_arm(state: SharedState) -> _Arm | None:
    """The arm for the retained config or kernel overlay, or ``None`` when there is nothing to compare."""
    cb = state.current_best or {}
    config = normalize_proposal(cb)
    overlay = str(cb.get("final_overlay") or "").strip()
    if not (is_executable(config) or overlay):
        return None
    return _Arm(
        name="optimized",
        args=config["extra_args"],
        envs=config["extra_envs"],
        overlay=overlay,
        controls=controls_of(config),
    )


def _budget_skip_result(variant: GridVariant) -> VariantResult:
    """Synthetic VariantResult for a budget-exhausted variant; ``skipped`` status distinguishes \"out of time\" from \"Magpie crashed\"."""
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


@dataclass(frozen=True)
class _RungNote:
    """How one rung came to run and whether it is part of the published curve.

    ``stage`` is a ``STAGE_*`` value, ``committed`` says whether the rung is part
    of the published curve, and the seconds fields are wall clock.
    """

    stage: str
    committed: bool = True
    start_time: str = ""
    wall_duration_sec: float | None = None
    budget_remaining_sec: float | None = None


@dataclass(frozen=True)
class _RungServer:
    """The serving lease and server lifecycle a rung runs under."""

    serving_lease: Any
    base_extra_envs: dict[str, str] | None = None
    server_lifecycle: dict[str, Any] | None = None
    server_already_ready: bool = False
    warmup_before_measure: bool | None = None
    lifecycle_boot_only: bool = False


@dataclass(frozen=True)
class _ArmCurves:
    """The measured points of both arms of a sweep."""

    baseline: list[dict[str, Any]]
    optimized: list[dict[str, Any]]


def _order_concs_desc(concs: list[int]) -> list[int]:
    """Return a strictly descending, deduplicated copy of the CONC ladder."""
    return sorted(set(concs), reverse=True)


def _build_arm_grid(
    arm: _Arm,
    concs_desc: list[int],
    *,
    isl: int,
    osl: int,
    num_prompts_factor: int,
) -> list[GridVariant]:
    """Build a single-arm grid in descending CONC order."""
    out: list[GridVariant] = []
    for conc in concs_desc:
        num_prompts = max(int(conc) * int(num_prompts_factor), int(conc))
        envs = dict(arm.envs)
        envs.update(
            {
                "CONC": str(conc),
                "ISL": str(isl),
                "OSL": str(osl),
                "NUM_PROMPTS": str(num_prompts),
            }
        )
        envs["RUN_EVAL"] = "false"
        variant = GridVariant(
            name=f"{arm.name}_conc{conc}",
            extra_server_args=arm.args,
            extra_envs=envs,
            note=f"arm={arm.name} conc={conc} isl={isl} osl={osl}",
            **arm.controls,
        )
        variant.overlay_pythonpath = arm.overlay  # type: ignore[attr-defined]
        out.append(variant)
    return out
