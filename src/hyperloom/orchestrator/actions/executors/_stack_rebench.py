# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared full-stack rebench step.

Re-benches a variant layered on the current stack and compares it against a
stability floor (``base_tput * (1 + threshold%)``). Used by the explore ledger
(post-KEEP confirmation) and by integrate_patch (KEEP gate for patches).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..stop_attribution import stopped_by_the_run_class
from ._grid_runner import GridVariant, run_grid


# Single source of truth for the post-KEEP confirmation floor shared by every
# executor that layers a variant on the live stack and re-benches it (the
# explore ledger and integrate_patch). A KEPT gain must re-clear at least this
# margin over ``base_tput`` -- i.e. still reproduce a positive gain above grid
# noise -- rather than merely not regressing (a 0.0 floor only checks
# non-regression and lets noise-level "wins" survive). Kept below the KEEP gate
# (the grid noise floor) so confirmation is a stability check, not a second
# discovery gate -- see :func:`resolve_stable_threshold_pct`, which enforces
# that relationship. Override the request per task via
# ``params['rebench_stable_threshold_pct']``.
DEFAULT_STACK_STABLE_PCT = 0.5


def resolve_stable_threshold_pct(requested_pct: float, keep_threshold_pct: float) -> float:
    """The floor a confirmation rebench is graded against, clamped to half the KEEP gate.

    A confirmation must stay a stability check rather than become a stricter
    second discovery gate as the per-cycle KEEP threshold decays
    (``decaying_keep_threshold_pct`` falls toward 0.1%). An absolute floor
    inverts past cycle 1: at a 0.40% KEEP gate a fixed 0.5% band rejects
    variants the same round admitted. An explicit lower per-task floor stays
    valid; it simply cannot exceed half of the threshold that admitted this
    candidate.

    Args:
        requested_pct: The per-task request, defaulting to
            :data:`DEFAULT_STACK_STABLE_PCT`.
        keep_threshold_pct: The KEEP gate this candidate cleared.

    Returns:
        float: The stability floor as a percentage over the base throughput.
    """
    return min(float(requested_pct), max(0.0, float(keep_threshold_pct) / 2.0))


@dataclass
class StackRebenchResult:
    """Outcome of a single stack rebench measurement."""

    tput: float | None
    workspace: str | None
    warnings: list[str] = field(default_factory=list)
    stable_floor: float = 0.0
    # Set to the ledger class from :mod:`..stop_attribution` when the run itself
    # stopped the round, empty otherwise. Lets a caller tell "the confirmation
    # did not happen" apart from "the confirmation failed": :attr:`stable` is
    # ``False`` for both, and only one of them is evidence about the variant.
    error_class: str = ""

    @property
    def stable(self) -> bool:
        """True when the measured throughput cleared the stability floor."""
        return self.tput is not None and self.tput >= self.stable_floor


async def measure_stack_rebench(
    *,
    config_path: Path | str,
    base_extra_args: str,
    variant: GridVariant,
    base_tput: float,
    stable_threshold_pct: float,
    output_slot: Path,
    variant_timeout_sec: int,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
    result_dir: str | None = None,
    magpie_python: str | None = None,
    server_lifecycle: dict[str, Any] | None = None,
    base_args_mode: str = "append",
    preclean_before_run: bool = True,
    soft_deadline_sec: float | None = None,
    server_already_ready: bool = False,
    serving_lease: Any = None,
    session_deadline_sec: float | None = None,
    variant_expected_sec: float | None = None,
) -> StackRebenchResult:
    """Run ``variant`` once on the stack and grade it against the floor.

    ``soft_deadline_sec`` applies the overtime soft-kill so a pathological
    post-sample server drain is bounded rather than running until the hard
    ``variant_timeout_sec``. ``server_already_ready`` should be ``True`` when
    ``server_lifecycle`` enables cleanup-reuse (the subprocess re-attaches to a
    hot server and never writes a ready marker, which would otherwise leave the
    from-ready soft clock un-armed). ``serving_lease`` (Ray-managed GPU
    execution, §12 T1) routes the round through the caller's held Ray lease so
    the rebench shares the same lease as the warmup/decision rounds it reuses
    the hot server from; ``None`` keeps the local path.

    ``session_deadline_sec`` bounds the rebench by the session wall-clock, so a
    confirmation round cannot outlive the run it is confirming for.
    A rebench dropped for lack of budget is reported as its own warning rather
    than as a failed measurement: not measuring a variant is not evidence that
    the variant is unstable, and the caller grades on the distinction.
    """
    output_slot.mkdir(parents=True, exist_ok=True)
    results = await run_grid(
        base_yaml_path=config_path,
        base_extra_args=base_extra_args,
        grid=[variant],
        output_root=output_slot,
        variant_timeout_sec=variant_timeout_sec,
        model_path=model_path,
        gpu_type=gpu_type,
        benchmark_script=benchmark_script,
        result_dir=result_dir,
        magpie_python=magpie_python,
        server_lifecycle=server_lifecycle,
        base_args_mode=base_args_mode,
        preclean_before_run=preclean_before_run,
        soft_deadline_sec=soft_deadline_sec,
        server_already_ready=server_already_ready,
        serving_lease=serving_lease,
        session_deadline_sec=session_deadline_sec,
        variant_expected_sec=variant_expected_sec,
    )
    rb = results[0] if results else None
    tput: float | None = None
    workspace: str | None = None
    warnings: list[str] = []
    error_class = ""
    if rb is not None and rb.status == "succeeded":
        tput = rb.output_throughput
        workspace = rb.workspace
        warnings = list(rb.nonfatal_warnings)
    elif rb is not None and stopped_by_the_run_class(getattr(rb, "error_class", "")) is not None:
        error_class = rb.error_class
        warnings.append(f"stack_rebench_skipped:{error_class}")
    elif rb is not None:
        warnings.append(f"stack_rebench_failed:{(rb.error or '')[-120:]}")
    else:
        warnings.append("stack_rebench_no_result")
    stable_floor = base_tput * (1.0 + stable_threshold_pct / 100.0)
    return StackRebenchResult(
        tput=tput,
        workspace=workspace,
        warnings=warnings,
        stable_floor=stable_floor,
        error_class=error_class,
    )


__all__ = [
    "DEFAULT_STACK_STABLE_PCT",
    "StackRebenchResult",
    "measure_stack_rebench",
    "resolve_stable_threshold_pct",
]
