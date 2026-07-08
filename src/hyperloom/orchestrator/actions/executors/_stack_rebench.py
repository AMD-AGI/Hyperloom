# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared full-stack rebench step.

Re-benches a variant layered on the current stack and compares it against a
stability floor (``base_tput * (1 + threshold%)``). Used by the explore ledger
(post-KEEP confirmation) and by integrate_patch (KEEP gate for patches).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._grid_runner import GridVariant, run_grid


@dataclass
class StackRebenchResult:
    """Outcome of a single stack rebench measurement."""

    tput: float | None
    workspace: str | None
    warnings: list[str] = field(default_factory=list)
    stable_floor: float = 0.0

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
    preclean_before_run: bool = True,
) -> StackRebenchResult:
    """Run ``variant`` once on the stack and grade it against the floor."""
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
        preclean_before_run=preclean_before_run,
    )
    rb = results[0] if results else None
    tput: float | None = None
    workspace: str | None = None
    warnings: list[str] = []
    if rb is not None and rb.status == "succeeded":
        tput = rb.output_throughput
        workspace = rb.workspace
        warnings = list(rb.nonfatal_warnings)
    elif rb is not None:
        warnings.append(f"stack_rebench_failed:{(rb.error or '')[-120:]}")
    else:
        warnings.append("stack_rebench_no_result")
    stable_floor = base_tput * (1.0 + stable_threshold_pct / 100.0)
    return StackRebenchResult(tput=tput, workspace=workspace, warnings=warnings, stable_floor=stable_floor)


__all__ = ["StackRebenchResult", "measure_stack_rebench"]
