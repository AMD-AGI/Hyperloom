"""ExecutionMode selection — DESIGN §3.4.1.

``MAX_HOURS`` is the *only* knob that picks the run-time mode.

    < 2h   →  QUICK_PARAM_SWEEP
    2-6h   →  GUIDED_KERNEL_OPT       (inclusive at 2 and at 6)
    > 6h   →  MARATHON_MULTI_AGENT
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum


class ExecutionMode(str, Enum):
    QUICK_PARAM_SWEEP = "quick_param_sweep"
    GUIDED_KERNEL_OPT = "guided_kernel_opt"
    MARATHON_MULTI_AGENT = "marathon_multi_agent"


def choose_execution_mode(env: Mapping[str, str]) -> ExecutionMode:
    """Pick the mode from the user's env block."""
    if "MAX_HOURS" not in env:
        raise ValueError("MAX_HOURS is required (DESIGN §8.1)")
    try:
        max_hours = float(env["MAX_HOURS"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MAX_HOURS must be numeric, got {env['MAX_HOURS']!r}") from exc
    if max_hours <= 0:
        raise ValueError("MAX_HOURS must be > 0 (DESIGN §8.1)")
    if max_hours < 2:
        return ExecutionMode.QUICK_PARAM_SWEEP
    if max_hours <= 6:
        return ExecutionMode.GUIDED_KERNEL_OPT
    return ExecutionMode.MARATHON_MULTI_AGENT
