# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether this macro cycle's Forge rewrite controller has stopped."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CONTROLLER_TERMINAL_STATUSES",
    "kernel_controller_settled",
]

#: Every status the rewrite controller can end on. All of them are terminal for
#: the phase: the controller is not re-run inside one macro cycle, so a failure
#: is as final as a published patch.
CONTROLLER_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "completed",
        "failed",
        "no_opportunity",
        "no_result",
        "partial",
    }
)


def kernel_controller_settled(state: Any) -> bool:
    """Return true when this macro cycle's rewrite controller has stopped."""
    if str(state.kernel_optimizer or "").strip().lower() != "forge":
        return False
    result = state.kernel_rewrite_controller_result or {}
    if not isinstance(result, dict):
        return False
    try:
        result_cycle = int(result.get("macro_cycle", -1))
        current_cycle = int(state.macro_cycle or 0)
    except (TypeError, ValueError):
        return False
    status = str(result.get("status") or "").strip().lower()
    return result_cycle == current_cycle and status in CONTROLLER_TERMINAL_STATUSES
