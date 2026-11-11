# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forge controller time grants and accounting across KERNEL macro cycles."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from time import monotonic
from typing import Any

from . import machine_state as _phase_state
from .kernel_round_budget import measurement_round_sec

log = logging.getLogger(__name__)

_CONTROLLER_ATTEMPT_FACTS = ("macro_cycle", "handoff_dir", "budget_minutes", "hard_timeout_sec")
_CONTROLLER_REMAINING_KEY = "controller_budget_remaining_sec"
_CONTROLLER_GRACE_SEC = 30


class _ForgeControllerMixin:
    """The KERNEL phase's controller grant and durable attempt receipt."""

    def _kernel_rewrite_controller_timeouts(self) -> tuple[int, int]:
        """Return the Controller soft budget and Hyperloom hard timeout."""
        candidates: list[float] = []
        session_remaining = _phase_state.session_remaining_seconds(self.shared_state)
        if session_remaining is not None:
            candidates.append(max(0.0, float(session_remaining) - self.shared_state.closing_reserve_sec()))
        phase_remaining = _phase_state.phase_budget_remaining_seconds(self.shared_state)
        if phase_remaining is not None:
            candidates.append(max(0.0, float(phase_remaining)))
        hard_timeout = int(min(candidates) - measurement_round_sec(self.shared_state)) if candidates else 90 * 60
        allocation = getattr(self.shared_state, "forge_controller_allocation_sec", None)
        if allocation is not None:
            remaining = self.shared_state.kernel_rewrite_controller_result.get(_CONTROLLER_REMAINING_KEY, allocation)
            hard_timeout = min(hard_timeout, int(max(0.0, float(remaining))) + _CONTROLLER_GRACE_SEC)
        return max(0, hard_timeout - _CONTROLLER_GRACE_SEC), max(0, hard_timeout)

    async def _controller_attempt_result(self, handoff_dir: Path, output_dir: Path) -> dict[str, Any]:
        from ..kernel.controller_submit import run_controller_subprocess

        controller_budget_sec, hard_timeout_sec = self._kernel_rewrite_controller_timeouts()
        allocation = getattr(self.shared_state, "forge_controller_allocation_sec", None)
        attempt = {
            "macro_cycle": int(self.shared_state.macro_cycle or 0),
            "handoff_dir": str(handoff_dir),
            "budget_minutes": controller_budget_sec / 60.0,
            "hard_timeout_sec": hard_timeout_sec,
            "patch_count": 0,
            "task_count": 0,
            "output_dir": str(output_dir),
        }
        if allocation is not None:
            remaining = self.shared_state.kernel_rewrite_controller_result.get(_CONTROLLER_REMAINING_KEY, allocation)
            attempt[_CONTROLLER_REMAINING_KEY] = remaining
        if controller_budget_sec <= 0 or hard_timeout_sec <= 0:
            return {
                **attempt,
                "status": "no_result",
                "reason": (
                    "no KERNEL controller time remains"
                    if allocation is not None
                    else "no KERNEL phase budget remains for the rewrite controller"
                ),
            }
        started_monotonic = monotonic()
        try:
            result = await asyncio.to_thread(
                run_controller_subprocess,
                handoff_dir=handoff_dir,
                output_dir=output_dir,
                budget_minutes=attempt["budget_minutes"],
                hard_timeout_sec=hard_timeout_sec,
                explicit_allocation=allocation is not None,
            )
        except (OSError, ValueError) as error:
            log.exception("KERNEL entry: kernel rewrite controller failed")
            result = {"status": "failed", "reason": f"controller invocation failed: {error}"}
            if allocation is None:
                return {**attempt, **result}
        if allocation is not None:
            result[_CONTROLLER_REMAINING_KEY] = max(0.0, remaining - (monotonic() - started_monotonic))
        return {**attempt, **result, **{key: attempt[key] for key in _CONTROLLER_ATTEMPT_FACTS}}
