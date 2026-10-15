# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one more measured round costs in the KERNEL phase, and whether the phase can still fit one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..actions.stop_attribution import PHASE_TIME_EXHAUSTED_CLASS, STOPPED_BY_THE_RUN, StoppedByTheRun
from . import machine_state as _phase_state


def measurement_round_sec(state: Any) -> float:
    """What measuring one more result in this phase costs: one more round of the baseline's measurement."""
    measured = _phase_state.one_more_measurement_sec(state) or _phase_state.measured_seconds(
        state, "baseline_runtime_sec"
    )
    if measured is not None:
        return measured
    from ..actions.executors._subprocess_kill import resolve_benchmark_timeouts

    return resolve_benchmark_timeouts()[1]


def round_stop(state: Any) -> StoppedByTheRun | None:
    """How to record a measured round the KERNEL budget can no longer fit, or ``None`` while one still fits."""
    remaining = _phase_state.phase_budget_remaining_seconds(state)
    if remaining is None or remaining >= measurement_round_sec(state):
        return None
    return STOPPED_BY_THE_RUN[PHASE_TIME_EXHAUSTED_CLASS]


def controller_round_stop(state: Any, rounds_run: int) -> StoppedByTheRun | None:
    """Admit the E2E round of one more Controller patch.

    The first round is paid for by the round the Controller's own timeout holds back; every later one has to fit what
    is left of the phase.
    """
    return None if rounds_run == 0 else round_stop(state)


async def integrate_within_phase(payload: dict[str, Any], *, state: Any, session_dir: Path) -> dict[str, Any]:
    """Run one integrate round, or report it never started when the KERNEL budget cannot fit another round."""
    from ..kernel.request_handlers import integrate_handler

    stop = round_stop(state)
    if stop is not None:
        return {"status": "skipped", "error_class": stop.error_class, "error": stop.never_started}
    return await integrate_handler(payload, session_dir=session_dir)
