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


# Kept past the rebench's own round, as a share of one baseline round. GEAK stopping late and the wait for its GPUs
# come out of it before explore checks that the round fits; a candidate slower than the baseline comes out of it after.
_REBENCH_SPARE_OF_A_ROUND = 0.25


def rebench_round_sec(state: Any) -> float:
    """What re-measuring a GEAK result costs: the round explore will admit it as, and a quarter of a round to spare.

    The rebench is an explore round measured the way the baseline was. After a baseline that ran a warm pass, explore
    admits it only when its warmup and its decision pass both fit, so holding back one pass leaves the warmup to GEAK
    and the rebench refused for want of the time it was supposed to have. A baseline that could not afford its warm
    pass never leaves PRELUDE, so one without a warm figure is single-pass, as AgentX's is, and so is its rebench.
    """
    first_pass_sec = _phase_state.measured_seconds(state, "baseline_runtime_sec")
    if first_pass_sec is None:
        from ..actions.executors._subprocess_kill import resolve_benchmark_timeouts

        return resolve_benchmark_timeouts()[1]
    warm_pass_sec = _phase_state.measured_seconds(state, "baseline_warm_runtime_sec") or 0.0
    return first_pass_sec + warm_pass_sec + _REBENCH_SPARE_OF_A_ROUND * first_pass_sec


def round_stop(state: Any) -> StoppedByTheRun | None:
    """How to record a measured round the KERNEL budget can no longer fit, or ``None`` while one still fits."""
    remaining = _phase_state.phase_seconds_left(state)
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
