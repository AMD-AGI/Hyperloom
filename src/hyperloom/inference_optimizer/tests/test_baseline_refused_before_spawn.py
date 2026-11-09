# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A baseline inference_suite refused before it spawned a server stops the run at once and opens no launch lane."""

from __future__ import annotations

import pytest

from inference_suite.contracts import STEPS_BEFORE_SPAWN, FailedStep

from hyperloom.inference_optimizer.breakdown.stop_reasons import ARGV_INVALID, HOST_GPUS_UNAVAILABLE
from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
from hyperloom.orchestrator.phases import machine_state as phase_state
from hyperloom.orchestrator.state.task_registry import Task

from .conftest import make_coordinator

_OWN_STOP_REASONS = {FailedStep.FLAG_UNKNOWN: ARGV_INVALID, FailedStep.GPUS_UNAVAILABLE: HOST_GPUS_UNAVAILABLE}


async def _fail_baseline(session_dir, phase: str, step: FailedStep, detail: str):
    coord = make_coordinator(session_dir, shared_state_overrides={"enablement_mode": "auto"})
    coord.shared_state.phase = phase
    task = Task(task_id="bl", kind="baseline", state="running", params={}, idempotency_key="bl")
    error_class = failed_step_error_class(step)
    await coord.writeback.handle_unpromotable_result(
        task, {"status": "failed", "error_class": error_class, "error": f"{error_class}: {detail}"}
    )
    return coord.shared_state


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [phase_state.PHASE_PRELUDE, phase_state.PHASE_ENABLEMENT])
@pytest.mark.parametrize("step", sorted(STEPS_BEFORE_SPAWN, key=lambda s: s.value))
async def test_a_refusal_before_spawn_stops_the_run_with_its_cause(session_dir, phase, step):
    state = await _fail_baseline(session_dir, phase, step, "/tmp/cache holds no staging manifest")

    assert state.stop_reason == _OWN_STOP_REASONS.get(step, "baseline_failed")
    assert state.last_action_failures[-1]["error_class"] == failed_step_error_class(step)
    assert "holds no staging manifest" in state.last_action_failures[-1]["error_excerpt"]
    assert not (state.enablement.launch_log or "").strip()


@pytest.mark.asyncio
async def test_a_server_that_exited_before_ready_opens_the_launch_lane(session_dir):
    state = await _fail_baseline(
        session_dir, phase_state.PHASE_PRELUDE, FailedStep.SERVER_EXITED_BEFORE_READY, "KeyError: 'qwen3_moe'"
    )

    assert not state.stop_reason
    assert "KeyError: 'qwen3_moe'" in state.enablement.launch_log
