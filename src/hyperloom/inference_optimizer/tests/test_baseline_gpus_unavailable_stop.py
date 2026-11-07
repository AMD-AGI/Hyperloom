# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A baseline inference_suite refused because its GPUs are held or not visible stops the run at once, naming them."""

from __future__ import annotations

import pytest

from inference_suite.contracts import FailedStep

from hyperloom.inference_optimizer.breakdown.stop_reasons import HOST_GPUS_UNAVAILABLE
from hyperloom.orchestrator.actions.executors._failed_step import failed_step_error_class
from hyperloom.orchestrator.phases import machine_state as phase_state
from hyperloom.orchestrator.state.task_registry import Task

from .conftest import make_coordinator

_DETAIL = "GPU 3: held by pid 4242 (python3); GPU 9: not visible (the host lists GPUs 0, 1, 2, 3, 4, 5, 6, 7)"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [phase_state.PHASE_PRELUDE, phase_state.PHASE_ENABLEMENT])
async def test_a_baseline_refused_for_its_gpus_stops_the_run_on_the_first_occurrence(session_dir, phase):
    coord = make_coordinator(session_dir, shared_state_overrides={"enablement_mode": "auto"})
    coord.shared_state.phase = phase
    task = Task(task_id="bl-gpus", kind="baseline", state="running", params={}, idempotency_key="bl-gpus")
    error_class = failed_step_error_class(FailedStep.GPUS_UNAVAILABLE)

    await coord.writeback.handle_unpromotable_result(
        task, {"status": "failed", "error_class": error_class, "error": _DETAIL}
    )

    state = coord.shared_state
    assert error_class == HOST_GPUS_UNAVAILABLE
    assert state.stop_reason == HOST_GPUS_UNAVAILABLE
    assert state.last_action_failures[-1]["error_class"] == HOST_GPUS_UNAVAILABLE
    assert "GPU 3: held by pid 4242" in state.last_action_failures[-1]["error_excerpt"]
    assert not (state.enablement.launch_log or "").strip()
