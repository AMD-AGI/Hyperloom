# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A baseline whose benchmark client refused the result stops the run on the first occurrence, on either runner."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.state.task_registry import Task

from .conftest import make_coordinator


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", ["client_failed", "magpie_nonzero_invalid_measurement"])
async def test_a_refused_baseline_client_stops_the_run_at_once(session_dir, error_class):
    coord = make_coordinator(session_dir, shared_state_overrides={"enablement_mode": "off"})
    task = Task(task_id="bl-client", kind="baseline", state="running", params={}, idempotency_key="bl-client")

    await coord.writeback.handle_unpromotable_result(
        task, {"status": "failed", "error_class": error_class, "error": "client exited with code 1"}
    )

    assert coord.shared_state.stop_reason == "baseline_failed"
