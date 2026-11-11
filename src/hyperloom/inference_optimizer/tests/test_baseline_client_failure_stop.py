# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A baseline the harness refused after its whole protocol stops the run at once; a client that exited nonzero is
retried on the failure streak, since inference_suite names any nonzero client exit ``client_failed``, a transient one
included."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.state.task_registry import Task

from .conftest import make_coordinator


def _task(n: int) -> Task:
    return Task(task_id=f"bl-{n}", kind="baseline", state="running", params={}, idempotency_key=f"bl-{n}")


@pytest.mark.asyncio
async def test_a_refused_measurement_stops_the_run_at_once(session_dir):
    coord = make_coordinator(session_dir, shared_state_overrides={"enablement_mode": "off"})

    await coord.writeback.handle_unpromotable_result(
        _task(1), {"status": "failed", "error_class": "round_nonzero_invalid_measurement", "error": "refused"}
    )

    assert coord.shared_state.stop_reason == "baseline_failed"


@pytest.mark.asyncio
async def test_a_failed_client_is_retried_on_the_streak_and_stops_the_run_at_its_third_failure(session_dir):
    coord = make_coordinator(session_dir, shared_state_overrides={"enablement_mode": "off"})
    failed = {"status": "failed", "error_class": "client_failed", "error": "429 Too Many Requests"}

    for n in (1, 2):
        await coord.writeback.handle_unpromotable_result(_task(n), dict(failed))
        assert not coord.shared_state.stop_reason, f"failure {n} must leave a retry"

    await coord.writeback.handle_unpromotable_result(_task(3), dict(failed))

    assert coord.shared_state.stop_reason == "baseline_failed"
