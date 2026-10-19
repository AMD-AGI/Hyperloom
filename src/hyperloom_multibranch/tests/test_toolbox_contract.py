# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom_multibranch.contracts import toolbox
from hyperloom_multibranch.contracts.toolbox import (
    REJECTION_CODES,
    Backpressure,
    CancelOutcome,
    QueueState,
    RequestRejected,
    ToolboxError,
    UnknownTestRequest,
)


def test_rejection_carries_the_reason_and_a_stable_code() -> None:
    error = RequestRejected("patch patch/vllm/0001 does not apply on c0", "delta_not_applicable")
    assert isinstance(error, ToolboxError)
    assert error.reason == "patch patch/vllm/0001 does not apply on c0"
    assert error.code == "delta_not_applicable"
    assert "delta_not_applicable" in str(error)
    assert RequestRejected("bad").code in REJECTION_CODES


def test_backpressure_is_not_a_rejection() -> None:
    error = Backpressure("b1", queued=9, limit=8)
    assert isinstance(error, ToolboxError)
    assert not isinstance(error, RequestRejected)
    assert (error.branch_id, error.queued, error.limit) == ("b1", 9, 8)


def test_unknown_request_is_a_lookup_error() -> None:
    assert issubclass(UnknownTestRequest, LookupError)
    assert issubclass(UnknownTestRequest, ToolboxError)


def test_empty_queue_state_lists_every_priority() -> None:
    state = QueueState()
    assert dict(state.queued_by_priority) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    assert (state.queued, state.running, state.cold_start_seconds) == (0, 0, None)


def test_queue_state_totals() -> None:
    state = QueueState({0: 1, 1: 0, 2: 0, 3: 2, 4: 3}, {"b1": 2, "b2": 1}, {"b1": 4, "b2": 2})
    assert (state.queued, state.running) == (6, 3)


def test_the_toolbox_protocol_is_the_three_calls_of_the_design() -> None:
    names = {name for name in vars(toolbox.Toolbox) if not name.startswith("_")}
    assert names == {"submit_test", "cancel", "queue_state", "progress", "measurement_capacity", "put_attachment"}
    assert {o.value for o in CancelOutcome} == {"withdrawn", "finished"}


def test_the_notebook_writer_protocols_are_re_exported() -> None:
    from hyperloom_multibranch.contracts.notebook.interfaces import NodeHealth, TestRecordSink

    assert toolbox.TestRecordSink is TestRecordSink
    assert toolbox.NodeHealth is NodeHealth


def test_a_minimal_implementation_satisfies_the_protocol_shape() -> None:
    class Minimal:
        async def submit_test(self, request: TestRequest) -> str:
            return "r1"

        async def cancel(self, request_id: str) -> CancelOutcome:
            return CancelOutcome.FINISHED

        async def queue_state(self) -> QueueState:
            return QueueState()

        async def measurement_capacity(self) -> toolbox.MeasurementCapacity:
            return toolbox.MeasurementCapacity(8, 8)

    impl: toolbox.Toolbox = Minimal()
    request = TestRequest(TestKind.TEST, Purpose.EXPLORE, "b1", "c0")
    import asyncio

    assert asyncio.run(impl.submit_test(request)) == "r1"
    with pytest.raises(UnknownTestRequest):
        raise UnknownTestRequest("r9")
