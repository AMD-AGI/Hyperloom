# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The integrate gate's verdict against the real production writer.

``test_sbd_v6_kernel_timeline.py`` pins what the recorder does with a verdict;
these pin that the orchestrator actually hands it one. The seam is
``SharedState.record_kernel_integrate_result``, which every one of the three
settle sites funnels through -- so a verdict that stops reaching the timeline
stops reaching it everywhere at once, and that is what these tests would
catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import (
    ROUTE_FORGE,
    make_kernel_recorder,
)
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _integrate_rows(session_dir: Path) -> list[dict[str, Any]]:
    events = [event for event in read_timeline_events(session_dir) if event.get("type") == "kernel"]
    return events[0]["ext"]["integrate"] if events else []


def _visited_kernel(*, macro_cycle: int) -> None:
    """Run one KERNEL visit that produced a rewrite, and close it."""
    recorder = make_kernel_recorder(macro_cycle=macro_cycle, route=ROUTE_FORGE)
    assert recorder is not None
    recorder.begin(tput_before=1000.0)
    recorder.record_kernel_rewrite(run_id="attempt-1", kernel_id="k001", status="success", micro_decision="keep")
    recorder.finish(verdict="needs_review", status="succeeded", tput_after=1000.0)


def _state(*, macro_cycle: int) -> SharedState:
    state = SharedState()
    state.macro_cycle = macro_cycle
    return state


def _result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "ok",
        "decision": "KEEP",
        "kernel_id": "k001",
        "integration_id": "int-1",
        "patch_path": "/tmp/k001_opt.py",
        "target_file": "vllm/attention.py",
        "gain_pct": 5.5,
        "accuracy_pass": True,
    }
    result.update(overrides)
    return result


def test_a_settled_keep_reaches_the_kernel_event(tmp_path: Path) -> None:
    """The gate runs after the visit closed, and the verdict still lands."""
    _visited_kernel(macro_cycle=2)

    _state(macro_cycle=2).record_kernel_integrate_result(_result())

    rows = _integrate_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["kernel_id"] == "k001"
    assert rows[0]["decision"] == "KEEP"
    assert rows[0]["gain_pct"] == 5.5
    assert rows[0]["accuracy_pass"] is True
    assert rows[0]["settled_in_macro_cycle"] == 2


def test_the_verdict_also_reaches_the_rewrite_row_it_ruled_on(tmp_path: Path) -> None:
    """The lane row's e2e block is the projection of these rows, not a second copy."""
    _visited_kernel(macro_cycle=2)

    _state(macro_cycle=2).record_kernel_integrate_result(_result())

    events = [event for event in read_timeline_events(tmp_path) if event.get("type") == "kernel"]
    e2e = events[0]["ext"]["forge"]["lanes"]["kernel_rewrites"][0]["e2e"]
    assert e2e["integrated"] is True
    assert e2e["e2e_gain_pct"] == 5.5
    assert e2e["target_file"] == "vllm/attention.py"


def test_the_adoptions_server_args_reach_the_verdict_that_carried_them(tmp_path: Path) -> None:
    """The stack entry an adoption introduces is part of what the gate kept."""
    _visited_kernel(macro_cycle=2)

    _state(macro_cycle=2).record_kernel_integrate_result(
        _result(extra_server_args="--enable-chunked-prefill")
    )

    assert _integrate_rows(tmp_path)[0]["extra_server_args"] == "--enable-chunked-prefill"


def test_a_gain_the_queue_did_not_qualify_is_taken_as_this_kernels_own(tmp_path: Path) -> None:
    """The integrate queue measures one patch at a time, so its gain is pinned."""
    _visited_kernel(macro_cycle=2)

    _state(macro_cycle=2).record_kernel_integrate_result(_result())

    assert _integrate_rows(tmp_path)[0]["gain_attributed"] is True


def test_a_revert_is_recorded_as_a_verdict_not_as_a_dropped_patch(tmp_path: Path) -> None:
    """``decision`` rules on the patch; ``rejected_reason`` is an exhausted budget."""
    _visited_kernel(macro_cycle=2)

    _state(macro_cycle=2).record_kernel_integrate_result(_result(decision="REVERT", gain_pct=0.2))

    rows = _integrate_rows(tmp_path)
    assert rows[0]["decision"] == "REVERT"
    assert rows[0]["rejected_reason"] == "revert_decision"
    assert rows[0]["retryable"] is False


def test_an_integration_fault_is_counted_apart_and_stays_retryable(tmp_path: Path) -> None:
    """A fault never measured the patch, so it is not a verdict against it."""
    _visited_kernel(macro_cycle=2)

    _state(macro_cycle=2).record_kernel_integrate_result(
        _result(decision=None, status="failed", error_class="integration_failed"),
    )

    rows = _integrate_rows(tmp_path)
    assert rows[0]["fault_count"] == 1
    assert rows[0]["retryable"] is True
    assert rows[0]["rejected_reason"] is None
    assert rows[0]["error_class"] == "integration_failed"
