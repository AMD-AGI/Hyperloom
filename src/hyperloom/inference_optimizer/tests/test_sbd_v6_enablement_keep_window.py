# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which KEEP rounds landed, and which are still awaiting revalidation, from the real coordinator.

Each window row names the KEEP it judges and states how it ended, so the counts follow from the
rows alone. These tests drive the lane, the writeback, the resume recovery and the reconcile pass
that end a window, rather than writing window rows by hand.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import enablement_event
from hyperloom.inference_optimizer.breakdown.recorder.assembler import event_parts
from hyperloom.inference_optimizer.breakdown.recorder.sections import ENABLEMENT_EVENT_SECTIONS
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult

from .conftest import make_coordinator


@pytest.fixture(autouse=True)
def _single_node(monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env as mne

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)


@pytest.fixture
def coord(tmp_path: Path):
    c = make_coordinator(tmp_path)
    state = c.shared_state
    state.phase = "ENABLEMENT"
    state.enablement.origin = "eval"
    state.enablement.accepted_config_path = "/s/accepted.yaml"
    yield c
    c.db.close()


def _counts() -> dict[str, Any]:
    ext, _status = enablement_event.assemble_enablement_ext(
        event_parts(ENABLEMENT_EVENT_SECTIONS), event=enablement_event.enablement_event_id()
    )
    attempts = ext["attempts"]
    return {
        "landed": attempts["landed"],
        "validation_pending": attempts["validation_pending"],
        "rows": {
            row["task_id"]: (row["landed"], row["validation_pending"])
            for row in attempts["rows"]
            if row["status"] == "kept"
        },
        "windows": [(row["generation"], row.get("verdict")) for row in ext["revalidations"]["rows"]],
    }


async def _keep(coord, task_id: str) -> str:
    """Settle one KEEP through the lane and open the window that judges it."""
    coord.shared_state.enablement.last_specialist_task_id = task_id
    await coord.enablement_lane.maybe_rearm_enablement(
        {"enablement": True, "status": "kept", "specialist_task_id": task_id}
    )
    window = await coord.enablement_revalidation.maybe_enqueue_enablement_baseline_revalidation()
    assert window
    return window


def _measured(tmp_path: Path, accuracy: float = 1.0) -> dict[str, Any]:
    return {
        "status": "succeeded",
        "reported_success": True,
        "valid_measurement": True,
        "submission_valid": True,
        "submission_invalid_reasons": [],
        "output_throughput": 804.35,
        "completed_requests": 3047,
        "duration_seconds": 3627.0,
        "request_error_rate": 0.0,
        "accuracy": accuracy,
        "accuracy_metric": "request_error_rate",
        "returncode": 0,
        "workspace": str(tmp_path / "runs" / "baseline" / "reval"),
    }


async def _finish_window(coord, window: str, tmp_path: Path, *, accuracy: float = 1.0, booked: bool = True) -> None:
    """End the window's baseline task, and let the dispatcher book its result when ``booked``."""
    await coord.tasks.transition(window, "running")
    await coord.tasks.transition(window, "succeeded")
    if booked:
        result = SubAgentResult(task_id=window, state="succeeded", result=_measured(tmp_path, accuracy))
        coord.dispatcher._completion_queue.append((await coord.tasks.get(window), result))
        await coord.dispatcher.pump_dispatcher_once()


async def _stop_window(coord, window: str) -> None:
    """The wall-clock budget stops the window's task, which the writeback routes."""
    await coord.tasks.transition(window, "running")
    await coord.tasks.transition(window, "cancelled")
    await coord.writeback._record_revalidation_not_promoted(
        task=await coord.tasks.get(window),
        result_payload={},
        err_class="",
        stopped_by_the_run=True,
    )


async def test_a_kept_round_lands_when_its_window_promotes(coord, tmp_path):
    window = await _keep(coord, "spec-1")
    await _finish_window(coord, window, tmp_path)

    assert _counts() == {
        "landed": 1,
        "validation_pending": 0,
        "rows": {"spec-1": (True, False)},
        "windows": [(1, "promoted")],
    }


async def test_a_kept_round_nobody_revalidated_is_pending(coord):
    await _keep(coord, "spec-1")

    assert _counts() == {
        "landed": 0,
        "validation_pending": 1,
        "rows": {"spec-1": (False, True)},
        "windows": [(1, None)],
    }


async def test_a_kept_round_below_the_accuracy_floor_is_rejected(coord, tmp_path):
    coord.shared_state.enablement.accuracy_floor = 0.9
    window = await _keep(coord, "spec-1")
    await _finish_window(coord, window, tmp_path, accuracy=0.2)

    assert _counts() == {
        "landed": 0,
        "validation_pending": 0,
        "rows": {"spec-1": (False, False)},
        "windows": [(1, "rejected")],
    }


async def test_a_window_the_budget_stopped_leaves_its_kept_round_pending(coord):
    window = await _keep(coord, "spec-1")
    await _stop_window(coord, window)

    assert _counts() == {
        "landed": 0,
        "validation_pending": 1,
        "rows": {"spec-1": (False, True)},
        "windows": [(1, "stopped")],
    }


async def test_a_stopped_window_reopened_and_promoted_lands_the_kept_round(coord, tmp_path):
    window = await _keep(coord, "spec-1")
    await _stop_window(coord, window)
    reopened = await coord.enablement_revalidation.maybe_enqueue_enablement_baseline_revalidation()
    assert reopened and reopened != window
    await _finish_window(coord, reopened, tmp_path)

    assert _counts() == {
        "landed": 1,
        "validation_pending": 0,
        "rows": {"spec-1": (True, False)},
        "windows": [(1, "stopped"), (2, "promoted")],
    }


async def test_a_window_cleared_by_the_resume_recovery_is_neither_landed_nor_pending(coord, tmp_path):
    window = await _keep(coord, "spec-1")
    await _finish_window(coord, window, tmp_path, booked=False)
    await coord.writeback._resume_recover_pending_revalidation({"fixes": []})

    assert _counts() == {
        "landed": 0,
        "validation_pending": 0,
        "rows": {"spec-1": (False, False)},
        "windows": [(1, "cleared")],
    }


async def test_a_window_cleared_by_the_reconcile_pass_is_neither_landed_nor_pending(coord, tmp_path):
    window = await _keep(coord, "spec-1")
    await _finish_window(coord, window, tmp_path, booked=False)
    report = await coord.reconciler.run(time.time())

    assert report.closed_windows == [window]
    assert _counts() == {
        "landed": 0,
        "validation_pending": 0,
        "rows": {"spec-1": (False, False)},
        "windows": [(1, "cleared")],
    }


async def test_a_cancelled_window_reopened_by_the_resume_recovery_stays_pending(coord):
    window = await _keep(coord, "spec-1")
    await coord.tasks.transition(window, "running")
    await coord.tasks.transition(window, "cancelled")
    await coord.writeback._resume_recover_pending_revalidation({"fixes": []})

    assert _counts() == {
        "landed": 0,
        "validation_pending": 1,
        "rows": {"spec-1": (False, True)},
        "windows": [(1, None)],
    }


async def test_a_rejected_keep_is_not_pending_once_a_later_keep_lands(coord, tmp_path):
    coord.shared_state.enablement.accuracy_floor = 0.9
    first = await _keep(coord, "spec-1")
    await _finish_window(coord, first, tmp_path, accuracy=0.2)
    coord.shared_state.enablement.accuracy_floor = 0.0
    second = await _keep(coord, "spec-2")
    await _stop_window(coord, second)
    reopened = await coord.enablement_revalidation.maybe_enqueue_enablement_baseline_revalidation()
    await _finish_window(coord, reopened, tmp_path)

    assert _counts() == {
        "landed": 1,
        "validation_pending": 0,
        "rows": {"spec-1": (False, False), "spec-2": (True, False)},
        "windows": [(1, "rejected"), (2, "stopped"), (3, "promoted")],
    }


async def test_a_superseded_keep_that_was_never_judged_is_not_pending(coord, tmp_path):
    first = await _keep(coord, "spec-1")
    await _finish_window(coord, first, tmp_path, booked=False)
    await coord.writeback._resume_recover_pending_revalidation({"fixes": []})
    await _keep(coord, "spec-2")

    assert _counts() == {
        "landed": 0,
        "validation_pending": 1,
        "rows": {"spec-1": (False, False), "spec-2": (False, True)},
        "windows": [(1, "cleared"), (2, None)],
    }


def test_rows_recorded_without_a_keep_reference_count_as_before():
    event = enablement_event.enablement_event_id()
    parts = {
        enablement_event.SECTION_ATTEMPT: [
            {"event_id": event, "attempt": 1, "task_id": "t1", "status": "kept", "landed": False},
            {"event_id": event, "attempt": 2, "task_id": "t2", "status": "reverted", "landed": False},
        ],
        enablement_event.SECTION_REVALIDATION: [
            {"event_id": event, "generation": 1, "closed_at": "x", "promoted": True}
        ],
    }
    ext, _status = enablement_event.assemble_enablement_ext(parts, event=event)

    assert {key: value for key, value in ext["attempts"].items() if key != "rows"} == {
        "count": 2,
        "settled": 2,
        "landed": 1,
        "advanced": 0,
    }
    assert [row["landed"] for row in ext["attempts"]["rows"]] == [False, False]


def test_the_real_vllm_dsv4pro_rows_pair_to_one_landed_keep():
    event = enablement_event.enablement_event_id()
    parts = {
        enablement_event.SECTION_EVENT: [{"event_id": event, "validating_keep": "t1"}],
        enablement_event.SECTION_ATTEMPT: [
            {"event_id": event, "attempt": 1, "task_id": "t1", "status": "kept", "landed": False},
        ],
        enablement_event.SECTION_REVALIDATION: [
            {"event_id": event, "generation": 1, "opened_at": "2026-10-12T14:25:22+00:00"},
            {
                "event_id": event,
                "generation": 2,
                "opened_at": "2026-10-12T17:38:33+00:00",
                "closed_at": "2026-10-12T19:20:41+00:00",
                "promoted": True,
                "keep_ref": "t1",
                "verdict": "promoted",
            },
        ],
    }
    ext, _status = enablement_event.assemble_enablement_ext(parts, event=event)

    assert (ext["attempts"]["landed"], ext["attempts"]["validation_pending"]) == (1, 0)
    assert [(r["landed"], r["validation_pending"]) for r in ext["attempts"]["rows"]] == [(True, False)]
