# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist that never ran is not a specialist that found nothing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hyperloom.orchestrator.lever import LEVER_SOURCE_PATCH
from hyperloom.orchestrator.phases.machine_state import (
    _lever_attempts,
    _trailing_no_keep,
)
from hyperloom.orchestrator.phases.framework import FrameworkPhase

from .test_framework_agent_authoring import _Stub


def _task(cand: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_id="spec-1",
        params={
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand,
            "framework_batch_id": "",
            "framework_audit": {},
        },
    )


_GATE_ERROR = (
    "role='orchestration' delegate payload field 'source_file'="
    "'hyvideo/models/transformers/modules/attention.py(206): sequence_parallel_attention_vision' "
    "is not under session_dir or a trusted installed source scope"
)


def test_dispatch_failure_is_not_recorded_as_authored_empty(tmp_path: Path):
    """A run that failed before delivering must not claim the specialist authored nothing."""

    stub = _Stub(tmp_path, authoring=True)

    FrameworkPhase._record_framework_agent_authoring_empty_outcome(  # type: ignore[arg-type]
        stub,
        task=_task("local_explore:0"),
        done_payload={},
        run_error=_GATE_ERROR,
    )

    rows = stub.shared_state.framework_agent_phase_progress
    assert len(rows) == 1, "the row must still exist, or the pump re-dispatches forever"
    assert rows[0]["status"] == "dispatch_failed"
    assert rows[0]["kept"] is False
    assert _GATE_ERROR[:40] in rows[0]["rationale"]


def test_genuine_empty_deliverable_is_still_authored_empty(tmp_path: Path):
    """A specialist that ran and found nothing keeps its existing status."""

    stub = _Stub(tmp_path, authoring=True)

    FrameworkPhase._record_framework_agent_authoring_empty_outcome(  # type: ignore[arg-type]
        stub,
        task=_task("local_explore:1"),
        done_payload={
            "patches_written": [],
            "proposal_set": [],
            "summary": "no host-side redundancy left in the rollout loop",
        },
    )

    rows = stub.shared_state.framework_agent_phase_progress
    assert rows[0]["status"] == "author_empty"


def test_recovery_path_also_separates_a_failed_run(tmp_path: Path):
    """The bus-replay path sees the error on the envelope, not in the result."""

    stub = _Stub(tmp_path, authoring=True)

    FrameworkPhase._record_framework_agent_authoring_empty_outcome(  # type: ignore[arg-type]
        stub,
        task=_task("local_explore:2"),
        # What a replayed delegated_result carries: no specialist_done at all.
        done_payload={},
        run_error=_GATE_ERROR,
    )

    assert stub.shared_state.framework_agent_phase_progress[0]["status"] == "dispatch_failed"


def test_dispatch_failure_leaves_no_attempt_for_the_plateau_to_count(tmp_path: Path):
    """The dispatch row settles on the progress ledger; the plateau reads attempts, and finds none."""

    stub = _Stub(tmp_path, authoring=True)

    FrameworkPhase._record_framework_agent_authoring_empty_outcome(  # type: ignore[arg-type]
        stub,
        task=_task("local_explore:3"),
        done_payload={},
        run_error=_GATE_ERROR,
    )

    assert stub.shared_state.framework_agent_phase_progress[0]["status"] == "dispatch_failed"
    assert _lever_attempts(stub.shared_state, LEVER_SOURCE_PATCH) == []


def test_real_outcomes_still_trip_the_plateau():
    """The behaviour the plateau exists for is unchanged."""
    state = SimpleNamespace(
        macro_cycle=0,
        attempts=[
            {"lever_kind": LEVER_SOURCE_PATCH, "outcome": "REVERT", "adopted": False},
            {"lever_kind": LEVER_SOURCE_PATCH, "outcome": "REVERT", "adopted": False},
            {"lever_kind": LEVER_SOURCE_PATCH, "outcome": "REVERT", "adopted": False},
        ],
    )
    assert _trailing_no_keep(_lever_attempts(state, LEVER_SOURCE_PATCH)) == 3
