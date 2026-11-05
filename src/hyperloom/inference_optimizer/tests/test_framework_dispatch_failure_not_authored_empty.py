# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist that never ran is not a specialist that found nothing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.phases.exit_predicates import framework_plateau

from hyperloom.orchestrator.lever import LEVER_SOURCE_PATCH, LEVER_UPSTREAM_PR
from hyperloom.orchestrator.phases.machine_state import (
    DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK,
    _lever_attempts,
    _patch_lever_evidence,
    _trailing_no_keep,
)

from .test_framework_agent_authoring import _stub


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

    stub = _stub(tmp_path)

    stub.phase_framework._record_framework_agent_authoring_empty_outcome(
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

    stub = _stub(tmp_path)

    stub.phase_framework._record_framework_agent_authoring_empty_outcome(
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

    stub = _stub(tmp_path)

    stub.phase_framework._record_framework_agent_authoring_empty_outcome(
        task=_task("local_explore:2"),
        # What a replayed delegated_result carries: no specialist_done at all.
        done_payload={},
        run_error=_GATE_ERROR,
    )

    assert stub.shared_state.framework_agent_phase_progress[0]["status"] == "dispatch_failed"


@pytest.mark.parametrize(
    ("done_payload", "run_error", "outcome"),
    [
        ({}, _GATE_ERROR, "dispatch_failed"),
        ({"patches_written": [], "summary": "nothing left to remove"}, "", "author_empty"),
    ],
)
def test_an_undelivered_round_counts_once_against_the_source_lever(
    tmp_path: Path, done_payload: dict, run_error: str, outcome: str
):
    """A lane that cannot run, like one that finds nothing, must be able to run the source arm dry."""

    stub = _stub(tmp_path)
    task = _task("local_explore:3")

    for _ in range(2):  # the bus-replay path re-delivers the same settlement
        stub.phase_framework._record_framework_agent_authoring_empty_outcome(
            task=task, done_payload=done_payload, run_error=run_error
        )

    rows = _lever_attempts(stub.shared_state, LEVER_SOURCE_PATCH)
    assert [(r["outcome"], r["adopted"], r["candidate_id"]) for r in rows] == [(outcome, False, "local_explore:3")]


def test_an_empty_upstream_authoring_round_counts_against_the_upstream_lever(tmp_path: Path):
    stub = _stub(tmp_path)
    task = _task("https://github.com/org/repo/pull/7")
    task.params["lever_kind"] = LEVER_UPSTREAM_PR

    stub.phase_framework._record_framework_agent_authoring_empty_outcome(task=task, done_payload={"summary": "none"})

    assert [r["outcome"] for r in _lever_attempts(stub.shared_state, LEVER_UPSTREAM_PR)] == ["author_empty"]
    assert _lever_attempts(stub.shared_state, LEVER_SOURCE_PATCH) == []


def test_empty_rounds_alone_run_the_source_arm_dry(tmp_path: Path):
    stub = _stub(tmp_path)
    for i in range(DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK):
        stub.phase_framework._record_framework_agent_authoring_empty_outcome(
            task=_task(f"local_explore:{i}"), done_payload={}, run_error=_GATE_ERROR
        )

    evidence = _patch_lever_evidence(stub.shared_state, {})
    assert framework_plateau(evidence).source_dry is True
    assert evidence["source_consecutive_no_keep"] == DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK


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
