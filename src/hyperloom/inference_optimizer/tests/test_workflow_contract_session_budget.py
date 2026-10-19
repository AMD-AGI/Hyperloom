# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A phase exit's frozen budget validates against the workflow contract."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
from jsonschema import validate

from hyperloom.inference_optimizer.breakdown.exporter import build
from hyperloom.inference_optimizer.breakdown.workflow_contract import workflow_schema
from hyperloom.inference_optimizer.session.manifest import write_manifest
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.phases.machine_state import compute_next_phase, record_phase_transition
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.parametrize("phase", ["FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP"])
def test_phase_exit_budget_exports_under_contract(tmp_path, phase: str):
    write_manifest(tmp_path, session_id="session-budget")
    state = SharedState(session_id="session-budget", phase=phase, baseline_tput=100.0)
    with session_scope(tmp_path):
        record_phase_transition(state, to_phase=phase, reason="phase_entered")
        state.set_stop_reason("target_reached")
        target, reason, evidence = compute_next_phase(state)
        assert (target, reason) == ("CLOSE", "target_reached")
        budget = evidence["predicate_inputs"]["budget"]
        assert budget["current_balance"] == budget["remaining_sec"]
        record_phase_transition(state, to_phase=target, reason=reason, evidence=evidence)
    state.save(tmp_path)

    exported = build(tmp_path)
    exits = [
        segment["exit_evidence"]
        for event in exported["timeline"]
        if event.get("type") == "phase"
        for segment in event["ext"]["segments"]
        if segment.get("exit_reason") == "target_reached"
    ]
    assert [row["predicate_inputs"]["budget"] for row in exits] == [budget]
    validate(instance=exported, schema=workflow_schema())


def test_sweep_exit_with_live_grant_exports_under_contract(tmp_path):
    """A session-wide exit while SWEEP still waits on its grant freezes a widened budget the schema accepts."""
    write_manifest(tmp_path, session_id="session-budget")
    state = SharedState(
        session_id="session-budget",
        phase="SWEEP",
        baseline_tput=100.0,
        start_ts=(datetime.now(timezone.utc) - timedelta(minutes=175)).isoformat(),
        max_minutes=180,
        phase_budget_pct={"SWEEP": 0.05, "CLOSE": 0.02},
    )
    # Five minutes of session left, less the CLOSE reserve; the sweep's grant runs past the session deadline.
    grant_sec = 900.0
    with session_scope(tmp_path):
        record_phase_transition(state, to_phase="SWEEP", reason="phase_entered")
        state.conc_sweep_granted_until_unix = time.time() + grant_sec
        state.set_stop_reason("target_reached")
        target, reason, evidence = compute_next_phase(state)
        assert (target, reason) == ("CLOSE", "target_reached")
        budget = evidence["predicate_inputs"]["budget"]
        assert budget["remaining_sec"] > grant_sec - 60.0, "the grant did not widen the budget"
        assert budget["current_balance"] == budget["remaining_sec"]
        record_phase_transition(state, to_phase=target, reason=reason, evidence=evidence)
    state.save(tmp_path)

    validate(instance=build(tmp_path), schema=workflow_schema())
