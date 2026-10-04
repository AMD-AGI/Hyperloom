# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A phase exit's frozen budget follows the contract the session was stamped with."""

from __future__ import annotations

import json

import pytest
from jsonschema import validate

from hyperloom.inference_optimizer.breakdown.exporter import build
from hyperloom.inference_optimizer.breakdown.recorder.session_metadata import record_metadata_identity
from hyperloom.inference_optimizer.breakdown.workflow_contract import WORKFLOW_CONTRACT_V1, workflow_schema
from hyperloom.inference_optimizer.session.manifest import write_manifest
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.phases.machine_state import compute_next_phase, record_phase_transition
from hyperloom.orchestrator.phases.session_contract import bound_session_declares
from hyperloom.orchestrator.state.shared_state import SharedState


def _stamp_session(tmp_path, *, legacy: bool) -> str:
    manifest = write_manifest(tmp_path, session_id="session-budget")
    if not legacy:
        return manifest["workflow_contract_version"]
    unstamped = {
        key: value
        for key, value in manifest.items()
        if key not in {"workflow_contract_version", "workflow_contract_digest"}
    }
    (tmp_path / "manifest.json").write_text(json.dumps(unstamped), encoding="utf-8")
    record_metadata_identity(tmp_path, unstamped)
    return WORKFLOW_CONTRACT_V1


@pytest.mark.parametrize("phase", ["FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP"])
@pytest.mark.parametrize("legacy", [True, False], ids=["v1", "v2"])
def test_phase_exit_budget_exports_under_session_contract(tmp_path, phase: str, legacy: bool):
    """A v1 session resumed on v2 code exports a phase exit its own schema accepts."""
    version = _stamp_session(tmp_path, legacy=legacy)
    state = SharedState(session_id="session-budget", phase=phase, baseline_tput=100.0)
    with session_scope(tmp_path):
        record_phase_transition(state, to_phase=phase, reason="phase_entered")
        state.set_stop_reason("target_reached")
        target, reason, evidence = compute_next_phase(state)
        assert (target, reason) == ("CLOSE", "target_reached")
        budget = evidence["predicate_inputs"]["budget"]
        assert ("current_balance" in budget) is not legacy
        if not legacy:
            assert budget["current_balance"] == budget["remaining_sec"]
        record_phase_transition(state, to_phase=target, reason=reason, evidence=evidence)
    state.save(tmp_path)

    exported = build(tmp_path)
    assert exported["metadata"]["workflow"]["workflow_contract_version"] == version
    exits = [
        segment["exit_evidence"]
        for event in exported["timeline"]
        if event.get("type") == "phase"
        for segment in event["ext"]["segments"]
        if segment.get("exit_reason") == "target_reached"
    ]
    assert [row["predicate_inputs"]["budget"] for row in exits] == [budget]
    validate(instance=exported, schema=workflow_schema(version))


@pytest.mark.parametrize(
    ("manifest_text", "declared"),
    [
        (None, True),
        (json.dumps({"workflow_contract_version": "hyperloom.workflow_evaluation.v2"}), True),
        (json.dumps({}), False),
        (json.dumps({"workflow_contract_version": "hyperloom.workflow_evaluation.v9"}), False),
        ("{not json", False),
        (json.dumps(["not", "a", "manifest"]), False),
    ],
    ids=["no-manifest", "v2", "unstamped", "unknown-version", "corrupt", "not-a-mapping"],
)
def test_bound_session_declares_follows_manifest_identity(tmp_path, manifest_text, declared: bool):
    if manifest_text is not None:
        (tmp_path / "manifest.json").write_text(manifest_text, encoding="utf-8")
    with session_scope(tmp_path):
        assert bound_session_declares("phase_budget", "current_balance") is declared
    assert bound_session_declares("phase_budget", "current_balance") is True
