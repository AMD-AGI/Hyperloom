# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Executable workflow contract coverage over real recorder/exporter output."""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from jsonschema import ValidationError, validate

from hyperloom.inference_optimizer.breakdown.exporter import build, patch_breakdown_close, write_breakdown_json
from hyperloom.inference_optimizer.breakdown.session_package import package_session_artifacts
from hyperloom.inference_optimizer.breakdown.recorder import phase_event, record_stage_reached, stack_event
from hyperloom.inference_optimizer.breakdown.recorder.close_out import (
    record_close_opened,
    record_close_safety_net,
    record_close_settled,
    record_close_step,
)
from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import ROUTE_FORGE, make_kernel_recorder
from hyperloom.inference_optimizer.breakdown.recorder.session_metadata import record_metadata_identity
from hyperloom.inference_optimizer.breakdown.recorder.warm_start_event import MATCH_MISS, make_warm_start_recorder
from hyperloom.inference_optimizer.breakdown.workflow_contract import (
    WORKFLOW_CONTRACT_V1,
    workflow_contract,
    workflow_contract_digest,
    workflow_schema,
)
from hyperloom.inference_optimizer.session.manifest import write_manifest
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.phases.machine_state import record_phase_transition, workflow_predicate_inputs
from hyperloom.orchestrator.state.shared_state import SharedState


def _writer_fixture(tmp_path, *, denied: bool) -> dict:
    """Produce a schema fixture through the manifest, recorder, and exporter."""
    session_id = "workflow-contract-fixture"
    write_manifest(tmp_path, session_id=session_id)
    state = SharedState(
        session_id=session_id,
        phase="CLOSE",
        baseline_tput=0.0 if denied else 100.0,
    )
    state.set_stop_reason("baseline_failed" if denied else "target_reached")
    state.save(tmp_path)
    predicate_inputs = workflow_predicate_inputs(
        SimpleNamespace(
            phase="PRELUDE",
            macro_cycle=0,
            baseline_tput=100.0,
            warm_replay_outcome={},
        ),
        budget_pct=None,
        kernel_enabled=True,
        optimize_enabled=True,
        enablement_enabled=True,
    )
    with session_scope(tmp_path):
        phase_event.record_entry(
            phase="PRELUDE",
            macro_cycle=0,
            sequence=1,
            reason="session_start",
            entered_unix=1.0,
        )
        phase_event.record_dispatch(
            action="baseline",
            task_id="task-1",
            phase="PRELUDE",
            macro_cycle=0,
            dispatch_class="llm",
            dispatched_unix=2.0,
        )
        phase_event.record_settle(
            task_id="task-1",
            status="failed" if denied else "succeeded",
            decision="rejected" if denied else "accepted",
            settled_unix=3.0,
        )
        if denied:
            phase_event.record_denial(
                actor="framework-agent",
                proposal_msg_id=None,
                action="profile",
                phase="PRELUDE",
                macro_cycle=0,
                rule="phase_action_not_allowed",
                hint="use an allowed PRELUDE action",
            )
        phase_event.record_exit(
            phase="PRELUDE",
            macro_cycle=0,
            to_phase="CLOSE",
            reason="baseline_failed" if denied else "target_reached",
            evidence={"predicate_inputs": predicate_inputs},
            exited_unix=4.0,
        )
    record_stage_reached(tmp_path, "close" if not denied else "prelude")
    if denied:
        record_close_safety_net(tmp_path)
    else:
        record_close_opened(tmp_path)
        record_close_step(tmp_path, step="report", status="done", optional=False)
        record_close_settled(tmp_path, stop_reason="target_reached")
    return build(tmp_path)


@pytest.mark.parametrize("denied", [False, True], ids=["success", "fail-fast"])
def test_real_writer_fixtures_cover_declared_consumer_paths(tmp_path, denied: bool):
    fixture = _writer_fixture(tmp_path, denied=denied)

    validate(instance=fixture, schema=workflow_schema())
    workflow = fixture["metadata"]["workflow"]
    assert workflow["workflow_contract_version"] == "hyperloom.workflow_evaluation.v2"
    assert workflow["contract_digest"] == workflow_contract_digest()
    assert fixture["outcome"]["stop_reason"] == ("baseline_failed" if denied else "target_reached")

    event = next(row for row in fixture["timeline"] if row["type"] == "phase")
    assert set(("process_status", "business_outcome", "failure", "blocked_by")) <= event.keys()
    assert event["ext"]["segments"][0]["exit_evidence"]["predicate_inputs"]["macro_cycle"] == 0
    action = event["ext"]["actions"]["rows"][0]
    assert (action["dispatch_class"], action["allowed"], action["denial_rule"]) == ("llm", True, None)
    if denied:
        denial = event["ext"]["denials"]["rows"][0]
        assert denial["proposal_msg_id"] is None
        assert denial["rule"] == "phase_action_not_allowed"


def _declared_consumer_values(fixture: dict) -> dict[str, object]:
    workflow = fixture["metadata"]["workflow"]
    phase = next(row for row in fixture["timeline"] if row["type"] == "phase")
    actions = phase["ext"]["actions"]["rows"]
    denials = phase["ext"]["denials"]["rows"]
    return {
        "metadata.workflow.workflow_contract_version": workflow["workflow_contract_version"],
        "metadata.workflow.contract_digest": workflow["contract_digest"],
        "metadata.workflow.run_flags": workflow["run_flags"],
        "metadata.workflow.phase_actions": workflow["phase_actions"],
        "metadata.workflow.llm_proposable_actions": workflow["llm_proposable_actions"],
        "metadata.workflow.coordinator_internal_actions": workflow["coordinator_internal_actions"],
        "metadata.workflow.coordinator_reserved_actions": workflow["coordinator_reserved_actions"],
        "metadata.workflow.kernel_lane_task_kinds": workflow["kernel_lane_task_kinds"],
        "timeline[].process_status": [row["process_status"] for row in fixture["timeline"]],
        "timeline[].business_outcome": [row["business_outcome"] for row in fixture["timeline"]],
        "timeline[].failure": [row["failure"] for row in fixture["timeline"]],
        "timeline[].blocked_by": [row["blocked_by"] for row in fixture["timeline"]],
        "outcome.stop_reason": fixture["outcome"]["stop_reason"],
        "outcome.stage_reached_recorded": fixture["outcome"].get("stage_reached_recorded"),
        "outcome.derived_from_event_ids": fixture["outcome"].get("derived_from_event_ids"),
        "timeline[type=phase].ext.segments[].exit_evidence.predicate_inputs.budget.current_balance": [
            ((row.get("exit_evidence") or {}).get("predicate_inputs", {}).get("budget") or {}).get("current_balance")
            for row in phase["ext"]["segments"]
            if row.get("exited_at")
        ],
        "timeline[type=stack].ext.adoptions.rows[].stack_entry_id": [
            row.get("stack_entry_id")
            for event in fixture["timeline"]
            if event["type"] == "stack"
            for row in event["ext"]["adoptions"]["rows"]
        ],
        "timeline[type=stack].ext.adoptions.rows[].integration_id": [
            row.get("integration_id")
            for event in fixture["timeline"]
            if event["type"] == "stack"
            for row in event["ext"]["adoptions"]["rows"]
        ],
        "close.sequence_schema_version": fixture["close"].get("sequence_schema_version"),
        "close.source": fixture["close"].get("source"),
        "close.steps[].optional": [row.get("optional") for row in fixture["close"]["steps"]],
        "close.steps[].artifact_path": [row.get("artifact_path") for row in fixture["close"]["steps"]],
        "close.steps[].artifact_digest": [row.get("artifact_digest") for row in fixture["close"]["steps"]],
        "close.steps[].error": [row.get("error") for row in fixture["close"]["steps"]],
        "timeline[type=phase].ext.segments[].exit_evidence.predicate_inputs": [
            row["exit_evidence"]["predicate_inputs"] for row in phase["ext"]["segments"] if row.get("exited_at")
        ],
        "timeline[type=phase].ext.actions.rows[].dispatch_class": [row["dispatch_class"] for row in actions],
        "timeline[type=phase].ext.actions.rows[].allowed": [row["allowed"] for row in actions],
        "timeline[type=phase].ext.actions.rows[].denial_rule": [row["denial_rule"] for row in actions],
        "timeline[type=phase].ext.denials.rows[].actor": [row["actor"] for row in denials],
        "timeline[type=phase].ext.denials.rows[].proposal_msg_id": [row["proposal_msg_id"] for row in denials],
        "timeline[type=phase].ext.denials.rows[].action": [row["action"] for row in denials],
        "timeline[type=phase].ext.denials.rows[].phase": [row["phase"] for row in denials],
        "timeline[type=phase].ext.denials.rows[].rule": [row["rule"] for row in denials],
        "timeline[type=phase].ext.denials.rows[].hint": [row["hint"] for row in denials],
    }


def test_declared_consumer_paths_are_mechanically_covered(tmp_path):
    fixture = _writer_fixture(tmp_path, denied=True)

    values = _declared_consumer_values(fixture)
    assert set(values) == set(workflow_contract()["consumer_paths"])
    assert all(value is not None for value in values.values())


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_manifest_keeps_v1_identity_on_export(tmp_path, explicit: bool):
    manifest = write_manifest(tmp_path, session_id="legacy-workflow")
    legacy = {
        key: value
        for key, value in manifest.items()
        if key not in {"workflow_contract_version", "workflow_contract_digest"}
    }
    if explicit:
        legacy["workflow_contract_version"] = WORKFLOW_CONTRACT_V1
        legacy["workflow_contract_digest"] = workflow_contract_digest(WORKFLOW_CONTRACT_V1)
    record_metadata_identity(tmp_path, legacy)
    workflow = build(tmp_path)["metadata"]["workflow"]
    assert workflow["workflow_contract_version"] == WORKFLOW_CONTRACT_V1
    assert workflow["contract_digest"] == workflow_contract_digest(WORKFLOW_CONTRACT_V1)
    assert "stage_reached_recorded" not in build(tmp_path)["outcome"]
    validate(instance=build(tmp_path), schema=workflow_schema(WORKFLOW_CONTRACT_V1))

    state = SharedState(session_id="legacy-workflow")
    with session_scope(tmp_path):
        record_phase_transition(state, to_phase="PRELUDE", reason="phase_entered")
    state.save(tmp_path)
    refreshed = build(tmp_path)
    assert refreshed["metadata"]["workflow"]["workflow_contract_version"] == WORKFLOW_CONTRACT_V1
    assert refreshed["outcome"]["stage_reached_recorded"] == "prelude"
    validate(instance=refreshed, schema=workflow_schema(WORKFLOW_CONTRACT_V1))


def test_authored_milestones_survive_sbd_export(tmp_path):
    write_manifest(tmp_path, session_id="v2-milestones")
    state = SharedState(session_id="v2-milestones")
    with session_scope(tmp_path):
        record_phase_transition(state, to_phase="PRELUDE", reason="phase_entered")
        warm_start = make_warm_start_recorder(requested_canonical_id="model-1", scope={})
        assert warm_start is not None
        warm_start.finish(match_status=MATCH_MISS)
        assert build(tmp_path)["outcome"]["stage_reached_recorded"] == "warm_start"

        assert phase_event.record_dispatch(action="profile", task_id="profile-1", phase="PRELUDE", macro_cycle=0)
        assert build(tmp_path)["outcome"]["stage_reached_recorded"] == "profile"

        record_phase_transition(state, to_phase="KERNEL_AGENT", reason="phase_entered")
        kernel = make_kernel_recorder(macro_cycle=0, route=ROUTE_FORGE)
        assert kernel is not None
        kernel.begin(tput_before=100.0)
        assert build(tmp_path)["outcome"]["stage_reached_recorded"] == "kernel"


def test_v2_budget_and_stack_ids_come_from_writers(tmp_path):
    write_manifest(tmp_path, session_id="v2-evidence")
    state = SharedState(session_id="v2-evidence", phase="FRAMEWORK_AGENT", baseline_tput=100.0)
    state.set_stop_reason("target_reached")
    state.save(tmp_path)
    inputs = workflow_predicate_inputs(state, now_unix=1.0)
    with session_scope(tmp_path):
        phase_event.record_entry(phase="FRAMEWORK_AGENT", macro_cycle=0, sequence=1, entered_unix=1.0)
        phase_event.record_exit(
            phase="FRAMEWORK_AGENT",
            macro_cycle=0,
            to_phase="CLOSE",
            reason="target_reached",
            evidence={"predicate_inputs": inputs},
            exited_unix=2.0,
        )
        stack_event.record_adoption(
            stack_index=0,
            entry={"action": "integrate", "stack_entry_id": "entry-1", "integration_id": "int-1"},
            throughput_before=100.0,
            throughput_after=110.0,
            baseline_tput=100.0,
        )
    record_stage_reached(tmp_path, "close")
    record_close_opened(tmp_path)
    record_close_settled(tmp_path, stop_reason="target_reached")

    fixture = build(tmp_path)
    validate(instance=fixture, schema=workflow_schema())
    phase = next(event for event in fixture["timeline"] if event["type"] == "phase")
    budget = phase["ext"]["segments"][0]["exit_evidence"]["predicate_inputs"]["budget"]
    assert budget["current_balance"] == budget["remaining_sec"]
    stack = next(event for event in fixture["timeline"] if event["type"] == "stack")
    adoption = stack["ext"]["adoptions"]["rows"][0]
    assert (adoption["stack_entry_id"], adoption["integration_id"]) == ("entry-1", "int-1")
    assert fixture["outcome"]["derived_from_event_ids"]["validation"] == [stack["id"]]
    del adoption["stack_entry_id"]
    with pytest.raises(ValidationError):
        validate(instance=fixture, schema=workflow_schema())


def test_final_close_refresh_matches_rebuilt_package(tmp_path):
    write_manifest(tmp_path, session_id="v2-close")
    state = SharedState(session_id="v2-close", phase="CLOSE", baseline_tput=100.0)
    state.set_stop_reason("target_reached")
    state.save(tmp_path)
    record_close_opened(tmp_path)
    record_close_step(tmp_path, step="session_breakdown", status="done", optional=False)
    sbd_path = write_breakdown_json(tmp_path)
    package = package_session_artifacts(tmp_path, session_id="v2-close", dest_root=tmp_path / "workspace")
    assert package is not None
    record_close_step(tmp_path, step="artifact_package", status="done", optional=False, artifact_path=str(package))
    record_close_step(tmp_path, step="done", status="done", optional=False)
    record_close_settled(tmp_path, stop_reason="target_reached")
    record_stage_reached(tmp_path, "close")
    assert patch_breakdown_close(tmp_path)
    rebuilt = package_session_artifacts(tmp_path, session_id="v2-close", dest_root=tmp_path / "workspace")
    assert rebuilt == package

    on_disk = json.loads(sbd_path.read_text(encoding="utf-8"))
    validate(instance=on_disk, schema=workflow_schema())
    assert on_disk["close"]["close_sequence_done"] is True
    assert on_disk["outcome"]["stage_reached_recorded"] == "close"
    with ZipFile(package) as archive:
        bundled = json.loads(archive.read("session_breakdown.json"))
    assert bundled["close"] == on_disk["close"]
    assert bundled["outcome"] == on_disk["outcome"]
    del on_disk["close"]["steps"][0]["optional"]
    with pytest.raises(ValidationError):
        validate(instance=on_disk, schema=workflow_schema())


@pytest.mark.parametrize(
    "mutation",
    [
        "allowlist",
        "event_type",
        "status_mismatch",
        "action",
        "phase",
        "stop_reason",
        "missing_outcome",
        "missing_stop_reason",
        "missing_phase_ext",
        "missing_segment_identity",
        "missing_denied_at",
        "failure_shape",
        "extra_phase_ext",
    ],
)
def test_schema_rejects_contract_drift(tmp_path, mutation: str):
    fixture = deepcopy(_writer_fixture(tmp_path, denied=mutation == "missing_denied_at"))
    event = next(row for row in fixture["timeline"] if row["type"] == "phase")
    if mutation == "allowlist":
        fixture["metadata"]["workflow"]["phase_actions"]["PRELUDE"].append("undeclared")
    elif mutation == "event_type":
        event["type"] = "undeclared"
    elif mutation == "status_mismatch":
        event["process_status"] = "failed"
    elif mutation == "action":
        event["ext"]["actions"]["rows"][0]["action"] = "undeclared"
    elif mutation == "phase":
        event["ext"]["actions"]["rows"][0]["phase"] = "UNDECLARED"
    elif mutation == "stop_reason":
        fixture["outcome"]["stop_reason"] = "undeclared"
    elif mutation == "missing_outcome":
        del fixture["outcome"]
    elif mutation == "missing_stop_reason":
        del fixture["outcome"]["stop_reason"]
    elif mutation == "missing_phase_ext":
        del event["ext"]["phase"]
    elif mutation == "missing_segment_identity":
        del event["ext"]["segments"][0]["sequence"]
    elif mutation == "missing_denied_at":
        del event["ext"]["denials"]["rows"][0]["denied_at"]
    elif mutation == "failure_shape":
        event["failure"] = {"unknown": "shape"}
    else:
        event["ext"]["extra"] = True

    with pytest.raises(ValidationError):
        validate(instance=fixture, schema=workflow_schema())


def test_unknown_contract_digest_fails_schema_validation(tmp_path):
    fixture = deepcopy(_writer_fixture(tmp_path, denied=False))
    fixture["metadata"]["workflow"]["contract_digest"] = "0" * 64

    with pytest.raises(ValidationError):
        validate(instance=fixture, schema=workflow_schema())


def test_contract_vocabularies_match_runtime_surfaces():
    from hyperloom.inference_optimizer.breakdown.stop_reasons import STOP_REASON_VOCAB
    from hyperloom.inference_optimizer.session.sbd_v6 import _EVENT_TYPES
    from hyperloom.orchestrator.phases.machine_state import (
        PHASE_ALLOWED_ACTIONS,
        PHASE_COORDINATOR_RESERVED,
        PHASE_KERNEL_AGENT,
        allowed_actions_for,
    )
    from hyperloom.inference_optimizer.protocol.action_surfaces import COORDINATOR_INTERNAL_ACTIONS

    contract = workflow_contract()
    assert contract["vocabularies"]["event_types"] == list(_EVENT_TYPES)
    assert set(contract["vocabularies"]["stop_reasons"]) == set(STOP_REASON_VOCAB)
    assert set(workflow_schema()["properties"]["outcome"]["properties"]["stop_reason"]["enum"]) == set(
        STOP_REASON_VOCAB
    ) | {""}
    assert {phase: set(actions) for phase, actions in contract["phase_actions"].items()} == {
        phase: set(actions) for phase, actions in PHASE_ALLOWED_ACTIONS.items()
    }
    assert {phase: set(actions) for phase, actions in contract["llm_proposable_actions"].items()} == {
        phase: set(allowed_actions_for(phase)) for phase in PHASE_ALLOWED_ACTIONS
    }
    assert set(contract["coordinator_internal_actions"]) == set(COORDINATOR_INTERNAL_ACTIONS)
    assert {phase: set(actions) for phase, actions in contract["coordinator_reserved_actions"].items()} == {
        phase: set(actions) for phase, actions in PHASE_COORDINATOR_RESERVED.items()
    }
    assert set(contract["kernel_lane_task_kinds"]) == set(PHASE_ALLOWED_ACTIONS[PHASE_KERNEL_AGENT])


def test_process_status_and_business_outcome_remain_separate():
    from hyperloom.inference_optimizer.breakdown.recorder.event_timeline import build_envelope

    rejected = build_envelope(
        event_type="warm_replay",
        event="prelude:0:warm_replay_1",
        status="rejected",
        ext={"verdict": {"outcome_status": "quality_failed"}},
    )
    no_gain = build_envelope(
        event_type="kernel",
        event="kernel_agent:0:kernel_1",
        status="succeeded",
        ext={"outcome": {"verdict": "no_improvement"}},
    )

    assert (rejected["process_status"], rejected["business_outcome"]) == ("rejected", "quality_failed")
    assert (no_gain["process_status"], no_gain["business_outcome"]) == ("succeeded", "no_improvement")
    assert rejected["failure"] is None
    assert no_gain["failure"] is None


def test_malformed_failure_is_preserved_for_schema_rejection(tmp_path):
    from hyperloom.inference_optimizer.breakdown.recorder.event_timeline import build_envelope

    fixture = _writer_fixture(tmp_path, denied=False)
    event = build_envelope(
        event_type="roofline",
        event="prelude:0:roofline",
        status="failed",
        ext={"failure": "boom"},
    )
    fixture["timeline"].append(event)

    assert event["failure"] == "boom"
    with pytest.raises(ValidationError):
        validate(instance=fixture, schema=workflow_schema())


def test_unknown_business_outcome_is_preserved_for_schema_rejection():
    from hyperloom.inference_optimizer.breakdown.recorder.event_timeline import build_envelope

    event = build_envelope(
        event_type="kernel",
        event="kernel_agent:0:kernel_1",
        status="succeeded",
        ext={"outcome": {"verdict": "undeclared_new_state"}},
    )

    assert event["business_outcome"] == "undeclared_new_state"
    schema = workflow_schema()
    event_schema = {
        chr(36) + "schema": schema[chr(36) + "schema"],
        chr(36) + "defs": schema[chr(36) + "defs"],
        **schema["properties"]["timeline"]["items"],
    }
    with pytest.raises(ValidationError):
        validate(instance=event, schema=event_schema)
