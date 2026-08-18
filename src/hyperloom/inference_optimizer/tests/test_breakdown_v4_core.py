# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from types import SimpleNamespace

from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.recorder.assembler import (
    _deep_merge,
    _kb_writes_summary,
    _kernel_outcome,
    _merge_lists,
    assemble_parts,
    assemble_v4_parts,
)
from hyperloom.inference_optimizer.breakdown.recorder.instrument import (
    record_adoption,
    record_artifact,
    record_critic_iteration,
    record_measurement,
    record_operation,
    record_phase_event,
    record_phase_transition,
    record_geak_operation,
    record_gemm_tuning_operation,
    record_kernel_backend_result,
    record_kernel_dispatch,
    record_kernel_e2e,
    record_kernel_strategy_selection,
    record_run_snapshot,
    record_specialist_round,
    record_subject,
    record_tool_version,
    record_trace_event,
    snapshot_state_sections,
)
from hyperloom.inference_optimizer.breakdown.recorder.recorder import SECTION_SHAPES
from hyperloom.inference_optimizer.breakdown.schema import (
    ArtifactRef,
    Measurement,
    SCHEMA_VERSION_V5,
    Operation,
    SessionBreakdownV4,
)


def test_v4_schema_and_stream_registry_are_optional():
    assert not Operation.__required_keys__
    assert not Measurement.__required_keys__
    assert not ArtifactRef.__required_keys__
    assert not SessionBreakdownV4.__required_keys__
    assert {
        "root_operation_id",
        "macro_cycle",
        "source",
        "executor_class",
        "purpose",
        "scope",
        "strategy_group",
        "strategy",
        "measurement_refs",
        "artifact_refs",
        "adoption_refs",
        "extensions",
    } <= Operation.__annotations__.keys()
    assert {"metric_basis", "harness", "workload", "samples", "aggregation"} <= Measurement.__annotations__.keys()
    assert {
        "present",
        "producer_operation_id",
        "consumers",
        "coverage",
        "retention",
    } <= ArtifactRef.__annotations__.keys()
    assert SECTION_SHAPES["run_snapshot"] == "singleton"
    for name in (
        "phase_transitions",
        "subjects",
        "operations",
        "measurements",
        "adoptions",
        "artifacts",
        "trace_events",
    ):
        assert SECTION_SHAPES[name] == "item"


def test_v4_missing_streams_are_unavailable_and_poison_files_are_ignored(tmp_path, monkeypatch):
    poison = "must-not-enter-v4"
    (tmp_path / "state.json").write_text(json.dumps({"operations": [{"operation_id": poison}]}))
    (tmp_path / "manifest.json").write_text(json.dumps({"session_id": poison}))
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "result.json").write_text(json.dumps({"measurements": [{"measurement_id": poison}]}))
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "report.json").write_text(json.dumps({"artifacts": [{"artifact_id": poison}]}))

    class RejectCollectors:
        def __getattr__(self, name):
            raise AssertionError(f"collector access is forbidden: {name}")

    monkeypatch.setattr(exporter, "collectors", RejectCollectors())
    out = exporter.build_v4_live(tmp_path)

    assert out["schema_version"] == SCHEMA_VERSION_V5
    assert out["operations"] == []
    assert out["measurements"] == []
    assert out["artifacts"] == []
    assert out["run"] == {}
    assert out["integrity"]["status"] == "unavailable"
    for field in ("run", "operations", "measurements", "adoptions", "artifacts", "trace"):
        assert out["integrity"]["fields"][field]["status"] == "unavailable"
    assert poison not in json.dumps(
        {
            "run": out["run"],
            "operations": out["operations"],
            "measurements": out["measurements"],
            "artifacts": out["artifacts"],
        }
    )


def test_v4_operation_upsert_merges_state_and_nested_attempts(tmp_path):
    record_operation(
        tmp_path,
        operation_id="op-1",
        kind="benchmark",
        status="running",
        attempts=[{"attempt_id": "attempt-1", "status": "running"}],
    )
    record_operation(
        tmp_path,
        operation_id="op-1",
        status="succeeded",
        outputs={"throughput": 42.0},
        attempts=[
            {"attempt_id": "attempt-1", "status": "succeeded"},
            {"attempt_id": "attempt-2", "status": "skipped"},
        ],
    )

    operation = exporter.build_v4_live(tmp_path)["operations"][0]
    assert operation["operation_id"] == "op-1"
    assert operation["kind"] == "benchmark"
    assert operation["status"] == "succeeded"
    assert operation["outputs"] == {"throughput": 42.0}
    assert operation["attempts"] == [
        {"attempt_id": "attempt-1", "status": "succeeded"},
        {"attempt_id": "attempt-2", "status": "skipped"},
    ]


def test_v4_stable_entities_dedupe_and_preserve_updates(tmp_path):
    record_measurement(
        tmp_path,
        measurement_id="measurement-1",
        name="throughput",
        value=10.0,
    )
    record_measurement(
        tmp_path,
        measurement_id="measurement-1",
        value=12.0,
        unit="tok/s",
    )
    record_artifact(
        tmp_path,
        artifact_id="artifact-1",
        kind="report",
        path="reports/result.json",
    )
    record_artifact(
        tmp_path,
        artifact_id="artifact-1",
        digest="sha256:abc",
    )
    record_adoption(
        tmp_path,
        adoption_id="adoption-1",
        operation_id="op-1",
        status="pending",
    )
    record_adoption(
        tmp_path,
        adoption_id="adoption-1",
        status="adopted",
        validated=True,
    )

    out = exporter.build_v4_live(tmp_path)
    assert out["measurements"] == [
        {
            "measurement_id": "measurement-1",
            "name": "throughput",
            "unit": "tok/s",
            "value": 12.0,
        }
    ]
    assert out["artifacts"] == [
        {
            "artifact_id": "artifact-1",
            "digest": "sha256:abc",
            "kind": "report",
            "path": "reports/result.json",
        }
    ]
    assert out["adoptions"] == [
        {
            "adoption_id": "adoption-1",
            "operation_id": "op-1",
            "status": "adopted",
            "decision": "KEEP",
            "validated": True,
        }
    ]


def test_v4_canonical_streams_and_legacy_aliases_match(tmp_path):
    record_run_snapshot(
        tmp_path,
        {
            "run": {"session_id": "session-1"},
            "workload": {"model_name": "model-1"},
            "model": {"model_type": "test"},
            "versions": {"hyperloom": {"version": "test"}},
            "outcome": {"status": "succeeded"},
        },
    )
    record_phase_transition(
        tmp_path,
        {"transition_id": "transition-1", "from_phase": "PRELUDE", "phase": "EXPLORE"},
    )
    record_subject(tmp_path, subject_id="subject-1", subject_type="kernel")
    record_trace_event(tmp_path, event_id="event-1", kind="operation_started")

    out = exporter.build_v4_live(tmp_path)
    assert out["session"] == out["run"] == out["compat"]["session"]
    assert out["model_info"] == out["model"]
    assert out["phases"]["transitions"] == out["phase_timeline"]
    assert out["phases"]["transitions"] == out["phase_timeline"]
    assert out["trace"]["events"] == [{"event_id": "event-1", "kind": "operation_started"}]
    assert out["integrity"]["status"] == "partial"


def test_v4_integrity_is_exact_when_every_canonical_field_is_authored(tmp_path):
    record_run_snapshot(
        tmp_path,
        {
            "run": {"session_id": "session-exact"},
            "workload": {"name": "workload"},
            "model": {"name": "model"},
            "versions": {"hyperloom": {"version": "test"}},
            "outcome": {"status": "succeeded"},
        },
    )
    record_phase_transition(tmp_path, {"transition_id": "transition-exact", "phase": "DONE"})
    record_subject(tmp_path, subject_id="subject-exact", subject_type="kernel")
    record_operation(tmp_path, operation_id="operation-exact", status="succeeded")
    record_measurement(tmp_path, measurement_id="measurement-exact", value=1.0)
    record_adoption(
        tmp_path,
        adoption_id="adoption-exact",
        operation_id="operation-exact",
        validated=True,
    )
    record_artifact(tmp_path, artifact_id="artifact-exact", present=True)
    record_trace_event(tmp_path, event_id="event-exact", kind="completed")

    out = exporter.build_v4_live(tmp_path)
    integrity = out["integrity"]

    assert integrity["status"] == "exact"
    assert {field["status"] for field in integrity["fields"].values()} == {"exact"}


def test_write_breakdown_json_uses_v4_feature_flag(tmp_path, monkeypatch):
    record_operation(tmp_path, operation_id="op-flag", status="succeeded")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BREAKDOWN_V4", "1")
    target = exporter.write_breakdown_json(tmp_path)
    flagged = json.loads(target.read_text())
    assert flagged["schema_version"] == SCHEMA_VERSION_V5
    assert "run" in flagged

    monkeypatch.delenv("INFERENCE_OPTIMIZER_BREAKDOWN_V4")
    target = exporter.write_breakdown_json(tmp_path)
    fallback = json.loads(target.read_text())
    assert fallback["schema_version"] == SCHEMA_VERSION_V5
    assert "session" in fallback


def test_v4_helpers_reject_invalid_producer_without_raising(tmp_path):
    record_operation(
        tmp_path,
        operation_id="invalid-producer",
        producer="not valid",
        status="succeeded",
    )
    assert exporter.build_v4_live(tmp_path)["operations"] == []


def test_v4_baseline_action_mirrors_operation_eval_measurements_and_artifacts(tmp_path):
    record_phase_event(
        tmp_path,
        action="baseline",
        entry={
            "ts": "2026-07-22T10:00:00Z",
            "task_id": "baseline-1",
            "status": "succeeded",
            "decision": "promoted",
            "extras": {"fingerprint": "baseline-config"},
        },
        result={
            "status": "succeeded",
            "output_throughput": 123.0,
            "accuracy": 0.77,
            "accuracy_source": "lm-eval",
            "run_eval_disabled": False,
            "benchmark_report_path": "/session/runs/baseline/report.json",
            "workspace": "/session/runs/baseline",
        },
        phase="PRELUDE",
        macro_cycle=2,
        tick=9,
    )

    out = exporter.build_v4_live(tmp_path)
    operation = out["operations"][0]
    assert operation["name"] == "baseline"
    assert operation["status"] == "succeeded"
    assert operation["macro_cycle"] == 2
    assert operation["executor_class"] == "deterministic"
    assert {step["kind"] for step in operation["substeps"]} == {"benchmark", "evaluation"}
    assert next(step for step in operation["substeps"] if step["kind"] == "evaluation")["status"] == "succeeded"
    assert {measurement["name"] for measurement in out["measurements"]} == {"throughput", "accuracy"}
    assert {artifact["kind"] for artifact in out["artifacts"]} == {"workspace", "benchmark_report_path"}
    assert out["phases"]["transitions"][0]["operation_id"] == operation["operation_id"]
    assert out["trace"]["events"][0]["kind"] == "operation_finalized"


def test_v4_framework_action_records_candidate_gates_and_adoption(tmp_path):
    record_phase_event(
        tmp_path,
        action="framework_agent",
        entry={
            "ts": "2026-07-22T10:05:00Z",
            "task_id": "framework-1",
            "status": "succeeded",
            "decision": "promoted",
            "extras": {"candidate_id": "repo-pr-42"},
        },
        result={
            "status": "kept",
            "candidate": {"repo": "org/repo", "pr_number": 42},
            "output_throughput": 130.0,
            "delta_pct": 5.5,
            "accuracy_pass": True,
            "patches_applied": ["/session/runs/framework/repo-pr-42.patch"],
        },
        phase="FRAMEWORK_AGENT",
    )

    out = exporter.build_v4_live(tmp_path)
    operation = out["operations"][0]
    assert operation["executor_class"] == "deterministic"
    assert operation["subject"]["subject_type"] == "framework_candidate"
    assert operation["gates"][0]["decision"] == "allow"
    assert out["adoptions"][0]["decision"] == "KEEP"
    assert out["adoptions"][0]["operation_id"] == operation["operation_id"]


def test_v4_specialist_and_sweep_are_proposal_or_discovery_only(tmp_path):
    record_specialist_round(
        tmp_path,
        {
            "round_id": "round-3",
            "dispatched_at": "2026-07-22T10:10:00Z",
            "completed_at": "2026-07-22T10:11:00Z",
            "domains": ["memory", "scheduler"],
            "proposals_total": 3,
        },
    )
    record_phase_event(
        tmp_path,
        action="sweep",
        entry={
            "ts": "2026-07-22T10:12:00Z",
            "task_id": "sweep-1",
            "status": "succeeded",
            "decision": "discarded",
            "extras": {},
        },
        result={
            "status": "succeeded",
            "all_variants": [
                {
                    "variant_name": "c16",
                    "conc": 16,
                    "output_throughput_tok_s": 140.0,
                    "status": "ok",
                }
            ],
        },
        phase="SWEEP",
    )

    out = exporter.build_v4_live(tmp_path)
    specialist = next(operation for operation in out["operations"] if operation["kind"] == "specialist")
    sweep = next(operation for operation in out["operations"] if operation["name"] == "sweep")
    assert specialist["extensions"]["downstream_relation"] == "proposal_only"
    assert specialist["adoption_refs"] == []
    assert sweep["purpose"] == "discovery"
    assert sweep["adoption_refs"] == []
    assert out["adoptions"] == []
    assert any(measurement["dimensions"]["conc"] == 16 for measurement in out["measurements"])


def test_v4_state_snapshot_does_not_infer_canonical_entities_from_stack(tmp_path):
    state = SimpleNamespace(
        session_id="session-4",
        claw_session_id="claw-4",
        sandbox_user_id="sandbox-4",
        start_ts="2026-07-22T10:00:00Z",
        stop_reason="",
        max_minutes=30,
        tick=4,
        phase="EXPLORE",
        macro_cycle=1,
        framework="sglang",
        model_name="model-4",
        model_path="/models/model-4",
        model_class="decoder",
        model_type="qwen",
        model_architectures=["QwenForCausalLM"],
        model_info={"hidden_size": 4096},
        model_arch={},
        gpu_type="mi355x",
        precision="fp8",
        tp=8,
        ep=1,
        conc=32,
        isl=1024,
        osl=128,
        max_model_len=8192,
        target_gain_pct=10.0,
        target_tput=None,
        baseline_tput=100.0,
        baseline_accuracy=0.75,
        current_best={"tput": 110.0},
        cumulative_gain_validated=10.0,
        cumulative_gain_validated_ts="",
        versions={},
        optimization_stack=[
            {"action": "baseline", "variant_name": "anchor", "tput": 100.0},
            {
                "action": "replay_warm_recipe",
                "task_id": "warm-1",
                "variant_name": "recipe",
                "tput": 110.0,
                "gain_pct": 10.0,
                "decision": "KEEP",
            },
            {"action": "sweep", "variant_name": "discovery", "tput": 120.0},
        ],
        gain_per_stack_entry=[None, 10.0, None],
        explore_search={},
        last_sweep={},
        roofline_snapshots=[],
    )

    snapshot_state_sections(tmp_path, state)
    out = exporter.build_v4_live(tmp_path)
    assert out["run"]["session_id"] == "session-4"
    assert out["workload"]["tp"] == 8
    assert out["model"]["hidden_size"] == 4096
    assert out["outcome"]["optimization_stack_size"] == 3
    assert out["operations"] == []
    assert out["measurements"] == []
    assert out["artifacts"] == []
    assert out["adoptions"] == []
    assert out["optimizations"]["entries"] == []
    assert len(assemble_parts(tmp_path)["optimization_stack"]) == 3


def _session_state(*, stop_reason: str, stop_ts: str) -> SimpleNamespace:
    """A minimal state carrying only what the ``session`` snapshot reads.

    Args:
        stop_reason (str): The state's stop reason (empty while running).
        stop_ts (str): The state's recorded stop timestamp.

    Returns:
        SimpleNamespace: The state stand-in to snapshot.
    """
    return SimpleNamespace(
        session_id="session-5",
        claw_session_id="claw-5",
        sandbox_user_id="sandbox-5",
        start_ts="2026-07-22T10:00:00Z",
        stop_reason=stop_reason,
        stop_ts=stop_ts,
        max_minutes=30,
        tick=5,
        phase="CLOSE",
    )


def test_session_snapshot_carries_the_end_time_of_a_stopped_run(tmp_path):
    snapshot_state_sections(tmp_path, _session_state(stop_reason="time_exhausted", stop_ts="2026-07-22T10:30:00+00:00"))
    session = assemble_parts(tmp_path)["session"]
    assert session["ended_at_utc"] == "2026-07-22T10:30:00Z"


def test_session_snapshot_leaves_a_running_run_without_an_end_time(tmp_path):
    # A resume clears the reason, so a leftover stop_ts must not end the session.
    snapshot_state_sections(tmp_path, _session_state(stop_reason="", stop_ts="2026-07-22T10:30:00+00:00"))
    session = assemble_parts(tmp_path)["session"]
    assert session["ended_at_utc"] == ""


def test_v4_critic_hook_mirrors_structured_kb_writes(tmp_path):
    record_critic_iteration(
        tmp_path,
        iter_n=7,
        review={"verdict": "approve", "summary": "safe"},
        emit={
            "ts": "2026-07-22T10:20:00Z",
            "topic": "framework:42",
            "kb_writes": [
                {
                    "write_id": "write-1",
                    "kind": "lesson",
                    "result": {"status": "succeeded", "point_id": "point-1"},
                }
            ],
        },
        workdir=tmp_path / "critic" / "7",
    )

    out = exporter.build_v4_live(tmp_path)
    critic = next(operation for operation in out["operations"] if operation["kind"] == "critic")
    kb_write = next(operation for operation in out["operations"] if operation["kind"] == "kb_write")
    assert kb_write["parent_operation_id"] == critic["operation_id"]
    assert kb_write["executor_class"] == "deterministic"
    assert kb_write["outputs"]["point_id"] == "point-1"
    assert any(event["kind"] == "kb_write_finalized" for event in out["trace"]["events"])


def test_v4_kernel_route_records_only_selected_xor_candidate(tmp_path):
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="geak",
        actual_path="geak",
        macro_cycle=3,
    )

    out = exporter.build_v4_live(tmp_path)
    selection = next(operation for operation in out["operations"] if operation["kind"] == "strategy_selection")
    routes = [operation for operation in out["operations"] if operation["kind"] == "kernel_optimizer_run"]

    assert len(routes) == 1
    assert routes[0]["strategy"] == "geak"
    assert {route["parent_operation_id"] for route in routes} == {selection["operation_id"]}
    assert routes[0]["status"] == "running"
    assert routes[0]["executor_class"] == "llm_tool"
    assert selection["outputs"]["candidates"] == ["geak", "kernel_agent_forge"]
    assert out["kernel_route"]["xor"] is True
    assert out["kernel_route"]["selected_strategy"] == "geak"
    assert out["kernel_route"]["routes"] == [out["kernel_route"]["executed_route"]]
    assert out["kernel_route"]["executed_route"]["strategy"] == "geak"


def test_v4_versions_stream_merges_with_snapshot_precedence(tmp_path):
    record_run_snapshot(
        tmp_path,
        {
            "versions": {
                "geak": {
                    "tool": "geak",
                    "version": "snapshot-version",
                }
            }
        },
    )
    record_tool_version(tmp_path, tool="geak", version="stream-version")
    record_tool_version(tmp_path, tool="forge", version="forge-version")

    out = exporter.build_v4_live(tmp_path)
    assembled = assemble_v4_parts(tmp_path)

    assert out["versions"]["geak"]["version"] == "snapshot-version"
    assert out["versions"]["forge"]["version"] == "forge-version"
    assert assembled["run_snapshot"]["versions"] == out["versions"]
    assert out["integrity"]["fields"]["versions"]["status"] == "exact"


def test_v4_geak_internal_keep_is_provisional_without_adoption(tmp_path):
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="geak",
        actual_path="geak",
    )
    record_kernel_e2e(
        tmp_path,
        kernel_id="geak-k1",
        integrated=True,
        e2e_gain_pct=8.0,
        validated=True,
        decision="KEEP",
        result={"new_tput": 108.0, "base_tput": 100.0},
        route_strategy="geak_internal",
    )

    out = exporter.build_v4_live(tmp_path)
    assert out["adoptions"] == []
    assert not any(operation["kind"] == "kernel_optimization" for operation in out["operations"])
    geak = next(
        operation
        for operation in out["operations"]
        if operation.get("kind") == "kernel_optimizer_run"
        and operation.get("strategy") == "geak"
    )
    refs = geak["extensions"]["geak"]["internal_refs"]
    assert refs[0]["kernel_id"] == "geak-k1"
    assert refs[0]["provisional"] is True


def test_v4_geak_final_validation_creates_validated_adoption(tmp_path):
    result = {
        "status": "ok",
        "baseline_throughput_tok_s": 100.0,
        "final_throughput_tok_s": 112.0,
        "accepted_config": {"flags": "--foo", "env": "BAR=1"},
        "output_parity": True,
        "alignment_metrics": {"final_basis": "hot"},
        "bench_client": "geak",
    }
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="geak",
        actual_path="geak",
    )
    record_geak_operation(tmp_path, stage="candidate", result=result, status="running")
    assert exporter.build_v4_live(tmp_path)["adoptions"] == []

    record_geak_operation(
        tmp_path,
        stage="final_validation",
        result=result,
        status="succeeded",
        validated=True,
        measured_tput=111.0,
        validation_source="geak_orch_harness_validated",
    )
    out = exporter.build_v4_live(tmp_path)

    assert len(out["adoptions"]) == 1
    assert out["adoptions"][0]["validated"] is True
    assert out["adoptions"][0]["metadata"]["validation_tier"] == "orchestrator_final"
    final = next(
        measurement
        for measurement in out["measurements"]
        if measurement["name"] == "final_throughput"
        and measurement["dimensions"]["headline_eligible"] is True
    )
    assert final["value"] == 111.0
    assert final["metric_basis"] == "output"
    assert out["optimizations"]["entries"][0]["source"] == "kernel_agent"
    assert out["optimizations"]["entries"][0]["backend"] == "geak"


def test_v4_forge_operation_carries_correctness_evidence(tmp_path):
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="kernel_agent_forge",
        actual_path="kernel_agent_forge",
    )
    record_kernel_dispatch(
        tmp_path,
        kernel_id="k001",
        dispatched=True,
        backends=["forge"],
    )
    record_kernel_backend_result(
        tmp_path,
        {
            "kernel_id": "k001",
            "run_id": "run-1",
            "proposal": {"decision": "KEEP", "reasons": ["verified"]},
            "verification": {
                "status": "passed",
                "compile_passed": True,
                "correctness_passed": True,
                "correctness_source": "generated_harness",
                "best_attempt_id": "attempt-1",
                "best_backend": "forge",
            },
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "backend": "forge",
                    "status": "succeeded",
                    "compile_passed": True,
                    "correctness_passed": True,
                    "correctness_source": "generated_harness",
                    "micro_speedup": 1.25,
                }
            ],
            "status": "ok",
        },
    )

    out = exporter.build_v4_live(tmp_path)
    operation = next(operation for operation in out["operations"] if operation["kind"] == "kernel_optimization")
    native_route = next(route for route in out["kernel_route"]["routes"] if route["strategy"] == "kernel_agent_forge")
    assert len(out["kernel_route"]["routes"]) == 1
    assert out["kernel_route"]["executed_route"] == native_route
    assert native_route["selected"] is True
    assert operation["executor_class"] == "llm_tool"
    assert not any(
        candidate.get("strategy") == "geak"
        for candidate in out["operations"]
        if candidate.get("kind") == "kernel_optimizer_run"
    )
    assert operation["parent_operation_id"] == native_route["operation_id"]
    assert operation["attempts"][0]["outputs"]["correctness_source"] == "generated_harness"
    assert out["optimizations"]["backend_attempts"][0]["attempt_id"] == "attempt-1"
    assert out["optimizations"]["backend_attempts"][0]["kernel_id"] == "k001"
    assert out["optimizations"]["backend_attempts"][0]["backend"] == "forge"
    assert out["optimizations"]["backend_attempts"][0]["micro_speedup"] == 1.25
    assert out["optimizations"]["backend_attempts"][0]["compile_passed"] is True
    assert (
        out["optimizations"]["backend_attempts"][0]["correctness_passed"]
        is True
    )
    gate_names = {gate["name"] for gate in operation["gates"]}
    assert {"compile", "correctness", "kernel_verification"} <= gate_names
    measurement = next(item for item in out["measurements"] if item["name"] == "micro_speedup")
    assert measurement["metric_basis"] == "kernel_time_ratio"
    assert measurement["metadata"]["completeness"] == "partial"


def test_v4_gemm_adoption_is_keep_only(tmp_path):
    keep_dir = tmp_path / "keep"
    revert_dir = tmp_path / "revert"
    record_gemm_tuning_operation(
        keep_dir,
        payload={"task_id": "gemm-keep", "gemm_tuning_backend": "forge"},
        result={
            "status": "complete",
            "decision": "KEEP",
            "backend": "forge",
            "best_speedup": 1.2,
            "e2e_validated": True,
            "e2e_gain_pct": 5.0,
            "artifacts": {"tuned": "/tmp/tuned.json"},
        },
    )
    record_gemm_tuning_operation(
        revert_dir,
        payload={"task_id": "gemm-revert", "gemm_tuning_backend": "forge"},
        result={
            "status": "complete",
            "decision": "REVERT",
            "backend": "forge",
            "best_speedup": 0.98,
        },
    )

    keep = exporter.build_v4_live(keep_dir)
    revert = exporter.build_v4_live(revert_dir)
    assert [adoption["decision"] for adoption in keep["adoptions"]] == ["KEEP"]
    assert revert["adoptions"][0]["status"] == "revoked"
    assert revert["adoptions"][0]["validated"] is False
    assert revert["optimizations"]["entries"] == []
    assert keep["optimizations"]["entries"][0]["backend"] == "forge"
    assert keep["optimizations"]["entries"][0]["optimization_kind"] == "gemm_tuning"
    assert keep["optimizations"]["gemm_tuning_runs"][0]["engine"] == "forge"
    assert keep["optimizations"]["gemm_tuning_runs"][0]["adopted"] is True
    assert keep["optimizations"]["validation"]["method"] == "validated"
    assert keep["optimizations"]["validation"]["validated_total_gain_pct"] == 5.0


def test_v4_kernel_keep_then_revert_revokes_adoption(tmp_path):
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="kernel_agent_forge",
        actual_path="kernel_agent_forge",
        macro_cycle=1,
    )
    record_kernel_e2e(
        tmp_path,
        kernel_id="kernel-revoke",
        integrated=True,
        validated=True,
        decision="KEEP",
        e2e_gain_pct=4.0,
        result={"decision_reason": "validated"},
    )
    record_kernel_e2e(
        tmp_path,
        kernel_id="kernel-revoke",
        integrated=False,
        validated=False,
        decision="REVERT",
        e2e_gain_pct=-2.0,
        result={"decision_reason": "regression"},
    )

    out = exporter.build_v4_live(tmp_path)

    assert len(out["adoptions"]) == 1
    assert out["adoptions"][0]["status"] == "revoked"
    assert out["adoptions"][0]["decision"] == "REVERT"
    assert out["adoptions"][0]["validated"] is False
    assert out["optimizations"]["entries"] == []
    assert all(
        summary["keeps"] == 0
        for summary in out["optimizations"]["summary_by_source"].values()
    )


def test_v4_warm_replay_keep_then_revert_revokes_same_adoption(tmp_path):
    record_phase_event(
        tmp_path,
        action="replay_warm_recipe",
        entry={
            "action": "replay_warm_recipe",
            "task_id": "warm-revoke",
            "status": "succeeded",
            "decision": "KEEP",
            "ts": "2026-07-22T10:00:00Z",
        },
        result={"validated": True, "best_gain_pct": 4.0},
        phase="EXPLORE",
        macro_cycle=1,
        tick=1,
    )
    record_phase_event(
        tmp_path,
        action="replay_warm_recipe",
        entry={
            "action": "replay_warm_recipe",
            "task_id": "warm-revoke",
            "status": "succeeded",
            "decision": "REVERT",
            "ts": "2026-07-22T10:01:00Z",
        },
        result={"validated": False, "reason": "final validation failed"},
        phase="EXPLORE",
        macro_cycle=1,
        tick=2,
    )

    out = exporter.build_v4_live(tmp_path)

    assert len(out["adoptions"]) == 1
    assert out["adoptions"][0]["status"] == "revoked"
    assert out["adoptions"][0]["validated"] is False
    assert out["optimizations"]["entries"] == []


def test_v4_geak_final_validation_failure_revokes_prior_keep(tmp_path):
    result = {
        "status": "ok",
        "baseline_throughput_tok_s": 100.0,
        "final_throughput_tok_s": 110.0,
        "accepted_config": {"flags": "--candidate"},
    }
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="geak",
        actual_path="geak",
        macro_cycle=2,
    )
    record_geak_operation(
        tmp_path,
        stage="final_validation",
        result=result,
        status="succeeded",
        validated=True,
        measured_tput=110.0,
        validation_source="same_harness",
        macro_cycle=2,
    )
    record_geak_operation(
        tmp_path,
        stage="final_validation_failed",
        result={**result, "status": "failed"},
        status="failed",
        validated=False,
        validation_source="same_harness",
        macro_cycle=2,
    )

    out = exporter.build_v4_live(tmp_path)

    assert len(out["adoptions"]) == 1
    assert out["adoptions"][0]["status"] == "revoked"
    assert out["adoptions"][0]["validated"] is False
    assert out["optimizations"]["entries"] == []


def test_v4_gemm_micro_keep_is_provisional_until_e2e_keep(tmp_path):
    payload = {
        "task_id": "gemm-e2e",
        "gemm_tuning_backend": "forge",
        "macro_cycle": 3,
    }
    micro = {
        "status": "complete",
        "decision": "KEEP",
        "best_speedup": 1.2,
        "e2e_validated": False,
    }
    record_gemm_tuning_operation(tmp_path, payload=payload, result=micro)
    assert exporter.build_v4_live(tmp_path)["adoptions"] == []

    record_gemm_tuning_operation(
        tmp_path,
        payload=payload,
        result={
            **micro,
            "e2e_validated": True,
            "e2e_gain_pct": 3.5,
        },
    )
    out = exporter.build_v4_live(tmp_path)

    assert len(out["adoptions"]) == 1
    assert out["adoptions"][0]["status"] == "adopted"
    assert out["adoptions"][0]["validated"] is True


def test_v4_kernel_routes_remain_xor_across_macro_cycles(tmp_path):
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="geak",
        actual_path="geak",
        macro_cycle=1,
    )
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="kernel_agent_forge",
        actual_path="kernel_agent_forge",
        macro_cycle=2,
    )

    out = exporter.build_v4_live(tmp_path)
    selections = [
        operation
        for operation in out["operations"]
        if operation.get("kind") == "strategy_selection"
    ]
    routes = [
        operation
        for operation in out["operations"]
        if operation.get("kind") == "kernel_optimizer_run"
    ]

    assert len(selections) == 2
    assert len(routes) == 2
    assert len({operation["operation_id"] for operation in selections}) == 2
    assert len({operation["operation_id"] for operation in routes}) == 2
    assert out["kernel_route"]["xor"] is True
    assert [cycle["xor"] for cycle in out["kernel_route"]["cycles"]] == [True, True]
    assert out["integrity"]["conflicts"] == []


def test_v4_kernel_route_reselection_supersedes_prior_strategy_in_same_cycle(tmp_path):
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="geak",
        actual_path="geak",
        macro_cycle=7,
    )
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="kernel_agent_forge",
        actual_path="kernel_agent_forge",
        macro_cycle=7,
    )

    out = exporter.build_v4_live(tmp_path)
    selections = [
        operation
        for operation in out["operations"]
        if operation.get("kind") == "strategy_selection"
    ]
    geak_route = next(
        route
        for route in out["operations"]
        if route.get("kind") == "kernel_optimizer_run"
        and route.get("strategy") == "geak"
    )

    assert len(selections) == 1
    assert selections[0]["outputs"]["selected_strategy"] == "kernel_agent_forge"
    assert selections[0]["outputs"]["selection_version"] == 2
    assert geak_route["status"] == "superseded"
    assert out["kernel_route"]["xor"] is True
    assert out["kernel_route"]["executed_route"]["strategy"] == "kernel_agent_forge"
    assert len(out["kernel_route"]["routes"]) == 2
    assert sum(route["active"] is True for route in out["kernel_route"]["routes"]) == 1
    assert out["integrity"]["conflicts"] == []


def test_v4_cross_process_reselection_is_normalized_from_operation_fragments(tmp_path):
    from hyperloom.inference_optimizer.breakdown.recorder import instrument
    from hyperloom.inference_optimizer.breakdown.recorder import recorder as recorder_module

    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="geak",
        actual_path="geak",
        macro_cycle=8,
    )
    instrument._KERNEL_ROUTE_CONTEXT.clear()
    recorder_module._RECORDERS.clear()
    record_kernel_strategy_selection(
        tmp_path,
        selected_strategy="kernel_agent_forge",
        actual_path="kernel_agent_forge",
        macro_cycle=8,
    )

    assembled = assemble_v4_parts(tmp_path)
    route_operations = [
        operation
        for operation in assembled["operations"]
        if operation.get("kind") == "kernel_optimizer_run"
    ]
    geak_route = next(
        operation
        for operation in route_operations
        if operation.get("strategy") == "geak"
    )
    forge_route = next(
        operation
        for operation in route_operations
        if operation.get("strategy") == "kernel_agent_forge"
    )
    out = exporter.build_v4_live(tmp_path)

    assert geak_route["status"] == "superseded"
    assert geak_route["extensions"]["route_competition"]["active"] is False
    assert geak_route["extensions"]["route_competition"]["historical_executed"] is True
    assert forge_route["extensions"]["route_competition"]["active"] is True
    assert out["operations"] == assembled["operations"]
    assert out["kernel_route"]["executed_route"]["strategy"] == "kernel_agent_forge"
    assert out["kernel_route"]["xor"] is True
    assert out["integrity"]["conflicts"] == []


def test_v4_phase_transition_ids_do_not_collide_with_same_timestamp(tmp_path):
    from hyperloom.orchestrator.phases import machine_state

    state = SimpleNamespace(
        phase="EXPLORE",
        phase_history=[],
        macro_cycle=4,
        tick=8,
        _session_dir=tmp_path,
    )
    machine_state.record_phase_transition(
        state,
        to_phase="KERNEL_AGENT",
        reason="first",
        ts="2026-07-22T12:00:00Z",
        ts_unix=100.0,
    )
    state.phase = "EXPLORE"
    machine_state.record_phase_transition(
        state,
        to_phase="KERNEL_AGENT",
        reason="second",
        ts="2026-07-22T12:00:00Z",
        ts_unix=100.0,
    )

    transitions = exporter.build_v4_live(tmp_path)["phase_timeline"]
    assert len(transitions) == 2
    assert len({transition["transition_id"] for transition in transitions}) == 2


def test_v4_integrity_reports_dangling_refs_and_route_conflicts(tmp_path):
    record_operation(
        tmp_path,
        operation_id="selection-conflict",
        kind="strategy_selection",
        strategy_group="kernel_optimizer",
        outputs={"selected_strategy": "geak"},
        executor_class="invalid",
        measurement_refs=["missing-measurement"],
    )

    out = exporter.build_v4_live(tmp_path)
    integrity = out["integrity"]
    codes = {conflict["code"] for conflict in integrity["conflicts"]}

    assert integrity["status"] == "partial"
    assert integrity["fields"]["operations"]["status"] == "partial"
    assert {
        "invalid_executor_class",
        "dangling_reference",
        "kernel_route_xor_violation",
    } <= codes


def test_v4_integrity_reports_dangling_transition_and_trace_operations(tmp_path):
    record_phase_transition(
        tmp_path,
        transition_id="transition-dangling",
        operation_id="missing-transition-operation",
        phase="EXPLORE",
    )
    record_trace_event(
        tmp_path,
        event_id="trace-dangling",
        kind="operation_finalized",
        operation_id="missing-trace-operation",
        parent_operation_id="missing-trace-parent",
    )

    integrity = exporter.build_v4_live(tmp_path)["integrity"]
    phase_conflicts = integrity["fields"]["phases"]["status"]
    trace_conflicts = integrity["fields"]["trace"]["status"]
    dangling = [
        conflict
        for conflict in integrity["conflicts"]
        if conflict["code"] == "dangling_operation_reference"
    ]

    assert phase_conflicts == "partial"
    assert trace_conflicts == "partial"
    assert len(dangling) == 3


def test_v4_integrity_requires_operation_for_business_entities(tmp_path):
    record_measurement(
        tmp_path,
        measurement_id="measurement-without-operation",
        kind="throughput",
        name="throughput",
        value=1.0,
    )
    record_adoption(
        tmp_path,
        adoption_id="adoption-without-operation",
        status="adopted",
    )
    record_artifact(
        tmp_path,
        artifact_id="artifact-without-operation",
        kind="report",
        path="/tmp/report.json",
    )

    integrity = exporter.build_v4_live(tmp_path)["integrity"]
    conflicts = [
        conflict
        for conflict in integrity["conflicts"]
        if conflict["code"] == "missing_operation_reference"
    ]

    assert {conflict["field"] for conflict in conflicts} == {
        "measurements",
        "adoptions",
        "artifacts",
    }


def test_v4_integrity_rejects_multiple_selection_ids_in_one_cycle(tmp_path):
    for suffix, strategy in (("a", "geak"), ("b", "kernel_agent_forge")):
        selection_id = f"selection-{suffix}"
        route_id = f"route-{suffix}"
        record_operation(
            tmp_path,
            operation_id=selection_id,
            kind="strategy_selection",
            strategy_group="kernel_optimizer",
            macro_cycle=9,
            outputs={
                "candidates": ["geak", "kernel_agent_forge"],
                "selected_strategy": strategy,
            },
        )
        record_operation(
            tmp_path,
            operation_id=route_id,
            kind="kernel_optimizer_run",
            strategy_group="kernel_optimizer",
            strategy=strategy,
            status="running",
            macro_cycle=9,
            parent_operation_id=selection_id,
            root_operation_id=selection_id,
        )

    out = exporter.build_v4_live(tmp_path)
    integrity = out["integrity"]
    codes = {conflict["code"] for conflict in integrity["conflicts"]}

    assert "multiple_active_selections_in_cycle" in codes
    assert integrity["fields"]["operations"]["status"] == "partial"
    assert out["kernel_route"]["xor"] is False


def test_v4_integrity_validates_nested_operation_references(tmp_path):
    record_operation(
        tmp_path,
        operation_id="nested-op",
        kind="composite",
        relations=[
            {
                "relation_id": "relation-1",
                "operation_id": "nested-op",
                "target_operation_id": "missing-related-operation",
                "subject": {"subject_id": "missing-related-subject"},
            }
        ],
        attempts=[
            {
                "attempt_id": "attempt-nested",
                "operation_id": "missing-attempt-operation",
                "measurements": ["missing-attempt-measurement"],
                "artifacts": ["missing-attempt-artifact"],
                "adoption_refs": ["missing-attempt-adoption"],
                "subject_id": "missing-attempt-subject",
            }
        ],
        substeps=[
            {
                "substep_id": "substep-nested",
                "measurements": ["missing-substep-measurement"],
                "artifacts": ["missing-substep-artifact"],
                "adoption_id": "missing-substep-adoption",
                "target_operation_id": "missing-substep-operation",
            }
        ],
    )

    integrity = exporter.build_v4_live(tmp_path)["integrity"]
    nested_conflicts = [
        conflict
        for conflict in integrity["conflicts"]
        if conflict["code"] == "dangling_nested_reference"
    ]
    reference_fields = {conflict["reference_field"] for conflict in nested_conflicts}

    assert integrity["fields"]["operations"]["status"] == "partial"
    assert {
        "target_operation_id",
        "subject",
        "operation_id",
        "measurements",
        "artifacts",
        "adoption_refs",
        "subject_id",
        "adoption_id",
    } <= reference_fields


def test_v4_compat_major_fields_project_only_from_canonical(tmp_path):
    record_operation(
        tmp_path,
        operation_id="baseline-op",
        kind="composite",
        name="baseline",
        outputs={"throughput": 100.0},
    )
    record_operation(
        tmp_path,
        operation_id="explore-op",
        kind="explore",
        outputs={"decision": "KEEP", "best_gain_pct": 5.0},
    )
    record_operation(
        tmp_path,
        operation_id="roofline-op",
        kind="roofline",
        outputs={"within_roofline_pct": 80.0},
    )
    record_operation(
        tmp_path,
        operation_id="capability-op",
        kind="capability_summary",
        outputs={"forge": True},
    )
    record_operation(
        tmp_path,
        operation_id="specialist-op",
        kind="specialist",
        outputs={"proposals_total": 2},
    )
    record_operation(
        tmp_path,
        operation_id="kb-op",
        kind="kb_write",
        outputs={"point_id": "point-1"},
    )
    record_operation(
        tmp_path,
        operation_id="kernel-op",
        kind="kernel_optimization",
        name="kernel-1",
        scope="kernel",
        strategy="forge",
        status="succeeded",
        attempts=[{"attempt_id": "attempt-1", "backend": "forge", "status": "succeeded"}],
        outputs={"decision": "KEEP"},
    )
    record_measurement(
        tmp_path,
        measurement_id="kernel-roofline-measurement",
        operation_id="kernel-op",
        kind="roofline",
        name="arithmetic_intensity",
        value=12.5,
        dimensions={"kernel_id": "kernel-1"},
    )
    record_operation(
        tmp_path,
        operation_id="conc-sweep-op",
        kind="conc_sweep",
        status="succeeded",
        outputs={"comparison": [{"conc": 16, "speedup": 1.1}]},
    )
    record_artifact(
        tmp_path,
        artifact_id="source-artifact",
        operation_id="explore-op",
        kind="source_file",
        path="/workspace/model.py",
    )
    record_trace_event(
        tmp_path,
        event_id="trace-compat",
        kind="operation_finalized",
        operation_id="explore-op",
    )
    record_trace_event(
        tmp_path,
        event_id="langfuse-compat",
        kind="langfuse_push_receipt",
        operation_id="explore-op",
        enabled=True,
        trace_id="trace-1",
        counts={"generations_sent": 2},
        counts_final=True,
        receipt_source="author_time_trace",
    )

    out = exporter.build_v4_live(tmp_path)

    assert out["baseline"]["throughput"] == 100.0
    assert out["param_search"]["operations"][0]["operation_id"] == "explore-op"
    assert out["roofline"][0]["outputs"]["within_roofline_pct"] == 80.0
    assert out["capability_summary"]["forge"] is True
    assert out["specialist_runs"][0]["outputs"]["proposals_total"] == 2
    assert out["kb_provenance"]["operations"][0]["outputs"]["point_id"] == "point-1"
    assert out["telemetry"]["events"][0]["event_id"] == "trace-compat"
    assert out["source_files"]["source-artifact"]["path"] == "/workspace/model.py"
    assert out["kernel_lifecycle"]["optimized"][0]["kernel_id"] == "kernel-1"
    assert out["kernel_roofline"]["kernels"][0]["arithmetic_intensity"] == 12.5
    assert out["kernel_optimization_summary"]["totals"]["attempted"] == 1
    assert out["conc_sweep_summary"]["comparison"][0]["conc"] == 16
    assert out["langfuse"]["enabled"] is True
    assert out["langfuse"]["counts"]["generations_sent"] == 2


# ---------------------------------------------------------------------------
# _merge_lists / _deep_merge — nested-id keyed merge vs. append semantics
# ---------------------------------------------------------------------------


def test_merge_lists_merges_entries_sharing_a_nested_id():
    """Two entries with the same ``attempt_id`` are deep-merged, not duplicated."""
    current = [{"attempt_id": "a1", "status": "running", "tags": ["x"]}]
    update = [{"attempt_id": "a1", "status": "done", "tags": ["y"]}]
    merged = _merge_lists(current, update)
    assert len(merged) == 1
    assert merged[0]["status"] == "done"
    # nested list under a keyed entry is itself merged (append of new scalar).
    assert merged[0]["tags"] == ["x", "y"]


def test_merge_lists_appends_new_keyed_and_dedupes_scalars():
    """A new nested id appends; duplicate scalars are not re-appended."""
    current = [{"subject_id": "s1", "name": "one"}, "scalar"]
    update = [
        {"subject_id": "s2", "name": "two"},  # new id -> append
        {"subject_id": "s1", "name": "one-updated"},  # existing id -> merge
        "scalar",  # duplicate scalar -> dropped
        "fresh",  # new scalar -> appended
    ]
    merged = _merge_lists(current, update)
    ids = [e["subject_id"] for e in merged if isinstance(e, dict)]
    assert ids == ["s1", "s2"]
    s1 = next(e for e in merged if isinstance(e, dict) and e["subject_id"] == "s1")
    assert s1["name"] == "one-updated"
    scalars = [e for e in merged if not isinstance(e, dict)]
    assert scalars == ["scalar", "fresh"]


def test_merge_lists_appends_unkeyed_dict_when_novel():
    """A dict with no nested id and not already present is appended verbatim."""
    current = [{"foo": 1}]
    update = [{"foo": 1}, {"bar": 2}]  # first is a dup, second is novel
    merged = _merge_lists(current, update)
    assert merged == [{"foo": 1}, {"bar": 2}]


def test_deep_merge_recurses_dicts_and_merges_nested_lists():
    """Nested dicts recurse; nested keyed lists route through _merge_lists."""
    current = {"a": {"b": 1}, "items": [{"gate_id": "g1", "v": 1}]}
    update = {"a": {"c": 2}, "items": [{"gate_id": "g1", "v": 2}], "d": 3}
    merged = _deep_merge(current, update)
    assert merged["a"] == {"b": 1, "c": 2}
    assert merged["d"] == 3
    assert merged["items"] == [{"gate_id": "g1", "v": 2}]


# ---------------------------------------------------------------------------
# _kernel_outcome — coarse per-kernel lifecycle label
# ---------------------------------------------------------------------------


def test_kernel_outcome_dispatched_vs_skipped_vs_discovered():
    assert _kernel_outcome({"dispatched": True}, [], {}) == "dispatched"
    assert _kernel_outcome({"dispatched": False}, [], {}) == "skipped"
    assert _kernel_outcome({}, [], {}) == "discovered"


def test_kernel_outcome_attempted_and_terminal_decisions():
    assert _kernel_outcome({}, [{"backend": "geak"}], {}) == "attempted"
    assert _kernel_outcome({}, [], {"decision": "REVERT"}) == "reverted"
    e2e_keep = {"decision": "KEEP", "validated": True, "integrated": True}
    assert _kernel_outcome({}, [], e2e_keep) == "adopted"


# ---------------------------------------------------------------------------
# _kb_writes_summary — verdict tally, skipping malformed rows
# ---------------------------------------------------------------------------


def test_kb_writes_summary_skips_non_dict_and_empty_verdict():
    rows = [
        {"verdict": "approve"},
        {"verdict": "approve"},
        {"verdict": ""},  # empty -> skipped
        "not-a-dict",  # non-dict -> skipped
        {"verdict": "reject"},
    ]
    summary = _kb_writes_summary(rows)
    assert summary["total"] == 3
    assert summary["by_verdict"] == {"APPROVE": 2, "REJECT": 1}
