# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``kernel`` event: what it records and what it settles."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.assembler import kernel_event_parts
from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import (
    REBENCH_FALLBACK,
    REBENCH_NO_PROMOTE,
    REBENCH_VALIDATED,
    ROUTE_FORGE,
    ROUTE_GEAK,
    SOURCE_GEAK_AUTHORED_KERNEL,
    SOURCE_GEAK_ENV_SELECTION,
    SOURCE_KERNEL_REWRITE,
    assemble_kernel_ext,
    kernel_event_id,
    make_kernel_recorder,
    record_integrate_verdict,
    record_trace_analyze_request,
)
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.phases.kernel import KernelPhase


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _kernel_events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "kernel"]


def _forge_recorder():
    recorder = make_kernel_recorder(
        macro_cycle=3,
        route=ROUTE_FORGE,
        route_reason="kernel_opt_backend_order=forge",
        code_revision="abc1234",
    )
    assert recorder is not None
    recorder.begin(
        stack_depth_in=2,
        tput_before=1000.0,
        session_baseline_tput=800.0,
        snapshot={"roofline_snapshot_id": 4, "ts": "2026-09-02T00:00:00"},
        snapshot_staleness="fresh",
    )
    return recorder


def _geak_recorder(*, macro_cycle: int = 1):
    recorder = make_kernel_recorder(macro_cycle=macro_cycle, route=ROUTE_GEAK)
    assert recorder is not None
    recorder.begin(tput_before=900.0)
    return recorder


def _phase_with_recorder(tmp_path: Path, recorder: Any) -> KernelPhase:
    phase = object.__new__(KernelPhase)
    phase.session_dir = tmp_path
    phase.shared_state = types.SimpleNamespace(macro_cycle=3)
    phase._kernel_timeline_recorder = recorder
    return phase


def test_one_event_per_entry_with_macro_cycle_at_the_top(tmp_path):
    recorder = _forge_recorder()
    recorder.finish(tput_after=1000.0)

    events = _kernel_events(tmp_path)
    assert len(events) == 1
    ext = events[0]["ext"]
    assert events[0]["kind"] == "kernel_agent"
    assert events[0]["id"] == kernel_event_id(3)
    assert ext["macro_cycle"] == 3
    assert "macro_cycle" not in (ext.get("forge") or {})
    assert ext["entry"]["route"] == ROUTE_FORGE
    assert ext["entry"]["roofline_snapshot_id"] == 4


def test_the_event_is_on_the_timeline_before_it_concludes(tmp_path):
    _forge_recorder()

    events = _kernel_events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == "running"
    assert "end_time" not in events[0]


def test_the_stage_in_flight_is_recoverable_from_the_rows_alone(tmp_path):
    recorder = _forge_recorder()
    recorder.enter_stage("gemm_tuning")

    assert _kernel_events(tmp_path)[0]["status"] == "running"
    ext, status = assemble_kernel_ext(kernel_event_parts(), event=recorder.event_id)
    assert ext["in_flight_stage"] == "gemm_tuning"
    # An event still in flight has reached no verdict, whatever its rows so far
    # add up to.
    assert status == "running"


def test_a_forge_keep_is_settled_by_the_lane_that_timed_it(tmp_path):
    """The integrate gate runs after the visit, so the lane is its own evidence.

    A verdict that waited for the gate could never be stated here: the gate
    records into the event of the cycle it settles in, which is not this one.
    """
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(
        run_id="attempt-7",
        kernel_id="k001",
        status="success",
        micro_decision="keep",
        speedup=1.4,
    )
    recorder.finish(tput_after=1000.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    row = ext["attempts"][0]
    assert row["route"] == ROUTE_FORGE
    assert row["source_kind"] == SOURCE_KERNEL_REWRITE
    assert row["micro_decision"] == "KEEP"
    assert row["accepted"] is True
    assert row["outcome"] == "adopted"
    assert row["settled_by"] == "lane"
    assert row["unsettled_reason"] == ""
    # The lane reports a ratio; the attempt states one axis for both routes.
    assert row["gain_pct"] == 40.0
    assert ext["outcome"]["verdict"] == "improved"
    assert [entry["settled_by"] for entry in ext["outcome"]["delivered"]] == ["lane"]


def test_a_keep_nothing_measured_is_delivered_without_claiming_a_gain(tmp_path):
    """Keeping a candidate and improving the model are two different facts."""
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="keep")
    recorder.finish(tput_after=1000.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["attempts"][0]["outcome"] == "adopted"
    assert ext["attempts"][0]["gain_pct"] is None
    assert [entry["ref"] for entry in ext["outcome"]["delivered"]] == ["attempt-7"]
    assert ext["outcome"]["verdict"] == "no_improvement"
    assert ext["outcome"]["reason"] == "nothing measured a gain on the candidates kept"


def test_a_bus_requested_analysis_accounts_for_the_snapshot_it_advanced(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="keep")
    record_trace_analyze_request(
        macro_cycle=3,
        run_id="msg-42",
        status="ok",
        result={"hot_kernels": [{"name": "fused_moe", "gpu_pct": 31.5}]},
        requested_by="kernel_agent",
        request_msg_id="msg-42",
        trace_input="traces/rank0.pt.trace.json",
        top_k=15,
        snapshot={"roofline_snapshot_id": 4, "analysis_md_path": "reports/analysis.md"},
    )
    recorder.finish(tput_after=1000.0)

    runs = _kernel_events(tmp_path)[0]["ext"]["forge"]["trace_analyze_runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "msg-42"
    assert runs[0]["trigger"] == "bus_request"
    assert runs[0]["requested_by"] == "kernel_agent"
    assert runs[0]["roofline_snapshot_id"] == 4
    assert runs[0]["cache_hit"] is False


def test_a_bus_requested_analysis_after_the_visit_closed_still_lands(tmp_path):
    recorder = _forge_recorder()
    recorder.finish(tput_after=1000.0)

    record_trace_analyze_request(
        macro_cycle=3,
        run_id="msg-42",
        status="ok",
        result={},
        snapshot={"roofline_snapshot_id": 4},
    )

    runs = _kernel_events(tmp_path)[0]["ext"]["forge"]["trace_analyze_runs"]
    assert [run["run_id"] for run in runs] == ["msg-42"]


def test_a_bus_request_with_no_visit_running_mints_no_event(tmp_path):
    record_trace_analyze_request(
        macro_cycle=3,
        run_id="msg-42",
        status="ok",
        result={},
        snapshot={"roofline_snapshot_id": 4},
    )

    assert _kernel_events(tmp_path) == []


def test_a_bus_requested_analysis_carries_the_kernel_table_it_produced(tmp_path):
    recorder = _forge_recorder()
    record_trace_analyze_request(
        macro_cycle=3,
        run_id="msg-42",
        status="ok",
        result={},
        snapshot={
            "roofline_snapshot_id": 4,
            "reusable_native_kernel_ids": ["k001"],
            "hot_kernels_top15": [{"kernel_id": "k001", "name": "fused_moe", "gpu_pct": 31.5}],
        },
    )
    recorder.finish(tput_after=1000.0)

    discovered = _kernel_events(tmp_path)[0]["ext"]["forge"]["discovered_kernels"]
    assert [row["kernel_id"] for row in discovered] == ["k001"]
    assert discovered[0]["provenance"] == "trace_analyze_run"


def test_an_adoption_states_the_basis_its_gain_was_measured_on(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="keep")
    record_integrate_verdict(
        macro_cycle=3,
        integration_id="geak-k001",
        kernel_id="k001",
        decision="KEEP",
        gain_pct=6.0,
        basis="hot",
        alignment_status="aligned",
        gain_attributed=False,
    )
    recorder.finish(tput_after=1060.0)

    row = _kernel_events(tmp_path)[0]["ext"]["integrate"][0]
    assert row["basis"] == "hot"
    assert row["alignment_status"] == "aligned"
    assert row["gain_attributed"] is False


def test_the_integrate_gate_is_what_settles_a_forge_candidate(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="keep")
    record_integrate_verdict(macro_cycle=3, integration_id="int-1", kernel_id="k001", decision="KEEP", gain_pct=6.0)
    recorder.finish(tput_after=1060.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    row = ext["attempts"][0]
    assert row["outcome"] == "adopted"
    # Forge runs no rebench of its own; the gate is its evidence.
    assert row["settled_by"] == "integrate"
    assert ext["outcome"]["verdict"] == "improved"
    assert ext["outcome"]["delivered"] == [
        {
            "route": ROUTE_FORGE,
            "source_kind": SOURCE_KERNEL_REWRITE,
            "ref": "attempt-7",
            "kernel_id": "k001",
            "gain_pct": 6.0,
            "settled_by": "integrate",
        }
    ]


def test_a_lane_that_declined_its_own_candidate_needs_no_gate(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="revert")
    recorder.finish(tput_after=1000.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    row = ext["attempts"][0]
    assert row["accepted"] is False
    assert row["outcome"] == "rejected"
    assert row["settled_by"] == "lane"
    assert ext["outcome"]["verdict"] == "no_improvement"
    assert ext["outcome"]["reason"] == "no candidate was kept"


def test_a_reverted_kernel_is_rejected_by_the_gate_not_left_pending(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="keep")
    record_integrate_verdict(macro_cycle=3, integration_id="int-1", kernel_id="k001", decision="REVERT")
    recorder.finish(tput_after=1000.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["attempts"][0]["outcome"] == "rejected"
    assert ext["attempts"][0]["settled_by"] == "integrate"
    assert ext["outcome"]["verdict"] == "no_improvement"


def test_a_kernel_gated_twice_is_settled_by_the_verdict_that_stands(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="keep")
    record_integrate_verdict(
        macro_cycle=3,
        integration_id="int-1",
        kernel_id="k001",
        decision="REVERT",
        settled_at="2026-09-02T00:01:00",
    )
    record_integrate_verdict(
        macro_cycle=3,
        integration_id="int-2",
        kernel_id="k001",
        decision="KEEP",
        gain_pct=4.0,
        settled_at="2026-09-02T00:09:00",
    )
    recorder.finish(tput_after=1000.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["attempts"][0]["outcome"] == "adopted"
    assert ext["attempts"][0]["e2e"]["decision"] == "KEEP"
    assert ext["attempts"][0]["e2e"]["e2e_gain_pct"] == 4.0


def test_an_adopted_rewrite_states_the_backend_that_produced_it(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(
        run_id="attempt-7",
        kernel_id="k001",
        status="success",
        adopted_backend="triton",
        micro_decision="keep",
    )
    record_integrate_verdict(
        macro_cycle=3,
        integration_id="int-1",
        kernel_id="k001",
        decision="KEEP",
        gain_pct=6.0,
    )
    recorder.finish(tput_after=1060.0)

    row = _kernel_events(tmp_path)[0]["ext"]["attempts"][0]
    assert row["outcome"] == "adopted"
    assert row["backend"] == "triton"


def test_an_integrate_verdict_lands_after_the_visit_has_closed(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(
        run_id="attempt-7",
        kernel_id="k001",
        status="success",
        micro_decision="keep",
    )
    recorder.finish(tput_after=1000.0)

    # Closing wrote the event's own fragment; nothing is assembled until export.
    assert _kernel_events(tmp_path)[0]["ext"]["integrate"] == []

    record_integrate_verdict(
        macro_cycle=3,
        integration_id="int-1",
        kernel_id="k001",
        decision="KEEP",
        status="succeeded",
        attempt_count=2,
        fault_count=1,
        gain_pct=4.5,
        accuracy_pass=True,
        patch_path="patches/k001.diff",
        target_file="vllm/attention.py",
        settled_at="2026-09-02T00:20:00",
    )

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["integrate"] == [
        {
            "integration_id": "int-1",
            "kernel_id": "k001",
            "decision": "KEEP",
            "status": "succeeded",
            "attempt_count": 2,
            "fault_count": 1,
            "gain_pct": 4.5,
            "accuracy_pass": True,
            "validation_tier": None,
            "patch_path": "patches/k001.diff",
            "target_file": "vllm/attention.py",
            "error_class": None,
            "rejected_reason": None,
            "retryable": False,
            "settled_at": "2026-09-02T00:20:00",
            "settled_in_macro_cycle": 3,
            "extra_server_args": None,
            "basis": None,
            "alignment_status": None,
            "gain_attributed": None,
        }
    ]
    # And the same verdict reaches the row it ruled on, joined by kernel_id.
    assert ext["attempts"][0]["e2e"] == {
        "integrated": True,
        "e2e_gain_pct": 4.5,
        "validated": True,
        "decision": "KEEP",
        "patch_path": "patches/k001.diff",
        "target_file": "vllm/attention.py",
    }


def test_the_standing_verdict_wins_and_the_best_gain_survives_a_later_fault(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success")
    recorder.finish(tput_after=1000.0)

    record_integrate_verdict(
        macro_cycle=3,
        integration_id="int-1",
        kernel_id="k001",
        decision="KEEP",
        gain_pct=6.0,
        settled_at="2026-09-02T00:10:00",
    )
    record_integrate_verdict(
        macro_cycle=3,
        integration_id="int-2",
        kernel_id="k001",
        decision="REVERT",
        gain_pct=None,
        rejected_reason="fault_attempts_exhausted_2",
        settled_at="2026-09-02T00:30:00",
    )

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert [row["integration_id"] for row in ext["integrate"]] == ["int-1", "int-2"]
    e2e = ext["attempts"][0]["e2e"]
    assert e2e["decision"] == "REVERT"
    assert e2e["integrated"] is False
    assert e2e["e2e_gain_pct"] == 6.0


def test_a_kernel_that_was_never_gated_has_no_e2e_block(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success")
    recorder.finish(tput_after=1000.0)

    assert _kernel_events(tmp_path)[0]["ext"]["attempts"][0]["e2e"] is None


def test_a_verdict_with_no_event_to_belong_to_is_dropped(tmp_path):
    recorder = _forge_recorder()
    recorder.finish(tput_after=1000.0)

    record_integrate_verdict(macro_cycle=9, integration_id="int-1", kernel_id="k001", decision="KEEP")

    assert [event["id"] for event in _kernel_events(tmp_path)] == [kernel_event_id(3)]


def test_a_fusion_is_settled_by_the_integration_it_queued(tmp_path):
    """Fusion produces no kernel of its own, so the gate is found by integration id."""
    recorder = _forge_recorder()
    recorder.record_fusion_run(
        run_id="fusion-1",
        status="success",
        pattern="rmsnorm+silu",
        applied=True,
        gain_pct=2.0,
        micro_decision="keep",
        integrate_ref="int-fusion-1",
    )
    record_integrate_verdict(
        macro_cycle=3,
        integration_id="int-fusion-1",
        kernel_id="",
        decision="REVERT",
    )
    recorder.finish(tput_after=1000.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    row = ext["attempts"][0]
    assert row["source_kind"] == "fusion"
    assert row["outcome"] == "rejected"
    assert row["settled_by"] == "integrate"
    assert ext["outcome"]["delivered"] == []


def test_each_gain_names_the_anchor_it_was_measured_against(tmp_path):
    """One visit, two anchors, two different true answers.

    The stage moved throughput 10%, and the session stands 37.5% above where it
    started. A single gain beside three anchors left which pair it came from to
    be inferred.
    """
    recorder = _forge_recorder()
    recorder.finish(tput_after=1100.0)

    throughput = _kernel_events(tmp_path)[0]["ext"]["outcome"]["throughput"]
    assert throughput["before"] == 1000.0
    assert throughput["after"] == 1100.0
    assert throughput["session_baseline"] == 800.0
    assert throughput["gain_pct"] == 10.0
    assert throughput["session_gain_pct"] == 37.5


def test_a_gain_is_not_claimed_against_an_anchor_that_was_never_measured(tmp_path):
    """A percentage of a missing or zero anchor states nothing, so it is absent."""
    recorder = make_kernel_recorder(macro_cycle=4, route=ROUTE_FORGE)
    assert recorder is not None
    recorder.begin(tput_before=0.0)
    recorder.finish(tput_after=1100.0)

    throughput = _kernel_events(tmp_path)[0]["ext"]["outcome"]["throughput"]
    assert throughput["after"] == 1100.0
    assert throughput["gain_pct"] is None
    assert throughput["session_gain_pct"] is None


def test_geak_handoff_and_product_do_not_share_the_accepted_flags_name(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_handoff(
        {
            "schema_version": 2,
            "accepted_flags": "--enable-torch-compile",
            "accepted_env": "A=1",
            "raw_baseline_tput": 800.0,
            "orchestrator_best_tput_same_config": 900.0,
            "baseline_env_spec": {"layers": []},
        }
    )
    recorder.record_geak_product(
        accepted_flags=["--enable-torch-compile", "--attention-backend=aiter"],
        accepted_envs={"A": "1", "B": "2"},
        cfg_hash="deadbeef",
        final_overlay="/s/geak/overlay",
        final_overlay_digest="d1",
    )
    recorder.finish(tput_after=900.0)

    geak = _kernel_events(tmp_path)[0]["ext"]["geak"]
    assert geak["handoff"]["baseline_flags"] == "--enable-torch-compile"
    assert geak["handoff"]["baseline_env_spec_present"] is True
    assert "accepted_flags" not in geak["handoff"]
    assert geak["product"]["accepted_flags"] == [
        "--enable-torch-compile",
        "--attention-backend=aiter",
    ]
    assert geak["product"]["cfg_hash"] == "deadbeef"


def test_the_handoff_records_which_cards_geak_was_given(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_handoff(
        {
            "schema_version": 3,
            "gpu_ids": "0,1",
            "gpu_ids_space": "logical",
            "gpu_pin": {"ids": "6,7", "var": "ROCR_VISIBLE_DEVICES", "source": "recipe"},
        }
    )
    recorder.finish(tput_after=900.0)

    handoff = _kernel_events(tmp_path)[0]["ext"]["geak"]["handoff"]
    assert handoff["gpu_ids"] == "0,1"
    assert handoff["gpu_ids_space"] == "logical"
    assert handoff["gpu_pin"]["ids"] == "6,7"
    assert handoff["gpu_pin"]["var"] == "ROCR_VISIBLE_DEVICES"


def test_a_handoff_without_a_pin_is_not_read_as_a_pin_to_card_zero(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_handoff({"schema_version": 3, "gpu_ids": "0,1"})
    recorder.finish(tput_after=900.0)

    handoff = _kernel_events(tmp_path)[0]["ext"]["geak"]["handoff"]
    assert handoff["schema_version"] == 3
    assert handoff["gpu_pin"] == {}


def test_geak_env_selections_are_recorded_as_their_own_source(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_claim(
        {"self_reported_gain_pct": 7.5, "geak_status": "ok"},
        specs=[
            {"short_name": "dsa_sparse_attn", "op_kind": "attn", "e2e_delta_pct": 4.0, "lane": "headQueue"},
            {
                "short_name": "moe_grouped_gemm_ck2stage",
                "kind": "env",
                "op_kind": "moe",
                "e2e_delta_pct": 3.5,
                "lane": "kernelQueue",
            },
        ],
    )
    recorder.finish(tput_after=900.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    claim = ext["geak"]["claim"]
    assert claim["verified"] is False
    assert claim["kernels_optimized"] == 1
    assert claim["authored_kernels"][0]["lane"] == "headQueue"
    assert claim["authored_kernels"][0]["name_source"] == "symbol"
    assert claim["env_selections"] == [
        {
            "selection": "moe_grouped_gemm_ck2stage",
            "op_kind": "moe",
            "lane": "kernelQueue",
            "e2e_delta_pct": 3.5,
        }
    ]
    # An env selection is never a kernel attempt, so it gets an attempt row of
    # its own rather than being folded onto one.
    kinds = {row["source_kind"] for row in ext["attempts"]}
    assert kinds == {SOURCE_GEAK_AUTHORED_KERNEL, SOURCE_GEAK_ENV_SELECTION}
    assert all(row["route"] == ROUTE_GEAK for row in ext["attempts"])
    assert ext["outcome"]["delivered"] == []


def test_geak_attempts_carry_what_it_tried_not_only_what_it_kept(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_attempts(
        {
            "discovery_runs": [{"source": "bypass", "status": "success", "hot_kernels": [1, 2, 3]}],
            "kernels": [
                {
                    "kernel_id": "k001",
                    "dispatch": {"dispatched": True, "backends": ["triton"]},
                    "backend_result": {"backend": "triton", "status": "ok", "speedup": 1.3},
                    "e2e": {"integrated": True, "e2e_gain_pct": 4.0},
                },
                {
                    "kernel_id": "k002",
                    "dispatch": {"dispatched": False, "skip_reason": "non_reusable_kernel"},
                },
            ],
        }
    )
    recorder.finish(tput_after=900.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["geak"]["discovery_runs"][0]["hot_kernel_count"] == 3
    rows = {row["kernel_id"]: row for row in ext["attempts"]}
    assert set(rows) == {"k001", "k002"}
    assert rows["k001"]["status"] == "ok"
    assert rows["k001"]["backend"] == "triton"
    assert rows["k002"]["dispatched"] is False
    assert rows["k002"]["skip_reason"] == "non_reusable_kernel"
    # A kernel GEAK never dispatched produced nothing to gate.
    assert rows["k002"]["outcome"] == "rejected"
    assert rows["k002"]["settled_by"] == "lane"


def test_an_attempt_carries_what_made_the_kernel_worth_trying(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_attempts(
        {
            "kernels": [
                {
                    "kernel_id": "k001",
                    "name": "fused_moe_kernel",
                    "op_kind": "moe",
                    "gpu_pct": 31.5,
                    "micro_speedup": 1.8,
                    "dispatch": {"dispatched": True, "backends": ["triton"]},
                }
            ]
        }
    )
    recorder.finish(tput_after=900.0)

    kernel = _kernel_events(tmp_path)[0]["ext"]["attempts"][0]
    assert kernel["name"] == "fused_moe_kernel"
    assert kernel["speedup"] == 1.8
    # What made it worth trying is GEAK's own framing, not a shared fact.
    assert kernel["detail"]["op_kind"] == "moe"
    assert kernel["detail"]["gpu_pct"] == 31.5


def test_the_isolated_speedup_is_read_from_wherever_the_journey_states_it(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_attempts(
        {
            "kernels": [
                {
                    "kernel_id": "k001",
                    "dispatch": {"dispatched": True},
                    "backend_result": {"backend": "triton", "verification": {"micro_speedup": 2.4}},
                }
            ]
        }
    )
    recorder.finish(tput_after=900.0)

    assert _kernel_events(tmp_path)[0]["ext"]["attempts"][0]["speedup"] == 2.4


def test_an_attempt_with_no_op_kind_of_its_own_takes_the_dispatchs(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_attempts(
        {"kernels": [{"kernel_id": "k001", "dispatch": {"dispatched": True, "op_kind": "attn"}}]}
    )
    recorder.finish(tput_after=900.0)

    assert _kernel_events(tmp_path)[0]["ext"]["attempts"][0]["detail"]["op_kind"] == "attn"


def test_the_latency_geak_measured_survives_a_run_that_kept_nothing(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_measurement(
        {
            "metric_basis": "output_throughput",
            "bench_client": "agentx",
            "ttft_ms": 42.5,
            "tpot_ms": 7.25,
            "output_parity": True,
            "status": "no_gain",
        }
    )
    recorder.finish(tput_after=900.0)

    claim = _kernel_events(tmp_path)[0]["ext"]["geak"]["claim"]
    assert claim["ttft_mean_ms"] == 42.5
    assert claim["tpot_mean_ms"] == 7.25
    assert claim["metric_basis"] == "output_throughput"
    assert claim["bench_client"] == "agentx"
    assert claim["output_parity"] is True


def test_the_measurement_lands_beside_the_claim_it_belongs_to(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_claim({"self_reported_gain_pct": 7.5, "geak_status": "ok"})
    recorder.record_geak_measurement({"ttft_ms": 42.5, "output_parity": False})
    recorder.finish(tput_after=1000.0)

    claim = _kernel_events(tmp_path)[0]["ext"]["geak"]["claim"]
    assert claim["self_reported_gain_pct"] == 7.5
    assert claim["ttft_mean_ms"] == 42.5
    assert claim["output_parity"] is False
    assert claim["verified"] is False


def test_a_collapsed_alias_twin_leaves_its_name_on_the_survivor(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_claim(
        {"geak_status": "ok"},
        specs=[
            {
                "short_name": "fused_moe",
                "op_kind": "moe",
                "e2e_delta_pct": 4.0,
                "lane": "kernelQueue",
                "alias_collapsed": True,
                "aliases": ["cand_c0_triton"],
            }
        ],
    )
    recorder.finish(tput_after=1000.0)

    authored = _kernel_events(tmp_path)[0]["ext"]["geak"]["claim"]["authored_kernels"][0]
    assert authored["alias_collapsed"] is True
    assert authored["aliases"] == ["cand_c0_triton"]


def test_a_geak_attempt_names_its_route_rather_than_engaging_forge(tmp_path):
    recorder = _geak_recorder()
    recorder.record_geak_attempts({"kernels": [{"kernel_id": "k001", "dispatch": {"dispatched": True}}]})
    recorder.finish(tput_after=900.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    # Both routes share the attempts array, so the row states which produced it
    # and the route block of the route that never ran stays absent.
    assert ext["forge"] is None
    assert [row["route"] for row in ext["attempts"]] == [ROUTE_GEAK]


def test_two_entries_in_one_session_assemble_from_their_own_rows_only(tmp_path):
    first = _geak_recorder(macro_cycle=1)
    first.record_geak_attempts({"kernels": [{"kernel_id": "k001", "dispatch": {"dispatched": True}}]})
    first.finish(tput_after=900.0)

    second = _geak_recorder(macro_cycle=2)
    second.record_geak_attempts(
        {
            "kernels": [
                {"kernel_id": "k002", "dispatch": {"dispatched": True}},
                {"kernel_id": "k003", "dispatch": {"dispatched": True}},
            ]
        }
    )
    second.finish(tput_after=900.0)

    events = {event["id"]: event for event in _kernel_events(tmp_path)}
    assert set(events) == {kernel_event_id(1), kernel_event_id(2)}
    assert [row["kernel_id"] for row in events[kernel_event_id(1)]["ext"]["attempts"]] == ["k001"]
    assert sorted(row["kernel_id"] for row in events[kernel_event_id(2)]["ext"]["attempts"]) == ["k002", "k003"]


def test_a_phase_crash_closes_the_event_naming_the_stage(tmp_path):
    recorder = _forge_recorder()
    recorder.enter_stage("forge_fusion")
    recorder.finish_crashed(RuntimeError("boom"))

    event = _kernel_events(tmp_path)[0]
    assert event["status"] == "failed"
    assert event["end_time"]
    outcome = event["ext"]["outcome"]
    assert outcome["verdict"] == "failed"
    assert outcome["failed_stage"] == "forge_fusion"
    assert outcome["error_class"] == "RuntimeError"
    assert "boom" in outcome["reason"]


def test_a_crash_after_a_measured_win_still_closes_failed(tmp_path):
    """A visit that raised is not a good one, whatever it delivered first.

    The verdict is derived from the instruments, and they hold a real win
    here, so deriving the status from the verdict too had a crashed visit
    closing ``succeeded``. The two answer different questions: the verdict
    keeps the win, because it was measured; the status reports that the visit
    did not get to the end, which is the phase's to say and only it knows.
    """
    recorder = _forge_recorder()
    recorder.enter_stage("forge_fusion")
    recorder.record_kernel_rewrite(
        run_id="attempt-7",
        kernel_id="k001",
        status="success",
        micro_decision="keep",
        speedup=1.4,
    )
    recorder.finish_crashed(RuntimeError("boom"))

    event = _kernel_events(tmp_path)[0]
    assert event["status"] == "failed"
    outcome = event["ext"]["outcome"]
    assert outcome["verdict"] == "improved"
    assert outcome["failed_stage"] == "forge_fusion"
    assert outcome["error_class"] == "RuntimeError"


def test_reassembling_a_crashed_visit_reaches_the_same_status(tmp_path):
    """The close is a fact on the rows, so a later re-read cannot soften it."""
    recorder = _forge_recorder()
    recorder.enter_stage("forge_fusion")
    recorder.record_kernel_rewrite(
        run_id="attempt-7",
        kernel_id="k001",
        status="success",
        micro_decision="keep",
        speedup=1.4,
    )
    recorder.finish_crashed(RuntimeError("boom"))

    event_id = _kernel_events(tmp_path)[0]["id"]
    _ext, status = assemble_kernel_ext(kernel_event_parts(), event=event_id)
    assert status == "failed"


def test_a_fault_mid_visit_is_named_without_ending_the_visit(tmp_path):
    """A raising tick is swallowed by the loop, so the visit outlives it.

    Closing here would cut short a visit that survived; recording nothing left
    the event closing clean, with the exception readable nowhere.
    """
    recorder = _forge_recorder()
    recorder.enter_stage("forge_fusion")
    recorder.record_fault(stage="tick_body", error_class="RuntimeError", message="boom")
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="revert")
    recorder.finish(tput_after=1000.0)

    event = _kernel_events(tmp_path)[0]
    outcome = event["ext"]["outcome"]
    assert outcome["failed_stage"] == "tick_body"
    assert outcome["error_class"] == "RuntimeError"
    assert outcome["verdict"] == "failed"
    assert "boom" in outcome["reason"]
    # The visit went on recording after the fault: the row is still here.
    assert [row["kernel_id"] for row in event["ext"]["attempts"]] == ["k001"]


def test_a_measured_win_outranks_a_fault_the_visit_survived(tmp_path):
    """A visit can blow up somewhere and still hand the stack a measured gain.

    Calling that failed would bury the delivery; the fault stays readable
    beside the verdict, which is what says the win was not come by cleanly.
    """
    recorder = _forge_recorder()
    recorder.record_fault(stage="reactor:optimizer", error_class="TimeoutError", message="turn never returned")
    recorder.record_kernel_rewrite(
        run_id="attempt-7",
        kernel_id="k001",
        status="success",
        micro_decision="keep",
        speedup=1.4,
    )
    recorder.finish(tput_after=1400.0)

    outcome = _kernel_events(tmp_path)[0]["ext"]["outcome"]
    assert outcome["verdict"] == "improved"
    assert outcome["error_class"] == "TimeoutError"
    assert outcome["failed_stage"] == "reactor:optimizer"
    assert [entry["ref"] for entry in outcome["delivered"]] == ["attempt-7"]


def test_the_first_fault_is_the_one_kept(tmp_path):
    """What follows a crash is generally its consequence, not a second cause."""
    recorder = _forge_recorder()
    recorder.record_fault(stage="tick_body", error_class="RuntimeError", message="the cause")
    recorder.record_fault(stage="advance_phase", error_class="KeyError", message="the consequence")
    recorder.finish(tput_after=1000.0)

    outcome = _kernel_events(tmp_path)[0]["ext"]["outcome"]
    assert outcome["error_class"] == "RuntimeError"
    assert outcome["failed_stage"] == "tick_body"


def test_the_phase_naming_the_stage_it_died_in_outranks_an_earlier_fault(tmp_path):
    recorder = _forge_recorder()
    recorder.record_fault(stage="tick_body", error_class="RuntimeError", message="survived this one")
    recorder.finish_failed(stage="geak_handoff", error_class="invalid_env_spec", message="cannot serialize")

    outcome = _kernel_events(tmp_path)[0]["ext"]["outcome"]
    assert outcome["error_class"] == "invalid_env_spec"
    assert outcome["failed_stage"] == "geak_handoff"


def test_an_unfinished_visit_states_no_verdict(tmp_path):
    """Finalize rebuilds a killed visit from its rows, which reached no verdict.

    The rows so far may add up to a gain, but nobody concluded the visit, and
    a verdict here read as a conclusion that contradicted the status beside it.
    """
    recorder = _forge_recorder()
    recorder.record_fault(stage="tick_body", error_class="RuntimeError", message="boom")
    recorder.record_kernel_rewrite(
        run_id="attempt-7",
        kernel_id="k001",
        status="success",
        micro_decision="keep",
        speedup=1.4,
    )

    ext, status = assemble_kernel_ext(kernel_event_parts(), event=recorder.event_id)
    assert status == "running"
    assert ext["outcome"]["verdict"] == ""
    assert ext["outcome"]["reason"] == ""
    # The fault is not a conclusion, so it survives: it is the most useful
    # thing a reader can learn about a visit that was killed.
    assert ext["outcome"]["error_class"] == "RuntimeError"
    assert ext["outcome"]["failed_stage"] == "tick_body"


def test_the_trace_analyze_run_records_the_only_legal_kernel_id_source(tmp_path):
    recorder = _forge_recorder()
    recorder.record_trace_analyze_run(
        run_id="ta-1",
        trigger="pre_run_optimization",
        status="ok",
        requested_by="orchestration",
        trace_input="/s/traces",
        top_k=10,
        result={
            "analysis_meta": {"route": "agent", "tool": "tracelens"},
            "hot_kernels_top15": [{"name": "aten::mm", "gpu_pct": 10.0}],
            "candidates_path": "/s/candidates.json",
        },
        snapshot={"roofline_snapshot_id": 5, "reusable_native_kernel_ids": ["k001", "k002"]},
    )
    recorder.finish(tput_after=1000.0)

    run = _kernel_events(tmp_path)[0]["ext"]["forge"]["trace_analyze_runs"][0]
    assert run["route"] == "agent"
    assert run["tool"] == "tracelens"
    assert run["roofline_snapshot_id"] == 5
    assert run["reusable_native_kernel_ids"] == ["k001", "k002"]
    assert run["artifacts"]["candidates_path"] == "/s/candidates.json"
    assert run["hot_kernels"]["count"] == 1


def _geak_rebench(recorder, attempt_id: str, decision: str | None, **overrides: Any) -> None:
    fields: dict[str, Any] = {
        "attempt_id": attempt_id,
        "idempotency_key": attempt_id,
        "task_id": attempt_id,
        "base_tput": 900.0,
        "measured_tput": 950.0,
        "decision": decision,
        "status": "settled",
    }
    fields.update(overrides)
    recorder.record_geak_rebench_attempt(max_attempts=4, **fields)


def _geak_with_one_acceptance():
    recorder = _geak_recorder()
    recorder.record_geak_claim(
        {"self_reported_gain_pct": 5.0},
        specs=[{"short_name": "dsa_sparse_attn", "op_kind": "attn", "e2e_delta_pct": 4.0, "lane": "headQueue"}],
    )
    return recorder


@pytest.mark.parametrize(
    "decisions",
    [(REBENCH_VALIDATED, REBENCH_NO_PROMOTE), (REBENCH_NO_PROMOTE, REBENCH_VALIDATED)],
)
def test_conflicting_geak_rebenches_leave_the_candidate_pending(tmp_path, decisions):
    recorder = _geak_with_one_acceptance()
    _geak_rebench(recorder, "geak-rb-1", decisions[0])
    _geak_rebench(recorder, "geak-rb-2", decisions[1])
    recorder.finish(tput_after=950.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["geak"]["rebench"]["conflicting_decisions"] == sorted(set(decisions))
    assert ext["geak"]["rebench"]["settled_against"] == ""
    assert ext["outcome"]["delivered"] == []
    assert ext["outcome"]["verdict"] == "no_improvement"
    row = ext["attempts"][0]
    assert row["outcome"] == "needs_review"
    assert row["unsettled_reason"] == "rebench_conflict"


def test_agreeing_geak_rebenches_still_settle_the_candidate(tmp_path):
    recorder = _geak_with_one_acceptance()
    _geak_rebench(recorder, "geak-rb-1", REBENCH_VALIDATED)
    _geak_rebench(recorder, "geak-rb-2", REBENCH_VALIDATED)
    recorder.finish(tput_after=950.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert "conflicting_decisions" not in ext["geak"]["rebench"]
    assert ext["geak"]["rebench"]["attempts_used"] == 2
    assert [row["ref"] for row in ext["outcome"]["delivered"]] == ["dsa_sparse_attn"]
    assert ext["attempts"][0]["settled_by"] == "rebench"


def test_the_rebench_ledger_is_shared_rather_than_nested_in_the_route(tmp_path):
    recorder = _geak_with_one_acceptance()
    _geak_rebench(recorder, "geak-rb-1", REBENCH_VALIDATED)
    recorder.finish(tput_after=950.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert [row["attempt_id"] for row in ext["rebench"]] == ["geak-rb-1"]
    # The attempt points at the row that settled it, across the layer boundary.
    assert ext["attempts"][0]["rebench_ref"] == "geak-rb-1"
    assert "attempts" not in ext["geak"]["rebench"]


def test_a_visit_that_produced_nothing_ran_without_improving_anything(tmp_path):
    """Producing no candidate is a kind of no-improvement, not a kind of skip.

    The visit ran; how much it did is ``attempts`` being empty, which is a
    fact the array already states without the verdict encoding it twice.
    """
    recorder = _forge_recorder()
    recorder.finish(tput_after=1000.0)

    outcome = _kernel_events(tmp_path)[0]["ext"]["outcome"]
    assert outcome["verdict"] == "no_improvement"
    assert outcome["reason"] == "the visit produced no candidate"
    assert _kernel_events(tmp_path)[0]["status"] == "succeeded"


def test_the_verdict_comes_from_the_gate_and_not_from_the_caller(tmp_path):
    recorder = _forge_recorder()
    _forge_rewrite_with_gate(recorder, decision="KEEP", gain_pct=10.0)
    recorder.finish(tput_after=1100.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["outcome"]["verdict"] == "improved"
    # The gate measured the patch end to end, which outranks the lane's timing.
    assert ext["attempts"][0]["gain_pct"] == 10.0


def _forge_rewrite_with_gate(recorder, **verdict: Any) -> None:
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="keep")
    record_integrate_verdict(macro_cycle=3, integration_id="int-1", kernel_id="k001", **verdict)


def test_a_gate_concluding_against_the_candidate_still_succeeds_the_entry(tmp_path):
    recorder = _forge_recorder()
    _forge_rewrite_with_gate(recorder, decision="REVERT")
    recorder.finish(tput_after=1000.0)

    assert _kernel_events(tmp_path)[0]["status"] == "succeeded"


def test_a_visit_that_found_no_win_still_closes_as_a_completed_visit(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="revert")
    recorder.finish(tput_after=1000.0)

    assert _kernel_events(tmp_path)[0]["status"] == "succeeded"


def test_an_entry_whose_every_attempt_failed_is_failed(tmp_path):
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="failed")
    recorder.finish(tput_after=1000.0)

    event = _kernel_events(tmp_path)[0]
    assert event["status"] == "failed"
    assert event["ext"]["attempts"][0]["outcome"] == "failed"
    assert event["ext"]["outcome"]["verdict"] == "failed"


def test_a_failed_attempt_is_not_read_as_a_rejection(tmp_path):
    """A candidate that never built was never judged; the two are not the same."""
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(
        run_id="attempt-7",
        kernel_id="k001",
        status="failed",
        error_class="compile_error",
        failure_reason="triton compile failed",
    )
    recorder.record_kernel_rewrite(run_id="attempt-8", kernel_id="k002", status="success", micro_decision="revert")
    recorder.finish(tput_after=1000.0)

    rows = {row["attempt_id"]: row for row in _kernel_events(tmp_path)[0]["ext"]["attempts"]}
    assert rows["attempt-7"]["outcome"] == "failed"
    assert rows["attempt-7"]["error_class"] == "compile_error"
    assert rows["attempt-7"]["failure_reason"] == "triton compile failed"
    assert rows["attempt-8"]["outcome"] == "rejected"


def test_a_fallback_verdict_is_inconclusive_rather_than_a_rejection(tmp_path):
    recorder = _geak_with_one_acceptance()
    _geak_rebench(
        recorder,
        "geak-rb-1",
        REBENCH_FALLBACK,
        measured_tput=None,
        engagement={"config_matched": True, "overlay_loaded": False},
    )
    recorder.finish(tput_after=900.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["rebench"][0]["engagement"]["overlay_loaded"] is False
    assert ext["outcome"]["delivered"] == []
    assert ext["attempts"][0]["outcome"] == "needs_review"
    assert ext["attempts"][0]["unsettled_reason"] == "rebench_inconclusive"


def test_recording_the_same_rebench_twice_updates_one_row(tmp_path):
    recorder = _geak_with_one_acceptance()
    _geak_rebench(recorder, "geak-rb-1", None, measured_tput=None, status="dispatched", settled_at=None)
    _geak_rebench(recorder, "geak-rb-1", REBENCH_VALIDATED, settled_at="2026-09-02T00:09:00")
    recorder.finish(tput_after=950.0)

    ledger = _kernel_events(tmp_path)[0]["ext"]["rebench"]
    assert len(ledger) == 1
    assert ledger[0]["decision"] == REBENCH_VALIDATED
    assert ledger[0]["settled_at"] == "2026-09-02T00:09:00"


def test_attempts_are_ordered_by_when_they_started(tmp_path):
    recorder = _forge_recorder()
    recorder.record_fusion_run(run_id="late", status="success", started_at="2026-09-02T00:05:00")
    recorder.record_fusion_run(run_id="early", status="success", started_at="2026-09-02T00:01:00", applied=True)
    recorder.finish(tput_after=1000.0)

    rows = _kernel_events(tmp_path)[0]["ext"]["attempts"]
    assert [row["attempt_id"] for row in rows] == ["early", "late"]
    assert rows[0]["detail"]["applied"] is True


def _record_inline_reprofile(recorder, *, task_id: str = "rp-1") -> None:
    """Dispatch a re-profile inline, the way the entry hook does."""
    from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
    from hyperloom.inference_optimizer.breakdown.recorder.roofline_event import make_roofline_recorder

    recorder.record_reprofile(ran=True, task_kind="roofline", trigger="gain", task_id=task_id)
    inline = make_roofline_recorder(
        make_sink(recorder.event_id, producer="orchestrator"),
        task_id=task_id,
        task_kind="roofline",
        reason="kernel_entry_reprofile",
        owns_event=False,
    )
    assert inline is not None
    return inline


def test_an_inline_reprofile_survives_the_active_close(tmp_path):
    recorder = _forge_recorder()
    _record_inline_reprofile(recorder)
    recorder.finish(tput_after=1050.0)

    reprofile = _kernel_events(tmp_path)[0]["ext"]["forge"]["reprofile"]
    assert reprofile["ran"] is True
    assert reprofile["run"]["task_id"] == "rp-1"


def test_a_recovered_kernel_event_keeps_its_inline_reprofile(tmp_path):
    from hyperloom.inference_optimizer.breakdown.recorder.event_finalize import finalize_events

    recorder = _forge_recorder()
    _record_inline_reprofile(recorder)
    # No finish(): the session was killed after the re-profile was dispatched.

    assert finalize_events(tmp_path) == [recorder.event_id]

    event = _kernel_events(tmp_path)[0]
    assert event["status"] == "interrupted"
    reprofile = event["ext"]["forge"]["reprofile"]
    assert reprofile["ran"] is True
    assert reprofile["run"]["task_id"] == "rp-1"


def test_the_visits_own_measurements_are_held_by_the_event_that_asked_for_them(tmp_path):
    """A candidate A/B or stack validation measures through the baseline executor as a sub-step.

    Recording it here rather than as a top-level baseline event is what keeps the gate's verdict
    and the measurement behind it in one place.
    """
    from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import make_baseline_recorder
    from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink

    recorder = _forge_recorder()
    guest = make_baseline_recorder(
        make_sink(recorder.event_id, producer="orchestrator"),
        task_id="integrate-k001",
        task_kind="baseline",
        reason="candidate_ab",
        framework="sglang",
        owns_event=False,
    )
    assert guest is not None
    guest.finish({"status": "succeeded", "output_throughput": 1100.0})
    recorder.finish(tput_after=1100.0)

    events = _kernel_events(tmp_path)
    assert len(events) == 1
    measurements = events[0]["ext"]["measurements"]
    assert [row["task_id"] for row in measurements] == ["integrate-k001"]
    assert measurements[0]["status"] == "succeeded"
    assert measurements[0]["request"]["reason"] == "candidate_ab"


def test_a_measurement_that_lands_after_the_visit_closed_is_still_published(tmp_path):
    """Stack validation and the integrate drain both measure on SWEEP entry, after the visit closed.

    The verdict they settle republishes the event, and that is what carries the measurement in.
    """
    from hyperloom.inference_optimizer.breakdown.recorder.baseline_event import make_baseline_recorder
    from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import record_integrate_verdict

    recorder = _forge_recorder()
    recorder.finish(tput_after=1000.0)
    assert _kernel_events(tmp_path)[0]["ext"]["measurements"] == []

    guest = make_baseline_recorder(
        make_sink(recorder.event_id, producer="orchestrator"),
        task_id="integrate-stack-k001+k004",
        task_kind="baseline",
        owns_event=False,
    )
    assert guest is not None
    guest.finish({"status": "succeeded", "output_throughput": 1100.0})
    record_integrate_verdict(
        macro_cycle=3,
        integration_id="stack-k001+k004",
        kernel_id="k001+k004",
        decision="KEEP",
        status="ok",
    )

    event = _kernel_events(tmp_path)[0]
    assert [row["task_id"] for row in event["ext"]["measurements"]] == ["integrate-stack-k001+k004"]
    assert [row["integration_id"] for row in event["ext"]["integrate"]] == ["stack-k001+k004"]


def test_discovered_kernels_carry_profiling_fields(tmp_path):
    recorder = _forge_recorder()
    recorder.record_discovered_kernels(
        {
            "roofline_snapshot_id": 9,
            "reusable_native_kernel_ids": ["k-hot"],
            "hot_kernels_top15": [
                {
                    "kernel_id": "k-hot",
                    "name": "gemm",
                    "gpu_pct": 61.0,
                    "duration_us": 900.0,
                    "call_count": 120,
                    "kernel_category": "gemm",
                    "bound_type": "compute",
                    "arithmetic_intensity": 180.0,
                    "bandwidth_utilization_pct": 12.0,
                    "compute_utilization_pct": 77.0,
                    "recommended_backends": ["triton"],
                    "recommended_actions": ["tile"],
                    "reusable_native_kernel": True,
                    "source_file": "ops/gemm.py",
                },
                {
                    "kernel_id": "k-cheap",
                    "name": "rms_norm",
                    "gpu_pct": 4.0,
                    "duration_us": 12.0,
                    "efficiency_percent": 3.0,
                    "bound_type": "memory",
                },
            ],
        },
        provenance="entry_snapshot",
    )
    recorder.finish(tput_after=1000.0)

    forge = _kernel_events(tmp_path)[0]["ext"]["forge"]
    assert len(forge["discovered_kernels"]) == 2
    hot = forge["discovered_kernels"][0]
    assert hot["kernel_id"] == "k-hot"
    assert hot["duration_us"] == 900.0
    assert hot["call_count"] == 120
    assert hot["bound_type"] == "compute"
    assert hot["bandwidth_util_pct"] == 12.0
    assert hot["recommended_actions"] == ["tile"]
    assert forge["recommended_kernels"][0]["kernel_id"] == "k-hot"


def test_trace_analyze_run_also_records_discovered_kernels(tmp_path):
    recorder = _forge_recorder()
    recorder.record_trace_analyze_run(
        run_id="ta-2",
        trigger="pre_run_optimization",
        status="ok",
        result={"analysis_meta": {"route": "agent", "tool": "tracelens"}},
        snapshot={
            "roofline_snapshot_id": 6,
            "hot_kernels_top15": [
                {
                    "kernel_id": "k001",
                    "name": "aten::mm",
                    "gpu_pct": 10.0,
                    "duration_us": 50.0,
                    "call_count": 8,
                    "bound_type": "memory",
                }
            ],
            "reusable_native_kernel_ids": ["k001"],
        },
    )
    recorder.finish(tput_after=1000.0)

    discovered = _kernel_events(tmp_path)[0]["ext"]["forge"]["discovered_kernels"]
    assert discovered[0]["kernel_id"] == "k001"
    assert discovered[0]["duration_us"] == 50.0
    assert discovered[0]["provenance"] == "trace_analyze_run"


def test_record_backend_versions_and_timeline_mirrors_each_attempt(tmp_path):
    from hyperloom.inference_optimizer.breakdown.recorder.instrument import (
        record_backend_versions_and_timeline,
    )

    recorder = _forge_recorder()
    record_backend_versions_and_timeline(
        tmp_path,
        {
            "kernel_id": "k001",
            "run_id": "forge-run-1",
            "status": "ok",
            "attempts": [
                {
                    "attempt_id": "a1",
                    "backend": "triton",
                    "status": "failed",
                    "micro_speedup": 0.9,
                    "decision": "REVERT",
                },
                {
                    "attempt_id": "a2",
                    "backend": "aiter",
                    "status": "success",
                    "micro_speedup": 1.4,
                    "decision": "KEEP",
                },
            ],
            "verification": {
                "best_attempt_id": "a2",
                "micro_speedup": 1.4,
                "best_artifact_path": "/w/out.py",
            },
            "proposal": {"decision": "KEEP"},
        },
    )
    recorder.finish(tput_after=1000.0)

    rows = _kernel_events(tmp_path)[0]["ext"]["attempts"]
    assert [row["attempt_id"] for row in rows] == ["a1", "a2"]
    assert rows[1]["backend"] == "aiter"
    assert rows[1]["speedup"] == 1.4
    assert rows[0]["micro_decision"] == "REVERT"


def test_controller_integrations_are_wired_to_forge_rewrites(tmp_path):
    recorder = _forge_recorder()
    phase = _phase_with_recorder(tmp_path, recorder)
    phase._record_kernel_rewrite_controller_timeline(
        {
            "macro_cycle": 3,
            "run_id": "controller-3",
            "integration": {
                "results": [
                    {
                        "operator_id": "kernel:forge:rmsnorm:sglang:v1:aiter:mi355x",
                        "status": "kept",
                        "gain_pct": 4.0,
                    },
                    {
                        "operator_id": "kernel:forge:rope:sglang:v1:triton:mi355x",
                        "status": "reverted_e2e_failed",
                        "reason": "no gain",
                    },
                ]
            },
        }
    )
    recorder.finish(tput_after=1000.0)

    rows = _kernel_events(tmp_path)[0]["ext"]["attempts"]
    assert [row["name"] for row in rows] == ["rmsnorm", "rope"]
    assert rows[0]["backend"] == "forge"
    assert rows[0]["speedup"] == 1.04
    assert rows[1]["micro_decision"] == "REVERT"
    assert rows[1]["failure_reason"] == "no gain"
    # The integration status is this route's failure taxonomy, and the row
    # carries no separate class: without stamping it, ``error_class`` was the
    # one field a reader could ask both other routes but never forge.
    assert rows[0]["error_class"] is None
    assert rows[1]["error_class"] == "reverted_e2e_failed"


def test_gemm_and_fusion_handlers_record_their_forge_lanes(tmp_path):
    recorder = _forge_recorder()
    phase = _phase_with_recorder(tmp_path, recorder)
    phase._record_gemm_tuning_timeline(
        {
            "task_id": "gemm-1",
            "status": "ok",
            "backend": "forge",
            "decision": "KEEP",
            "best_speedup": 1.08,
            "shape_capture": {"shape_count": 12},
            "tuners_run": [{"tuner": "dense", "improved_shapes": 3}],
            "tuned_file": "/tmp/tuned.csv",
        }
    )
    phase._record_fusion_timeline(
        {
            "fusion_run_id": "fusion-1",
            "status": "ok",
            "kept": True,
            "decision": "KEEP",
            "pattern": "rmsnorm+silu",
            "target_module": "decoder",
            "patch_path": "/tmp/fusion.patch",
        }
    )
    recorder.finish(tput_after=1000.0)

    rows = {row["source_kind"]: row for row in _kernel_events(tmp_path)[0]["ext"]["attempts"]}
    assert rows["gemm_tuning"]["detail"]["shapes_total"] == 12
    assert rows["gemm_tuning"]["detail"]["shapes_tuned"] == 3
    assert rows["gemm_tuning"]["detail"]["tuner"] == "dense"
    assert rows["fusion"]["detail"]["pattern"] == "rmsnorm+silu"
    assert rows["fusion"]["detail"]["applied"] is True


def test_geak_runner_outcome_is_wired_to_geak_delegation(tmp_path):
    recorder = _geak_recorder()
    phase = _phase_with_recorder(tmp_path, recorder)
    phase._record_geak_delegation_timeline(
        {
            "status": "ok",
            "returncode": 0,
            "eval_dir": "/tmp/geak/eval",
            "report_path": "/tmp/geak/report.json",
            "stages_reached": ["discover", "optimize"],
        },
        handoff={"exp_root": "/tmp/geak"},
        started_at="2026-09-02T00:00:00+00:00",
        duration_sec=12.5,
        runner_timeout_sec=300,
        kill_timeout_sec=360,
    )
    recorder.finish(tput_after=900.0)

    delegation = _kernel_events(tmp_path)[0]["ext"]["geak"]["delegation"]
    assert delegation["runner_status"] == "ok"
    assert delegation["returncode"] == 0
    assert delegation["duration_sec"] == 12.5
    assert delegation["stages_reached"] == ["discover", "optimize"]


def test_finish_records_stack_delta(tmp_path):
    recorder = _forge_recorder()
    recorder.finish(
        tput_after=1100.0,
        stack_added=[{"action": "integrate_patch", "variant_name": "fused_gemm"}],
        stack_removed=[],
    )

    delta = _kernel_events(tmp_path)[0]["ext"]["outcome"]["stack_delta"]
    assert delta["added"][0]["variant_name"] == "fused_gemm"
    assert delta["removed"] == []


def test_a_dropped_candidates_verdict_survives_the_slot_it_was_read_from(tmp_path):
    recorder = _geak_with_one_acceptance()
    _geak_rebench(recorder, "geak-rb-1", REBENCH_NO_PROMOTE)
    recorder.record_geak_rebench_conclusion(
        final_status="fallback_failed",
        final_error_class="subprocess_nonzero",
        final_error="GEAK harness fallback did not validate",
    )
    recorder.finish(tput_after=900.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    rebench = ext["geak"]["rebench"]
    assert rebench["final_status"] == "fallback_failed"
    assert rebench["final_error_class"] == "subprocess_nonzero"
    assert rebench["final_error"] == "GEAK harness fallback did not validate"
    # The attempt that produced it is still there: the conclusion merges onto
    # the event rather than replacing what the attempts said.
    assert [row["attempt_id"] for row in ext["rebench"]] == ["geak-rb-1"]


def test_a_verdict_with_no_failure_behind_it_carries_no_error(tmp_path):
    recorder = _geak_with_one_acceptance()
    _geak_rebench(recorder, "geak-rb-1", REBENCH_NO_PROMOTE)
    recorder.record_geak_rebench_conclusion(final_status="no_promote")
    recorder.finish(tput_after=900.0)

    rebench = _kernel_events(tmp_path)[0]["ext"]["geak"]["rebench"]
    assert rebench["final_status"] == "no_promote"
    # Absent rather than blank: nothing failed, so there is no class to name.
    assert rebench["final_error_class"] is None
    assert rebench["final_error"] is None
