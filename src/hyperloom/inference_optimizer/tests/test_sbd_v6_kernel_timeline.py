# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``kernel`` event: what it records and what it settles."""

from __future__ import annotations

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
)
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope


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


def test_one_event_per_entry_with_macro_cycle_at_the_top(tmp_path):
    """The entry identifier belongs to the event, not to either route block."""
    recorder = _forge_recorder()
    recorder.finish(verdict="no_gain", status="succeeded", tput_after=1000.0)

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
    """A live session shows the entry as soon as it opens, not when it ends."""
    _forge_recorder()

    events = _kernel_events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == "running"
    assert "end_time" not in events[0]


def test_the_stage_in_flight_is_recoverable_from_the_rows_alone(tmp_path):
    """A session killed mid-phase is closed out of its fragments by finalize.

    The timeline entry keeps the status and stage it opened with, because
    rewriting it on every sub-step is the write amplification the fragments
    exist to avoid. What the kill has to leave behind is enough to *assemble*
    the stage, and that is a property of the rows rather than of the entry.
    """
    recorder = _forge_recorder()
    recorder.enter_stage("gemm_tuning")

    assert _kernel_events(tmp_path)[0]["status"] == "running"
    ext, status = assemble_kernel_ext(kernel_event_parts(), event=recorder.event_id)
    assert ext["in_flight_stage"] == "gemm_tuning"
    assert status == "skipped"


def test_a_keep_without_a_rebench_cannot_read_as_adopted(tmp_path):
    """The candidate layer's own verdict never settles end-to-end adoption."""
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(
        run_id="attempt-7",
        kernel_id="k001",
        status="success",
        micro_decision="keep",
        speedup=1.4,
        e2e={"integrated": True, "e2e_gain_pct": 6.0, "decision": "KEEP"},
    )
    recorder.finish(verdict="needs_review", status="succeeded", tput_after=1000.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    row = ext["forge"]["lanes"]["kernel_rewrites"][0]
    assert row["micro_decision"] == "keep"
    assert row["outcome"] == "needs_review"
    assert row["rebench_ref"] is None
    assert ext["outcome"]["adopted"] == []
    assert ext["outcome"]["pending_review"] == [
        {"source_kind": SOURCE_KERNEL_REWRITE, "ref": "attempt-7", "why": "no_rebench"}
    ]
    assert ext["outcome"]["by_source"][SOURCE_KERNEL_REWRITE] == {
        "attempted": 1,
        "adopted": 0,
        "needs_review": 1,
        "rejected": 0,
    }


def test_a_validated_rebench_is_what_promotes_a_candidate(tmp_path):
    """Adoption is carried by the rebench row the lane points at."""
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(
        run_id="attempt-7",
        kernel_id="k001",
        status="success",
        adopted_backend="triton",
        micro_decision="keep",
        rebench_ref="rb-1",
        e2e={"integrated": True, "e2e_gain_pct": 6.0},
    )
    recorder.record_rebench_attempt(
        attempt_id="rb-1",
        source_kind=SOURCE_KERNEL_REWRITE,
        source_ref="attempt-7",
        idempotency_key="rebench-c3",
        task_id="task-9",
        dispatched_at="2026-09-02T00:01:00",
        settled_at="2026-09-02T00:09:00",
        base_tput=1000.0,
        measured_tput=1060.0,
        decision=REBENCH_VALIDATED,
        decision_reason="beat current_best",
        status="settled",
        engagement={"config_matched": True, "overlay_loaded": True},
    )
    recorder.finish(verdict="adopted", status="succeeded", tput_after=1060.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["forge"]["lanes"]["kernel_rewrites"][0]["outcome"] == "adopted"
    assert ext["forge"]["lanes"]["kernel_rewrites"][0]["adopted_backend"] == "triton"
    assert ext["outcome"]["adopted"] == [
        {
            "source_kind": SOURCE_KERNEL_REWRITE,
            "ref": "attempt-7",
            "gain_pct": 6.0,
            "rebench_ref": "rb-1",
        }
    ]
    ledger = ext["forge"]["rebench_ledger"][0]
    assert ledger["delta_pct"] == 6.0
    assert ledger["engagement"]["config_matched"] is True
    assert ledger["engagement"]["overlay_loaded"] is True


def test_a_rebench_that_did_not_promote_rejects_the_candidate(tmp_path):
    """Measured truthfully and did not beat current_best is a rejection."""
    recorder = _forge_recorder()
    recorder.record_fusion_run(
        run_id="fusion-1",
        status="success",
        pattern="rmsnorm+silu",
        applied=True,
        gain_pct=2.0,
        micro_decision="keep",
        rebench_ref="rb-2",
    )
    recorder.record_rebench_attempt(
        attempt_id="rb-2",
        source_kind="fusion",
        source_ref="fusion-1",
        base_tput=1000.0,
        measured_tput=995.0,
        decision=REBENCH_NO_PROMOTE,
        status="settled",
    )
    recorder.finish(verdict="no_gain", status="succeeded", tput_after=1000.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["forge"]["lanes"]["fusion_runs"][0]["outcome"] == "rejected"
    assert ext["outcome"]["adopted"] == []
    assert ext["outcome"]["by_source"]["fusion"]["rejected"] == 1


def test_net_gain_is_measured_against_the_previous_stage_not_the_baseline(tmp_path):
    """The entry's own gain is relative to what it was handed."""
    recorder = _forge_recorder()
    recorder.finish(verdict="adopted", status="succeeded", tput_after=1100.0)

    outcome = _kernel_events(tmp_path)[0]["ext"]["outcome"]
    assert outcome["tput_before"] == 1000.0
    assert outcome["session_baseline_tput"] == 800.0
    assert outcome["net_gain_pct"] == 10.0


def test_geak_handoff_and_product_do_not_share_the_accepted_flags_name(tmp_path):
    """The starting config and the produced config must stay distinguishable."""
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
    recorder.finish(verdict="not_run", status="succeeded", tput_after=900.0)

    geak = _kernel_events(tmp_path)[0]["ext"]["geak"]
    assert geak["handoff"]["baseline_flags"] == "--enable-torch-compile"
    assert geak["handoff"]["baseline_env_spec_present"] is True
    assert "accepted_flags" not in geak["handoff"]
    assert geak["product"]["accepted_flags"] == [
        "--enable-torch-compile",
        "--attention-backend=aiter",
    ]
    assert geak["product"]["cfg_hash"] == "deadbeef"


def test_geak_env_selections_are_recorded_as_their_own_source(tmp_path):
    """A library or env acceptance authors no kernel but still carries gain."""
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
    recorder.finish(verdict="inconclusive", status="succeeded", tput_after=900.0)

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
            "outcome": "needs_review",
        }
    ]
    by_source = ext["outcome"]["by_source"]
    assert by_source[SOURCE_GEAK_AUTHORED_KERNEL]["attempted"] == 1
    assert by_source[SOURCE_GEAK_ENV_SELECTION]["attempted"] == 1
    assert by_source[SOURCE_GEAK_ENV_SELECTION]["needs_review"] == 1
    assert ext["outcome"]["adopted"] == []


def test_geak_attempts_carry_what_it_tried_not_only_what_it_kept(tmp_path):
    """The journey names every kernel considered, including the rejects."""
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
    recorder.finish(verdict="inconclusive", status="succeeded", tput_after=900.0)

    attempts = _kernel_events(tmp_path)[0]["ext"]["geak"]["attempts"]
    assert attempts["discovery_runs"][0]["hot_kernel_count"] == 3
    assert attempts["counts"] == {
        "discovered": 2,
        "dispatched": 1,
        "skipped": 1,
        "backend_ok": 1,
        "backend_fail": 0,
        "integrated": 1,
    }
    assert attempts["kernels"][1]["skip_reason"] == "non_reusable_kernel"
    assert attempts["kernels"][1]["backend_result"] is None


def test_geak_attempts_are_not_appended_to_the_forge_lane(tmp_path):
    """The two routes stay isolated even though the rows are shaped alike."""
    recorder = _geak_recorder()
    recorder.record_geak_attempts({"kernels": [{"kernel_id": "k001", "dispatch": {"dispatched": True}}]})
    recorder.finish(verdict="inconclusive", status="succeeded", tput_after=900.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["forge"] is None
    assert len(ext["geak"]["attempts"]["kernels"]) == 1


def test_two_entries_in_one_session_assemble_from_their_own_rows_only(tmp_path):
    """The event id is what separates two entries sharing one fragment spool."""
    first = _geak_recorder(macro_cycle=1)
    first.record_geak_attempts({"kernels": [{"kernel_id": "k001", "dispatch": {"dispatched": True}}]})
    first.finish(verdict="inconclusive", tput_after=900.0)

    second = _geak_recorder(macro_cycle=2)
    second.record_geak_attempts(
        {
            "kernels": [
                {"kernel_id": "k002", "dispatch": {"dispatched": True}},
                {"kernel_id": "k003", "dispatch": {"dispatched": True}},
            ]
        }
    )
    second.finish(verdict="inconclusive", tput_after=900.0)

    events = {event["id"]: event for event in _kernel_events(tmp_path)}
    assert set(events) == {kernel_event_id(1), kernel_event_id(2)}
    assert [row["kernel_id"] for row in events[kernel_event_id(1)]["ext"]["geak"]["attempts"]["kernels"]] == ["k001"]
    assert [row["kernel_id"] for row in events[kernel_event_id(2)]["ext"]["geak"]["attempts"]["kernels"]] == [
        "k002",
        "k003",
    ]


def test_a_phase_crash_closes_the_event_naming_the_stage(tmp_path):
    """A raised phase is distinguishable from a session killed mid-phase."""
    recorder = _forge_recorder()
    recorder.enter_stage("forge_fusion")
    recorder.finish_crashed(RuntimeError("boom"))

    event = _kernel_events(tmp_path)[0]
    assert event["status"] == "failed"
    assert event["end_time"]
    assert event["ext"]["failure"]["phase"] == "forge_fusion"
    assert event["ext"]["failure"]["error_class"] == "RuntimeError"


def test_the_trace_analyze_run_records_the_only_legal_kernel_id_source(tmp_path):
    """A phase-requested analysis has no roofline event to carry its evidence."""
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
    recorder.finish(verdict="no_gain", status="succeeded", tput_after=1000.0)

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
    """Two settled verdicts that disagree must not collapse to the newest one.

    GEAK rebenches the same candidate up to its per-cycle ceiling, so taking
    the last verdict would let a validation after a rejection read as an
    adoption -- and would make the outcome depend on dispatch order.
    """
    recorder = _geak_with_one_acceptance()
    _geak_rebench(recorder, "geak-rb-1", decisions[0])
    _geak_rebench(recorder, "geak-rb-2", decisions[1])
    recorder.finish(verdict="adopted", tput_after=950.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["geak"]["rebench"]["conflicting_decisions"] == sorted(set(decisions))
    assert ext["outcome"]["adopted"] == []
    assert ext["outcome"]["pending_review"] == [
        {"source_kind": SOURCE_GEAK_AUTHORED_KERNEL, "ref": "dsa_sparse_attn", "why": "rebench_conflict"}
    ]


def test_agreeing_geak_rebenches_still_settle_the_candidate(tmp_path):
    """Repeated attempts are the normal path; only disagreement is a conflict."""
    recorder = _geak_with_one_acceptance()
    _geak_rebench(recorder, "geak-rb-1", REBENCH_VALIDATED)
    _geak_rebench(recorder, "geak-rb-2", REBENCH_VALIDATED)
    recorder.finish(verdict="adopted", tput_after=950.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert "conflicting_decisions" not in ext["geak"]["rebench"]
    assert ext["geak"]["rebench"]["attempts_used"] == 2
    assert [row["ref"] for row in ext["outcome"]["adopted"]] == ["dsa_sparse_attn"]


def test_the_geak_ledger_stays_out_of_the_forge_ledger(tmp_path):
    """Which ledger asked for a re-measurement is recorded, not inferred."""
    recorder = _geak_with_one_acceptance()
    _geak_rebench(recorder, "geak-rb-1", REBENCH_VALIDATED)
    recorder.record_rebench_attempt(
        attempt_id="forge-rb-1",
        source_kind=SOURCE_KERNEL_REWRITE,
        source_ref="attempt-7",
        base_tput=900.0,
        measured_tput=910.0,
        decision=REBENCH_VALIDATED,
        status="settled",
    )
    recorder.finish(verdict="adopted", tput_after=950.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert [row["attempt_id"] for row in ext["geak"]["rebench"]["attempts"]] == ["geak-rb-1"]
    assert [row["attempt_id"] for row in ext["forge"]["rebench_ledger"]] == ["forge-rb-1"]


def test_the_verdict_stays_unstamped_when_nothing_was_adopted(tmp_path):
    """An entry that adopted nothing concluded nothing about a candidate."""
    recorder = _forge_recorder()
    recorder.record_kernel_rewrite(run_id="attempt-7", kernel_id="k001", status="success", micro_decision="keep")
    recorder.finish(verdict="adopted", tput_after=1000.0)

    assert _kernel_events(tmp_path)[0]["ext"]["outcome"]["verdict"] is None


def test_the_verdict_comes_from_the_rebench_and_not_from_the_caller(tmp_path):
    """The phase seam names no verdict, so assembly must supply one.

    The seam is the only production caller and it has nothing to judge with;
    it used to default to "adopted", which stamped the word on any entry whose
    rows happened to settle that way and on any that did not.
    """
    recorder = _forge_recorder()
    _forge_rewrite_with_rebench(recorder, decision=REBENCH_VALIDATED, measured_tput=1100.0, status="settled")
    recorder.finish(tput_after=1100.0)

    assert _kernel_events(tmp_path)[0]["ext"]["outcome"]["verdict"] == "adopted"


def test_an_unnamed_verdict_stays_unstamped_when_the_rebench_rejected(tmp_path):
    recorder = _forge_recorder()
    _forge_rewrite_with_rebench(recorder, decision=REBENCH_NO_PROMOTE, measured_tput=1000.0, status="settled")
    recorder.finish(tput_after=1000.0)

    assert _kernel_events(tmp_path)[0]["ext"]["outcome"]["verdict"] is None


def _forge_rewrite_with_rebench(recorder, **rebench: Any) -> None:
    recorder.record_kernel_rewrite(
        run_id="attempt-7", kernel_id="k001", status="success", micro_decision="keep", rebench_ref="rb-1"
    )
    recorder.record_rebench_attempt(
        attempt_id="rb-1",
        source_kind=SOURCE_KERNEL_REWRITE,
        source_ref="attempt-7",
        base_tput=1000.0,
        **rebench,
    )


def test_a_rebench_concluding_against_the_candidate_still_succeeds_the_entry(tmp_path):
    """The distinction that matters is measuring, not what was measured."""
    recorder = _forge_recorder()
    _forge_rewrite_with_rebench(recorder, measured_tput=995.0, decision=REBENCH_NO_PROMOTE, status="settled")
    recorder.finish(verdict="no_gain", tput_after=1000.0)

    assert _kernel_events(tmp_path)[0]["status"] == "succeeded"


def test_a_candidate_whose_rebench_measured_nothing_leaves_the_entry_degraded(tmp_path):
    """The work happened and nothing settled it, which is not a clean run."""
    recorder = _forge_recorder()
    _forge_rewrite_with_rebench(recorder, measured_tput=None, decision=None, status="skipped")
    recorder.finish(verdict="needs_review", tput_after=1000.0)

    assert _kernel_events(tmp_path)[0]["status"] == "degraded"


def test_an_entry_with_no_candidate_and_no_measurement_is_skipped(tmp_path):
    """Nothing was produced, so there is nothing to call degraded."""
    recorder = _forge_recorder()
    recorder.finish(verdict="not_run", tput_after=1000.0)

    assert _kernel_events(tmp_path)[0]["status"] == "skipped"


def test_an_entry_whose_every_rebench_faulted_is_failed(tmp_path):
    """A faulted rebench is a different operational state from a skipped one."""
    recorder = _forge_recorder()
    _forge_rewrite_with_rebench(recorder, measured_tput=None, decision=None, status="failed")
    recorder.finish(verdict="needs_review", tput_after=1000.0)

    assert _kernel_events(tmp_path)[0]["status"] == "failed"


def test_a_fallback_verdict_is_inconclusive_rather_than_a_rejection(tmp_path):
    """The config under test did not engage, so nothing was measured about it.

    The projection this replaces read ``fallback_failed`` as a rejection, which
    credits the rebench with a conclusion it did not reach: a run whose overlay
    dropped out measured plain flags, so it says nothing about the candidate
    either way. ``no_material`` and ``no_promote`` do reject, because both were
    measured with the configuration engaged.
    """
    recorder = _geak_with_one_acceptance()
    _geak_rebench(
        recorder,
        "geak-rb-1",
        REBENCH_FALLBACK,
        measured_tput=None,
        engagement={"config_matched": True, "overlay_loaded": False},
    )
    recorder.finish(verdict="adopted", tput_after=900.0)

    ext = _kernel_events(tmp_path)[0]["ext"]
    assert ext["geak"]["rebench"]["attempts"][0]["engagement"]["overlay_loaded"] is False
    assert ext["outcome"]["adopted"] == []
    assert ext["outcome"]["pending_review"][0]["why"] == "rebench_inconclusive"
    assert ext["outcome"]["by_source"][SOURCE_GEAK_AUTHORED_KERNEL]["rejected"] == 0


def test_a_lane_that_produced_nothing_stays_empty(tmp_path):
    """An absent lane is an empty list, not an empty shell of a row."""
    recorder = _forge_recorder()
    recorder.record_fusion_run(run_id="fusion-1", status="success", applied=True)
    recorder.finish(verdict="needs_review", tput_after=1000.0)

    lanes = _kernel_events(tmp_path)[0]["ext"]["forge"]["lanes"]
    assert len(lanes["fusion_runs"]) == 1
    assert lanes["kernel_rewrites"] == []
    assert lanes["gemm_tuning_runs"] == []
    assert lanes["collective_runs"] == []


def test_recording_the_same_rebench_twice_updates_one_row(tmp_path):
    """A dispatch and its later verdict describe one attempt, not two."""
    recorder = _forge_recorder()
    for decision in (None, REBENCH_VALIDATED):
        recorder.record_rebench_attempt(
            attempt_id="rb-1",
            source_kind=SOURCE_KERNEL_REWRITE,
            source_ref="attempt-7",
            idempotency_key="rebench-c3",
            task_id="task-9",
            dispatched_at="2026-09-02T00:01:00",
            settled_at=None if decision is None else "2026-09-02T00:09:00",
            base_tput=1000.0,
            measured_tput=None if decision is None else 1050.0,
            decision=decision,
            status="dispatched" if decision is None else "settled",
        )
    recorder.finish(verdict="adopted", status="succeeded", tput_after=1050.0)

    ledger = _kernel_events(tmp_path)[0]["ext"]["forge"]["rebench_ledger"]
    assert len(ledger) == 1
    assert ledger[0]["decision"] == REBENCH_VALIDATED
    assert ledger[0]["settled_at"] == "2026-09-02T00:09:00"


def test_lane_rows_are_ordered_by_when_they_started(tmp_path):
    """Row order comes from the rows, not from the order they were written in.

    Writing the later run first is exactly the case a fragment envelope's
    ``seq`` or ``ts`` would sort backwards, since both name the write rather
    than the run.
    """
    recorder = _forge_recorder()
    recorder.record_fusion_run(run_id="late", status="success", started_at="2026-09-02T00:05:00")
    recorder.record_fusion_run(run_id="early", status="success", started_at="2026-09-02T00:01:00", applied=True)
    recorder.finish(verdict="needs_review", tput_after=1000.0)

    lanes = _kernel_events(tmp_path)[0]["ext"]["forge"]["lanes"]["fusion_runs"]
    assert [row["run_id"] for row in lanes] == ["early", "late"]
    assert lanes[0]["applied"] is True


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
    """The baseline for the recovery test below: closing normally keeps the run."""
    recorder = _forge_recorder()
    _record_inline_reprofile(recorder)
    recorder.finish(verdict="adopted", tput_after=1050.0)

    reprofile = _kernel_events(tmp_path)[0]["ext"]["forge"]["reprofile"]
    assert reprofile["ran"] is True
    assert reprofile["run"]["task_id"] == "rp-1"


def test_a_recovered_kernel_event_keeps_its_inline_reprofile(tmp_path):
    """Recovery must read the roofline sections the active close reads.

    An inline re-profile records into the kernel event, so its rows live in the
    ``roofline_*`` sections under this event's id. Recovering from only the
    ``kernel_*`` sections gave the assembler nothing to fold into
    ``forge.reprofile.run`` -- an interrupted entry is exactly the case that
    cannot be reconstructed any other way.
    """
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
