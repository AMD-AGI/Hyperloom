# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``conc_sweep`` event.

The event this replaces was projected out of ``conc_sweep_summary.json``, and
the projection's limit is what most of these tests pin: that file is a result
document, so a ladder that came out short looked the same whether the budget
refused its rungs, the server would not boot at them, or the benchmark failed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.assembler import conc_sweep_event_parts
from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import (
    ARM_BASELINE,
    ARM_OPTIMIZED,
    GRID_MODE_DEFAULT,
    GRID_REQUESTED,
    PRODUCER,
    STAGE_BOOT,
    STAGE_BOOT_ATTEMPT,
    STAGE_BUDGET_SKIP,
    STAGE_REUSE,
    STAGE_SERVER_RESTART,
    STRATEGY_REFUSED,
    STRATEGY_SERVER_RESTART,
    STRATEGY_SINGLE_SERVER,
    assemble_conc_sweep_ext,
    conc_sweep_event_id,
    make_conc_sweep_recorder,
)
from hyperloom.inference_optimizer.breakdown.recorder.event_finalize import finalize_events
from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
from hyperloom.inference_optimizer.breakdown.recorder.event_timeline import EVENT_STATUS_INTERRUPTED
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "conc_sweep"]


def _ext(session_dir: Path, index: int = 0) -> dict[str, Any]:
    return _events(session_dir)[index]["ext"]


def _recorder(
    *,
    task_id: str = "cs-1",
    phase: str = "sweep",
    macro_cycle: int = 2,
    reason: str = "phase_entry",
    params: dict[str, Any] | None = None,
):
    """Build a recorder the way a dispatched sweep gets one."""
    recorder = make_conc_sweep_recorder(
        make_sink(conc_sweep_event_id(phase=phase, macro_cycle=macro_cycle), producer=PRODUCER),
        task_id=task_id,
        task_kind="conc_sweep",
        reason=reason,
        params=params or {"concs": [64, 32], "variant_timeout_sec": 1800, "total_budget_sec": 9000},
    )
    assert recorder is not None
    return recorder


def _point(conc: int, *, arm: str = ARM_OPTIMIZED, status: str = "succeeded", **overrides: Any) -> dict[str, Any]:
    """One flattened rung, shaped the way the sweep's own report writes it."""
    point: dict[str, Any] = {
        "arm": arm,
        "conc": conc,
        "status": status,
        "output_throughput": 100.0 * conc,
        "request_throughput": 1.5 * conc,
        "total_token_throughput": 200.0 * conc,
        "input_throughput": 100.0 * conc,
        "e2e_norm_intvty_p90": 42.0,
        "tpot_p90_ms": 13.5,
        "ttft_mean_ms": 130.0,
        "e2el_mean_ms": 4000.0,
        "duration_seconds": 240.0,
        "completed_requests": conc * 5,
        "error": None,
        "error_class": None,
        "killed_overtime": False,
        "estimated_output_throughput": None,
        "workspace": f"/w/{arm}_conc{conc}",
        "report_path": f"/w/{arm}_conc{conc}/benchmark_report.json",
    }
    point.update(overrides)
    return point


def _plan(recorder) -> None:
    """Record the plan block a real sweep records before it runs an arm."""
    recorder.record_workload(session_id="sess-7", isl=1024, osl=1024, tp=8, benchmark_mode="synthetic")
    recorder.record_anchor(
        baseline_tput=1000.0,
        anchor_tput=1400.0,
        tp=8,
        variant_id="env_tuning_3",
        action="explore",
        extra_server_args="--enable-torch-compile",
        extra_envs={"SGLANG_X": "1"},
    )
    recorder.record_plan(
        concs_requested=[64, 32],
        concs_ordered=[64, 32],
        grid_source=GRID_REQUESTED,
        num_prompts_factor=5,
        variant_timeout_sec=1800,
        arms_order=[ARM_OPTIMIZED, ARM_BASELINE],
    )
    recorder.record_budget(
        declared_total_sec=9000,
        granted_total_sec=21600,
        rung_cost_sec=10800.0,
        raised=True,
        gate_active=True,
        deadline=1780000000.0,
        session_soft_deadline_sec=10800.0,
    )
    recorder.record_environment(
        sweep_task_id="conc_sweep_20260907T020304Z",
        workspace="/runs/conc_sweep/conc_sweep_20260907T020304Z",
        model_path="/models/glm",
        gpu_type="mi300x",
        base_config_path="/w/conc_sweep_base.with_envs.yaml",
        report_json_path="/reports/conc_sweep_summary.json",
        report_csv_path="/reports/conc_sweep_raw.csv",
    )


def _final(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "succeeded",
        "elapsed_sec": 812.5,
        "budget_exhausted": False,
        "summary": {
            "metric": "output_throughput",
            "successful_pairs": 2,
            "failed_pairs": 0,
            "best_conc": 64,
            "best_speedup": 1.4,
            "median_speedup": 1.35,
            "mean_speedup": 1.35,
        },
        "report_json_path": "/reports/conc_sweep_summary.json",
        "report_csv_path": "/reports/conc_sweep_raw.csv",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# The window the projection could not report
# ---------------------------------------------------------------------------
def test_the_event_opens_when_the_sweep_starts_rather_than_when_it_ends(_bound_session):
    recorder = _recorder()
    open_events = [event for event in read_timeline_events(_bound_session) if event.get("type") == "conc_sweep"]
    assert [event["status"] for event in open_events] == ["running"]
    assert open_events[0]["start_time"]

    _plan(recorder)
    recorder.finish(_final())

    event = _events(_bound_session)[0]
    assert event["status"] == "succeeded"
    assert event["start_time"] and event["end_time"]
    assert event["start_time"] <= event["end_time"]


def test_a_sweep_killed_mid_ladder_still_reaches_the_timeline(_bound_session):
    recorder = _recorder()
    _plan(recorder)
    recorder.open_arm(ARM_OPTIMIZED, extra_server_args="--x", extra_envs={})
    recorder.record_variant(ARM_OPTIMIZED, stage=STAGE_BOOT, conc=64, point=_point(64))

    assert finalize_events(_bound_session) == ["sweep:2:conc_sweep"]
    event = _events(_bound_session)[0]
    assert event["status"] == EVENT_STATUS_INTERRUPTED
    assert [point["conc"] for point in event["ext"]["arms"][ARM_OPTIMIZED]["points"]] == [64]


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
def test_the_ladder_says_whether_it_was_chosen_or_handed_over(_bound_session):
    recorder = _recorder()
    _plan(recorder)
    recorder.finish(_final())

    plan = _ext(_bound_session)["plan"]
    assert plan["grid_source"] == GRID_REQUESTED
    assert plan["concs_requested"] == [64, 32]
    assert plan["concs_ordered"] == [64, 32]
    assert plan["num_prompts_factor"] == 5
    # The optimized arm runs first, which is why a budget that runs out takes
    # the baseline arm with it rather than the arm being asked about.
    assert plan["arms_order"] == [ARM_OPTIMIZED, ARM_BASELINE]


def test_a_ladder_the_mode_picked_says_so(_bound_session):
    recorder = _recorder()
    recorder.record_plan(concs_requested=[1, 4, 8], concs_ordered=[8, 4, 1], grid_source=GRID_MODE_DEFAULT)
    recorder.finish(_final())

    assert _ext(_bound_session)["plan"]["grid_source"] == GRID_MODE_DEFAULT


def test_the_event_names_the_configuration_it_was_asked_to_compare(_bound_session):
    recorder = _recorder()
    _plan(recorder)
    recorder.finish(_final())

    anchor = _ext(_bound_session)["input_anchor"]
    assert anchor["base_variant_id"] == "env_tuning_3"
    assert anchor["base_action"] == "explore"
    assert anchor["anchor_tput"] == 1400.0
    assert anchor["baseline_tput"] == 1000.0
    # 1400 over eight cards, computed where both numbers are in hand.
    assert anchor["input_throughput_tok_s_per_gpu"] == 175.0
    assert anchor["extra_server_args"] == "--enable-torch-compile"
    assert anchor["extra_envs"] == {"SGLANG_X": "1"}


def test_an_anchor_with_no_throughput_reports_no_per_gpu_number(_bound_session):
    recorder = _recorder()
    recorder.record_anchor(baseline_tput=1000.0, anchor_tput=None, tp=8)
    recorder.finish(_final())

    anchor = _ext(_bound_session)["input_anchor"]
    assert anchor["anchor_tput"] is None
    assert anchor["input_throughput_tok_s_per_gpu"] is None


def test_the_budget_reports_both_the_number_asked_for_and_the_number_spent(_bound_session):
    recorder = _recorder()
    _plan(recorder)
    recorder.finish(_final())

    budget = _ext(_bound_session)["budget"]
    assert budget["declared_total_sec"] == 9000
    assert budget["granted_total_sec"] == 21600
    assert budget["raised"] is True
    assert budget["rung_cost_sec"] == 10800.0
    assert budget["gate_active"] is True
    assert budget["session_soft_deadline_sec"] == 10800.0


def test_the_sweeps_own_task_id_is_recorded_apart_from_the_dispatched_one(_bound_session):
    recorder = _recorder(task_id="cs-9")
    _plan(recorder)
    recorder.finish(_final())

    ext = _ext(_bound_session)
    assert ext["request"]["task_id"] == "cs-9"
    assert ext["environment"]["sweep_task_id"] == "conc_sweep_20260907T020304Z"
    assert ext["environment"]["model_path"] == "/models/glm"
    assert ext["environment"]["gpu_type"] == "mi300x"
    assert ext["environment"]["base_config_path"] == "/w/conc_sweep_base.with_envs.yaml"


def test_the_dispatch_says_why_the_sweep_ran(_bound_session):
    recorder = _recorder(reason="phase_entry")
    recorder.finish(_final())

    request = _ext(_bound_session)["request"]
    assert request["reason"] == "phase_entry"
    assert request["task_kind"] == "conc_sweep"
    assert request["requested_total_budget_sec"] == 9000


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def test_an_arm_says_how_it_ran_its_ladder(_bound_session):
    recorder = _recorder()
    _plan(recorder)
    recorder.open_arm(ARM_OPTIMIZED, extra_server_args="--x", extra_envs={"SGLANG_X": "1"})
    recorder.record_arm_grid(
        ARM_OPTIMIZED,
        rungs=[
            {"name": "optimized_conc64", "conc": 64, "num_prompts": 320},
            {"name": "optimized_conc32", "conc": 32, "num_prompts": 160},
        ],
    )
    recorder.record_arm_strategy(
        ARM_OPTIMIZED,
        strategy=STRATEGY_SINGLE_SERVER,
        lifecycle_eligible=True,
        lifecycle_reason="supported",
        port=8888,
        framework="sglang",
        serving_lease_held=True,
    )
    recorder.finish_arm(ARM_OPTIMIZED, status="succeeded")
    recorder.finish(_final())

    arm = _ext(_bound_session)["arms"][ARM_OPTIMIZED]
    assert arm["strategy"] == STRATEGY_SINGLE_SERVER
    assert arm["lifecycle"] == {"eligible": True, "reason": "supported", "port": 8888, "framework": "sglang"}
    assert arm["serving_lease_held"] is True
    assert arm["status"] == "succeeded"
    # The load each rung carries is derived from its concurrency and was never
    # written down, so the run could not be reproduced from the report alone.
    assert arm["grid"] == [
        {"name": "optimized_conc64", "conc": 64, "num_prompts": 320},
        {"name": "optimized_conc32", "conc": 32, "num_prompts": 160},
    ]


def test_an_arm_that_restarted_per_rung_says_why(_bound_session):
    recorder = _recorder()
    recorder.open_arm(ARM_BASELINE, extra_server_args="", extra_envs={})
    recorder.record_arm_strategy(
        ARM_BASELINE,
        strategy=STRATEGY_SERVER_RESTART,
        reason="framework_not_lifecycle_eligible",
        lifecycle_eligible=False,
        lifecycle_reason="no_server_lifecycle",
        port=8888,
        framework="vllm",
        serving_lease_held=False,
    )
    recorder.finish_arm(ARM_BASELINE, status="succeeded")
    recorder.finish(_final())

    arm = _ext(_bound_session)["arms"][ARM_BASELINE]
    assert arm["strategy"] == STRATEGY_SERVER_RESTART
    assert arm["strategy_reason"] == "framework_not_lifecycle_eligible"
    assert arm["lifecycle"]["eligible"] is False


def test_the_baseline_arms_empty_args_are_an_answer_rather_than_a_gap(_bound_session):
    recorder = _recorder()
    recorder.open_arm(ARM_BASELINE, extra_server_args="", extra_envs={})
    recorder.finish_arm(ARM_BASELINE, status="succeeded")
    recorder.finish(_final())

    arm = _ext(_bound_session)["arms"][ARM_BASELINE]
    assert arm["extra_server_args"] == ""
    # The projection carried no envs at all for either arm.
    assert arm["extra_envs"] == {}


def test_an_arm_the_budget_turned_away_records_the_gate_that_did_it(_bound_session):
    recorder = _recorder()
    recorder.record_arm_refused(ARM_BASELINE, reason="total_budget_exhausted", remaining_sec=0.0)
    recorder.finish(_final(status="failed", budget_exhausted=True))

    arm = _ext(_bound_session)["arms"][ARM_BASELINE]
    assert arm["strategy"] == STRATEGY_REFUSED
    assert arm["status"] == "skipped"
    assert arm["refused"] == {"reason": "total_budget_exhausted", "remaining_sec": 0.0}
    assert arm["points"] == []


def test_an_arm_that_never_ran_reports_no_strategy_rather_than_a_wrong_one(_bound_session):
    recorder = _recorder()
    recorder.open_arm(ARM_OPTIMIZED, extra_server_args="--x", extra_envs={})
    recorder.finish_arm(ARM_OPTIMIZED, status="succeeded")
    recorder.finish(_final())

    baseline = _ext(_bound_session)["arms"][ARM_BASELINE]
    assert baseline["arm"] == ARM_BASELINE
    assert baseline["strategy"] == ""
    assert baseline["refused"] is None
    assert baseline["points"] == []


# ---------------------------------------------------------------------------
# Rungs
# ---------------------------------------------------------------------------
def test_a_rung_carries_the_agentic_axis_the_projection_dropped(_bound_session):
    recorder = _recorder()
    recorder.open_arm(ARM_OPTIMIZED, extra_server_args="--x", extra_envs={})
    recorder.record_variant(
        ARM_OPTIMIZED,
        stage=STAGE_BOOT,
        conc=64,
        point=_point(64),
        num_prompts=320,
        start_time="2026-09-07T02:03:04+00:00",
        wall_duration_sec=311.5,
        granted_cap_sec=10800.0,
        budget_remaining_sec=20000.0,
    )
    recorder.finish_arm(ARM_OPTIMIZED, status="succeeded")
    recorder.finish(_final())

    point = _ext(_bound_session)["arms"][ARM_OPTIMIZED]["points"][0]
    assert point["arm"] == ARM_OPTIMIZED
    assert point["total_token_throughput"] == 12800.0
    assert point["e2e_norm_intvty_p90"] == 42.0
    assert point["tpot_p90_ms"] == 13.5
    assert point["request_throughput"] == 96.0
    assert point["input_throughput"] == 6400.0
    assert point["completed_requests"] == 320
    assert point["killed_overtime"] is False
    assert point["workspace"] == "/w/optimized_conc64"
    # The report carries one elapsed_sec for the whole sweep, so a rung's own
    # window was not recoverable.
    assert point["stage"] == STAGE_BOOT
    assert point["num_prompts"] == 320
    assert point["start_time"] == "2026-09-07T02:03:04+00:00"
    assert point["wall_duration_sec"] == 311.5
    assert point["granted_cap_sec"] == 10800.0
    assert point["budget_remaining_sec"] == 20000.0


def test_the_curve_reads_upward_whatever_order_the_ladder_ran_in(_bound_session):
    recorder = _recorder()
    recorder.open_arm(ARM_OPTIMIZED, extra_server_args="--x", extra_envs={})
    recorder.record_variant(ARM_OPTIMIZED, stage=STAGE_BOOT, conc=64, point=_point(64))
    recorder.record_variant(ARM_OPTIMIZED, stage=STAGE_REUSE, conc=32, point=_point(32))
    recorder.record_variant(ARM_OPTIMIZED, stage=STAGE_REUSE, conc=16, point=_point(16))
    recorder.finish_arm(ARM_OPTIMIZED, status="succeeded")
    recorder.finish(_final())

    points = _ext(_bound_session)["arms"][ARM_OPTIMIZED]["points"]
    assert [point["conc"] for point in points] == [16, 32, 64]
    assert [point["stage"] for point in points] == [STAGE_REUSE, STAGE_REUSE, STAGE_BOOT]


def test_a_rung_the_budget_refused_is_not_a_benchmark_failure(_bound_session):
    recorder = _recorder()
    recorder.open_arm(ARM_BASELINE, extra_server_args="", extra_envs={})
    recorder.record_variant(
        ARM_BASELINE,
        stage=STAGE_BUDGET_SKIP,
        conc=32,
        point=_point(32, arm=ARM_BASELINE, status="skipped", error_class="budget_exhausted", output_throughput=None),
        budget_remaining_sec=140.0,
        granted_cap_sec=1800.0,
    )
    recorder.finish_arm(ARM_BASELINE, status="skipped")
    recorder.finish(_final(status="skipped", budget_exhausted=True))

    point = _ext(_bound_session)["arms"][ARM_BASELINE]["points"][0]
    assert point["stage"] == STAGE_BUDGET_SKIP
    assert point["error_class"] == "budget_exhausted"
    # It was refused at 140s left against a rung priced at 1800s, which is the
    # arithmetic that refused it.
    assert point["budget_remaining_sec"] == 140.0
    assert point["granted_cap_sec"] == 1800.0


# ---------------------------------------------------------------------------
# Boot-retry-descend
# ---------------------------------------------------------------------------
def test_the_concurrencies_the_server_would_not_boot_at_are_reported(_bound_session):
    recorder = _recorder()
    recorder.open_arm(ARM_OPTIMIZED, extra_server_args="--x", extra_envs={})
    recorder.record_variant(
        ARM_OPTIMIZED,
        stage=STAGE_BOOT_ATTEMPT,
        conc=64,
        point=_point(64, status="failed", error="OOM", error_class="single_server_boot_failed"),
        committed=False,
        wall_duration_sec=61.0,
    )
    recorder.record_variant(ARM_OPTIMIZED, stage=STAGE_BOOT, conc=32, point=_point(32))
    recorder.commit_variant(
        ARM_OPTIMIZED,
        stage=STAGE_BOOT_ATTEMPT,
        conc=64,
        point=_point(64, status="failed", error="OOM", error_class="single_server_boot_failed"),
    )
    recorder.record_arm_boot(
        ARM_OPTIMIZED,
        succeeded=True,
        booted_conc=32,
        attempted_concs=[64, 32],
        failed_concs=[64],
    )
    recorder.finish_arm(ARM_OPTIMIZED, status="degraded")
    recorder.finish(_final())

    boot = _ext(_bound_session)["arms"][ARM_OPTIMIZED]["boot"]
    assert boot["succeeded"] is True
    assert boot["booted_conc"] == 32
    assert boot["attempted_concs"] == [64, 32]
    assert boot["failed_concs"] == [64]
    assert [(row["conc"], row["status"], row["committed"]) for row in boot["attempts"]] == [
        (64, "failed", True),
        (32, "succeeded", True),
    ]
    assert boot["attempts"][0]["error"] == "OOM"


def test_a_failed_boot_counts_only_once_a_lower_rung_came_up(_bound_session):
    recorder = _recorder()
    recorder.open_arm(ARM_OPTIMIZED, extra_server_args="--x", extra_envs={})
    for conc in (64, 32):
        recorder.record_variant(
            ARM_OPTIMIZED,
            stage=STAGE_BOOT_ATTEMPT,
            conc=conc,
            point=_point(conc, status="failed", error_class="single_server_boot_failed"),
            committed=False,
        )
    recorder.record_arm_boot(ARM_OPTIMIZED, succeeded=False, attempted_concs=[64, 32], failed_concs=[64, 32])
    recorder.record_arm_strategy(
        ARM_OPTIMIZED,
        strategy=STRATEGY_SERVER_RESTART,
        reason="all_boot_attempts_failed",
        lifecycle_eligible=True,
        lifecycle_reason="supported",
    )
    recorder.record_variant(ARM_OPTIMIZED, stage=STAGE_SERVER_RESTART, conc=64, point=_point(64))
    recorder.record_variant(ARM_OPTIMIZED, stage=STAGE_SERVER_RESTART, conc=32, point=_point(32))
    recorder.finish_arm(ARM_OPTIMIZED, status="succeeded")
    recorder.finish(_final())

    arm = _ext(_bound_session)["arms"][ARM_OPTIMIZED]
    # The curve is the retry's results, one per concurrency, not four rows.
    assert [(point["conc"], point["stage"]) for point in arm["points"]] == [
        (32, STAGE_SERVER_RESTART),
        (64, STAGE_SERVER_RESTART),
    ]
    assert arm["boot"]["succeeded"] is False
    assert [row["committed"] for row in arm["boot"]["attempts"]] == [False, False]
    assert arm["strategy"] == STRATEGY_SERVER_RESTART
    assert arm["strategy_reason"] == "all_boot_attempts_failed"


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def test_the_pair_table_keeps_the_gain_columns_the_projection_dropped(_bound_session):
    recorder = _recorder()
    recorder.record_progress(
        comparison=[
            {
                "conc": 32,
                "baseline_value": 1000.0,
                "optimized_value": 1350.0,
                "speedup": 1.35,
                "delta_pct": 35.0,
                "baseline_status": "succeeded",
                "optimized_status": "succeeded",
            }
        ],
        summary={
            "metric": "output_throughput",
            "successful_pairs": 1,
            "failed_pairs": 0,
            "best_conc": 32,
            "best_speedup": 1.35,
            "median_speedup": 1.35,
            "mean_speedup": 1.35,
        },
    )
    recorder.finish(_final())

    pair = _ext(_bound_session)["comparison"][0]
    assert pair["conc"] == 32
    assert pair["baseline_value"] == 1000.0
    assert pair["optimized_value"] == 1350.0
    assert pair["speedup"] == 1.35
    assert pair["delta_pct"] == 35.0
    assert pair["baseline_status"] == "succeeded"
    assert pair["optimized_status"] == "succeeded"
    assert pair["error"] is None
    # The output objective has no second axis to hold, so the guard columns stay null rather than reading as a
    # rung whose throughput was measured and fell outside the band.
    assert pair["baseline_guard"] is None
    assert pair["optimized_guard"] is None
    assert pair["guard_holds"] is None


def test_a_failed_pair_is_explained_by_the_arm_that_broke(_bound_session):
    recorder = _recorder()
    recorder.open_arm(ARM_OPTIMIZED, extra_server_args="--x", extra_envs={})
    recorder.record_variant(
        ARM_OPTIMIZED,
        stage=STAGE_BOOT,
        conc=32,
        point=_point(32, status="failed", error="server died", error_class="server_crash"),
    )
    recorder.record_progress(
        comparison=[
            {
                "conc": 32,
                "baseline_tput": 1000.0,
                "optimized_tput": None,
                "speedup": None,
                "baseline_status": "succeeded",
                "optimized_status": "failed",
            }
        ],
        summary={"metric": "output_throughput", "successful_pairs": 0, "failed_pairs": 1},
    )
    recorder.finish(_final(status="failed"))

    assert _ext(_bound_session)["comparison"][0]["error"] == "optimized: server died"


def test_a_pair_with_no_point_at_all_says_so(_bound_session):
    recorder = _recorder()
    recorder.record_progress(
        comparison=[
            {
                "conc": 8,
                "baseline_tput": None,
                "optimized_tput": 900.0,
                "speedup": None,
                "baseline_status": "",
                "optimized_status": "succeeded",
            }
        ],
        summary={"successful_pairs": 0, "failed_pairs": 1},
    )
    recorder.finish(_final(status="failed"))

    assert _ext(_bound_session)["comparison"][0]["error"] == "baseline: no point recorded"


def test_a_later_pass_revises_a_pair_rather_than_adding_one(_bound_session):
    recorder = _recorder()
    recorder.record_progress(
        comparison=[{"conc": 32, "baseline_tput": None, "speedup": None, "baseline_status": ""}],
        summary={"successful_pairs": 0, "failed_pairs": 1},
    )
    recorder.record_progress(
        comparison=[
            {
                "conc": 32,
                "baseline_tput": 1000.0,
                "optimized_tput": 1350.0,
                "speedup": 1.35,
                "baseline_status": "succeeded",
                "optimized_status": "succeeded",
            }
        ],
        summary={"successful_pairs": 1, "failed_pairs": 0},
    )
    recorder.finish(_final())

    pairs = _ext(_bound_session)["comparison"]
    assert len(pairs) == 1
    assert pairs[0]["speedup"] == 1.35
    assert pairs[0]["error"] is None


# ---------------------------------------------------------------------------
# Result, ceiling and status
# ---------------------------------------------------------------------------
def test_the_result_carries_the_roll_up_statistics(_bound_session):
    recorder = _recorder()
    recorder.finish(_final())

    result = _ext(_bound_session)["result"]
    assert result["metric"] == "output_throughput"
    assert result["successful_pairs"] == 2
    assert result["failed_pairs"] == 0
    assert result["median_speedup"] == 1.35
    assert result["mean_speedup"] == 1.35
    assert result["best_conc"] == 64
    assert result["best_speedup"] == 1.4


def test_the_theoretical_ceiling_reaches_the_event(_bound_session):
    ceiling = {
        "schema_version": 1,
        "source": "roofline_ceiling.py",
        "gpu_type": "mi300x",
        "precision": "fp8",
        "tp": 8,
        "isl": 1024,
        "osl": 1024,
        "model_meta": {"num_layers": 61, "weight_dtype_bytes": 1},
        "rows": [
            {
                "conc": 64,
                "t_mem_tok_s": 9000.0,
                "t_cmp_tok_s": 14000.0,
                "t_peak_tok_s": 9000.0,
                "bound_kind": "memory",
                "mbu_baseline_pct": 61.2,
                "mbu_optimized_pct": 82.4,
            }
        ],
    }
    recorder = _recorder()
    recorder.finish(_final(roofline_ceiling=ceiling))

    assert _ext(_bound_session)["roofline_ceiling"] == ceiling


def test_a_sweep_with_no_ceiling_reports_none_rather_than_an_empty_block(_bound_session):
    recorder = _recorder()
    recorder.finish(_final())

    assert _ext(_bound_session)["roofline_ceiling"] is None


def test_a_curve_cut_short_by_the_budget_is_degraded(_bound_session):
    recorder = _recorder()
    recorder.finish(_final(budget_exhausted=True, budget_skip_reason="total_budget_exhausted"))

    event = _events(_bound_session)[0]
    assert event["status"] == "degraded"
    # ``result.status`` keeps the sweep's own word for it, so the two cannot be
    # confused for each other.
    assert event["ext"]["result"]["status"] == "succeeded"
    assert event["ext"]["runtime"]["budget_skip_reason"] == "total_budget_exhausted"


def test_a_sweep_that_declined_before_running_says_which_prerequisite_failed(_bound_session):
    recorder = _recorder()
    recorder.record_declined({"status": "skipped", "skip_reason": "no_optimization_to_compare"})

    event = _events(_bound_session)[0]
    assert event["status"] == "skipped"
    result = event["ext"]["result"]
    assert result["skip_reason"] == "no_optimization_to_compare"
    # A decline is not the same outcome as a ladder that ran and produced
    # nothing usable, which is what ``was_skipped`` alone conflates.
    assert result["declined"] is True


def test_a_ladder_that_ran_and_produced_no_pair_is_not_a_decline(_bound_session):
    recorder = _recorder()
    recorder.finish(
        _final(
            status="skipped",
            was_skipped=True,
            skip_reason="budget_exhausted_no_successful_pairs",
            budget_exhausted=True,
            summary={"successful_pairs": 0, "failed_pairs": 2},
        )
    )

    result = _ext(_bound_session)["result"]
    assert result["was_skipped"] is True
    assert result["declined"] is False
    assert result["skip_reason"] == "budget_exhausted_no_successful_pairs"


def test_a_sweep_that_raised_closes_the_event_on_the_failure(_bound_session):
    recorder = _recorder()
    recorder.finish_crashed(RuntimeError("grid runner exploded"))

    event = _events(_bound_session)[0]
    assert event["status"] == "failed"
    assert event["ext"]["failure"]["error_class"] == "RuntimeError"
    assert "grid runner exploded" in event["ext"]["failure"]["message"]


def test_closing_twice_does_not_publish_two_verdicts(_bound_session):
    recorder = _recorder()
    recorder.finish(_final())
    recorder.finish_crashed(RuntimeError("late"))

    assert [event["status"] for event in _events(_bound_session)] == ["succeeded"]


def test_the_session_stop_reason_reaches_the_failure_block(_bound_session):
    recorder = _recorder()
    recorder.finish(_final(status="failed"), stop_reason="sweep_timeout")

    assert _ext(_bound_session)["failure"]["stop_reason"] == "sweep_timeout"


# ---------------------------------------------------------------------------
# Assembly edges
# ---------------------------------------------------------------------------
def test_an_event_nothing_recorded_assembles_to_nothing(_bound_session):
    assert assemble_conc_sweep_ext({}, event="sweep:2:conc_sweep") == ({}, "")


def test_rows_of_another_event_are_not_pulled_in(_bound_session):
    first = _recorder(task_id="cs-1", macro_cycle=1)
    first.record_variant(ARM_OPTIMIZED, stage=STAGE_BOOT, conc=64, point=_point(64))
    first.finish(_final())
    second = _recorder(task_id="cs-2", macro_cycle=2)
    second.record_variant(ARM_OPTIMIZED, stage=STAGE_BOOT, conc=8, point=_point(8))
    second.finish(_final())

    parts = conc_sweep_event_parts()
    first_ext, _ = assemble_conc_sweep_ext(parts, event="sweep:1:conc_sweep")
    second_ext, _ = assemble_conc_sweep_ext(parts, event="sweep:2:conc_sweep")
    assert [point["conc"] for point in first_ext["arms"][ARM_OPTIMIZED]["points"]] == [64]
    assert [point["conc"] for point in second_ext["arms"][ARM_OPTIMIZED]["points"]] == [8]


def test_a_second_sweep_in_one_cycle_is_named_rather_than_dropped(_bound_session):
    first = _recorder(task_id="cs-1")
    first.finish(_final(elapsed_sec=10.0))
    second = _recorder(task_id="cs-2")
    second.finish(_final(elapsed_sec=20.0))

    ext, _ = assemble_conc_sweep_ext(conc_sweep_event_parts(), event="sweep:2:conc_sweep")
    assert ext["request"]["task_id"] == "cs-2"
    assert ext["superseded_sweeps"] == ["cs-1"]


def test_a_recorder_with_no_sink_declines_instead_of_raising(_bound_session):
    assert make_conc_sweep_recorder(None, task_id="cs-1") is None


def test_a_malformed_event_id_declines_instead_of_breaking_the_sweep(_bound_session):
    with pytest.raises(ValueError):
        make_sink("not-an-event-id", producer=PRODUCER)
