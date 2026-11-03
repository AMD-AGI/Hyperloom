# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the recorded SBD V6 ``close`` section.

Most of these pin the defect the projection could not fix from where it stood:
``close.status`` was derived at export from which steps were present, but the
breakdown is written mid-sequence, so every healthy session read ``degraded``.
The sequencer's last act now records the verdict, so an un-settled section says
``running`` and ``degraded`` is free to mean a step actually failed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.collectors.v6_close import collect_v6_close
from hyperloom.inference_optimizer.breakdown.recorder.assembler import assemble_parts
from hyperloom.inference_optimizer.breakdown.recorder.close_out import (
    CloseStep,
    FinalRecipe,
    RooflineProgress,
    WriteBackSettlement,
    RESULT_AGENTX_BLOCKED,
    RESULT_TRANSPORT_FAILED,
    RESULT_WRITTEN,
    record_close_artifacts,
    record_close_opened,
    record_close_safety_net,
    record_close_settled,
    record_baseline_progress,
    record_close_step,
    record_final_recipe,
    record_roofline_progress,
    record_write_back_opened,
    record_write_back_settled,
)


@pytest.fixture
def sd(tmp_path: Path) -> Path:
    return tmp_path


def _close(session_dir: Path, *, warnings: list[str] | None = None) -> dict[str, Any]:
    """Build the ``close`` key the way the exporter does, from fragments only."""
    assembled = assemble_parts(session_dir, warnings=[])
    return collect_v6_close(
        warnings if warnings is not None else [],
        recorded=assembled.get("close"),
    )


def _run_sequence(session_dir: Path, *, fail: str = "") -> None:
    """Record a full close sequence, optionally failing one step."""
    record_close_opened(session_dir)
    for step in ("sequencer_started", "fact_finalize", "report", "session_breakdown", "ndjson_drain", "done"):
        status = "running" if step == "sequencer_started" else "done"
        if step == fail:
            status = "failed"
        record_close_step(session_dir, CloseStep(step=step, status=status))


def test_mid_sequence_reports_running_not_degraded(sd: Path) -> None:
    record_close_opened(sd)
    record_close_step(sd, CloseStep(step="sequencer_started", status="running"))
    record_close_step(sd, CloseStep(step="fact_finalize", status="done"))
    record_close_step(sd, CloseStep(step="report", status="done"))
    record_close_step(sd, CloseStep(step="session_breakdown", status="done"))

    close = _close(sd)
    assert close["status"] == "running"
    assert close["close_sequence_done"] is False
    # Not a verdict, and not empty either: the steps that did settle are on the wire.
    assert [row["step"] for row in close["steps"]] == [
        "sequencer_started",
        "fact_finalize",
        "report",
        "session_breakdown",
    ]


def test_settled_clean_sequence_succeeds(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd)

    close = _close(sd)
    assert close["status"] == "succeeded"
    assert close["close_sequence_done"] is True
    assert close["start_time"] and close["end_time"]


def test_failed_step_settles_degraded(sd: Path) -> None:
    _run_sequence(sd, fail="report")
    record_close_settled(sd)

    close = _close(sd)
    assert close["status"] == "degraded"
    assert close["close_sequence_done"] is True
    assert [row["status"] for row in close["steps"] if row["step"] == "report"] == ["failed"]


def test_unsettled_step_does_not_count_against_the_verdict(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd)
    assert _close(sd)["status"] == "succeeded"


def test_never_entered_close_has_no_section(sd: Path) -> None:
    assert assemble_parts(sd, warnings=[]).get("close") is None
    # The collector falls back to the projection, which reports a session that never closed.
    assert _close(sd)["status"] == "failed"


def test_safety_net_marks_only_a_close_that_never_started(sd: Path) -> None:
    record_close_safety_net(sd)
    assert (_close(sd)["source"], _close(sd)["status"]) == ("safety_net", "failed")

    record_close_opened(sd)
    record_close_safety_net(sd)
    assert (_close(sd)["source"], _close(sd)["status"]) == ("normal_close", "running")


def test_artifacts_are_recorded_not_probed(sd: Path) -> None:
    reports = sd / "reports"
    reports.mkdir()
    final_json = reports / "final.json"
    final_json.write_text("{}", encoding="utf-8")
    package = sd.parent / "workspace-pkg" / "session.zip"
    package.parent.mkdir(parents=True)
    package.write_text("zip", encoding="utf-8")

    _run_sequence(sd)
    record_close_artifacts(sd, final_json_path=final_json)
    record_close_artifacts(sd, artifact_package_path=package)
    record_close_settled(sd)
    final_json.unlink()

    artifacts = _close(sd)["artifacts"]
    assert artifacts["final_json_path"] == "reports/final.json"
    # Recorded by a later call, so the singleton merge must not have erased the earlier one.
    assert artifacts["artifact_package_path"] == str(package)
    # Never recorded, so it stays absent rather than being invented.
    assert artifacts["final_md_path"] is None
    assert artifacts["session_breakdown_path"] == "session_breakdown.json"


def test_step_row_carries_task_id_and_detail(sd: Path) -> None:
    record_close_opened(sd)
    record_close_step(
        sd,
        CloseStep(
            step="report",
            status="failed",
            task_id="t-42",
            detail="task_state='failed'",
            optional=False,
            error="task_state='failed'",
        ),
    )

    row = _close(sd)["steps"][0]
    assert row["task_id"] == "t-42"
    assert row["detail"] == "task_state='failed'"
    assert (row["optional"], row["error"]) == (False, "task_state='failed'")
    assert row["ts"]


def test_step_substream_does_not_leak_into_the_envelope(sd: Path) -> None:
    _run_sequence(sd)
    assembled = assemble_parts(sd, warnings=[])
    assert "close_step" not in assembled
    assert len(assembled["close"]["steps"]) == 6


def test_unknown_step_name_is_passed_through_and_warned(sd: Path) -> None:
    record_close_opened(sd)
    record_close_step(sd, CloseStep(step="brand_new_step", status="done"))

    warnings: list[str] = []
    close = _close(sd, warnings=warnings)
    assert [row["step"] for row in close["steps"]] == ["brand_new_step"]
    assert any("brand_new_step" in warning for warning in warnings)


def test_no_publication_attempt_leaves_no_write_back_key(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd)
    assert "kb_write_back" not in _close(sd)


def test_published_recipe_reports_its_identity_and_verdict(sd: Path) -> None:
    record_close_opened(sd)
    record_write_back_opened(sd, attempt=1, source="close")
    record_write_back_settled(
        sd,
        WriteBackSettlement(
            attempt=1,
            source="close",
            status="written",
            result_type=RESULT_WRITTEN,
            backend="local",
            canonical_id="cid-1",
            session_id="sid-1",
            scope={"kernel_optimizer": "geak", "tp": 8, "conc": 64, "isl": 128, "osl": 128},
            optimized_throughput=1234.5,
            validated_gain_pct=12.5,
        ),
    )

    write_back = _close(sd)["kb_write_back"]
    assert write_back["status"] == "written"
    assert write_back["result_type"] == RESULT_WRITTEN
    assert write_back["backend"] == "local"
    assert write_back["canonical_id"] == "cid-1"
    assert write_back["optimized_throughput"] == 1234.5
    assert write_back["validated_gain_pct"] == 12.5
    assert write_back["scope"]["tp"] == 8
    assert [row["status"] for row in write_back["attempts"]] == ["written"]


def test_attempt_opened_and_never_settled_stays_pending(sd: Path) -> None:
    record_close_opened(sd)
    record_write_back_opened(sd, attempt=1, source="close")

    write_back = _close(sd)["kb_write_back"]
    assert write_back["status"] == "pending"
    assert write_back["attempts"] == [
        {"attempt": 1, "source": "close", "status": "pending", "opened_at": write_back["attempts"][0]["opened_at"]}
    ]
    # Never settled, so there is no verdict to report and none is invented.
    assert "result_type" not in write_back
    assert "failure" not in write_back


def test_t4_fallback_retry_appends_a_second_attempt(sd: Path) -> None:
    record_close_opened(sd)
    record_write_back_opened(sd, attempt=1, source="close")
    record_write_back_settled(
        sd,
        WriteBackSettlement(
            attempt=1,
            source="close",
            status="error",
            result_type=RESULT_TRANSPORT_FAILED,
            raw_reason="ConnectionError",
            error_class="ConnectionError",
            backend="kb-store",
        ),
    )
    record_write_back_opened(sd, attempt=2, source="t4_fallback")
    record_write_back_settled(
        sd,
        WriteBackSettlement(
            attempt=2,
            source="t4_fallback",
            status="written",
            result_type=RESULT_WRITTEN,
            backend="kb-store",
            canonical_id="cid-2",
        ),
    )

    write_back = _close(sd)["kb_write_back"]
    assert [(row["attempt"], row["source"], row["status"]) for row in write_back["attempts"]] == [
        (1, "close", "error"),
        (2, "t4_fallback", "written"),
    ]
    # The arc carries the last settlement: the retry is what stands.
    assert write_back["status"] == "written"
    assert write_back["canonical_id"] == "cid-2"


def test_failure_keeps_the_error_class_apart_from_the_reason(sd: Path) -> None:
    record_close_opened(sd)
    record_write_back_opened(sd, attempt=1, source="close")
    record_write_back_settled(
        sd,
        WriteBackSettlement(
            attempt=1,
            source="close",
            status="error",
            result_type=RESULT_TRANSPORT_FAILED,
            raw_reason="upload rejected by store",
            error_class="RemoteRecipeUploadError",
            backend="kb-store",
        ),
    )

    write_back = _close(sd)["kb_write_back"]
    assert write_back["failure"] == {
        "error_class": "RemoteRecipeUploadError",
        "error": "upload rejected by store",
    }
    assert write_back["raw_reason"] == "upload rejected by store"
    assert write_back["attempts"][0]["error_class"] == "RemoteRecipeUploadError"


def test_skip_reason_is_a_recorded_code_not_a_matched_substring(sd: Path) -> None:
    record_close_opened(sd)
    record_write_back_opened(sd, attempt=1, source="close")
    record_write_back_settled(
        sd,
        WriteBackSettlement(
            attempt=1,
            source="close",
            status="skipped",
            result_type=RESULT_AGENTX_BLOCKED,
            raw_reason="agentx",
            backend="local",
        ),
    )

    write_back = _close(sd)["kb_write_back"]
    assert write_back["result_type"] == RESULT_AGENTX_BLOCKED
    assert write_back["status"] == "skipped"
    assert "failure" not in write_back


def test_second_pass_appends_rather_than_replacing(sd: Path) -> None:
    _run_sequence(sd, fail="report")
    record_close_settled(sd)
    assert _close(sd)["status"] == "degraded"

    record_close_opened(sd)
    assert _close(sd)["status"] == "running", "re-entering CLOSE reopens the section"
    record_close_step(sd, CloseStep(step="done", status="done"))

    close = _close(sd)
    assert [row["step"] for row in close["steps"]].count("done") == 2
    # The first pass's rows are still on the record after the second pass appended.
    assert close["steps"][0]["step"] == "sequencer_started"


def test_roofline_progress_is_snapshotted_at_the_close(sd: Path) -> None:
    record_close_opened(sd)
    record_roofline_progress(
        sd,
        RooflineProgress(
            baseline_tput=1000.0,
            baseline_ts="2026-09-06T09:00:00Z",
            optimization_stack=[
                {"ts": "2026-09-06T10:00:00Z", "tput": 1100.0, "action": "env_tune", "variant_name": "chunked"},
                {"ts": "2026-09-06T11:00:00Z", "tput": 1250.0, "action": "kernel_opt", "variant_name": "fused_rms"},
            ],
            latest_snapshot={"snapshot_id": 4, "theoretical_peak_tok_per_sec": 4000.0},
            current_best_tput=1250.0,
            cumulative_gain_pct=25.0,
            failure_streak=1,
        ),
    )

    progress = _close(sd)["roofline_progress"]
    assert progress["ceiling_available"] is True
    assert progress["ceiling_tok_per_sec"] == 4000.0
    assert progress["target_tok_per_sec"] == 2800.0
    assert progress["current_best_tput"] == 1250.0
    assert progress["current_best_pct_of_ceiling"] == 31.25
    assert progress["current_best_pct_of_target"] == 44.6429
    assert progress["roofline_failure_streak"] == 1
    assert progress["latest_snapshot_id"] == 4
    assert progress["trajectory_incomplete"] is False
    assert [point["label"] for point in progress["trajectory"]] == ["baseline", "chunked", "fused_rms"]
    assert progress["trajectory"][-1]["gain_pct"] == 25.0
    # The snapshot history stays in the roofline events; two copies could disagree.
    assert "snapshots" not in progress


def test_progress_says_so_when_the_curve_misses_a_promotion(sd: Path) -> None:
    record_close_opened(sd)
    record_roofline_progress(
        sd,
        RooflineProgress(
            baseline_tput=1000.0,
            optimization_stack=[{"ts": "2026-09-06T10:00:00Z", "tput": 1100.0, "action": "env_tune"}],
            latest_snapshot={"snapshot_id": 1, "theoretical_peak_tok_per_sec": 4000.0},
            current_best_tput=1400.0,
        ),
    )

    progress = _close(sd)["roofline_progress"]
    assert progress["trajectory_incomplete"] is True
    assert progress["current_best_tput"] == 1100.0
    assert progress["current_best_tput_declared"] == 1400.0


def test_a_session_that_never_analyzed_reports_no_ceiling_rather_than_zero(sd: Path) -> None:
    record_close_opened(sd)
    record_roofline_progress(sd, RooflineProgress(baseline_tput=1000.0, optimization_stack=[], latest_snapshot=None))

    progress = _close(sd)["roofline_progress"]
    assert progress["ceiling_available"] is False
    assert progress["ceiling_tok_per_sec"] is None
    assert progress["target_tok_per_sec"] is None
    assert progress["current_best_pct_of_ceiling"] is None
    assert progress["latest_snapshot_id"] is None


def test_a_close_that_never_snapshotted_progress_omits_the_key(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd)

    assert "roofline_progress" not in _close(sd)


def test_the_baseline_failure_tally_is_snapshotted_at_the_close(sd: Path) -> None:
    record_close_opened(sd)
    record_baseline_progress(sd, failure_streak=2, total_failures=5, arg_error_streak=1)

    progress = _close(sd)["baseline_progress"]
    assert progress["failure_streak"] == 2
    assert progress["total_failures"] == 5
    # Counted apart: a rejected server arg is correctable, a dying server is not.
    assert progress["arg_error_streak"] == 1


def test_a_session_whose_baselines_all_landed_reports_zeroes(sd: Path) -> None:
    record_close_opened(sd)
    record_baseline_progress(sd)

    assert _close(sd)["baseline_progress"] == {
        "failure_streak": 0,
        "total_failures": 0,
        "arg_error_streak": 0,
    }


def test_the_recipe_that_shipped_is_snapshotted_at_the_close(sd: Path) -> None:
    record_close_opened(sd)
    record_final_recipe(
        sd,
        FinalRecipe(
            throughput=142.5,
            ttft_mean_ms=31.2,
            e2el_mean_ms=980.0,
            action_path=["baseline", "explore:cuda_graph"],
            extra_server_args="--enable-torch-compile",
            extra_envs={"HSA_ENABLE": 1},
        ),
    )

    recipe = _close(sd)["final_recipe"]
    assert recipe["throughput"] == 142.5
    assert recipe["action_path"] == ["baseline", "explore:cuda_graph"]
    assert recipe["extra_server_args"] == "--enable-torch-compile"
    # Stringified on the way in, because that is what the launch actually exported.
    assert recipe["extra_envs"] == {"HSA_ENABLE": "1"}
    # Carried here because a session with no whole-stack validation has no row to read the pair off.
    assert (recipe["ttft_mean_ms"], recipe["e2el_mean_ms"]) == (31.2, 980.0)


def test_a_close_that_never_settled_a_recipe_omits_the_key(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd)

    assert "final_recipe" not in _close(sd)


def test_a_close_that_never_snapshotted_the_tally_omits_the_key(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd)

    assert "baseline_progress" not in _close(sd)


def test_a_close_that_never_drained_omits_the_candidate(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd)

    assert "geak_candidate" not in _close(sd)
