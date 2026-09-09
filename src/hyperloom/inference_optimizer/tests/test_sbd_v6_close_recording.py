# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the recorded SBD V6 ``close`` section.

The section this replaces was projected, and the defect these tests mostly pin
is the one the projection could not fix from where it stood: ``close.status``
was derived at export from which steps were present, but the breakdown is
written by a step in the middle of the sequence, so the later steps were
legitimately absent and every healthy session was labelled ``degraded``. The
word therefore had to mean "the record is incomplete" rather than "the
close-out went badly", and a reader could not tell the two apart.

Recording splits them: the sequencer's last act records the verdict, so an
un-settled section says ``running`` and ``degraded`` is free to mean a step
actually failed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.collectors.v6_close import collect_v6_close
from hyperloom.inference_optimizer.breakdown.recorder.assembler import assemble_parts
from hyperloom.inference_optimizer.breakdown.recorder.close_out import (
    RESULT_AGENTX_BLOCKED,
    RESULT_TRANSPORT_FAILED,
    RESULT_WRITTEN,
    record_close_artifacts,
    record_close_opened,
    record_close_settled,
    record_baseline_progress,
    record_close_step,
    record_final_recipe,
    record_geak_candidate,
    record_roofline_progress,
    record_write_back_opened,
    record_write_back_settled,
)
from hyperloom.inference_optimizer.session.session_paths import recipe_kb_dead_letter_ndjson


@pytest.fixture
def sd(tmp_path: Path) -> Path:
    return tmp_path


def _close(session_dir: Path, *, warnings: list[str] | None = None) -> dict[str, Any]:
    """Build the ``close`` key the way the exporter does, from fragments only."""
    assembled = assemble_parts(session_dir, warnings=[])
    return collect_v6_close(
        warnings if warnings is not None else [],
        recorded=assembled.get("close"),
        robustness=assembled.get("robustness"),
    )


def _write_findings(session_dir: Path, rows: list[dict[str, Any]], *, name: str = "s1") -> None:
    """Write findings the way the robustness ladder's sink does."""
    directory = session_dir / "agents" / "robustness" / "findings"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _finding(**overrides: Any) -> dict[str, Any]:
    row = {
        "tick_index": 4,
        "timestamp_unix": 1000.0,
        "symptom_name": "server_crash_loop",
        "severity": "high",
        "summary": "three boots failed in a row",
        "intents": [{"intent_type": "alert", "payload": {"detail": "noisy"}}],
        "evidence": {"crashes": 3},
        "rca_text": "the flag is unsupported on this build",
    }
    row.update(overrides)
    return row


def _run_sequence(session_dir: Path, *, fail: str = "") -> None:
    """Record a full close sequence, optionally failing one step."""
    record_close_opened(session_dir)
    for step in ("sequencer_started", "fact_finalize", "report", "session_breakdown", "ndjson_drain", "done"):
        status = "running" if step == "sequencer_started" else "done"
        if step == fail:
            status = "failed"
        record_close_step(session_dir, step=step, status=status)


def test_mid_sequence_reports_running_not_degraded(sd: Path) -> None:
    """The defect this conversion exists to fix.

    At the moment the breakdown is written, the steps after it have not run.
    The projection read that as ``degraded``; the recording has no verdict yet
    and says so.
    """
    record_close_opened(sd)
    record_close_step(sd, step="sequencer_started", status="running")
    record_close_step(sd, step="fact_finalize", status="done")
    record_close_step(sd, step="report", status="done")
    record_close_step(sd, step="session_breakdown", status="done")

    close = _close(sd)
    assert close["status"] == "running"
    assert close["close_sequence_done"] is False
    # Not a verdict, and not empty either: the steps that did settle are on
    # the wire, which is what lets a reader see how far the close-out got.
    assert [row["step"] for row in close["steps"]] == [
        "sequencer_started",
        "fact_finalize",
        "report",
        "session_breakdown",
    ]


def test_settled_clean_sequence_succeeds(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="target_reached")

    close = _close(sd)
    assert close["status"] == "succeeded"
    assert close["close_sequence_done"] is True
    assert close["start_time"] and close["end_time"]


def test_failed_step_settles_degraded(sd: Path) -> None:
    """``degraded`` now means a step failed, not that the record is partial."""
    _run_sequence(sd, fail="report")
    record_close_settled(sd, stop_reason="time_exhausted")

    close = _close(sd)
    assert close["status"] == "degraded"
    assert close["close_sequence_done"] is True
    assert [row["status"] for row in close["steps"] if row["step"] == "report"] == ["failed"]


def test_unsettled_step_does_not_count_against_the_verdict(sd: Path) -> None:
    """A step recorded ``running`` is not a failure.

    ``sequencer_started`` is recorded once as ``running`` and never revisited,
    so counting it as unsettled would have put every session in ``degraded``.
    The projection needed a special case for it; the verdict does not.
    """
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="target_reached")
    assert _close(sd)["status"] == "succeeded"


def test_never_entered_close_has_no_section(sd: Path) -> None:
    """No fragment at all is distinguishable from an un-settled close-out."""
    assert assemble_parts(sd, warnings=[]).get("close") is None
    # The collector falls back to the projection, which reports the session
    # never reached its close-out.
    assert _close(sd)["status"] == "failed"


def test_artifacts_are_recorded_not_probed(sd: Path) -> None:
    """Paths come from the step that wrote the file, not an export-time probe.

    Pinned by deleting the file after recording it: a projection that tested
    for the file would drop it, which is the behaviour being replaced.
    """
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
    record_close_settled(sd, stop_reason="target_reached")
    final_json.unlink()

    artifacts = _close(sd)["artifacts"]
    assert artifacts["final_json_path"] == "reports/final.json"
    # Recorded by a later call that knew nothing about the reports, and the
    # singleton merge must not have erased them.
    assert artifacts["artifact_package_path"] == str(package)
    # Never recorded, so it stays absent rather than being invented.
    assert artifacts["final_md_path"] is None
    assert artifacts["session_breakdown_path"] == "session_breakdown.json"


def test_escalation_comes_from_the_recorded_stop_reason(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="robustness_escalated")

    robustness = _close(sd)["robustness"]
    assert robustness["escalated"] is True
    # Recorded alongside the verdict so the escalation can be checked against
    # the reason it was drawn from.
    assert robustness["stop_reason"] == "robustness_escalated"


def test_ordinary_stop_reason_is_not_an_escalation(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="target_reached")
    assert _close(sd)["robustness"]["escalated"] is False


def test_the_close_out_says_what_the_ladder_found_not_just_that_it_escalated(sd: Path) -> None:
    """The verdict alone says a session was escalated without saying what for."""
    _write_findings(sd, [_finding()])
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="robustness_escalated")

    robustness = _close(sd)["robustness"]
    assert robustness["findings_total"] == 1
    (finding,) = robustness["findings"]
    assert finding["symptom_name"] == "server_crash_loop"
    assert finding["severity"] == "high"
    assert finding["tick_index"] == 4
    assert finding["rca_text"] == "the flag is unsupported on this build"
    assert finding["evidence"] == {"crashes": 3}
    # The payloads are the ladder's own working detail; the types are what the
    # close-out is reporting.
    assert finding["intents"] == ["alert"]


def test_an_unescalated_session_still_reports_what_fired(sd: Path) -> None:
    """These are the findings that were judged survivable, which had nowhere to be read."""
    _write_findings(sd, [_finding(severity="low")])
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="target_reached")

    robustness = _close(sd)["robustness"]
    assert robustness["escalated"] is False
    assert [row["severity"] for row in robustness["findings"]] == ["low"]


def test_a_session_whose_ladder_never_wrote_reports_no_findings_key(sd: Path) -> None:
    """An empty list would claim a ladder that ran and found nothing."""
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="target_reached")

    assert "findings" not in _close(sd)["robustness"]


def test_findings_are_ordered_and_the_total_survives_the_cap(sd: Path) -> None:
    """A truncated list must never read as the whole of it."""
    _write_findings(sd, [_finding(tick_index=n, timestamp_unix=float(2000 - n)) for n in range(60)])
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="target_reached")

    robustness = _close(sd)["robustness"]
    assert robustness["findings_total"] == 60
    assert len(robustness["findings"]) == 50
    ticks = [row["tick_index"] for row in robustness["findings"]]
    assert ticks == sorted(ticks, reverse=True)


def test_findings_from_several_sink_files_are_read_together(sd: Path) -> None:
    """The sink names its file after the session id, and a resume starts another."""
    _write_findings(sd, [_finding(symptom_name="first", timestamp_unix=1.0)], name="s1")
    _write_findings(sd, [_finding(symptom_name="second", timestamp_unix=2.0)], name="s2")
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="target_reached")

    robustness = _close(sd)["robustness"]
    assert [row["symptom_name"] for row in robustness["findings"]] == ["first", "second"]


def test_the_agents_turns_sit_with_the_verdict_drawn_from_them(sd: Path) -> None:
    """Reading the agent's account used to require knowing to look in two keys."""
    from hyperloom.inference_optimizer.breakdown.recorder.robustness_out import record_robustness_turn

    record_robustness_turn(sd, turn_idx=0, outcome="intents", intents=[{"type": "alert"}])
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="target_reached")

    turns = _close(sd)["robustness"]["turns"]
    assert [row["turn_idx"] for row in turns] == [0]
    assert turns[0]["outcome"] == "intents"


def test_step_row_carries_task_id_and_detail(sd: Path) -> None:
    record_close_opened(sd)
    record_close_step(sd, step="report", status="failed", task_id="t-42", detail="task_state='failed'")

    row = _close(sd)["steps"][0]
    assert row["task_id"] == "t-42"
    assert row["detail"] == "task_state='failed'"
    assert row["ts"]


def test_step_substream_does_not_leak_into_the_envelope(sd: Path) -> None:
    """``close_step`` is folded into ``close.steps`` and popped."""
    _run_sequence(sd)
    assembled = assemble_parts(sd, warnings=[])
    assert "close_step" not in assembled
    assert len(assembled["close"]["steps"]) == 6


def test_unknown_step_name_is_passed_through_and_warned(sd: Path) -> None:
    record_close_opened(sd)
    record_close_step(sd, step="brand_new_step", status="done")

    warnings: list[str] = []
    close = _close(sd, warnings=warnings)
    assert [row["step"] for row in close["steps"]] == ["brand_new_step"]
    assert any("brand_new_step" in warning for warning in warnings)


# ---------------------------------------------------------------------------
# close.kb_write_back
# ---------------------------------------------------------------------------


def test_no_publication_attempt_leaves_no_write_back_key(sd: Path) -> None:
    """Absence is the record that it was never tried.

    The key is only there because an attempt was made, so an empty one would
    claim a publication that never happened.
    """
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="target_reached")
    assert "kb_write_back" not in _close(sd)


def test_published_recipe_reports_its_identity_and_verdict(sd: Path) -> None:
    record_close_opened(sd)
    record_write_back_opened(sd, attempt=1, source="close")
    record_write_back_settled(
        sd,
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
    """The defect this replaces.

    A session killed mid-publish leaves an opened attempt behind. The
    projection read that leftover marker as ``failed``, which claimed the KB
    had refused the write when in fact it was never asked to answer.
    """
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
    """Two seams try the publication, and the row says which one settled it.

    The CLOSE path attempts it first; if that never settled, the T4 teardown
    hook tries again. ``source`` lives on the attempt rather than on the arc
    because the two attempts can disagree about it.
    """
    record_close_opened(sd)
    record_write_back_opened(sd, attempt=1, source="close")
    record_write_back_settled(
        sd,
        attempt=1,
        source="close",
        status="error",
        result_type=RESULT_TRANSPORT_FAILED,
        raw_reason="ConnectionError",
        error_class="ConnectionError",
        backend="kb-store",
    )
    record_write_back_opened(sd, attempt=2, source="t4_fallback")
    record_write_back_settled(
        sd,
        attempt=2,
        source="t4_fallback",
        status="written",
        result_type=RESULT_WRITTEN,
        backend="kb-store",
        canonical_id="cid-2",
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
    """``error_class`` and ``raw_reason`` used to be the same string.

    The publisher reports transport failures as a bare exception class name,
    so the projection put that one value in both fields and a reader could not
    tell a class name from a reason token.
    """
    record_close_opened(sd)
    record_write_back_opened(sd, attempt=1, source="close")
    record_write_back_settled(
        sd,
        attempt=1,
        source="close",
        status="error",
        result_type=RESULT_TRANSPORT_FAILED,
        raw_reason="upload rejected by store",
        error_class="RemoteRecipeUploadError",
        backend="kb-store",
    )

    write_back = _close(sd)["kb_write_back"]
    assert write_back["failure"] == {
        "error_class": "RemoteRecipeUploadError",
        "error": "upload rejected by store",
    }
    assert write_back["raw_reason"] == "upload rejected by store"
    assert write_back["attempts"][0]["error_class"] == "RemoteRecipeUploadError"


def test_skip_reason_is_a_recorded_code_not_a_matched_substring(sd: Path) -> None:
    """``agentx`` was the collision the projection's rule order worked around.

    The needle appears both as this skip reason and inside exception class
    names such as ``configuration:AgentXConfigError``, so the export had to
    keep its 14 rules in a particular order to stop them claiming each
    other's cases. The publisher states the code instead.
    """
    record_close_opened(sd)
    record_write_back_opened(sd, attempt=1, source="close")
    record_write_back_settled(
        sd,
        attempt=1,
        source="close",
        status="skipped",
        result_type=RESULT_AGENTX_BLOCKED,
        raw_reason="agentx",
        backend="local",
    )

    write_back = _close(sd)["kb_write_back"]
    assert write_back["result_type"] == RESULT_AGENTX_BLOCKED
    assert write_back["status"] == "skipped"
    assert "failure" not in write_back


def test_queue_depth_is_snapshotted_when_the_attempt_settles(sd: Path) -> None:
    """Counted at settlement, not at export: the queues keep moving."""
    dead_letter = recipe_kb_dead_letter_ndjson(sd)
    dead_letter.parent.mkdir(parents=True, exist_ok=True)
    dead_letter.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")

    record_close_opened(sd)
    record_write_back_opened(sd, attempt=1, source="close")
    record_write_back_settled(sd, attempt=1, source="close", status="written", result_type=RESULT_WRITTEN)
    # Moves after the settlement; the recorded depth must not follow it.
    dead_letter.write_text("", encoding="utf-8")

    queue = _close(sd)["kb_write_back"]["queue"]
    assert queue["dead_letter_lines"] == 2
    assert queue["pending_lines"] == 0


def test_second_pass_appends_rather_than_replacing(sd: Path) -> None:
    """A resumed session re-enters CLOSE and closes again.

    The second pass describes a second attempt, so its rows are appended; its
    verdict, which is the one that stands, replaces the first.
    """
    _run_sequence(sd, fail="report")
    record_close_settled(sd, stop_reason="time_exhausted")
    assert _close(sd)["status"] == "degraded"

    record_close_opened(sd)
    assert _close(sd)["status"] == "running", "re-entering CLOSE reopens the section"
    record_close_step(sd, step="done", status="done")

    close = _close(sd)
    assert [row["step"] for row in close["steps"]].count("done") == 2
    # The first pass's failure is still on the record, so the verdict stands
    # at degraded even though the second pass failed nothing.
    assert close["steps"][0]["step"] == "sequencer_started"


def test_roofline_progress_is_snapshotted_at_the_close(sd: Path) -> None:
    """The ceiling and the curve are session-level, so the close-out states them.

    Neither belongs to any single roofline run: the ceiling comes from the last
    analysis and the curve from every promotion between them, which is why this
    is the one roofline fact recorded outside a roofline event.
    """
    record_close_opened(sd)
    record_roofline_progress(
        sd,
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
    )

    progress = _close(sd)["roofline_progress"]
    assert progress["ceiling_kind"] == "throughput"
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
    # The snapshot history stays in the roofline events that recorded it; two
    # copies of it would be free to disagree.
    assert "snapshots" not in progress


def test_progress_reports_the_latency_domain_when_there_is_no_token_ceiling(sd: Path) -> None:
    """Diffusion models decode no tokens, so a null tok/s ceiling is not a failure."""
    record_close_opened(sd)
    record_roofline_progress(
        sd,
        baseline_tput=0.4,
        baseline_ts="2026-09-06T09:00:00Z",
        optimization_stack=[],
        latest_snapshot={"snapshot_id": 2, "roofline_ideal_ms": 800.0, "e2e_mean_ms": 2000.0},
        current_best_tput=0.4,
    )

    progress = _close(sd)["roofline_progress"]
    assert progress["ceiling_kind"] == "latency"
    assert progress["ceiling_available"] is False
    assert progress["latency_ceiling_available"] is True
    assert progress["latency_ceiling_ms"] == 800.0
    assert progress["achieved_latency_ms"] == 2000.0
    assert progress["current_best_pct_of_latency_ceiling"] == 40.0


def test_progress_says_so_when_the_curve_misses_a_promotion(sd: Path) -> None:
    """A resume interrupted mid-promote leaves the stack behind the session's best.

    Recorded rather than left to a warnings list the reader may not be reading:
    a curve that ends below the session's own headline number is a property of
    the record, so the record says it.
    """
    record_close_opened(sd)
    record_roofline_progress(
        sd,
        baseline_tput=1000.0,
        optimization_stack=[{"ts": "2026-09-06T10:00:00Z", "tput": 1100.0, "action": "env_tune"}],
        latest_snapshot={"snapshot_id": 1, "theoretical_peak_tok_per_sec": 4000.0},
        current_best_tput=1400.0,
    )

    progress = _close(sd)["roofline_progress"]
    assert progress["trajectory_incomplete"] is True
    assert progress["current_best_tput"] == 1100.0
    assert progress["current_best_tput_declared"] == 1400.0


def test_a_session_that_never_analyzed_reports_no_ceiling_rather_than_zero(sd: Path) -> None:
    """Zero would read as a measured ceiling of zero, which is a different claim."""
    record_close_opened(sd)
    record_roofline_progress(sd, baseline_tput=1000.0, optimization_stack=[], latest_snapshot=None)

    progress = _close(sd)["roofline_progress"]
    assert progress["ceiling_kind"] == "none"
    assert progress["ceiling_tok_per_sec"] is None
    assert progress["target_tok_per_sec"] is None
    assert progress["current_best_pct_of_ceiling"] is None
    assert progress["latest_snapshot_id"] is None


def test_a_close_that_never_snapshotted_progress_omits_the_key(sd: Path) -> None:
    """An empty curve would claim a session that made no progress."""
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="time_exhausted")

    assert "roofline_progress" not in _close(sd)


def test_the_baseline_failure_tally_is_snapshotted_at_the_close(sd: Path) -> None:
    """The two baseline facts no baseline event can hold.

    Each baseline event closes when its own measurement ends, and the counters
    are advanced by the write-back that accounts for it afterwards -- so an
    event records the count it was dispatched under and the session's total is
    only final here.
    """
    record_close_opened(sd)
    record_baseline_progress(sd, failure_streak=2, total_failures=5, arg_error_streak=1)

    progress = _close(sd)["baseline_progress"]
    assert progress["failure_streak"] == 2
    assert progress["total_failures"] == 5
    # Counted apart from the general streak: a rejected server arg is a
    # configuration error the session can correct, a dying server is not.
    assert progress["arg_error_streak"] == 1


def test_a_session_whose_baselines_all_landed_reports_zeroes(sd: Path) -> None:
    """Zero is a claim worth making, so it is recorded rather than omitted."""
    record_close_opened(sd)
    record_baseline_progress(sd)

    assert _close(sd)["baseline_progress"] == {
        "failure_streak": 0,
        "total_failures": 0,
        "arg_error_streak": 0,
    }


def test_the_recipe_that_shipped_is_snapshotted_at_the_close(sd: Path) -> None:
    """The terminal configuration is a close-out fact for the same reason.

    The stack ledger records every adoption and a revert does not retract the
    row that adopted it, so what was still standing at the end is stated here.
    """
    record_close_opened(sd)
    record_final_recipe(
        sd,
        throughput=142.5,
        ttft_mean_ms=31.2,
        e2el_mean_ms=980.0,
        action_path=["baseline", "explore:cuda_graph"],
        extra_server_args="--enable-torch-compile",
        extra_envs={"HSA_ENABLE": 1},
    )

    recipe = _close(sd)["final_recipe"]
    assert recipe["throughput"] == 142.5
    assert recipe["action_path"] == ["baseline", "explore:cuda_graph"]
    assert recipe["extra_server_args"] == "--enable-torch-compile"
    # Stringified on the way in: the launch applies envs as strings, and a
    # recipe that reproduces the run must say what was actually exported.
    assert recipe["extra_envs"] == {"HSA_ENABLE": "1"}
    # Carried beside the throughput because a session with no whole-stack
    # validation has no validation row to read the pair off.
    assert (recipe["ttft_mean_ms"], recipe["e2el_mean_ms"]) == (31.2, 980.0)


def test_a_close_that_never_settled_a_recipe_omits_the_key(sd: Path) -> None:
    """An empty recipe would claim a session that shipped a bare config."""
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="time_exhausted")

    assert "final_recipe" not in _close(sd)


def test_a_close_that_never_snapshotted_the_tally_omits_the_key(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="time_exhausted")

    assert "baseline_progress" not in _close(sd)


def test_a_candidate_dropped_at_the_close_says_what_was_dropped(sd: Path) -> None:
    """The drop narrative outlives the slot the report used to read it from.

    The close drain empties ``geak_pending`` by the same write that settles it,
    and every kernel event of the session has closed by then, so the verdict
    has nowhere else to be read from.
    """
    record_close_opened(sd)
    record_geak_candidate(
        sd,
        pending={
            "status": "rebench_cancelled",
            "revalidation_error": "close_sequence",
            "self_reported_gain_pct": 12.5,
            "self_reported_tput": 16800.0,
            "self_reported_basis": "geak_internal_bench",
        },
        revalidation_pending=False,
    )

    candidate = _close(sd)["geak_candidate"]
    assert candidate["status"] == "rebench_cancelled"
    assert candidate["revalidation_error"] == "close_sequence"
    assert candidate["self_reported_gain_pct"] == pytest.approx(12.5)
    assert candidate["self_reported_tput"] == pytest.approx(16800.0)
    assert candidate["self_reported_basis"] == "geak_internal_bench"


def test_a_candidate_still_waiting_is_not_a_candidate_that_was_judged(sd: Path) -> None:
    record_close_opened(sd)
    record_geak_candidate(sd, pending={"status": "awaiting_rebench"}, revalidation_pending=True)

    candidate = _close(sd)["geak_candidate"]
    assert candidate["status"] == "awaiting_rebench"
    assert candidate["revalidation_pending"] is True
    assert candidate["revalidation_error"] is None


def test_a_session_with_no_candidate_records_an_empty_verdict(sd: Path) -> None:
    """Recorded rather than omitted: "no candidate" and "never looked" differ."""
    record_close_opened(sd)
    record_geak_candidate(sd)

    candidate = _close(sd)["geak_candidate"]
    assert candidate["status"] == ""
    assert candidate["revalidation_pending"] is False
    assert candidate["self_reported_gain_pct"] is None


def test_a_close_that_never_drained_omits_the_candidate(sd: Path) -> None:
    _run_sequence(sd)
    record_close_settled(sd, stop_reason="time_exhausted")

    assert "geak_candidate" not in _close(sd)
