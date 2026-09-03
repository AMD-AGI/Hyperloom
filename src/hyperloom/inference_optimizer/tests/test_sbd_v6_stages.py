# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SBD V6 measurement stage projections and the ``close`` key.

Companion to ``test_sbd_v6_initial.py``, which covers the durable ``install``
/ ``model_gate`` events and the Framework Agent projection. Everything here is
projected at export time from V5 sections, so most tests call
``collect_v6_timeline`` / ``collect_v6_close`` directly with the sections a
real exporter run would hand them; the additivity and ordering tests go
through ``exporter.build`` because that is where the isolation actually has to
hold.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown import exporter, session_package
from hyperloom.inference_optimizer.breakdown.collectors import v6 as v6_collectors
from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_timeline
from hyperloom.inference_optimizer.breakdown.collectors.v6_close import collect_v6_close
from hyperloom.inference_optimizer.session.sbd_v6 import write_timeline_event_at


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _event(timeline: list[dict], event_type: str) -> dict | None:
    return next((event for event in timeline if event["type"] == event_type), None)


def _events(timeline: list[dict], event_type: str) -> list[dict]:
    return [event for event in timeline if event["type"] == event_type]


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def _sweep_section() -> dict:
    return {
        "all_variants": [
            {
                "variant_name": "variant_c64_i1024_o1024",
                "conc": 64,
                "isl": 1024,
                "osl": 1024,
                "status": "ok",
                "output_throughput_tok_s": 130.0,
                "benchmark_report_path": "runs/sweep/v1/benchmark_report.json",
            },
            {
                "variant_name": "variant_c128_i1024_o512",
                "conc": 128,
                "isl": 1024,
                "osl": 512,
                "status": "failed",
                "output_throughput_tok_s": None,
                "error": "server crashed",
            },
        ],
        "best_overall": {"variant_name": "variant_c64_i1024_o1024"},
    }


def test_sweep_reads_the_grid_back_off_the_points_it_measured(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={
            "last_sweep": {"workspace": "runs/sweep"},
            "sweep_attempts": [{"ts": "2026-08-27T02:30:00+00:00", "status": "ok", "task_id": "t-sweep"}],
        },
        sweep=_sweep_section(),
        phase_timeline=[{"action": "sweep", "ts": "2026-08-27T02:00:00+00:00", "task_id": "t-sweep"}],
    )

    event = _event(timeline, "sweep")
    # One point measured, one lost: the grid ran but did not run whole.
    assert event["status"] == "degraded"
    assert event["ext"]["plan"]["conc_grid"] == [64, 128]
    assert event["ext"]["plan"]["isl_grid"] == [1024]
    assert event["ext"]["plan"]["osl_grid"] == [512, 1024]
    # No producer records where the grid came from.
    assert event["ext"]["plan"]["grid_source"] is None
    assert event["ext"]["artifacts"]["sweep_dir"] == "runs/sweep"
    assert event["ext"]["artifacts"]["sweep_report_paths"] == ["runs/sweep/v1/benchmark_report.json"]
    assert [variant["variant_id"] for variant in event["ext"]["sweep"]["all_variants"]] == [
        "variant_c64_i1024_o1024",
        "variant_c128_i1024_o512",
    ]


def test_sweep_input_anchor_does_not_borrow_the_end_of_session_throughput(tmp_path):
    """``current_best.tput`` is the final figure, not the sweep's entry point."""
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={"last_sweep": {"workspace": "runs/sweep"}, "current_best": {"tput": 175.0}},
        sweep=_sweep_section(),
        baseline={"attempts_history": [{"task_id": "t-base", "status": "ok"}]},
    )

    anchor = _event(timeline, "sweep")["ext"]["input_anchor"]
    assert anchor["baseline_task_id"] == "t-base"
    assert anchor["input_throughput_tok_s_per_gpu"] is None
    assert anchor["current_best_task_id"] is None


def test_sweep_without_any_evidence_produces_no_event(tmp_path):
    timeline = collect_v6_timeline(tmp_path, [], state={}, sweep={"all_variants": []})

    assert _event(timeline, "sweep") is None


def test_sweep_that_died_before_its_first_grid_point_is_failed_not_skipped(tmp_path):
    """A sweep can fail with an empty point list, and did not 'not happen'.

    Deriving status from ``all_variants`` alone reported ``skipped`` on the
    same event that carried ``stop_reason: sweep_failed``.
    """
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={
            "stop_reason": "sweep_failed",
            "sweep_attempts": [
                {
                    "task_id": "t-sweep",
                    "status": "failed",
                    "error": "server never became ready",
                    "ts": "2026-08-27T03:00:00+00:00",
                }
            ],
        },
        sweep={"all_variants": []},
    )

    event = _event(timeline, "sweep")
    assert event["status"] == "failed"
    assert event["ext"]["failure"]["stop_reason"] == "sweep_failed"
    assert event["ext"]["failure"]["failed_task_id"] == "t-sweep"
    assert event["ext"]["failure"]["message"] == "server never became ready"
    assert event["ext"]["sweep"]["all_variants"] == []


def test_sweep_with_measured_points_and_a_failed_attempt_is_degraded(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={
            "stop_reason": "sweep_failed",
            "sweep_attempts": [{"task_id": "t-sweep-2", "status": "failed", "ts": "2026-08-27T03:10:00+00:00"}],
        },
        sweep={"all_variants": [{"variant_name": "c8", "conc": 8, "status": "ok", "output_throughput": 90.0}]},
    )

    assert _event(timeline, "sweep")["status"] == "degraded"


# ---------------------------------------------------------------------------
# conc_sweep
# ---------------------------------------------------------------------------
def _conc_sweep_section() -> dict:
    return {
        "status": "ok",
        "budget_exhausted": False,
        "concs_requested": [8, 16],
        "total_budget_sec": 600,
        "baseline": {
            "extra_server_args": "",
            "points": [
                {"conc": 8, "status": "ok", "output_throughput": 90.0},
                {"conc": 16, "status": "ok", "output_throughput": 120.0},
            ],
        },
        "optimized": {
            "extra_server_args": "--enable-torch-compile",
            "points": [
                {"conc": 8, "status": "ok", "output_throughput": 99.0},
                {"conc": 16, "status": "failed", "output_throughput": None, "error": "server OOM at conc=16"},
            ],
        },
        # ``conc_pair_comparison`` stamps both arm statuses on every row, and
        # ``None`` where an arm has no point at that concurrency at all.
        "comparison": [
            {
                "conc": 8,
                "baseline_tput": 90.0,
                "optimized_tput": 99.0,
                "speedup": 1.1,
                "baseline_status": "ok",
                "optimized_status": "ok",
            },
            {
                "conc": 16,
                "baseline_tput": 120.0,
                "optimized_tput": None,
                "speedup": None,
                "baseline_status": "ok",
                "optimized_status": "failed",
            },
        ],
        "summary": {"best_conc": 8, "best_speedup": 1.1},
        "workspace": "runs/conc_sweep",
        "elapsed_sec": 300.0,
    }


def test_conc_sweep_renames_the_comparison_columns_and_keeps_the_arms(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={"last_conc_sweep": {"ts": "2026-08-27T04:00:00+00:00", "status": "succeeded"}},
        conc_sweep_summary=_conc_sweep_section(),
    )

    event = _event(timeline, "conc_sweep")
    assert event["status"] == "succeeded"
    # Only the completion time is recorded; the window closes rather than
    # collapsing onto a single instant.
    assert event["start_time"] == ""
    assert event["end_time"] == "2026-08-27T04:00:00+00:00"
    assert event["ext"]["comparison"][0] == {
        "conc": 8,
        "baseline_output_throughput": 90.0,
        "optimized_output_throughput": 99.0,
        "speedup": 1.1,
        "error": None,
    }
    # An unpaired point names the arm that broke and quotes that arm's own
    # error. Reporting the first of the two statuses would have said
    # "succeeded" here, since it is the baseline arm that came through.
    assert event["ext"]["comparison"][1]["speedup"] is None
    assert event["ext"]["comparison"][1]["error"] == "optimized: server OOM at conc=16"
    # "" is the baseline arm's defining value, not a missing one.
    assert event["ext"]["arms"]["baseline"]["extra_server_args"] == ""
    assert event["ext"]["arms"]["optimized"]["extra_server_args"] == "--enable-torch-compile"
    assert event["ext"]["result"]["best_conc"] == 8
    assert event["ext"]["runtime"]["elapsed_sec"] == 300.0


def test_conc_sweep_cut_short_by_its_budget_is_degraded_not_succeeded(tmp_path):
    summary = _conc_sweep_section() | {"budget_exhausted": True}
    timeline = collect_v6_timeline(tmp_path, [], state={}, conc_sweep_summary=summary)

    event = _event(timeline, "conc_sweep")
    assert event["status"] == "degraded"
    assert event["ext"]["result"]["budget_exhausted"] is True


def test_conc_sweep_without_any_evidence_produces_no_event(tmp_path):
    assert _event(collect_v6_timeline(tmp_path, [], state={}), "conc_sweep") is None


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------
def _close_state(steps: list[dict], **overrides) -> dict:
    return {
        "phase": "CLOSE",
        "phase_history": [
            {
                "from_phase": "SWEEP",
                "to_phase": "CLOSE",
                "ts": "2026-08-27T05:00:00+00:00",
                "evidence": {"close_steps": steps},
            },
        ],
    } | overrides


def test_close_is_degraded_while_the_breakdown_predates_the_rest_of_the_sequence(tmp_path):
    """The step-2 snapshot describes the close-out only as far as itself.

    ``session_breakdown`` is step 2, so at the moment it runs the four steps
    after it do not exist yet and ``close_sequence_done`` is false. Reporting
    ``degraded`` for that is correct — the record really is incomplete. The
    sequencer's final ``patch_breakdown_close`` is what supersedes it; see
    ``test_close_patch_*``.
    """
    state = _close_state(
        [
            {"step": "sequencer_started", "status": "running", "ts": "2026-08-27T05:00:01+00:00"},
            {"step": "geak_rebench_drain", "status": "skipped", "ts": "2026-08-27T05:00:02+00:00"},
            {"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00", "task_id": "t-report"},
            {"step": "session_breakdown", "status": "done", "ts": "2026-08-27T05:00:45+00:00", "task_id": "t-bd"},
        ]
    )

    close = collect_v6_close(tmp_path, state, {}, [])

    assert close["status"] == "degraded"
    assert close["close_sequence_done"] is False
    assert [step["step"] for step in close["steps"]] == [
        "sequencer_started",
        "geak_rebench_drain",
        "report",
        "session_breakdown",
    ]
    assert close["start_time"] == "2026-08-27T05:00:01+00:00"
    assert close["end_time"] == "2026-08-27T05:00:45+00:00"
    assert close["steps"][2]["task_id"] == "t-report"


def test_close_is_succeeded_only_when_every_step_settled_and_the_sequence_finished(tmp_path):
    state = _close_state(
        [
            {"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00"},
            {"step": "ndjson_drain", "status": "skipped", "ts": "2026-08-27T05:01:00+00:00"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )

    assert collect_v6_close(tmp_path, state, {}, [])["status"] == "succeeded"


def test_close_succeeds_even_though_sequencer_started_never_leaves_running(tmp_path):
    """``sequencer_started`` is a marker, not a unit of work.

    The sequencer records it once on entry and never revisits it, so treating
    ``running`` as unsettled made ``succeeded`` unreachable no matter how
    cleanly the session closed.
    """
    state = _close_state(
        [
            {"step": "sequencer_started", "status": "running", "ts": "2026-08-27T05:00:01+00:00"},
            {"step": "fact_finalize", "status": "done", "ts": "2026-08-27T05:00:10+00:00"},
            {"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00"},
            {"step": "session_breakdown", "status": "done", "ts": "2026-08-27T05:00:45+00:00"},
            {"step": "artifact_package", "status": "skipped", "ts": "2026-08-27T05:00:55+00:00"},
            {"step": "ndjson_drain", "status": "skipped", "ts": "2026-08-27T05:01:00+00:00"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )
    warnings: list[str] = []

    close = collect_v6_close(tmp_path, state, {}, warnings)

    assert close["status"] == "succeeded"
    # ``fact_finalize`` is emitted by the sequencer but was missing from the V6
    # field design's enum. It is a known step, not drift.
    assert "fact_finalize" in [step["step"] for step in close["steps"]]
    assert warnings == []


def test_close_still_waits_on_a_step_that_really_is_running(tmp_path):
    state = _close_state(
        [
            {"step": "sequencer_started", "status": "running", "ts": "2026-08-27T05:00:01+00:00"},
            {"step": "report", "status": "running", "ts": "2026-08-27T05:00:30+00:00"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )

    assert collect_v6_close(tmp_path, state, {}, [])["status"] == "degraded"


def test_close_passes_through_an_unknown_step_and_warns(tmp_path):
    state = _close_state(
        [
            {"step": "teleport_to_s3", "status": "done", "ts": "2026-08-27T05:00:30+00:00"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )
    warnings: list[str] = []

    close = collect_v6_close(tmp_path, state, {}, warnings)

    # Dropping it would lose a step the producer really recorded.
    assert [step["step"] for step in close["steps"]] == ["teleport_to_s3", "done"]
    assert close["status"] == "succeeded"
    assert any("teleport_to_s3" in warning for warning in warnings)


def test_close_reports_degraded_when_a_step_failed(tmp_path):
    state = _close_state(
        [
            {"step": "report", "status": "failed", "ts": "2026-08-27T05:00:30+00:00", "detail": "task_state='failed'"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )

    close = collect_v6_close(tmp_path, state, {}, [])
    assert close["status"] == "degraded"
    assert close["steps"][0]["detail"] == "task_state='failed'"


def test_close_without_any_step_reports_failed(tmp_path):
    close = collect_v6_close(tmp_path, {"phase_history": []}, {}, [])

    assert close["status"] == "failed"
    assert close["steps"] == []
    assert close["start_time"] == ""


def test_close_falls_back_to_the_phase_entry_when_no_step_was_recorded(tmp_path):
    state = {"phase_history": [{"from_phase": "SWEEP", "to_phase": "CLOSE", "ts": "2026-08-27T05:00:00+00:00"}]}

    assert collect_v6_close(tmp_path, state, {}, [])["start_time"] == "2026-08-27T05:00:00+00:00"


def test_close_collects_steps_split_across_phase_history_rows(tmp_path):
    state = {
        "phase_history": [
            {
                "to_phase": "CLOSE",
                "ts": "2026-08-27T05:00:00+00:00",
                "evidence": {"close_steps": [{"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00"}]},
            },
            {
                "to_phase": "CLOSE",
                "ts": "2026-08-27T05:01:00+00:00",
                "evidence": {"close_steps": [{"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"}]},
            },
        ],
        "close_sequence_done": True,
    }

    close = collect_v6_close(tmp_path, state, {}, [])
    assert [step["step"] for step in close["steps"]] == ["report", "done"]
    assert close["status"] == "succeeded"


def test_close_surfaces_robustness_escalation_and_its_signals(tmp_path):
    state = _close_state([{"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"}])
    state["stop_reason"] = "robustness_escalated"
    signals = [
        {"ts": "2026-08-27T04:00:00+00:00", "signal": "crash", "action": "restart", "workdir": "robustness-workdir/0"}
    ]

    close = collect_v6_close(tmp_path, state, {"robustness_signals": signals}, [])

    assert close["robustness"]["escalated"] is True
    assert close["robustness"]["signals"] == signals


def test_close_artifacts_point_only_at_files_that_exist(tmp_path):
    _write_json(tmp_path / "reports" / "final.json", {"ok": True})
    state = _close_state(
        [
            {
                "step": "artifact_package",
                "status": "done",
                "ts": "2026-08-27T05:02:00+00:00",
                "detail": str(tmp_path / "bundle.zip"),
            }
        ]
    )

    artifacts = collect_v6_close(tmp_path, state, {}, [])["artifacts"]

    assert artifacts["final_json_path"] == "reports/final.json"
    assert artifacts["final_md_path"] is None
    assert artifacts["session_breakdown_path"] == "session_breakdown.json"
    assert artifacts["artifact_package_path"] == "bundle.zip"


def test_close_ignores_a_skipped_artifact_package_detail(tmp_path):
    """``detail`` doubles as the skip reason; only a ``done`` row holds a path."""
    state = _close_state(
        [
            {
                "step": "artifact_package",
                "status": "skipped",
                "ts": "2026-08-27T05:02:00+00:00",
                "detail": "no artifacts matched or dest unwritable",
            }
        ]
    )

    assert collect_v6_close(tmp_path, state, {}, [])["artifacts"]["artifact_package_path"] is None


# ---------------------------------------------------------------------------
# close: the end-of-sequence refresh
# ---------------------------------------------------------------------------
_FULL_CLOSE_STEPS = [
    {"step": "sequencer_started", "status": "running", "ts": "2026-08-27T05:00:01+00:00"},
    {"step": "fact_finalize", "status": "done", "ts": "2026-08-27T05:00:05+00:00"},
    {"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00"},
    {"step": "session_breakdown", "status": "done", "ts": "2026-08-27T05:00:45+00:00"},
    {"step": "artifact_package", "status": "done", "ts": "2026-08-27T05:00:55+00:00", "detail": "/workspace/s.zip"},
    {"step": "ndjson_drain", "status": "skipped", "ts": "2026-08-27T05:01:00+00:00"},
    {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
]


def _session_with_step_two_breakdown(tmp_path: Path) -> Path:
    """Build a session whose breakdown was written mid-CLOSE, as step 2 does."""
    _write_json(tmp_path / "state.json", _close_state(_FULL_CLOSE_STEPS[:4]))
    exporter.write_breakdown_json(tmp_path)
    # The sequencer then finishes, persisting the remaining steps.
    _write_json(tmp_path / "state.json", _close_state(_FULL_CLOSE_STEPS, close_sequence_done=True))
    return tmp_path / exporter.BREAKDOWN_FILENAME


def test_close_patch_replaces_the_step_two_snapshot_with_the_finished_sequence(tmp_path):
    target = _session_with_step_two_breakdown(tmp_path)
    before = json.loads(target.read_text(encoding="utf-8"))
    assert before["close"]["status"] == "degraded"
    assert [step["step"] for step in before["close"]["steps"]] == [
        "sequencer_started",
        "fact_finalize",
        "report",
        "session_breakdown",
    ]

    assert exporter.patch_breakdown_close(tmp_path) is True

    after = json.loads(target.read_text(encoding="utf-8"))
    assert after["close"]["status"] == "succeeded"
    assert after["close"]["close_sequence_done"] is True
    assert [step["step"] for step in after["close"]["steps"]] == [step["step"] for step in _FULL_CLOSE_STEPS]
    assert after["close"]["end_time"] == "2026-08-27T05:01:05+00:00"


def test_close_patch_touches_nothing_but_the_close_key(tmp_path):
    """The whole point of a patch over a rebuild: every other key is frozen."""
    target = _session_with_step_two_breakdown(tmp_path)
    before = json.loads(target.read_text(encoding="utf-8"))

    exporter.patch_breakdown_close(tmp_path)

    after = json.loads(target.read_text(encoding="utf-8"))
    assert set(after) == set(before)
    assert {key: value for key, value in after.items() if key != "close"} == {
        key: value for key, value in before.items() if key != "close"
    }


def test_close_patch_is_idempotent(tmp_path):
    tmp_path_target = _session_with_step_two_breakdown(tmp_path)
    assert exporter.patch_breakdown_close(tmp_path) is True
    # Nothing changed the second time, so nothing is rewritten.
    assert exporter.patch_breakdown_close(tmp_path) is False
    assert json.loads(tmp_path_target.read_text(encoding="utf-8"))["close"]["status"] == "succeeded"


def test_close_patch_is_a_no_op_without_a_breakdown(tmp_path):
    _write_json(tmp_path / "state.json", _close_state(_FULL_CLOSE_STEPS, close_sequence_done=True))

    assert exporter.patch_breakdown_close(tmp_path) is False


def test_close_patch_leaves_a_payload_that_never_carried_close_alone(tmp_path):
    """A V5-only breakdown has no ``close`` key, and gaining one is a surface change."""
    target = tmp_path / exporter.BREAKDOWN_FILENAME
    _write_json(target, {"schema_version": "hyperloom.session_breakdown.v5.0", "baseline": {}})
    _write_json(tmp_path / "state.json", _close_state(_FULL_CLOSE_STEPS, close_sequence_done=True))

    assert exporter.patch_breakdown_close(tmp_path) is False
    assert "close" not in json.loads(target.read_text(encoding="utf-8"))


def test_close_patch_swallows_a_corrupt_breakdown(tmp_path):
    """It runs at shutdown and must never mask the session's stop_reason."""
    target = tmp_path / exporter.BREAKDOWN_FILENAME
    target.write_text("{not json", encoding="utf-8")

    assert exporter.patch_breakdown_close(tmp_path) is False
    assert target.read_text(encoding="utf-8") == "{not json"


# ---------------------------------------------------------------------------
# what the consumer actually receives
# ---------------------------------------------------------------------------
def _packaged_close(session_dir: Path, dest_root: Path) -> tuple[dict, dict]:
    """Return the ``close`` key as delivered, from inside the zip and loose.

    External sync ships the package, not the session directory, so these two
    copies — not the one under ``session_dir`` — are what a consumer reads.
    """
    zip_path = dest_root / session_package.PACKAGE_SUBDIR / "sess-1.zip"
    with zipfile.ZipFile(zip_path) as bundle:
        zipped = json.loads(bundle.read(exporter.BREAKDOWN_FILENAME))
    loose = json.loads((dest_root / exporter.BREAKDOWN_FILENAME).read_text(encoding="utf-8"))
    return zipped["close"], loose["close"]


def test_the_delivered_package_carries_the_finished_close_section(tmp_path):
    """Patching the session copy is not delivery; the package has to be rebuilt.

    Mirrors the sequencer's order: package (CLOSE step 5), then patch the close
    section, then rebuild the bundle so the copies that ship agree with it.
    """
    session_dir = tmp_path / "session"
    dest_root = tmp_path / "dest"
    _session_with_step_two_breakdown(session_dir)

    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)
    assert exporter.patch_breakdown_close(session_dir) is True
    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)

    zipped, loose = _packaged_close(session_dir, dest_root)
    for delivered in (zipped, loose):
        assert delivered["status"] == "succeeded"
        assert delivered["close_sequence_done"] is True
        assert [step["step"] for step in delivered["steps"]] == [step["step"] for step in _FULL_CLOSE_STEPS]


def test_a_package_built_before_the_patch_ships_the_step_two_snapshot(tmp_path):
    """The regression this guards: the fix reaching the session dir only.

    Without the rebuild the session copy reads ``succeeded`` while both
    delivered copies still say ``degraded`` and stop four steps in — the state
    that made the previous round's fix invisible to its consumers.
    """
    session_dir = tmp_path / "session"
    dest_root = tmp_path / "dest"
    target = _session_with_step_two_breakdown(session_dir)

    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)
    exporter.patch_breakdown_close(session_dir)

    assert json.loads(target.read_text(encoding="utf-8"))["close"]["status"] == "succeeded"
    zipped, loose = _packaged_close(session_dir, dest_root)
    for stale in (zipped, loose):
        assert stale["status"] == "degraded"
        assert "artifact_package" not in {step["step"] for step in stale["steps"]}


def test_the_delivered_manifest_describes_the_rebuilt_bundle(tmp_path):
    """A surgical member swap would leave the manifest describing the old file.

    Hence a full repackage: the manifest is rebuilt from the members that were
    actually written, so its digest of ``session_breakdown.json`` matches what
    the consumer unzips.
    """
    session_dir = tmp_path / "session"
    dest_root = tmp_path / "dest"
    _session_with_step_two_breakdown(session_dir)

    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)
    exporter.patch_breakdown_close(session_dir)
    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)

    zip_path = dest_root / session_package.PACKAGE_SUBDIR / "sess-1.zip"
    with zipfile.ZipFile(zip_path) as bundle:
        manifest = json.loads(bundle.read(session_package.MANIFEST_JSON_NAME))
        member = bundle.getinfo(exporter.BREAKDOWN_FILENAME)
    entry = next(row for row in manifest["included_files"] if row["path"] == exporter.BREAKDOWN_FILENAME)
    assert entry["bytes"] == member.file_size


# ---------------------------------------------------------------------------
# outcome
# ---------------------------------------------------------------------------
def _v6_outcome(optimizations: dict) -> dict:
    return v6_collectors.collect_v6_outcome(
        session={"stop_reason": "target_reached"},
        baseline={},
        final={},
        optimizations=optimizations,
        state={"phase": "CLOSE"},
        timeline=[],
    )


def test_outcome_projects_authoritative_gain_by_v6_source_and_kernel_backend():
    outcome = _v6_outcome(
        {
            "available": True,
            "summary_by_source": {
                "warm_replay": {"keeps": 1, "total_gain_pct": 1.25},
                "explore": {"keeps": 2, "total_gain_pct": 2.0},
                "framework_agent": {"keeps": 1, "total_gain_pct": 0.75},
                "kernel_agent": {
                    "keeps": 4,
                    "total_gain_pct": 5.5,
                    "by_backend": {
                        "geak": {"keeps": 2, "total_gain_pct": 4.25, "non_attributable_keeps": 1},
                        "forge": {"keeps": 1, "total_gain_pct": 1.25, "non_attributable_keeps": 0},
                    },
                },
            },
            "validation": {
                "attributed_total_gain_pct": 9.5,
                "unattributed_gain_pct": 0.5,
                "reconciliation_gap_pct": 0.5,
            },
        }
    )

    attribution = outcome["validation"]["attribution"]
    assert attribution == {
        "available": True,
        "by_source": {
            "warm_replay": {"total_gain_pct": 1.25, "keep_count": 1},
            "framework_agent": {"total_gain_pct": 2.75, "keep_count": 3},
            "kernel": {
                "total_gain_pct": 5.5,
                "keep_count": 4,
                "by_backend": {
                    "geak": {
                        "total_gain_pct": 4.25,
                        "keep_count": 2,
                        "non_attributable_keep_count": 1,
                    },
                    "forge": {
                        "total_gain_pct": 1.25,
                        "keep_count": 1,
                        "non_attributable_keep_count": 0,
                    },
                },
            },
        },
    }


def test_outcome_marks_gain_totals_unknown_when_the_canonical_ledger_is_unavailable():
    attribution = _v6_outcome({"available": False})["validation"]["attribution"]

    assert attribution["available"] is False
    assert attribution["by_source"]["warm_replay"]["total_gain_pct"] is None
    assert attribution["by_source"]["framework_agent"]["total_gain_pct"] is None
    assert attribution["by_source"]["kernel"]["total_gain_pct"] is None
    assert attribution["by_source"]["kernel"]["by_backend"]["geak"]["total_gain_pct"] is None
    assert attribution["by_source"]["kernel"]["by_backend"]["forge"]["total_gain_pct"] is None


# ---------------------------------------------------------------------------
# cross-cutting: ordering and additivity
# ---------------------------------------------------------------------------
def test_projected_stages_interleave_with_durable_events_by_time(tmp_path):
    write_timeline_event_at(
        tmp_path,
        {
            "type": "install",
            "kind": "install",
            "status": "succeeded",
            "start_time": "2026-08-27T00:58:00+00:00",
            "end_time": "2026-08-27T00:59:00+00:00",
            "ext": {"run_kind": "fresh", "hard_fail_step_id": None, "runtime_snapshot": {}, "steps": []},
        },
    )

    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={"phase": "CLOSE"},
        sweep=_sweep_section(),
        baseline={"throughput_tok_s_per_gpu": 100.0},
        phase_timeline=[{"action": "sweep", "ts": "2026-08-27T00:30:00+00:00"}],
        conc_sweep_summary=_conc_sweep_section(),
    )

    # The conc sweep has no recorded time at all, so it sorts last rather than
    # to the epoch.
    assert [event["type"] for event in timeline] == ["sweep", "install", "conc_sweep"]


@pytest.mark.parametrize(
    "projector",
    ["project_sweep_event", "project_conc_sweep_event"],
)
def test_a_raising_stage_projector_costs_only_its_own_stage(tmp_path, monkeypatch, projector):
    """One stage blowing up must not take the durable events or its peers down.

    The exporter wraps the whole timeline collector, so without per-projector
    isolation a sweep-stage bug discards the ``install`` event a session read
    off disk before the Coordinator existed -- the one record a run that never
    reached a measurement stage actually has.
    """
    _write_json(
        tmp_path / "state.json",
        {"session_id": "s1", "model_name": "M", "framework": "sglang", "baseline_tput": 100.0, "phase": "CLOSE"},
    )
    _write_json(tmp_path / "manifest.json", {"session_id": "s1", "model_name": "M", "framework": "sglang"})
    write_timeline_event_at(
        tmp_path,
        {"type": "install", "kind": "install", "status": "succeeded", "start_time": "", "end_time": ""},
    )
    before = exporter.build(tmp_path)
    stage = {
        "project_sweep_event": "sweep",
        "project_conc_sweep_event": "conc_sweep",
    }[projector]
    assert "install" in {event["type"] for event in before["timeline"]}

    def _boom(*args, **kwargs):
        raise RuntimeError(f"{projector} exploded")

    monkeypatch.setattr(v6_collectors, projector, _boom)
    after = exporter.build(tmp_path)

    v6_keys = {"exported_at_utc", "metadata", "outcome", "timeline", "close"}
    assert {key: value for key, value in after.items() if key not in v6_keys} == {
        key: value for key, value in before.items() if key not in v6_keys
    }
    assert after["warnings"] == before["warnings"]

    types_after = [event["type"] for event in after["timeline"]]
    # The durable event survives, and so does every stage that projected.
    assert "install" in types_after
    assert stage not in types_after
    assert types_after == [event["type"] for event in before["timeline"] if event["type"] != stage]
    assert any(f"v6.timeline.{stage}" in warning for warning in after["metadata"]["warnings"])


def test_a_raising_close_collector_cannot_disturb_the_v5_payload(tmp_path, monkeypatch):
    _write_json(tmp_path / "state.json", {"session_id": "s1", "model_name": "M", "phase": "CLOSE"})
    _write_json(tmp_path / "manifest.json", {"session_id": "s1", "model_name": "M"})
    before = exporter.build(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("close exploded")

    monkeypatch.setattr(exporter.collectors, "collect_v6_close", _boom)
    after = exporter.build(tmp_path)

    assert after["warnings"] == before["warnings"]
    assert after["close"] == {}
    assert any("close" in warning for warning in after["metadata"]["warnings"])


# ---------------------------------------------------------------------------
# fabrication, settlement, identity and vocabulary
# ---------------------------------------------------------------------------
def _events(timeline: list[dict], event_type: str) -> list[dict]:
    return [event for event in timeline if event["type"] == event_type]


def test_a_sweep_point_status_is_normalized_onto_the_sweep_enum(tmp_path):
    """The sweep lane counts usable points by the literal word ``ok``."""
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        sweep={"all_variants": [{"variant_name": "v1", "status": "succeeded", "output_throughput_tok_s": 120.0}]},
    )

    event = _events(timeline, "sweep")[0]
    assert event["ext"]["sweep"]["all_variants"][0]["status"] == "ok"
    assert event["status"] == "succeeded"


def test_a_malformed_conc_grid_does_not_raise_out_of_the_projector(tmp_path):
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        conc_sweep_summary={"status": "ok", "concs_requested": 64},
    )

    assert _events(timeline, "conc_sweep")[0]["ext"]["plan"]["concs_requested"] == [64]


def test_an_unknown_close_step_status_is_reported(tmp_path):
    warnings: list[str] = []
    state = {
        "close_sequence_done": True,
        "phase_history": [
            {
                "to_phase": "CLOSE",
                "ts": "2026-08-27T02:00:00+00:00",
                "evidence": {
                    "close_steps": [{"step": "report", "status": "completed", "ts": "2026-08-27T02:00:01+00:00"}]
                },
            }
        ],
    }
    section = collect_v6_close(tmp_path, state, {}, warnings)

    # Passed through unchanged -- inventing ``done`` is the one thing this key
    # cannot afford -- but no longer silent about it.
    assert section["steps"][0]["status"] == "completed"
    assert any("unrecognized close step status" in warning for warning in warnings)
