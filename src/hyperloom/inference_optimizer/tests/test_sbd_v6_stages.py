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
        "baseline_throughput": 90.0,
        "optimized_throughput": 99.0,
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
    # The axis the speedups were taken on, defaulted when the summary omits it.
    assert event["ext"]["result"]["metric"] == "output_throughput"
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

    close = collect_v6_close(tmp_path, state, [])

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

    assert collect_v6_close(tmp_path, state, [])["status"] == "succeeded"


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

    close = collect_v6_close(tmp_path, state, warnings)

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

    assert collect_v6_close(tmp_path, state, [])["status"] == "degraded"


def test_close_passes_through_an_unknown_step_and_warns(tmp_path):
    state = _close_state(
        [
            {"step": "teleport_to_s3", "status": "done", "ts": "2026-08-27T05:00:30+00:00"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )
    warnings: list[str] = []

    close = collect_v6_close(tmp_path, state, warnings)

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

    close = collect_v6_close(tmp_path, state, [])
    assert close["status"] == "degraded"
    assert close["steps"][0]["detail"] == "task_state='failed'"


def test_close_without_any_step_reports_failed(tmp_path):
    close = collect_v6_close(tmp_path, {"phase_history": []}, [])

    assert close["status"] == "failed"
    assert close["steps"] == []
    assert close["start_time"] == ""


def test_close_falls_back_to_the_phase_entry_when_no_step_was_recorded(tmp_path):
    state = {"phase_history": [{"from_phase": "SWEEP", "to_phase": "CLOSE", "ts": "2026-08-27T05:00:00+00:00"}]}

    assert collect_v6_close(tmp_path, state, [])["start_time"] == "2026-08-27T05:00:00+00:00"


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

    close = collect_v6_close(tmp_path, state, [])
    assert [step["step"] for step in close["steps"]] == ["report", "done"]
    assert close["status"] == "succeeded"


def test_close_surfaces_robustness_escalation(tmp_path):
    """The close-out reports its verdict about robustness, not what was raised.

    What the agent raised is the top-level ``robustness`` key's job; ``close``
    answers only whether the session was stopped over it.
    """
    state = _close_state([{"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"}])
    state["stop_reason"] = "robustness_escalated"

    close = collect_v6_close(tmp_path, state, [])

    assert close["robustness"]["escalated"] is True
    assert "signals" not in close["robustness"]


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

    artifacts = collect_v6_close(tmp_path, state, [])["artifacts"]

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

    assert collect_v6_close(tmp_path, state, [])["artifacts"]["artifact_package_path"] is None


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
def _v6_outcome(timeline: list | None = None) -> dict:
    return v6_collectors.collect_v6_outcome(
        session={"stop_reason": "target_reached"},
        final={},
        state={"phase": "CLOSE"},
        timeline=timeline or [],
    )


def _baseline_action(
    *,
    task_id: str,
    throughput: float,
    establishes_quality_ref: bool = True,
    status: str = "succeeded",
    end_time: str = "2026-01-01T00:00:00+00:00",
) -> dict:
    """One action on a ``baseline`` event, shaped as the recorder assembles it."""
    return {
        "task_id": task_id,
        "status": status,
        "end_time": end_time,
        "request": {"task_id": task_id, "establishes_quality_ref": establishes_quality_ref},
        "measurement": {
            "throughput_tok_s_per_gpu": throughput,
            "accuracy": 0.81,
            "ttft_mean_ms": 120.0,
            "e2el_mean_ms": 900.0,
        },
    }


def _baseline_event(*actions: dict) -> dict:
    return {"type": "baseline", "ext": {"actions": list(actions)}}


def test_outcome_baseline_reads_the_anchoring_measurement_off_the_timeline():
    """The four figures come from the event, latency included.

    Latency used to be parsed out of ``benchmark_report.json`` at export time;
    the executor reports it, so the event already holds it.
    """
    outcome = _v6_outcome([_baseline_event(_baseline_action(task_id="b-1", throughput=800.0))])

    assert outcome["baseline"] == {
        "throughput_tok_s_per_gpu": 800.0,
        "accuracy": 0.81,
        "ttft_mean_ms": 120.0,
        "e2el_mean_ms": 900.0,
    }


def test_outcome_baseline_ignores_a_kernel_probe_that_anchors_nothing():
    """The discrimination the dispatch kind cannot make.

    The kernel lane's integrate re-baseline and stack validation reach the same
    executor carrying ``kind="baseline"`` literally, and land actions on the
    same event. They measure against an already-anchored baseline, so reading
    the newest ``baseline``-kind action would publish an A/B probe as the
    session's reference.
    """
    outcome = _v6_outcome([
            _baseline_event(
                _baseline_action(task_id="b-1", throughput=800.0, end_time="2026-01-01T00:00:00+00:00"),
                _baseline_action(
                    task_id="k-probe",
                    throughput=915.0,
                    establishes_quality_ref=False,
                    end_time="2026-01-01T05:00:00+00:00",
                ),
            )
        ],
    )

    assert outcome["baseline"]["throughput_tok_s_per_gpu"] == 800.0


def test_outcome_baseline_re_anchors_on_the_latest_anchoring_measurement():
    """A baseline re-measured after an enablement fix legitimately re-anchors."""
    outcome = _v6_outcome([
            _baseline_event(_baseline_action(task_id="b-1", throughput=800.0, end_time="2026-01-01T00:00:00+00:00")),
            _baseline_event(_baseline_action(task_id="b-2", throughput=845.0, end_time="2026-01-01T02:00:00+00:00")),
        ],
    )

    assert outcome["baseline"]["throughput_tok_s_per_gpu"] == 845.0


def test_outcome_baseline_keeps_a_degraded_anchor_and_drops_a_failed_one():
    """``degraded`` is the number the session's gains were read against.

    It stands on the cold warmup round because the budget would not hold the hot
    pass -- knowingly depressed, but it is what the session actually used.
    """
    degraded = _v6_outcome([_baseline_event(_baseline_action(task_id="b-1", throughput=770.0, status="degraded"))]
    )
    failed = _v6_outcome([_baseline_event(_baseline_action(task_id="b-1", throughput=770.0, status="failed"))])

    assert degraded["baseline"]["throughput_tok_s_per_gpu"] == 770.0
    assert failed["baseline"] == {
        "throughput_tok_s_per_gpu": None,
        "accuracy": None,
        "ttft_mean_ms": None,
        "e2el_mean_ms": None,
    }


# ``outcome.validation``'s attribution is covered in
# ``test_sbd_v6_stack_ledger.py``, against a recorded ``stack`` event rather
# than a hand-built ``optimizations`` dict. The two tests that lived here fed
# the collector a summary nothing had produced, so they could pin the
# projection's arithmetic and not whether the figures it projected were right.


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
        phase_timeline=[{"action": "conc_sweep", "ts": "2026-08-27T00:30:00+00:00"}],
        conc_sweep_summary=_conc_sweep_section(),
    )

    # The durable event is read first and the stage is projected after it, so
    # ordering by the projection order rather than by the recorded time would
    # put ``install`` in front.
    assert [event["type"] for event in timeline] == ["conc_sweep", "install"]


@pytest.mark.parametrize("projector", ["project_conc_sweep_event"])
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
    stage = {"project_conc_sweep_event": "conc_sweep"}[projector]
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
    section = collect_v6_close(tmp_path, state, warnings)

    # Passed through unchanged -- inventing ``done`` is the one thing this key
    # cannot afford -- but no longer silent about it.
    assert section["steps"][0]["status"] == "completed"
    assert any("unrecognized close step status" in warning for warning in warnings)
