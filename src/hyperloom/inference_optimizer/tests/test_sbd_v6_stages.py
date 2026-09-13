# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``close`` key after the sequencer's last act, and what ships with it.

The breakdown is written from inside the CLOSE sequence, so the section it
carries can only describe the close-out up to its own step. These tests pin the
patch that replaces that snapshot once the sequence has finished, and pin that
the package a consumer unzips carries the patched section rather than the
snapshot. Companion to ``test_sbd_v6_close_recording.py``, which covers what the
sequencer records; and to ``test_sbd_v6_initial.py``, which covers the durable
``install`` / ``model_gate`` events.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


from hyperloom.inference_optimizer.breakdown import exporter, session_package
from hyperloom.inference_optimizer.breakdown.collectors import v6 as v6_collectors
from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_timeline
from hyperloom.inference_optimizer.breakdown.collectors.v6_close import collect_v6_close
from hyperloom.inference_optimizer.breakdown.recorder.close_out import (
    record_close_opened,
    record_close_settled,
    record_close_step,
)
from hyperloom.inference_optimizer.session.sbd_v6 import write_timeline_event_at


#: The sequence the CLOSE sequencer runs, in order. The breakdown is written by
#: ``session_breakdown``, so only the steps up to it exist when it is built.
_CLOSE_STEPS = ("sequencer_started", "fact_finalize", "report", "session_breakdown", "ndjson_drain", "done")

#: Where the sequence stands when ``session_breakdown`` writes the breakdown.
_STEPS_AT_BREAKDOWN = _CLOSE_STEPS[:4]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record_steps(session_dir: Path, steps: tuple[str, ...]) -> None:
    for step in steps:
        record_close_step(
            session_dir,
            step=step,
            status="running" if step == "sequencer_started" else "done",
        )


def _session_with_step_two_breakdown(tmp_path: Path) -> Path:
    """Build a session whose breakdown was written mid-CLOSE, as step 2 does."""
    _write_json(tmp_path / "state.json", {"session_id": "s1", "model_name": "M", "phase": "CLOSE"})
    _write_json(tmp_path / "manifest.json", {"session_id": "s1", "model_name": "M"})
    record_close_opened(tmp_path)
    _record_steps(tmp_path, _STEPS_AT_BREAKDOWN)
    exporter.write_breakdown_json(tmp_path)
    # The sequencer then finishes, recording the remaining steps and its verdict.
    _record_steps(tmp_path, _CLOSE_STEPS[4:])
    record_close_settled(tmp_path, stop_reason="target_reached")
    return tmp_path / exporter.BREAKDOWN_FILENAME


def test_close_patch_replaces_the_step_two_snapshot_with_the_finished_sequence(tmp_path):
    target = _session_with_step_two_breakdown(tmp_path)
    before = json.loads(target.read_text(encoding="utf-8"))
    # No verdict yet, and the steps after this one have not run.
    assert before["close"]["status"] == "running"
    assert [step["step"] for step in before["close"]["steps"]] == list(_STEPS_AT_BREAKDOWN)

    assert exporter.patch_breakdown_close(tmp_path) is True

    after = json.loads(target.read_text(encoding="utf-8"))
    assert after["close"]["status"] == "succeeded"
    assert after["close"]["close_sequence_done"] is True
    assert [step["step"] for step in after["close"]["steps"]] == list(_CLOSE_STEPS)


def test_close_patch_touches_nothing_but_the_close_key(tmp_path):
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
    record_close_opened(tmp_path)
    _record_steps(tmp_path, _CLOSE_STEPS)
    record_close_settled(tmp_path, stop_reason="target_reached")

    assert exporter.patch_breakdown_close(tmp_path) is False


def test_close_patch_leaves_a_payload_that_never_carried_close_alone(tmp_path):
    target = tmp_path / exporter.BREAKDOWN_FILENAME
    _write_json(target, {"schema_version": "hyperloom.session_breakdown.v5.0", "outcome": {}})
    record_close_opened(tmp_path)
    _record_steps(tmp_path, _CLOSE_STEPS)
    record_close_settled(tmp_path, stop_reason="target_reached")

    assert exporter.patch_breakdown_close(tmp_path) is False
    assert "close" not in json.loads(target.read_text(encoding="utf-8"))


def test_close_patch_swallows_a_corrupt_breakdown(tmp_path):
    target = tmp_path / exporter.BREAKDOWN_FILENAME
    target.write_text("{not json", encoding="utf-8")

    assert exporter.patch_breakdown_close(tmp_path) is False
    assert target.read_text(encoding="utf-8") == "{not json"


# what the consumer actually receives
def _packaged_close(session_dir: Path, dest_root: Path) -> tuple[dict, dict]:
    """Return the ``close`` key as delivered, from inside the zip and loose."""
    zip_path = dest_root / session_package.PACKAGE_SUBDIR / "sess-1.zip"
    with zipfile.ZipFile(zip_path) as bundle:
        zipped = json.loads(bundle.read(exporter.BREAKDOWN_FILENAME))
    loose = json.loads((dest_root / exporter.BREAKDOWN_FILENAME).read_text(encoding="utf-8"))
    return zipped["close"], loose["close"]


def test_the_delivered_package_carries_the_finished_close_section(tmp_path):
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
        assert [step["step"] for step in delivered["steps"]] == list(_CLOSE_STEPS)


def test_a_package_built_before_the_patch_ships_the_step_two_snapshot(tmp_path):
    session_dir = tmp_path / "session"
    dest_root = tmp_path / "dest"
    target = _session_with_step_two_breakdown(session_dir)

    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)
    exporter.patch_breakdown_close(session_dir)

    assert json.loads(target.read_text(encoding="utf-8"))["close"]["status"] == "succeeded"
    zipped, loose = _packaged_close(session_dir, dest_root)
    for stale in (zipped, loose):
        assert stale["status"] == "running"
        assert [step["step"] for step in stale["steps"]] == list(_STEPS_AT_BREAKDOWN)


def test_the_delivered_manifest_describes_the_rebuilt_bundle(tmp_path):
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


# outcome
# ---------------------------------------------------------------------------
def _v6_outcome(timeline: list | None = None) -> dict:
    return v6_collectors.collect_v6_outcome(
        session={"stop_reason": "target_reached"},
        close={},
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
    outcome = _v6_outcome([_baseline_event(_baseline_action(task_id="b-1", throughput=800.0))])

    assert outcome["baseline"] == {
        "throughput_tok_s_per_gpu": 800.0,
        "accuracy": 0.81,
        "ttft_mean_ms": 120.0,
        "e2el_mean_ms": 900.0,
    }


def test_outcome_baseline_ignores_a_kernel_probe_that_anchors_nothing():
    outcome = _v6_outcome(
        [
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
    outcome = _v6_outcome(
        [
            _baseline_event(_baseline_action(task_id="b-1", throughput=800.0, end_time="2026-01-01T00:00:00+00:00")),
            _baseline_event(_baseline_action(task_id="b-2", throughput=845.0, end_time="2026-01-01T02:00:00+00:00")),
        ],
    )

    assert outcome["baseline"]["throughput_tok_s_per_gpu"] == 845.0


def test_outcome_baseline_keeps_a_degraded_anchor_and_drops_a_failed_one():
    degraded = _v6_outcome([_baseline_event(_baseline_action(task_id="b-1", throughput=770.0, status="degraded"))])
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
# cross-cutting: ordering, isolation and vocabulary
# ---------------------------------------------------------------------------
def test_the_timeline_is_ordered_by_when_events_happened_not_when_they_were_read(tmp_path):
    for start, end, event_type in (
        ("2026-08-27T00:58:00+00:00", "2026-08-27T00:59:00+00:00", "install"),
        ("2026-08-27T00:30:00+00:00", "2026-08-27T00:31:00+00:00", "model_gate"),
    ):
        write_timeline_event_at(
            tmp_path,
            {
                "type": event_type,
                "kind": event_type,
                "status": "succeeded",
                "start_time": start,
                "end_time": end,
                "ext": {},
            },
        )

    timeline = collect_v6_timeline(tmp_path, [])

    assert [event["type"] for event in timeline] == ["model_gate", "install"]


def test_a_raising_close_collector_cannot_disturb_the_v5_payload(tmp_path, monkeypatch):
    _write_json(tmp_path / "state.json", {"session_id": "s1", "model_name": "M", "phase": "CLOSE"})
    _write_json(tmp_path / "manifest.json", {"session_id": "s1", "model_name": "M"})
    before = exporter.build(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("close exploded")

    monkeypatch.setattr(exporter.collectors, "collect_v6_close", _boom)
    after = exporter.build(tmp_path)

    assert after["timeline"] == before["timeline"]
    assert after["close"] == {}
    assert any("close" in warning for warning in after["metadata"]["warnings"])


def test_an_unknown_close_step_status_is_reported():
    warnings: list[str] = []
    recorded = {
        "status": "succeeded",
        "close_sequence_done": True,
        "steps": [{"step": "report", "status": "completed", "ts": "2026-08-27T02:00:01+00:00"}],
    }
    section = collect_v6_close(warnings, recorded=recorded)

    # Passed through unchanged -- inventing ``done`` is the one thing this key cannot afford -- but no longer silent
    # about it.
    assert section["steps"][0]["status"] == "completed"
    assert any("unrecognized close step status" in warning for warning in warnings)
