# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``enablement`` event.

These tests pin what a projection of ``SharedState.enablement`` could not say: which round landed the
fix, what each round was pointed at, why the lane opened, and the difference between a round that was
refused and one that never happened. The lane's facts come from six modules on different ticks, so no
recorder object is held and every entry point opens the event idempotently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import enablement_event
from hyperloom.inference_optimizer.breakdown.recorder.assembler import enablement_event_parts
from hyperloom.inference_optimizer.breakdown.recorder.event_finalize import finalize_events
from hyperloom.inference_optimizer.breakdown.recorder.event_timeline import EVENT_STATUS_INTERRUPTED
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "enablement"]


def _event(session_dir: Path) -> dict[str, Any]:
    events = _events(session_dir)
    assert len(events) == 1, f"expected one enablement event, got {len(events)}"
    return events[0]


def _ext(session_dir: Path) -> dict[str, Any]:
    """Assemble the lane's ``ext`` the way finalize would, since an open event holds only fragments."""
    ext, _status = enablement_event.assemble_enablement_ext(
        enablement_event_parts(),
        event=enablement_event.enablement_event_id(),
    )
    return ext


def _boot_trigger(**overrides: Any) -> None:
    """Open the lane the way an unpromotable baseline does."""
    kwargs: dict[str, Any] = {
        "origin": enablement_event.ORIGIN_BOOT,
        "mode": "all",
        "kind": "import_error",
        "evidence": "ImportError: cannot import name 'fused_moe'",
    }
    kwargs.update(overrides)
    enablement_event.record_trigger(**kwargs)


def _kept(**overrides: Any) -> dict[str, Any]:
    """An integrate_patch result the gate KEPT."""
    result: dict[str, Any] = {
        "enablement": True,
        "status": enablement_event.ROUND_KEPT,
        "patches_applied": ["/s/patches/moe.diff"],
        "setup_commands_applied": ["pip install -e ."],
        "enablement_effective_config": {"extra_server_args": "--tp 8", "extra_envs": {"HIP_VISIBLE_DEVICES": "0"}},
    }
    result.update(overrides)
    return result


def test_the_lane_lands_on_the_timeline_when_it_is_triggered(_bound_session):
    _boot_trigger()

    event = _event(_bound_session)
    assert event["status"] == "running"
    assert event["ext"]["origin"] == enablement_event.ORIGIN_BOOT


def test_every_entry_point_writes_into_one_event(_bound_session):
    _boot_trigger()
    enablement_event.record_dispatch(task_id="spec-1", attempt=1, failure_kind="import_error")
    enablement_event.record_build(task_id="build-1", entry={"ok": True, "action": {"component": "aiter"}})
    enablement_event.record_revalidation(generation=1, task_id="base-9")
    enablement_event.record_human_review(digest="d0", failure_kind="UNKNOWN")

    assert len(_events(_bound_session)) == 1


def test_a_lane_nothing_triggered_leaves_no_event(_bound_session):
    assert _events(_bound_session) == []


def test_the_trigger_that_opened_the_lane_is_the_one_it_keeps(_bound_session):
    enablement_event.record_trigger(
        origin=enablement_event.ORIGIN_EVAL,
        mode="all",
        kind="accuracy_below_floor",
        evidence="gsm8k: 0.21",
        observed_accuracy=0.21,
        accuracy_floor=0.5,
        observed_task="gsm8k",
        observed_metric="exact_match",
    )
    enablement_event.record_trigger(
        origin=enablement_event.ORIGIN_EVAL,
        mode="all",
        kind="accuracy_unavailable",
        evidence="",
    )

    trigger = _ext(_bound_session)["trigger"]
    assert trigger["kind"] == "accuracy_below_floor"
    assert trigger["observed_accuracy"] == 0.21
    assert trigger["accuracy_floor"] == 0.5
    assert trigger["observed_task"] == "gsm8k"


def test_the_origin_survives_the_lane_succeeding(_bound_session):
    enablement_event.record_trigger(
        origin=enablement_event.ORIGIN_EVAL,
        mode="all",
        kind="accuracy_below_floor",
    )
    enablement_event.finish(outcome=enablement_event.OUTCOME_SUCCEEDED, reason="revalidation promoted")

    assert _ext(_bound_session)["origin"] == enablement_event.ORIGIN_EVAL


def test_the_admitted_mode_is_recorded_even_when_it_is_off(_bound_session):
    _boot_trigger(mode="off")

    assert _ext(_bound_session)["mode"] == "off"


def test_a_trigger_log_is_kept_by_its_tail(_bound_session):
    _boot_trigger(evidence="x" * 5000 + "ImportError: the gap")

    excerpt = _ext(_bound_session)["trigger"]["evidence_excerpt"]
    assert len(excerpt) == enablement_event.MAX_LOG_EXCERPT_CHARS
    assert excerpt.endswith("ImportError: the gap")


def test_a_round_is_recorded_with_what_it_was_pointed_at(_bound_session):
    _boot_trigger()
    enablement_event.record_dispatch(
        task_id="spec-1",
        attempt=1,
        failure_kind="missing_fused_moe",
        launch_log="ImportError: cannot import name 'fused_moe'",
        candidate_refs=["v0.4.1", "v0.4.2"],
    )

    rows = _ext(_bound_session)["attempts"]["rows"]
    assert len(rows) == 1
    assert rows[0]["failure_kind"] == "missing_fused_moe"
    assert rows[0]["candidate_refs"] == ["v0.4.1", "v0.4.2"]
    assert "fused_moe" in rows[0]["launch_log_excerpt"]


def test_the_dispatch_and_the_verdict_settle_one_row(_bound_session):
    _boot_trigger()
    enablement_event.record_dispatch(task_id="spec-1", attempt=1, failure_kind="missing_fused_moe")
    enablement_event.record_round(
        task_id="spec-1",
        attempt=1,
        result=_kept(),
        stall_streak=0,
        succeeded=True,
    )

    rows = _ext(_bound_session)["attempts"]["rows"]
    assert len(rows) == 1
    assert rows[0]["failure_kind"] == "missing_fused_moe"
    assert rows[0]["status"] == enablement_event.ROUND_KEPT
    assert rows[0]["landed"] is True


def test_a_round_dispatched_and_never_ruled_says_so(_bound_session):
    _boot_trigger()
    enablement_event.record_dispatch(task_id="spec-1", attempt=1, failure_kind="missing_fused_moe")

    finalize_events(_bound_session)

    event = _event(_bound_session)
    assert event["status"] == EVENT_STATUS_INTERRUPTED
    attempts = event["ext"]["attempts"]
    assert attempts["count"] == 1
    assert attempts["settled"] == 0
    assert attempts["rows"][0].get("status") in (None, "")


def test_the_round_that_landed_the_fix_is_identifiable(_bound_session):
    _boot_trigger()
    for attempt, status in enumerate(
        (
            enablement_event.ROUND_REVERTED,
            enablement_event.ROUND_ADVANCED,
            enablement_event.ROUND_KEPT,
        ),
        start=1,
    ):
        task = f"spec-{attempt}"
        enablement_event.record_dispatch(task_id=task, attempt=attempt, failure_kind=f"gap{attempt}")
        enablement_event.record_round(
            task_id=task,
            attempt=attempt,
            result={"enablement": True, "status": status, "patches_applied": [f"/s/p{attempt}.diff"]},
            stall_streak=1 if status == enablement_event.ROUND_REVERTED else 0,
            succeeded=status == enablement_event.ROUND_KEPT,
        )

    attempts = _ext(_bound_session)["attempts"]
    assert attempts["count"] == 3
    assert attempts["landed"] == 1
    assert attempts["advanced"] == 1
    assert [row["attempt"] for row in attempts["rows"]] == [1, 2, 3]
    landed = [row for row in attempts["rows"] if row["landed"]]
    assert [row["attempt"] for row in landed] == [3]


def test_the_gap_a_round_revealed_is_not_the_gap_it_faced(_bound_session):
    _boot_trigger(evidence="gap 1: ImportError fused_moe")
    enablement_event.record_dispatch(task_id="spec-1", attempt=1, launch_log="gap 1: ImportError fused_moe")
    enablement_event.record_round(
        task_id="spec-1",
        attempt=1,
        result={
            "enablement": True,
            "status": enablement_event.ROUND_ADVANCED,
            "enablement_launch_log": "gap 2: AttributeError rope_scaling",
        },
        stall_streak=0,
        succeeded=False,
    )

    ext = _ext(_bound_session)
    row = ext["attempts"]["rows"][0]
    assert "fused_moe" in row["launch_log_excerpt"]
    assert "rope_scaling" in row["next_launch_log_excerpt"]
    assert "fused_moe" in ext["trigger"]["evidence_excerpt"]


def test_a_round_the_lane_synthesised_keeps_its_own_row(_bound_session):
    _boot_trigger()
    enablement_event.record_dispatch(task_id="spec-1", attempt=1, failure_kind="gap1")
    enablement_event.record_round(
        task_id="",
        attempt=2,
        result={"enablement": True, "status": "reverted", "reason": "round_finished_without_rearm"},
        stall_streak=1,
        succeeded=False,
    )

    rows = _ext(_bound_session)["attempts"]["rows"]
    assert len(rows) == 2
    assert rows[0]["failure_kind"] == "gap1"
    assert rows[1]["reason"] == "round_finished_without_rearm"


def test_a_round_records_the_products_it_contributed(_bound_session):
    _boot_trigger()
    enablement_event.record_round(
        task_id="spec-1",
        attempt=1,
        result=_kept(
            artifacts_applied=[
                {"target": "/fw/layers/moe.py", "rel_target": "layers/moe.py", "kind": "replace", "backup": "/tmp/b"}
            ],
            enablement_kept_stack_action={
                "kind": "install",
                "framework": "sglang",
                "capability": "fused_moe",
                "acquisition_method": "wheel",
            },
            enablement_active_runtime={"venv_root": "/venv/a", "installed_versions": {"sglang": "0.4.2"}},
            enablement_localization_manifest={"files": 3},
        ),
        stall_streak=0,
        succeeded=True,
    )

    row = _ext(_bound_session)["attempts"]["rows"][0]
    assert row["patches_applied"] == ["/s/patches/moe.diff"]
    assert row["stack_action"]["capability"] == "fused_moe"
    assert row["runtime"]["venv_root"] == "/venv/a"
    assert row["localization_manifest"] == {"files": 3}
    assert row["effective_config"]["extra_server_args"] == "--tp 8"
    # The backup bookkeeping made the install reversible; it is not a fact about the repair.
    assert row["artifacts_applied"] == [
        {"target": "/fw/layers/moe.py", "rel_target": "layers/moe.py", "kind": "replace"}
    ]


def test_the_stall_streak_is_recorded_per_round(_bound_session):
    _boot_trigger()
    for attempt in (1, 2):
        enablement_event.record_round(
            task_id=f"spec-{attempt}",
            attempt=attempt,
            result={"enablement": True, "status": "reverted"},
            stall_streak=attempt,
            succeeded=False,
        )

    rows = _ext(_bound_session)["attempts"]["rows"]
    assert [row["stall_streak_after"] for row in rows] == [1, 2]


def test_a_build_the_lane_ran_is_recorded_with_its_verdict(_bound_session):
    _boot_trigger()
    enablement_event.record_build(
        task_id="build-1",
        entry={
            "ok": False,
            "failure_class": "compile_error",
            "failure_summary": "hipcc: no such arch",
            "action": {"component": "aiter", "gpu_arch": "gfx942", "max_jobs": 32},
            "installed_versions": {"aiter_ref": "abc123"},
            "build_log_path": "/s/enablement/builds/build-1/build.log",
        },
        novelty_key="aiter:abc123",
    )

    builds = _ext(_bound_session)["builds"]
    assert builds["count"] == 1
    assert builds["failed"] == 1
    row = builds["rows"][0]
    assert row["component"] == "aiter"
    assert row["ref"] == "abc123"
    assert row["gpu_arch"] == "gfx942"
    assert row["novelty_key"] == "aiter:abc123"
    assert row["failure_class"] == "compile_error"


def test_a_build_with_no_verdict_is_not_a_failed_build(_bound_session):
    _boot_trigger()
    enablement_event.record_build(task_id="build-1", entry={"action": {"component": "vllm"}})

    builds = _ext(_bound_session)["builds"]
    assert builds["count"] == 1
    assert builds["failed"] == 0
    assert "ok" not in builds["rows"][0]


def test_a_revalidation_window_records_opening_and_closing(_bound_session):
    enablement_event.record_trigger(
        origin=enablement_event.ORIGIN_EVAL, mode="all", kind="accuracy_below_floor", accuracy_floor=0.5
    )
    enablement_event.record_revalidation(generation=1, task_id="base-9", config_path="/s/accepted.yaml")
    enablement_event.record_revalidation_outcome(
        generation=1, promoted=True, task_id="base-9", accuracy=0.71, accuracy_floor=0.5
    )

    revalidations = _ext(_bound_session)["revalidations"]
    assert revalidations["count"] == 1
    assert revalidations["promoted"] == 1
    row = revalidations["rows"][0]
    assert row["config_path"] == "/s/accepted.yaml"
    assert row["accuracy"] == 0.71
    assert row["opened_at"] and row["closed_at"]


def test_each_generation_is_its_own_window(_bound_session):
    enablement_event.record_trigger(origin=enablement_event.ORIGIN_EVAL, mode="all", kind="accuracy_below_floor")
    enablement_event.record_revalidation(generation=1, task_id="base-1")
    enablement_event.record_revalidation_outcome(generation=1, promoted=False, reason="accuracy below floor")
    enablement_event.record_revalidation(generation=2, task_id="base-2")
    enablement_event.record_revalidation_outcome(generation=2, promoted=True, accuracy=0.8)

    rows = _ext(_bound_session)["revalidations"]["rows"]
    assert [row["generation"] for row in rows] == [1, 2]
    assert [row["promoted"] for row in rows] == [False, True]


def test_a_window_the_run_stopped_is_not_a_window_that_failed(_bound_session):
    enablement_event.record_trigger(origin=enablement_event.ORIGIN_EVAL, mode="all", kind="accuracy_below_floor")
    enablement_event.record_revalidation(generation=1, task_id="base-1")
    enablement_event.record_revalidation_outcome(generation=1, promoted=False, reason="stopped by the run")

    row = _ext(_bound_session)["revalidations"]["rows"][0]
    assert row["promoted"] is False
    assert row["reason"] == "stopped by the run"
    assert "error_class" not in row


def test_a_failure_too_unclassifiable_to_dispatch_is_still_recorded(_bound_session):
    _boot_trigger()
    enablement_event.record_human_review(
        digest="deadbeef",
        failure_kind="UNKNOWN",
        reason="did not match any actionable enablement signature",
        signature={"kind": "UNKNOWN", "raw_excerpt": "Segmentation fault"},
    )

    review = _ext(_bound_session)["human_review"]
    assert review["count"] == 1
    assert review["rows"][0]["failure_kind"] == "UNKNOWN"
    assert review["rows"][0]["signature"]["raw_excerpt"] == "Segmentation fault"
    # No round was dispatched, which is the point of the row.
    assert _ext(_bound_session)["attempts"]["count"] == 0


def test_the_same_failure_is_filed_once(_bound_session):
    _boot_trigger()
    for _ in range(3):
        enablement_event.record_human_review(digest="deadbeef", failure_kind="UNKNOWN")

    assert _ext(_bound_session)["human_review"]["count"] == 1


def test_a_lane_that_landed_its_repair_succeeds(_bound_session):
    _boot_trigger()
    enablement_event.record_round(task_id="spec-1", attempt=1, result=_kept(), stall_streak=0, succeeded=True)
    enablement_event.finish(
        outcome=enablement_event.OUTCOME_SUCCEEDED,
        reason="kept",
        kept_patches=["/s/patches/moe.diff"],
        setup_commands=["pip install -e ."],
        accepted_config={"extra_server_args": "--tp 8", "extra_envs": {}},
        setting_script="reports/enablement/enablement_setting.sh",
        active_runtime={"venv_root": "/venv/a"},
        attempt_runtimes=[{"venv_root": "/venv/old"}, {"venv_root": "/venv/a"}],
    )

    event = _event(_bound_session)
    assert event["status"] == "succeeded"
    result = event["ext"]["result"]
    assert result["outcome"] == enablement_event.OUTCOME_SUCCEEDED
    assert result["kept_patches"] == ["/s/patches/moe.diff"]
    assert result["setting_script"] == "reports/enablement/enablement_setting.sh"
    assert [runtime["promoted"] for runtime in result["attempt_runtimes"]] == [False, True]


def test_a_lane_that_hit_the_stall_cap_fails(_bound_session):
    _boot_trigger()
    enablement_event.finish(
        outcome=enablement_event.OUTCOME_STALLED,
        reason="enablement_stalled",
        stall_streak=5,
    )

    event = _event(_bound_session)
    assert event["status"] == "failed"
    assert event["ext"]["result"]["reason"] == "enablement_stalled"
    assert event["ext"]["result"]["stall_streak"] == 5


def test_a_lane_the_session_outlived_is_interrupted_not_judged(_bound_session):
    _boot_trigger()
    enablement_event.record_dispatch(task_id="spec-1", attempt=1, failure_kind="gap1")
    enablement_event.record_round(
        task_id="spec-1",
        attempt=1,
        result={"enablement": True, "status": enablement_event.ROUND_ADVANCED},
        stall_streak=0,
        succeeded=False,
    )

    finalize_events(_bound_session)

    event = _event(_bound_session)
    assert event["status"] == EVENT_STATUS_INTERRUPTED
    assert event["ext"]["result"] is None
    assert event["ext"]["attempts"]["advanced"] == 1


def test_the_lane_closes_the_event_it_opened(_bound_session):
    _boot_trigger()
    enablement_event.finish(outcome=enablement_event.OUTCOME_SUCCEEDED, reason="kept")

    events = _events(_bound_session)
    assert len(events) == 1
    assert events[0]["start_time"] and events[0]["end_time"]


def test_a_lane_that_closed_having_run_no_round_is_skipped(_bound_session):
    _boot_trigger()
    enablement_event.finish(outcome="", reason="nothing to author against")

    assert _event(_bound_session)["status"] == "skipped"


def test_engagement_is_a_property_of_the_event_existing(_bound_session):
    _boot_trigger()

    assert _ext(_bound_session)["engaged"] is True


def test_nothing_recorded_outside_a_session_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.session.session_binding.bound_session_or_none",
        lambda: None,
    )
    _boot_trigger()
    enablement_event.record_dispatch(task_id="spec-1", attempt=1)
    enablement_event.record_archive(task_id="spec-1", attempt=1, files=[{"path": "p", "role": "patch"}])
    enablement_event.record_round(task_id="spec-1", attempt=1, result={}, stall_streak=0, succeeded=False)
    enablement_event.record_build(task_id="build-1", entry={"ok": True})
    enablement_event.record_revalidation(generation=1)
    enablement_event.record_revalidation_outcome(generation=1, promoted=False)
    enablement_event.record_human_review(digest="d", failure_kind="UNKNOWN")
    enablement_event.finish(outcome=enablement_event.OUTCOME_SUCCEEDED)


def test_a_malformed_result_does_not_cost_the_round_its_row(_bound_session):
    _boot_trigger()
    enablement_event.record_round(task_id="spec-1", attempt=1, result=None, stall_streak=0, succeeded=False)

    assert _ext(_bound_session)["attempts"]["count"] == 1


def test_the_archive_merges_onto_the_round_the_dispatch_opened(_bound_session):
    _boot_trigger()
    enablement_event.record_dispatch(task_id="spec-1", attempt=1)
    enablement_event.record_archive(
        task_id="spec-1",
        attempt=1,
        files=[
            {"path": "reports/enablement/spec-1/patches/moe.diff", "role": "patch"},
            {"path": "reports/enablement/spec-1/launch_config.yaml", "role": "launch_config"},
            {"path": "reports/enablement/spec-1/server.log", "role": "server_log"},
        ],
    )
    enablement_event.record_round(
        task_id="spec-1",
        attempt=1,
        result=_kept(enablement_accepted_config_path="/s/runs/integrate_patch/spec-1/integrate_patch.with_envs.yaml"),
        stall_streak=0,
        succeeded=True,
    )

    rows = _ext(_bound_session)["attempts"]["rows"]
    assert len(rows) == 1
    assert rows[0]["files"] == [
        {"path": "reports/enablement/spec-1/patches/moe.diff", "role": "patch"},
        {"path": "reports/enablement/spec-1/launch_config.yaml", "role": "launch_config"},
        {"path": "reports/enablement/spec-1/server.log", "role": "server_log"},
    ]
    # Read out of the manifest, so it cannot name a copy the manifest lacks.
    assert rows[0]["accepted_config_path"] == "reports/enablement/spec-1/launch_config.yaml"
    # The round's own paths stay, as identity rather than as a way to fetch.
    assert rows[0]["patches_applied"] == ["/s/patches/moe.diff"]


def test_a_copy_the_archive_refused_is_named_nowhere(_bound_session):
    _boot_trigger()
    enablement_event.record_archive(task_id="spec-1", attempt=1, files=[])
    enablement_event.record_round(
        task_id="spec-1",
        attempt=1,
        result=_kept(enablement_accepted_config_path="/s/runs/integrate_patch/spec-1/integrate_patch.with_envs.yaml"),
        stall_streak=0,
        succeeded=True,
    )

    row = _ext(_bound_session)["attempts"]["rows"][0]
    assert row["files"] == []
    assert row["accepted_config_path"] is None
