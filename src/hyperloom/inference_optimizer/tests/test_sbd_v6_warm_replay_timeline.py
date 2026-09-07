# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``warm_replay`` event.

The event this replaces was projected, and two of the projection's defects are
what most of these tests pin: the gate verdicts were not recorded, so which
gate ended the arc had to be guessed from the terminal status, and the anchor
the gain was measured against was never persisted, so the event back-solved it
from the gain -- which is exact only until the session re-baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.event_finalize import finalize_events
from hyperloom.inference_optimizer.breakdown.recorder.event_timeline import EVENT_STATUS_INTERRUPTED
from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
    GATE_ACCURACY,
    GATE_KEEP_THRESHOLD,
    GATE_QUALITY,
    GATE_TPUT_VALID,
    make_warm_replay_recorder,
)
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "warm_replay"]


def _ext(session_dir: Path) -> dict[str, Any]:
    events = _events(session_dir)
    assert len(events) == 1, f"expected one warm_replay event, got {len(events)}"
    return events[0]["ext"]


def _recorder(**overrides: Any):
    """Build a recorder the way an enqueued replay gets one."""
    kwargs: dict[str, Any] = {
        "phase": "prelude",
        "macro_cycle": 0,
        "task_id": "t-warm-1",
        "tier": "T0",
        "config_source": "recipe-abc",
        "config_donor_tier": "T0",
        "donor": {"canonical_id": "recipe-abc", "session_id": "sess-donor"},
        "expected_gain_pct": 12.0,
        "confidence": 0.82,
        "min_reproduce_pct": 0.8,
        "session_baseline_tput": 15630.0,
        "kernel_count": 0,
        "recipe_suppressed": False,
    }
    kwargs.update(overrides)
    recorder = make_warm_replay_recorder(**kwargs)
    assert recorder is not None
    return recorder


def test_the_replay_lands_on_the_timeline_when_it_is_enqueued(_bound_session):
    """The event opens at enqueue, so its window is not collapsed onto its end."""
    _recorder()
    events = _events(_bound_session)
    assert len(events) == 1
    assert events[0]["ext"]["request"]["config_source"] == "recipe-abc"


def test_the_anchor_the_gain_was_measured_against_is_recorded_verbatim(_bound_session):
    """The enqueue anchor is a recorded fact, not a value re-derived from the gain."""
    recorder = _recorder(session_baseline_tput=15630.0)
    # The replay was enqueued against an older, lower baseline and the session
    # has since re-baselined upward. Back-solving from the gain would recover
    # the current baseline and report a gain the replay never measured.
    recorder.record_measurement(before_tput=14000.0, after_tput=15400.0, gain_pct=10.0)
    recorder.finish({"status": "reproduced"})

    ext = _ext(_bound_session)
    assert ext["measurement"]["before_tput"] == 14000.0
    # Both anchors stay on record, which is what makes the divergence readable.
    assert ext["request"]["session_baseline_tput"] == 15630.0


def test_a_gate_that_never_ran_is_absent_rather_than_failed(_bound_session):
    """An arc cut short leaves no row for the gates it never reached."""
    recorder = _recorder()
    recorder.record_measurement(before_tput=0.0, after_tput=0.0, gain_pct=None)
    recorder.record_gate(GATE_TPUT_VALID, passed=False, reason="invalid_tput tput=0 baseline=0")
    recorder.finish({"status": "failed", "reason": "invalid_tput tput=0 baseline=0"})

    ext = _ext(_bound_session)
    assert [row["gate"] for row in ext["gates"]] == [GATE_TPUT_VALID]
    assert ext["blocked_by"] == GATE_TPUT_VALID


def test_an_eval_that_ran_without_a_score_neither_passes_nor_fails(_bound_session):
    """``None`` is a verdict: no score was readable, not a score that failed."""
    recorder = _recorder()
    recorder.record_measurement(before_tput=14000.0, after_tput=15400.0, gain_pct=10.0, eval_ran=True)
    recorder.record_gate(GATE_ACCURACY, passed=None, reason="keep_verdict_unscored")
    recorder.finish({"status": "reproduced"})

    gate = _ext(_bound_session)["gates"][0]
    assert gate["passed"] is None
    assert gate["reason"] == "keep_verdict_unscored"


def test_a_threshold_rejection_after_an_unscored_eval_names_the_threshold(_bound_session):
    """An outright failure outranks a gate that merely could not rule."""
    recorder = _recorder()
    recorder.record_measurement(before_tput=14000.0, after_tput=14050.0, gain_pct=0.36)
    recorder.record_gate(GATE_ACCURACY, passed=None, reason="keep_verdict_unscored")
    recorder.record_gate(GATE_KEEP_THRESHOLD, passed=False, observed=0.36, threshold=2.0)
    recorder.finish({"status": "not_reproduced", "keep_threshold_pct": 2.0})

    ext = _ext(_bound_session)
    assert ext["blocked_by"] == GATE_KEEP_THRESHOLD
    assert ext["verdict"]["keep_threshold_pct"] == 2.0


def test_a_quality_rejection_reads_as_a_completed_arc_not_a_failure(_bound_session):
    """A replay measured and judged is a finished arc; only a lost measurement fails."""
    recorder = _recorder()
    recorder.record_measurement(before_tput=14000.0, after_tput=17000.0, gain_pct=21.4)
    recorder.record_gate(GATE_QUALITY, passed=False, reason="image-quality gate failed vs baseline reference")
    recorder.finish({"status": "quality_failed"})

    assert _events(_bound_session)[0]["status"] == "rejected"
    assert _ext(_bound_session)["blocked_by"] == GATE_QUALITY


def test_a_reproduced_replay_records_what_the_promotion_moved(_bound_session):
    """Promotion states the checkout and patches it changed, not just that it happened."""
    recorder = _recorder()
    recorder.record_measurement(before_tput=14000.0, after_tput=15400.0, gain_pct=10.0, accuracy=0.71)
    recorder.record_gate(GATE_ACCURACY, passed=True, observed=0.71, threshold=0.65)
    recorder.record_gate(GATE_KEEP_THRESHOLD, passed=True, observed=10.0, threshold=2.0)
    recorder.record_promotion(
        promoted_checkout="/opt/framework/sglang-warm",
        replayed_patch_refs=["/kb/download/fix-attn.patch"],
        stack_entry={"gain_pct": 10.0},
    )
    recorder.finish({"status": "reproduced", "settled_at": "2026-09-04T13:00:00Z"})

    ext = _ext(_bound_session)
    assert _events(_bound_session)[0]["status"] == "succeeded"
    assert ext["blocked_by"] is None
    assert ext["promotion"]["promoted_checkout"] == "/opt/framework/sglang-warm"
    assert ext["promotion"]["replayed_patch_refs"] == ["/kb/download/fix-attn.patch"]
    assert ext["verdict"]["settled_at"] == "2026-09-04T13:00:00Z"


def test_a_reproduced_replay_with_nothing_to_replay_is_degraded(_bound_session):
    """Measured uplift the session cannot act on is neither a success nor a failure."""
    recorder = _recorder()
    recorder.record_measurement(before_tput=14000.0, after_tput=15400.0, gain_pct=10.0)
    recorder.finish({"status": "reproduced_but_no_params", "reason": "task.params missing extra_server_args"})

    assert _events(_bound_session)[0]["status"] == "degraded"


def test_an_outcome_status_the_module_does_not_know_does_not_read_as_success(_bound_session):
    """An unmapped status is an arc this module cannot vouch for."""
    recorder = _recorder()
    recorder.finish({"status": "some_new_status_nobody_taught_us"})

    assert _events(_bound_session)[0]["status"] == "failed"


def test_a_replay_killed_mid_flight_is_recovered_as_interrupted(_bound_session):
    """A recorder that never closed leaves an event finalize can still settle."""
    recorder = _recorder()
    recorder.record_measurement(before_tput=14000.0, after_tput=15400.0, gain_pct=10.0)
    recorder.record_gate(GATE_ACCURACY, passed=True, observed=0.71, threshold=0.65)

    assert finalize_events(_bound_session) == [recorder.event_id]
    event = _events(_bound_session)[0]
    assert event["status"] == EVENT_STATUS_INTERRUPTED
    # The rows it did write survive: the point of recovery is the evidence, not
    # the status.
    assert event["ext"]["measurement"]["before_tput"] == 14000.0
    assert [row["gate"] for row in event["ext"]["gates"]] == [GATE_ACCURACY]


def test_a_crash_is_told_apart_from_a_kill(_bound_session):
    """An executor that raised records a failure row; a kill leaves none."""
    recorder = _recorder()
    recorder.finish_crashed(RuntimeError("boom"))

    ext = _ext(_bound_session)
    assert _events(_bound_session)[0]["status"] == "failed"
    assert ext["failure"]["error_class"] == "RuntimeError"


def test_the_gates_read_back_in_the_order_they_were_evaluated(_bound_session):
    """Assembly preserves evaluation order, which is what makes the arc readable."""
    recorder = _recorder()
    recorder.record_measurement(before_tput=14000.0, after_tput=15400.0, gain_pct=10.0)
    for gate in (GATE_TPUT_VALID, GATE_QUALITY, GATE_ACCURACY, GATE_KEEP_THRESHOLD):
        recorder.record_gate(gate, passed=True)
    recorder.finish({"status": "reproduced"})

    assert [row["gate"] for row in _ext(_bound_session)["gates"]] == [
        GATE_TPUT_VALID,
        GATE_QUALITY,
        GATE_ACCURACY,
        GATE_KEEP_THRESHOLD,
    ]


def test_recording_the_same_gate_twice_settles_rather_than_duplicates(_bound_session):
    """A gate re-ruled on better evidence is one row, not two."""
    recorder = _recorder()
    recorder.record_gate(GATE_ACCURACY, passed=None, reason="score pending")
    recorder.record_gate(GATE_ACCURACY, passed=True, observed=0.71, threshold=0.65)
    recorder.finish({"status": "reproduced"})

    gates = _ext(_bound_session)["gates"]
    assert len(gates) == 1
    assert gates[0]["passed"] is True
