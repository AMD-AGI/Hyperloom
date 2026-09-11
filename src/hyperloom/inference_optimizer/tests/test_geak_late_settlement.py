# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Late GEAK settlement updates the original durable KERNEL event."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.recorder import kernel_event
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.tests.test_geak_gain_alignment import _journey_with_validated_keeps
from hyperloom.inference_optimizer.tests.test_geak_promotion_retention import promotion as promotion, rebench_task
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.geak_rebench import geak_candidate_is_adjudicated
from hyperloom.orchestrator.phases.machine_state import PHASE_KERNEL_AGENT, PHASE_SWEEP
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["closed", "resumed", "later_cycle", "legacy_resume"])
@pytest.mark.parametrize("verdict", ["accuracy_drop", "no_material", "overlay_unproven"])
async def test_late_rejection_after_kernel_transition(promotion, monkeypatch, lifecycle, verdict):
    coord, result, recorder = promotion
    state = coord.shared_state
    recorder.begin(tput_before=110.0)
    result["final_throughput_tok_s"] = 150.0
    result["kernel_journey_path"] = _journey_with_validated_keeps(coord.session_dir, [1.5])
    raw_journey = json.loads(Path(result["kernel_journey_path"]).read_text())
    if verdict == "no_material":
        result.update(accepted_config={}, accepted_kernels=[], final_overlay="")
    state.geak_result = result
    coord._record_geak_candidate(result)
    coord._record_geak_kernel_journey(result)
    task = rebench_task(result)
    if verdict == "overlay_unproven":
        task.params.update(expected_overlay="", expected_overlay_digest="")
    state.geak_pending["revalidation_task_id"] = task.task_id

    monkeypatch.setattr(coord, "_on_enter_sweep", AsyncMock())
    await coord.phase_machine._on_phase_entered(
        from_phase=PHASE_KERNEL_AGENT, to_phase=PHASE_SWEEP, reason="geak_revalidation_queued"
    )
    assert coord.phase_kernel._kernel_timeline() is None
    before = next(event for event in read_timeline_events(coord.session_dir) if event["id"] == recorder.event_id)
    assert before["ext"]["geak"]["attempts"]["counts"]["integrated"] == 1
    header_before = kernel_event.event_parts((kernel_event.SECTION_EVENT,), event=recorder.event_id)
    Path(result["kernel_journey_path"]).unlink()

    if lifecycle == "legacy_resume":
        state.geak_result.pop("kernel_event_id", None)
    state.save(coord.session_dir)
    if lifecycle != "closed":
        old_coord = coord
        coord = Coordinator.__new__(Coordinator)
        coord.session_dir = old_coord.session_dir
        coord.shared_state = state = SharedState.load_or_init(coord.session_dir)
        assert coord.phase_kernel._kernel_timeline() is None

    later = None
    later_before = None
    if lifecycle == "later_cycle":
        state.macro_cycle = 1
        later = kernel_event.make_kernel_recorder(macro_cycle=1, route=kernel_event.ROUTE_GEAK)
        assert later is not None
        later.begin(tput_before=110.0)
        later.record_geak_attempts(raw_journey)
        coord.phase_kernel._kernel_timeline_recorder = later
        later_before = kernel_event.event_parts((kernel_event.SECTION_GEAK_ATTEMPT,), event=later.event_id)

    native_result = {
        "status": "succeeded",
        "output_throughput": 120.0,
        "best_variant": {"output_throughput": 120.0, "accuracy": 0.95, "fingerprint": "candidate"},
        "winners": [],
    }
    if verdict == "accuracy_drop":
        native_result["per_variant_outcomes"] = [
            {"outcome": "REVERT", "reason": "accuracy_drop", "fingerprint": "candidate"}
        ]
    replay = AsyncMock(side_effect=AssertionError("A conclusive native result cannot invoke fallback"))
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._geak_sweep.sweep_via_geak", replay)
    await coord._promote_to_shared_state("explore", native_result, task=task)
    replay.assert_not_awaited()
    assert state.geak_pending == {}
    assert state.current_best["tput"] == (120.0 if verdict == "overlay_unproven" else 110.0)

    events = read_timeline_events(coord.session_dir)
    settled = next(event for event in events if event["id"] == recorder.event_id)
    assert {key: value for key, value in settled.items() if key != "ext"} == {
        key: value for key, value in before.items() if key != "ext"
    }
    assert kernel_event.event_parts((kernel_event.SECTION_EVENT,), event=recorder.event_id) == header_before
    attempts = settled["ext"]["geak"]["attempts"]
    assert attempts["counts"]["integrated"] == 0
    e2e = attempts["kernels"][0]["e2e"]
    assert e2e["decision"] == "REVERT"
    assert e2e["integrated"] is False
    assert e2e["validated"] is False
    assert e2e["e2e_gain_pct"] is None
    assert e2e["self_reported_e2e_gain_pct"] == 50.0

    coord.phase_kernel._reject_geak_kernel_journey(
        state.geak_result,
        measured_tput=120.0,
        current_best_tput=110.0,
        provenance="repeated_writeback",
    )
    assert (
        next(event for event in read_timeline_events(coord.session_dir) if event["id"] == recorder.event_id) == settled
    )
    if later is not None:
        assert kernel_event.event_parts((kernel_event.SECTION_GEAK_ATTEMPT,), event=later.event_id) == later_before
        later.finish(tput_after=110.0)
    recorder.record_geak_attempts(raw_journey)
    state.save(coord.session_dir)
    exported = exporter.build(coord.session_dir)
    exported_event = next(event for event in exported["timeline"] if event["id"] == recorder.event_id)
    assert exported_event["ext"]["geak"]["attempts"]["counts"]["integrated"] == 0
    assert exported_event["ext"]["geak"]["attempts"]["kernels"][0]["e2e"] == e2e
    if later is not None:
        other = next(event for event in exported["timeline"] if event["id"] == later.event_id)
        assert other["ext"]["geak"]["attempts"]["counts"]["integrated"] == 1


def test_rejection_annotations_do_not_reopen_an_adjudicated_candidate(promotion):
    coord, result, _recorder = promotion
    result["final_throughput_tok_s"] = 150.0
    raw = deepcopy(result)
    coord._record_geak_candidate(result)
    coord._reject_geak_promotion(result, measured_tput=120.0, current_best_tput=110.0, reason="accuracy_drop")
    coord.shared_state.save(coord.session_dir)
    settled = SharedState.load_or_init(coord.session_dir).geak_result
    assert settled["kernel_event_id"] == kernel_event.kernel_event_id(0)
    assert settled["final_validation"]["decision"] == "REJECTED"
    assert geak_candidate_is_adjudicated(settled, raw, harness_can_replay=False)
    assert not geak_candidate_is_adjudicated(
        settled, {**raw, "final_throughput_tok_s": 160.0}, harness_can_replay=False
    )


def test_late_rejection_does_not_create_an_event(tmp_path):
    from hyperloom.inference_optimizer.session.session_binding import session_scope

    with session_scope(tmp_path):
        kernel_event.reject_geak_attempts(
            event=kernel_event.kernel_event_id(7),
            measured_tput=90.0,
            current_best_tput=100.0,
            provenance="native_rebench",
            rejection_reason="no_gain",
        )
        assert read_timeline_events(tmp_path) == []
        assert kernel_event.event_parts(kernel_event.EVENT_SECTIONS) == {
            section: [] for section in kernel_event.EVENT_SECTIONS
        }
