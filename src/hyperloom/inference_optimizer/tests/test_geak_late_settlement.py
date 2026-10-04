# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Late GEAK settlement updates the original durable KERNEL event."""

from __future__ import annotations

from copy import deepcopy


from hyperloom.inference_optimizer.breakdown.recorder import kernel_event
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.tests.test_geak_promotion_retention import (
    promotion as promotion,
)
from hyperloom.orchestrator.phases.geak_rebench import geak_candidate_is_adjudicated
from hyperloom.orchestrator.state.shared_state import SharedState


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
