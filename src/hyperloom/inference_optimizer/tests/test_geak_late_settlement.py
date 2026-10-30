# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Late GEAK settlement updates the original durable KERNEL event."""

from __future__ import annotations

from copy import deepcopy
import json
import subprocess

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import kernel_event
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.tests.test_geak_promotion_retention import (
    promotion as promotion,
)
from hyperloom.orchestrator.phases.geak_rebench import geak_candidate_is_adjudicated
from hyperloom.orchestrator.phases.kernel_geak_promotion import GeakRunnerFacts
from hyperloom.orchestrator.state.shared_state import SharedState


def test_rejection_annotations_do_not_reopen_an_adjudicated_candidate(promotion):
    coord, result, _recorder = promotion
    result["final_throughput_tok_s"] = 150.0
    raw = deepcopy(result)
    coord.phase_kernel._record_geak_candidate(result)
    coord.phase_kernel.reject_geak_promotion(
        result, measured_tput=120.0, current_best_tput=110.0, reason="accuracy_drop"
    )
    coord.shared_state.save(coord.session_dir)
    settled = SharedState.load_or_init(coord.session_dir).geak_result
    assert settled["kernel_event_id"] == kernel_event.kernel_event_id(0)
    assert settled["final_validation"]["decision"] == "REJECTED"
    assert geak_candidate_is_adjudicated(settled, raw, harness_can_replay=False)
    assert not geak_candidate_is_adjudicated(
        settled, {**raw, "final_throughput_tok_s": 160.0}, harness_can_replay=False
    )


@pytest.mark.asyncio
async def test_a_settled_agentx_candidate_is_not_re_promoted_on_the_next_kernel_entry(promotion, monkeypatch):
    """A settled candidate stays settled across a re-entry: the runner's timing never enters the record.

    Under AgentX the GEAK harness cannot replay the workload, so recovery compares the persisted
    ``geak_result`` against the ``result.json`` still on disk. Anything the phase merges into the
    persisted value that the file does not carry -- the dispatch's ``runner_timeout_s``,
    ``started_at`` and the rest -- makes that comparison fail, and the settled candidate is promoted
    and re-measured on every entry.
    """
    coord, result, _recorder = promotion
    state = coord.shared_state
    state.benchmark_mode = "agentx"
    phase = coord.phase_kernel
    # The replay refused because this overlay will not load; while that holds, the refusal is reusable.
    result["final_overlay"] = str(coord.session_dir / "overlay-that-never-loaded")
    geak_dir = coord.session_dir / "geak"
    geak_dir.mkdir(parents=True, exist_ok=True)
    (geak_dir / "result.json").write_text(json.dumps({**result, "returncode": 0}), encoding="utf-8")
    handoff = {"exp_root": str(coord.session_dir), "eval_dir": ""}
    exited_ok = subprocess.CompletedProcess(["geak_runner"], 0, "", "")

    settled = phase._settle_geak_runner_exit(
        exited_ok,
        handoff=handoff,
        agentx=True,
        facts=GeakRunnerFacts(
            started_at="2026-09-02T00:00:00+00:00", duration_sec=1200.0, runner_timeout_sec=3600, kill_timeout_sec=3900
        ),
    )
    assert settled is not None
    state.geak_result = settled

    async def refuse_as_incomparable(*, reason):
        return {"validated": False, "status": "incomparable", "reason": f"declined: {reason}"}

    monkeypatch.setattr(coord.writeback, "validate_geak_via_geak_harness", refuse_as_incomparable)
    await phase._revalidate_on_geak_harness(decline_reason="geak_no_grid_material")
    persisted = SharedState.load_or_init(coord.session_dir).geak_result
    assert persisted["revalidation_error_class"] == "incomparable"

    # Re-enter KERNEL: the same result.json, a clean exit, and a differently timed dispatch.
    state.geak_result = persisted
    again = phase._settle_geak_runner_exit(
        exited_ok,
        handoff=handoff,
        agentx=True,
        facts=GeakRunnerFacts(
            started_at="2026-09-02T09:00:00+00:00", duration_sec=90.0, runner_timeout_sec=1800, kill_timeout_sec=2100
        ),
    )

    assert again is None, "an already-adjudicated result.json must not be promoted a second time"
    # The verdict is left standing rather than retired by the skip that refused the replay.
    assert state.geak_result["revalidation_error_class"] == "incomparable"
    assert not state.geak_pending.get("status") == "awaiting_rebench"


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
