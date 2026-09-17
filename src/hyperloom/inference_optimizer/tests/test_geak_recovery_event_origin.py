# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Disk recovery keeps a GEAK candidate attached to its original KERNEL event."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.recorder import kernel_event
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.inference_optimizer.tests.test_geak_gain_alignment import _journey_with_validated_keeps
from hyperloom.inference_optimizer.tests.test_geak_revalidation_dispatch import coordinator as coordinator
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.machine_state import PHASE_KERNEL_AGENT, PHASE_SWEEP
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.asyncio
@pytest.mark.parametrize("new_candidate", [False, True], ids=["same_result", "new_result"])
async def test_later_cycle_disk_recovery_settles_the_candidate_event(coordinator, monkeypatch, new_candidate):
    coord = coordinator
    state = coord.shared_state
    state.framework = "sglang"
    state.benchmark_mode = "synthetic"
    state.baseline_tput = 100.0
    state.baseline_accuracy = 0.8
    state.current_best = {"action": "explore", "tput": 110.0}
    state.model_path = "/models/test"
    state.gpu_type = "mi355x"
    state.isl, state.osl, state.conc = 1024, 1024, 64
    state.optimization_stack = [{"action": "explore", "variant_name": "anchor", "tput": 110.0}]
    state.kernel_optimizer = "geak"
    state.conc_sweep_enabled = False
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "2")

    overlay = coord.session_dir / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("# CPU loadability fixture\n")
    eval_dir = coord.session_dir / "geak" / "e2e_cycle0"
    eval_dir.mkdir(parents=True)
    raw = {
        "status": "ok",
        "throughput_speedup": 1.5,
        "final_throughput_tok_s": 150.0,
        "accepted_config": {"flags": "--max-running-requests 64", "env": "SGLANG_USE_AITER=1"},
        "accepted_kernels": [{"short_name": "candidate_kernel", "kind": "authored", "e2e_delta_pct": 50.0}],
        "final_overlay": str(overlay),
        "eval_dir": str(eval_dir),
        "kernel_journey_path": _journey_with_validated_keeps(eval_dir, [1.5]),
    }
    result_path = eval_dir.parent / "result.json"
    result_path.write_text(json.dumps(raw))

    def no_runner(_name):
        pytest.fail("An existing unadjudicated result must use the disk recovery path")

    monkeypatch.setattr("hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path", no_runner)
    with session_scope(coord.session_dir):
        coord.phase_kernel._open_kernel_timeline(
            route=kernel_event.ROUTE_GEAK, route_reason="kernel_optimizer=geak", from_phase="FRAMEWORK_AGENT"
        )
        await coord._run_geak_kernel_phase(from_phase="FRAMEWORK_AGENT")
        origin = state.geak_result["kernel_event_id"]
        task = await coord.tasks.get(state.geak_pending["revalidation_task_id"])
        assert task.params["geak_fallback"] is True
        assert task.idempotency_key == "geak-revalidate-c0"
        await coord.phase_machine._on_phase_entered(
            from_phase=PHASE_KERNEL_AGENT, to_phase=PHASE_SWEEP, reason="kernel_budget_cap"
        )
        assert coord.phase_kernel._kernel_timeline() is None
        original = next(event for event in read_timeline_events(coord.session_dir) if event["id"] == origin)
        assert original["ext"]["geak"]["attempts"]["counts"]["integrated"] == 1
        state.save(coord.session_dir)

        resumed = Coordinator.__new__(Coordinator)
        resumed.session_dir = coord.session_dir
        resumed.tasks = coord.tasks
        resumed.bus = coord.bus
        resumed.shared_state = SharedState.load_or_init(coord.session_dir)
        resumed._apply_macro_cycle_reloop({})
        assert resumed.shared_state.macro_cycle == 1
        if new_candidate:
            await resumed.tasks.transition(task.task_id, "cancelled")
            eval_dir = result_path.parent / "e2e_cycle1"
            eval_dir.mkdir()
            raw.update(
                final_throughput_tok_s=160.0,
                eval_dir=str(eval_dir),
                kernel_journey_path=_journey_with_validated_keeps(eval_dir, [1.6]),
            )
            result_path.write_text(json.dumps(raw))

        resumed.phase_kernel._open_kernel_timeline(
            route=kernel_event.ROUTE_GEAK, route_reason="kernel_optimizer=geak", from_phase="FRAMEWORK_AGENT"
        )
        await resumed._run_geak_kernel_phase(from_phase="FRAMEWORK_AGENT")
        recovered_origin = resumed.shared_state.geak_result["kernel_event_id"]
        tracked_task = await resumed.tasks.get(resumed.shared_state.geak_pending["revalidation_task_id"])
        if new_candidate:
            assert tracked_task.task_id != task.task_id
            assert tracked_task.idempotency_key == "geak-revalidate-c1"
        else:
            assert tracked_task.task_id == task.task_id
            assert len(await resumed.tasks.queued()) == 1
        resumed._on_enter_sweep = AsyncMock()
        await resumed.phase_machine._on_phase_entered(
            from_phase=PHASE_KERNEL_AGENT, to_phase=PHASE_SWEEP, reason="kernel_budget_cap"
        )
        assert resumed.phase_kernel._kernel_timeline() is None
        for path in result_path.parent.glob("e2e_cycle*/kernel_journey.json"):
            path.unlink()
        await resumed.tasks.transition(tracked_task.task_id, "running")
        tracked_task = await resumed.tasks.transition(tracked_task.task_id, "succeeded")
        fingerprint = tracked_task.params["expected_cfg_hash"]
        native = {
            "status": "succeeded",
            "output_throughput": 120.0,
            "best_variant": {"output_throughput": 120.0, "accuracy": 0.95, "fingerprint": fingerprint},
            "winners": [],
            "per_variant_outcomes": [{"outcome": "REVERT", "reason": "accuracy_drop", "fingerprint": fingerprint}],
        }
        replay = AsyncMock(side_effect=AssertionError("A conclusive native rejection cannot invoke fallback"))
        monkeypatch.setattr("hyperloom.orchestrator.actions.executors._geak_sweep.sweep_via_geak", replay)
        await resumed._promote_to_shared_state("explore", native, task=tracked_task)
        replay.assert_not_awaited()
        assert resumed.shared_state.geak_pending == {}
        assert resumed.shared_state.current_best["tput"] == 110.0
        assert resumed.shared_state.geak_result["revalidation_status"] == "no_promote"
        resumed.shared_state.save(coord.session_dir)
        exported = exporter.build(coord.session_dir)
        events = {event["id"]: event for event in exported["timeline"]}
        earlier = events[origin]["ext"]["geak"]["attempts"]
        later = events[kernel_event.kernel_event_id(1)]["ext"]["geak"].get("attempts") or {}
        assert earlier["counts"]["integrated"] == (1 if new_candidate else 0)
        assert later.get("counts", {}).get("integrated", 0) == 0
        expected_origin = kernel_event.kernel_event_id(1) if new_candidate else origin
        assert recovered_origin == expected_origin
        rejected = events[expected_origin]["ext"]["geak"]["attempts"]["kernels"][0]["e2e"]
        assert rejected["decision"] == "REVERT"
        assert rejected["integrated"] is False
        assert rejected["validated"] is False
        assert rejected["e2e_gain_pct"] is None
