# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for GEAK same-harness revalidation dispatch (L1/L2)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from hyperloom.orchestrator.phases import geak_rebench as gr
from hyperloom.orchestrator.phases import machine_state as ps


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from hyperloom.inference_optimizer.session.paths import make_session_dir as _msd
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        MockRobustnessBackend,
        ScriptedPlan,
    )
    from .conftest import seed_target_analysis_marker

    sd = _msd()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    return Coordinator(sd, backends=backends)


def _geak_rebench_params(**extra: object) -> dict:
    return {
        "source": "resume_stack_revalidate",
        "geak_fallback": True,
        "reason": "geak_e2e_win",
        **extra,
    }


def _arm_kernel_to_sweep(st) -> None:
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_KERNEL_AGENT
    st.phase_started_ts = (now - timedelta(minutes=5)).isoformat()
    st.phase_started_unix = (now - timedelta(minutes=5)).timestamp()
    st.start_ts = (now - timedelta(minutes=10)).isoformat()
    st.max_minutes = 96 * 60
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok", "accepted_config": {"flags": "--foo", "env": ""}}
    st.geak_pending = {}
    st.set_pending_escalate_hint(ps.ESCALATE_HINT_SKIP_TO_SWEEP)


def test_geak_revalidate_idempotency_key_scopes_by_macro_cycle() -> None:
    assert gr.geak_revalidate_idempotency_key(0) == "geak-revalidate-c0"
    assert gr.geak_revalidate_idempotency_key(2) == "geak-revalidate-c2"
    assert gr.geak_revalidate_idempotency_key(2) != gr.geak_revalidate_idempotency_key(3)


@pytest.mark.asyncio
async def test_cancel_queued_not_allowed_spares_geak_rebench_into_sweep(coordinator) -> None:
    c = coordinator
    geak_task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key="geak-revalidate-c0",
    )
    regular = await c.tasks.create(
        kind="explore",
        params={"source": "normal_explore"},
        idempotency_key="regular-explore",
    )

    cancelled = await c.tasks.cancel_queued_not_allowed(
        allowed_kinds=ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_SWEEP],
        reason="phase_transition:KERNEL_AGENT->SWEEP",
        spare_queued=lambda _tid, kind, params: gr.spare_geak_rebench_on_phase_transition(
            target_phase=ps.PHASE_SWEEP,
            kind=kind,
            params=params,
        ),
    )

    assert geak_task.task_id not in cancelled
    assert regular.task_id in cancelled
    assert (await c.tasks.get(geak_task.task_id)).state == "queued"
    assert (await c.tasks.get(regular.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_cancel_queued_not_allowed_cancels_geak_rebench_into_close(coordinator) -> None:
    c = coordinator
    geak_task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key="geak-revalidate-c0",
    )

    cancelled = await c.tasks.cancel_queued_not_allowed(
        allowed_kinds=ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_CLOSE],
        reason="phase_transition:SWEEP->CLOSE",
        spare_queued=lambda _tid, kind, params: gr.spare_geak_rebench_on_phase_transition(
            target_phase=ps.PHASE_CLOSE,
            kind=kind,
            params=params,
        ),
    )

    assert geak_task.task_id in cancelled
    assert (await c.tasks.get(geak_task.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_cancel_family_cancels_geak_rebench_explore(coordinator) -> None:
    """Explore-family prune must cancel GEAK rebench (no implicit spare)."""
    c = coordinator
    geak_task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key="geak-revalidate-c1",
    )

    cancelled = await c.tasks.cancel_family(["explore"], reason="prune_branch")

    assert geak_task.task_id in cancelled
    assert (await c.tasks.get(geak_task.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_geak_revalidate_idempotency_key_allows_retry_per_macro_cycle(
    coordinator,
) -> None:
    c = coordinator
    cycle0_key = gr.geak_revalidate_idempotency_key(0)
    cycle1_key = gr.geak_revalidate_idempotency_key(1)

    settled = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=cycle0_key,
        task_id="geak-revalidate-c0-task",
    )
    await c.tasks.transition(settled.task_id, "cancelled", evidence={"reason": "test"})

    fresh, was_existing = await c.tasks.create_or_return_existing(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=cycle1_key,
    )

    assert not was_existing
    assert fresh.task_id != settled.task_id
    assert fresh.state == "queued"


@pytest.mark.asyncio
async def test_enqueue_internal_stack_rebench_uses_macro_cycle_idempotency_key(
    coordinator,
) -> None:
    """Production enqueue path must stamp cycle-scoped GEAK rebench keys."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--max-num-batched-tokens 8192", "env": ""},
    }

    st.macro_cycle = 0
    first = await c._enqueue_internal_stack_rebench(reason="geak_e2e_win")
    row0 = await c.tasks.get(str(first["task_id"]))
    assert row0.idempotency_key == "geak-revalidate-c0"

    st.macro_cycle = 1
    second = await c._enqueue_internal_stack_rebench(reason="geak_e2e_win")
    row1 = await c.tasks.get(str(second["task_id"]))
    assert row1.idempotency_key == "geak-revalidate-c1"
    assert row1.task_id != row0.task_id


@pytest.mark.asyncio
async def test_geak_rebench_survives_kernel_to_sweep_transition(coordinator) -> None:
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)

    geak_task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(st.macro_cycle),
        task_id="geak-rebench-survives-transition",
    )
    regular = await c.tasks.create(
        kind="explore",
        params={"source": "normal_explore"},
        idempotency_key="regular-explore-transition",
    )

    cancelled = await c.tasks.cancel_queued_not_allowed(
        allowed_kinds=ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_SWEEP],
        reason="phase_transition:KERNEL_AGENT->SWEEP",
        spare_queued=lambda _tid, kind, params: gr.spare_geak_rebench_on_phase_transition(
            target_phase=ps.PHASE_SWEEP,
            kind=kind,
            params=params,
        ),
    )

    assert geak_task.task_id not in cancelled
    assert regular.task_id in cancelled
    assert (await c.tasks.get(geak_task.task_id)).state == "queued"
    assert (await c.tasks.get(regular.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_duplicate_enqueue_skips_while_rebench_in_flight(coordinator, tmp_path) -> None:
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir()
    result = {
        "status": "ok",
        "final_throughput_tok_s": 116.0,
        "accepted_config": {"flags": "--foo", "env": ""},
    }
    (geak_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    inflight = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="cycle0-inflight",
    )

    coord = c
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        lambda _name: (_ for _ in ()).throw(RuntimeError("runner should not run")),
    )
    try:
        await coord._run_geak_kernel_phase(from_phase="KERNEL")
    finally:
        monkeypatch.undo()

    assert len([t for t in await c.tasks.queued() if gr.is_geak_same_harness_rebench_task(t.kind, t.params)]) == 1
    assert st.geak_pending["revalidation_task_id"] == inflight.task_id
    created = await c.tasks.get(inflight.task_id)
    assert created.state == "queued"


@pytest.mark.asyncio
async def test_geak_rebench_failure_clears_pending_when_placeholder_tracked(coordinator) -> None:
    c = coordinator
    st = c.shared_state
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok"}
    placeholder = gr.geak_revalidate_idempotency_key(0)
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": placeholder}

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=placeholder,
        task_id="geak-rebench-fail",
    )

    await c._handle_unpromotable_result(
        task,
        {"status": "failed", "error_class": "subprocess_nonzero", "error": "revalidation failed"},
    )

    assert not st.geak_pending
    assert st.geak_result["revalidation_status"] == "failed"


@pytest.mark.asyncio
async def test_resume_revalidation_with_empty_pending_still_promotes(coordinator) -> None:
    """Resume stack rebench with cleared geak_pending must still lift validated gain."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {"action": "geak_e2e", "tput": 116.0, "extra_server_args": "--foo"}
    st.optimization_stack = [{"action": "geak_e2e", "tput": 116.0}]
    st.geak_result = {"status": "ok", "accepted_config": {"flags": "--foo", "env": ""}}
    st.geak_pending = {}
    st.resume_pending_revalidation = True

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="resume-reverify",
    )

    await c._promote_to_shared_state(
        task.kind,
        {
            "output_throughput": 140.0,
            "best_variant": {"fingerprint": "abc"},
            "winners": [],
        },
        task=task,
    )

    assert st.cumulative_gain_validated == pytest.approx(40.0)
    assert st.cumulative_gain_provenance == "geak_orch_harness_validated"
    assert st.resume_pending_revalidation is False


@pytest.mark.asyncio
async def test_orphan_geak_rebench_success_does_not_promote(coordinator) -> None:
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {"action": "baseline", "tput": 100.0, "extra_server_args": ""}
    st.kernel_optimizer = "geak"
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--foo", "env": ""},
        "accepted_kernels": ["k1"],
    }
    tracked = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(0),
        task_id="tracked-rebench",
    )
    orphan = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key=gr.geak_revalidate_idempotency_key(1),
        task_id="orphan-rebench",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": tracked.task_id}

    await c._promote_to_shared_state(
        orphan.kind,
        {
            "output_throughput": 150.0,
            "best_variant": {"fingerprint": "abc", "gain_pct": 50.0},
            "winners": [{"fingerprint": "abc", "gain_pct": 50.0}],
        },
        task=orphan,
    )

    assert st.current_best["tput"] == 100.0
    assert st.geak_pending["revalidation_task_id"] == tracked.task_id
    assert not any(e.get("action") == "geak_e2e" for e in st.optimization_stack)
