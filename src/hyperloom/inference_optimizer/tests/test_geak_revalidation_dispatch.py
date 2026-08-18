# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for GEAK same-harness revalidation dispatch (L1/L2)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hyperloom.orchestrator.phases import machine_state as ps
from hyperloom.orchestrator.phases.machine_state import geak_revalidate_idempotency_key


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


async def _noop(*_args, **_kwargs):
    return None


def _arm_kernel_to_sweep(st) -> None:
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_KERNEL_AGENT
    st.phase_started_ts = (now - timedelta(minutes=5)).isoformat()
    st.phase_started_unix = (now - timedelta(minutes=5)).timestamp()
    st.start_ts = (now - timedelta(minutes=10)).isoformat()
    st.max_minutes = 96 * 60
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok"}
    st.geak_pending = {}
    st.set_pending_escalate_hint(ps.ESCALATE_HINT_SKIP_TO_SWEEP)


def test_geak_revalidate_idempotency_key_scopes_by_macro_cycle() -> None:
    assert geak_revalidate_idempotency_key(0) == "geak-revalidate-c0"
    assert geak_revalidate_idempotency_key(2) == "geak-revalidate-c2"
    assert geak_revalidate_idempotency_key(2) != geak_revalidate_idempotency_key(3)


@pytest.mark.asyncio
async def test_cancel_queued_not_allowed_spares_geak_fallback_rebench(coordinator) -> None:
    """L1: phase-boundary cleanup must not cancel pending GEAK revalidation tasks."""
    c = coordinator
    geak_task = await c.tasks.create(
        kind="explore",
        params={
            "source": "resume_stack_revalidate",
            "geak_fallback": True,
            "reason": "geak_e2e_win",
        },
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
    )

    assert geak_task.task_id not in cancelled
    assert regular.task_id in cancelled
    assert (await c.tasks.get(geak_task.task_id)).state == "queued"
    assert (await c.tasks.get(regular.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_cancel_family_spares_geak_fallback_rebench(coordinator) -> None:
    """Explore-family prune must not cancel the GEAK same-harness revalidation task."""
    c = coordinator
    geak_task = await c.tasks.create(
        kind="explore",
        params={"source": "resume_stack_revalidate", "geak_fallback": True},
        idempotency_key="geak-revalidate-c1",
    )
    regular = await c.tasks.create(
        kind="explore",
        params={"source": "normal_explore"},
        idempotency_key="regular-explore-prune",
    )

    cancelled = await c.tasks.cancel_family(["explore"], reason="prune_branch")

    assert geak_task.task_id not in cancelled
    assert regular.task_id in cancelled
    assert (await c.tasks.get(geak_task.task_id)).state == "queued"
    assert (await c.tasks.get(regular.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_geak_revalidate_idempotency_key_allows_retry_per_macro_cycle(
    coordinator,
) -> None:
    """L2: a settled cycle-0 rebench must not block enqueue on cycle 1."""
    c = coordinator
    cycle0_key = geak_revalidate_idempotency_key(0)
    cycle1_key = geak_revalidate_idempotency_key(1)
    assert cycle0_key != cycle1_key

    settled = await c.tasks.create(
        kind="explore",
        params={"source": "resume_stack_revalidate", "geak_fallback": True},
        idempotency_key=cycle0_key,
        task_id="geak-revalidate-c0-task",
    )
    await c.tasks.transition(settled.task_id, "cancelled", evidence={"reason": "test"})

    fresh, was_existing = await c.tasks.create_or_return_existing(
        kind="explore",
        params={"source": "resume_stack_revalidate", "geak_fallback": True},
        idempotency_key=cycle1_key,
    )

    assert not was_existing
    assert fresh.task_id != settled.task_id
    assert fresh.state == "queued"
    assert (await c.tasks.get(settled.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_geak_rebench_survives_kernel_to_sweep_transition(coordinator) -> None:
    """Queued GEAK revalidation must survive the real KERNEL→SWEEP phase transition."""
    c = coordinator
    c.phase_internal._maybe_enqueue_explore_research_scout = _noop  # type: ignore[method-assign]
    c.phase_explore._maybe_force_stalled_domain_specialist = _noop  # type: ignore[method-assign]
    c.phase_internal._maybe_enqueue_trajectory_reviewer = _noop  # type: ignore[method-assign]
    c.phase_machine._on_phase_entered = _noop  # type: ignore[method-assign]

    st = c.shared_state
    _arm_kernel_to_sweep(st)

    geak_task = await c.tasks.create(
        kind="explore",
        params={"source": "resume_stack_revalidate", "geak_fallback": True},
        idempotency_key=geak_revalidate_idempotency_key(st.macro_cycle),
        task_id="geak-rebench-survives-transition",
    )
    regular = await c.tasks.create(
        kind="explore",
        params={"source": "normal_explore"},
        idempotency_key="regular-explore-transition",
    )

    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_SWEEP
    assert (await c.tasks.get(geak_task.task_id)).state == "queued"
    assert (await c.tasks.get(regular.task_id)).state == "cancelled"


@pytest.mark.asyncio
async def test_geak_rebench_failure_clears_pending_when_placeholder_tracked(coordinator) -> None:
    """Failure handler must clear geak_pending when only the idempotency placeholder matches."""
    c = coordinator
    st = c.shared_state
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok"}
    placeholder = geak_revalidate_idempotency_key(0)
    st.geak_pending = {
        "status": "awaiting_rebench",
        "revalidation_task_id": placeholder,
    }

    task = await c.tasks.create(
        kind="explore",
        params={"source": "resume_stack_revalidate", "geak_fallback": True},
        idempotency_key=placeholder,
        task_id="geak-rebench-fail",
    )

    await c._handle_unpromotable_result(
        task,
        {
            "status": "failed",
            "error_class": "subprocess_nonzero",
            "error": "revalidation failed",
        },
    )

    assert not st.geak_pending
    assert st.geak_result["revalidation_status"] == "failed"
