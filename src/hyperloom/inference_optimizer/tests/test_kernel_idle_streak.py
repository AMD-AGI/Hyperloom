# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KERNEL idle-streak state and Controller progress fingerprint tests."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.phases import machine_state as ps


@pytest.fixture
def kernel_coordinator(tmp_path, monkeypatch):
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
    c = Coordinator(sd, backends=backends)

    async def _noop(*_args, **_kwargs):
        return None

    c.phase_internal._maybe_enqueue_explore_research_scout = _noop  # type: ignore[method-assign]
    c.phase_explore._maybe_force_stalled_domain_specialist = _noop  # type: ignore[method-assign]
    c.phase_internal._maybe_enqueue_trajectory_reviewer = _noop  # type: ignore[method-assign]
    c.phase_machine._on_phase_entered = _noop  # type: ignore[method-assign]
    yield c


@pytest.mark.asyncio
async def test_streak_state_is_cleared_outside_kernel(kernel_coordinator):
    c = kernel_coordinator
    st = c.shared_state
    st.phase = ps.PHASE_FRAMEWORK_AGENT
    st.kernel_idle_ticks = 9
    st.kernel_progress_fingerprint = "stale"
    st.kernel_idle_since_unix = 1.0

    await c.phase_machine._track_kernel_idle_streak()

    assert st.kernel_idle_ticks == 0
    assert st.kernel_progress_fingerprint == ""
    assert st.kernel_idle_since_unix == 0.0


def test_fingerprint_ignores_fields_that_are_not_progress():
    from types import SimpleNamespace

    base = SimpleNamespace(
        kernel_rewrite_controller_result={
            "macro_cycle": 1,
            "status": "running",
            "patch_count": 0,
            "finished_at": "",
            "diagnostic": "first",
        },
    )
    before = ps.compute_kernel_progress_fingerprint(base)

    base.kernel_rewrite_controller_result["diagnostic"] = "second"
    assert ps.compute_kernel_progress_fingerprint(base) == before

    base.kernel_rewrite_controller_result["patch_count"] = 1
    assert ps.compute_kernel_progress_fingerprint(base) != before


def test_fingerprint_tracks_inflight_task_ids():
    from types import SimpleNamespace

    state = SimpleNamespace(
        kernel_rewrite_controller_result={
            "macro_cycle": 1,
            "status": "running",
            "patch_count": 0,
            "finished_at": "",
        },
    )
    idle = ps.compute_kernel_progress_fingerprint(state)
    busy = ps.compute_kernel_progress_fingerprint(state, inflight_task_ids=("t1",))
    # A dispatch starting is progress in its own right, before any outcome lands.
    assert idle != busy
    # Order of the in-flight ids must not matter.
    assert ps.compute_kernel_progress_fingerprint(
        state, inflight_task_ids=("t2", "t1")
    ) == ps.compute_kernel_progress_fingerprint(state, inflight_task_ids=("t1", "t2"))


@pytest.mark.asyncio
async def test_running_specialist_counts_as_kernel_lane_work(kernel_coordinator):
    """A specialist admitted to KERNEL must reach ``_inflight_kernel_task_ids``.

    The kind filter is the phase allowlist, so admitting ``specialist`` there is
    what stops the idle guard from reading a live investigation as dead air.
    """
    c = kernel_coordinator
    task = await c.tasks.create(kind="specialist", params={}, idempotency_key="spec-idle")
    await c.tasks.transition(task.task_id, "running")
    assert task.task_id in await c.phase_machine._inflight_kernel_task_ids()


@pytest.mark.asyncio
async def test_queued_specialist_survives_the_transition_into_kernel(kernel_coordinator):
    """``cancel_queued_not_allowed`` reads the same allowlist, so the task lives."""
    c = kernel_coordinator
    task = await c.tasks.create(kind="specialist", params={}, idempotency_key="spec-keep")
    cancelled = await c.tasks.cancel_queued_not_allowed(
        allowed_kinds=ps.PHASE_ALLOWED_ACTIONS[ps.PHASE_KERNEL_AGENT],
        reason="phase_transition:EXPLORE->KERNEL_AGENT",
    )
    assert task.task_id not in cancelled
    assert (await c.tasks.get(task.task_id)).state == "queued"
