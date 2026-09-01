# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The KERNEL idle streak measures forward motion, not the ledger's opinion.

``MachinePhase._track_kernel_idle_streak`` used to reset ``kernel_idle_ticks``
whenever ``kernel_work_pending`` was true. That predicate reports whether the
attempt ledger still lists anything unresolved, so a session holding three
attempts that could never be advanced answered "yes" on all 1130 consecutive
idle ticks of a real 24h run: the counter never left zero, the wind-down guard
never even reached its branch, and KERNEL spun for 10.4h with all 8 GPUs idle.

The streak is now driven by a progress fingerprint plus the task registry:

* the fingerprint changes when the ledger, the rejected ids, the last kernel_opt
  result, the stack depth or the in-flight task set changes -> streak restarts;
* kernel-lane work queued or running freezes the streak, so a 30-minute build or
  benchmark is never mistaken for a stall;
* only genuine dead air grows it.

Covered here end-to-end through ``_advance_phase_if_needed`` so the counter and
the guard that reads it are exercised together.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def _arm_kernel_phase(st):
    """Park the session mid-KERNEL with plenty of budget and session time left."""
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_KERNEL_AGENT
    st.phase_started_ts = (now - timedelta(minutes=20)).isoformat()
    st.phase_started_unix = (now - timedelta(minutes=20)).timestamp()
    st.start_ts = (now - timedelta(minutes=30)).isoformat()
    st.max_minutes = 96 * 60


def _stall_the_ledger(st):
    """Three attempts that can never be advanced: kernel_work_pending stays True.

    Mirrors the production state — a PARTIAL/empty-decision attempt is always
    "work pending" to the ledger, no matter how long nothing happens to it.
    """
    st.kernel_opt_task_attempts = {
        f"k{idx:03d}": {
            "current_kernel_id": f"k{idx:03d}",
            "last_decision": "",
            "last_status": "",
            "last_source_file": f"/repo/op{idx}.py",
            "task_group_key": f"group-{idx}",
        }
        for idx in range(3)
    }
    assert ps.kernel_work_pending(st) is True


def _backdate_streak(st, seconds):
    """Age the streak's wall clock so the guard's floor is satisfied."""
    st.kernel_idle_since_unix = datetime.now(timezone.utc).timestamp() - seconds


@pytest.mark.asyncio
async def test_idle_kernel_winds_down_even_while_work_pending(kernel_coordinator):
    c = kernel_coordinator
    st = c.shared_state
    _arm_kernel_phase(st)
    _stall_the_ledger(st)

    # First scan opens the streak (no fingerprint stored yet), the rest observe
    # an unchanged fingerprint and no in-flight task, so the counter grows.
    await c._advance_phase_if_needed()
    assert st.kernel_idle_ticks == 0
    assert st.kernel_idle_since_unix > 0.0

    for _ in range(ps.KERNEL_IDLE_MAX_TICKS):
        await c._advance_phase_if_needed()
    assert st.kernel_idle_ticks >= ps.KERNEL_IDLE_MAX_TICKS
    # The wall-clock floor has not elapsed yet, so the phase is still KERNEL.
    assert st.phase == ps.PHASE_KERNEL_AGENT

    _backdate_streak(st, ps.KERNEL_IDLE_MIN_SECONDS + 1.0)
    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_SWEEP
    row = st.phase_history[-1]
    assert row["reason"] == "kernel_no_more_leverage"
    assert row["evidence"]["evidence"] == "kernel_idle_no_progress"
    # The ledger still says there is work; that must no longer suppress the exit.
    assert ps.kernel_work_pending(st) is True


@pytest.mark.asyncio
async def test_running_kernel_task_never_winds_down(kernel_coordinator):
    c = kernel_coordinator
    st = c.shared_state
    _arm_kernel_phase(st)
    _stall_the_ledger(st)

    build = await c.tasks.create(
        kind="kernel_opt",
        params={"kernel_id": "k000"},
        idempotency_key="long-running-build",
    )
    await c.tasks.transition(build.task_id, "running")

    # A real build compiles and benchmarks for 30+ minutes without writing a
    # single ledger field while ticks keep arriving every few seconds.
    for _ in range(ps.KERNEL_IDLE_MAX_TICKS * 20):
        await c._advance_phase_if_needed()
        # Even an aged streak clock must not help: the in-flight branch rebases
        # it every scan, so no idle window can accumulate under the build.
        _backdate_streak(st, ps.KERNEL_IDLE_MIN_SECONDS * 10)

    assert st.phase == ps.PHASE_KERNEL_AGENT
    assert st.kernel_idle_ticks == 0
    assert (await c.tasks.get(build.task_id)).state == "running"


@pytest.mark.asyncio
async def test_inline_kernel_request_never_winds_down(kernel_coordinator):
    """An ``integrate`` is awaited in the intent router, not queued as a task.

    So the registry probe reports nothing in flight for its whole duration. A
    real nine-minute ``integrate`` re-baseline accrued the full streak and wound
    KERNEL down to SWEEP four seconds after it returned, stranding nine selected
    candidates including the two hottest kernels on the trace.
    """
    c = kernel_coordinator
    st = c.shared_state
    _arm_kernel_phase(st)
    _stall_the_ledger(st)

    for _ in range(ps.KERNEL_IDLE_MAX_TICKS * 20):
        # What the intent router's heartbeat stamps while the handler runs.
        st.kernel_inline_step_seen_unix = datetime.now(timezone.utc).timestamp()
        await c._advance_phase_if_needed()
        _backdate_streak(st, ps.KERNEL_IDLE_MIN_SECONDS * 10)

    assert st.phase == ps.PHASE_KERNEL_AGENT
    assert st.kernel_idle_ticks == 0
    assert not await c.phase_machine._inflight_kernel_task_ids()


@pytest.mark.asyncio
async def test_orphaned_inline_step_stamp_still_winds_down(kernel_coordinator):
    """A stamp from a process that died mid-step must not mute the guard."""
    c = kernel_coordinator
    st = c.shared_state
    _arm_kernel_phase(st)
    _stall_the_ledger(st)
    st.kernel_inline_step_seen_unix = datetime.now(timezone.utc).timestamp() - ps.KERNEL_INLINE_STEP_STALE_SECONDS - 1.0

    await c._advance_phase_if_needed()
    for _ in range(ps.KERNEL_IDLE_MAX_TICKS):
        await c._advance_phase_if_needed()
    _backdate_streak(st, ps.KERNEL_IDLE_MIN_SECONDS + 1.0)
    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_SWEEP


@pytest.mark.asyncio
async def test_queued_kernel_task_never_winds_down(kernel_coordinator):
    # A task waiting on a resource lane is work the phase is committed to, not
    # dead air; it must hold the phase exactly like a running one.
    c = kernel_coordinator
    st = c.shared_state
    _arm_kernel_phase(st)
    _stall_the_ledger(st)

    await c.tasks.create(
        kind="integrate",
        params={"kernel_id": "k001"},
        idempotency_key="queued-integrate",
    )

    for _ in range(ps.KERNEL_IDLE_MAX_TICKS * 20):
        await c._advance_phase_if_needed()
        _backdate_streak(st, ps.KERNEL_IDLE_MIN_SECONDS * 10)

    assert st.phase == ps.PHASE_KERNEL_AGENT
    assert st.kernel_idle_ticks == 0


@pytest.mark.asyncio
async def test_ledger_progress_restarts_the_streak(kernel_coordinator):
    c = kernel_coordinator
    st = c.shared_state
    _arm_kernel_phase(st)
    _stall_the_ledger(st)

    for _ in range(ps.KERNEL_IDLE_MAX_TICKS + 1):
        await c._advance_phase_if_needed()
    assert st.kernel_idle_ticks >= ps.KERNEL_IDLE_MAX_TICKS

    # An attempt resolves: real forward motion resets the streak, and the aged
    # clock is re-stamped so the floor restarts from this moment too.
    _backdate_streak(st, ps.KERNEL_IDLE_MIN_SECONDS * 10)
    st.kernel_opt_task_attempts["k000"]["last_decision"] = "REVERT"
    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_KERNEL_AGENT
    assert st.kernel_idle_ticks == 0
    assert st.kernel_idle_since_unix >= datetime.now(timezone.utc).timestamp() - 5.0


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
        kernel_opt_task_attempts={"k000": {"last_decision": "", "last_micro_speedup": 1.0}},
        rejected_kernel_ids=[],
        last_kernel_opt={},
        optimization_stack=[],
        pending_kernel_integrations={},
    )
    before = ps.compute_kernel_progress_fingerprint(base)

    # A field outside the progress set churning must not look like progress,
    # otherwise incidental writes would keep restarting the streak forever.
    base.kernel_opt_task_attempts["k000"]["last_micro_speedup"] = 2.0
    assert ps.compute_kernel_progress_fingerprint(base) == before

    base.kernel_opt_task_attempts["k000"]["last_decision"] = "KEEP"
    assert ps.compute_kernel_progress_fingerprint(base) != before


def test_fingerprint_tracks_inflight_task_ids():
    from types import SimpleNamespace

    state = SimpleNamespace(
        kernel_opt_task_attempts={},
        rejected_kernel_ids=[],
        last_kernel_opt={},
        optimization_stack=[],
        pending_kernel_integrations={},
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
