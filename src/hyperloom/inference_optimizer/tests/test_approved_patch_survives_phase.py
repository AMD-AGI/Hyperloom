# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Critic-approved ``integrate_patch`` still queued at a phase boundary survives it.

``integrate_patch`` is outside KERNEL_AGENT's allowlist, so the FRAMEWORK_AGENT
-> KERNEL_AGENT transition cancelled every queued one -- including patches the
Critic had already approved and that were only waiting for the benchmark lane.

The transition tests go through the real ``_advance_phase_if_needed`` and the
real ``cancel_queued_not_allowed``; only the choice of target phase is pinned,
because why the phase moves is not what these assert.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hyperloom.orchestrator.phases import approved_patch as ap
from hyperloom.orchestrator.phases import machine_state as ps
from hyperloom.orchestrator.state.shared_state import SharedState

#: The GEAK same-harness rebench, as test_geak_revalidation_dispatch builds it.
_GEAK_REBENCH = {"source": "resume_stack_revalidate", "geak_fallback": True, "reason": "geak_e2e_win"}


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
    c = Coordinator(sd, backends=backends)

    async def _noop(*_args, **_kwargs):
        return None

    c.phase_internal._maybe_enqueue_explore_research_scout = _noop  # type: ignore[method-assign]
    c.phase_explore._maybe_force_stalled_domain_specialist = _noop  # type: ignore[method-assign]
    c.phase_internal._maybe_enqueue_trajectory_reviewer = _noop  # type: ignore[method-assign]
    c.phase_machine._on_phase_entered = _noop  # type: ignore[method-assign]
    c.shared_state.phase = ps.PHASE_FRAMEWORK_AGENT
    c.shared_state.phase_started_unix = datetime.now(timezone.utc).timestamp()
    yield c


def _pin_next_phase(monkeypatch, target: str) -> None:
    monkeypatch.setattr(
        ps,
        "compute_next_phase",
        lambda *_a, **_k: (target, "optimize_phase_budget_exhausted", {}),
    )


async def _integrate(c, sid: str, verdict: str | None, **extra):
    if verdict is not None:
        c.shared_state.record_specialist_patch_verdict(sid, verdict)
    return await c.tasks.create(
        kind="integrate_patch",
        params={"specialist_task_id": sid, **extra},
        idempotency_key=f"integrate-{sid}",
    )


async def _state_of(c, task) -> str:
    return (await c.tasks.get(task.task_id)).state


@pytest.mark.asyncio
async def test_framework_to_kernel_keeps_only_the_approved_patches(coordinator, monkeypatch):
    c = coordinator
    approved = await _integrate(c, "spec-approved", "approve")
    advised = await _integrate(c, "spec-advised", "advise")
    unreviewed = await _integrate(c, "spec-unreviewed", None)
    refused = await _integrate(c, "spec-refused", "reject")
    explore = await c.tasks.create(kind="explore", params={"source": "normal"}, idempotency_key="plain-explore")

    _pin_next_phase(monkeypatch, ps.PHASE_KERNEL_AGENT)
    await c._advance_phase_if_needed()

    assert c.shared_state.phase == ps.PHASE_KERNEL_AGENT
    assert await _state_of(c, approved) == "queued"
    assert await _state_of(c, advised) == "queued"
    assert await _state_of(c, unreviewed) == "cancelled"
    assert await _state_of(c, refused) == "cancelled"
    assert await _state_of(c, explore) == "cancelled"


@pytest.mark.asyncio
async def test_close_still_cancels_an_approved_patch(coordinator, monkeypatch):
    c = coordinator
    approved = await _integrate(c, "spec-approved", "approve")

    _pin_next_phase(monkeypatch, ps.PHASE_CLOSE)
    await c._advance_phase_if_needed()

    assert c.shared_state.phase == ps.PHASE_CLOSE
    assert await _state_of(c, approved) == "cancelled"


@pytest.mark.asyncio
async def test_the_geak_rebench_spare_still_works_beside_it(coordinator, monkeypatch):
    c = coordinator
    geak = await c.tasks.create(kind="explore", params=dict(_GEAK_REBENCH), idempotency_key="geak-revalidate-c0")
    approved = await _integrate(c, "spec-approved", "approve")

    _pin_next_phase(monkeypatch, ps.PHASE_KERNEL_AGENT)
    await c._advance_phase_if_needed()

    assert await _state_of(c, geak) == "queued"
    assert await _state_of(c, approved) == "queued"


@pytest.mark.asyncio
async def test_a_spared_patch_still_passes_dispatch_validation(coordinator, monkeypatch):
    """Surviving is only worth something if dispatch will not refuse it after the move."""
    c = coordinator
    approved = await _integrate(c, "spec-approved", "approve")
    _pin_next_phase(monkeypatch, ps.PHASE_KERNEL_AGENT)
    await c._advance_phase_if_needed()

    task = await c.tasks.get(approved.task_id)
    c.policy.validate_dispatched_task(task.kind, task.params, task_id=task.task_id)


class TestPredicate:
    def _state(self, **verdicts: str) -> SharedState:
        state = SharedState()
        for subject, verdict in verdicts.items():
            state.record_specialist_patch_verdict(subject, verdict)
        return state

    def test_an_upstream_pr_candidate_is_judged_by_its_candidate_id(self):
        state = self._state(**{"cand-7": "approve"})
        assert ap.spare_approved_integrate_patch_on_phase_transition(
            target_phase=ps.PHASE_KERNEL_AGENT,
            kind="integrate_patch",
            params={"framework_agent_candidate_id": "cand-7"},
            shared_state=state,
        )

    def test_other_kinds_are_never_spared_here(self):
        state = self._state(**{"spec-1": "approve"})
        assert not ap.spare_approved_integrate_patch_on_phase_transition(
            target_phase=ps.PHASE_KERNEL_AGENT,
            kind="explore",
            params={"specialist_task_id": "spec-1"},
            shared_state=state,
        )

    def test_a_patch_naming_no_subject_is_not_spared(self):
        state = self._state()
        assert not ap.spare_approved_integrate_patch_on_phase_transition(
            target_phase=ps.PHASE_KERNEL_AGENT,
            kind="integrate_patch",
            params={},
            shared_state=state,
        )
