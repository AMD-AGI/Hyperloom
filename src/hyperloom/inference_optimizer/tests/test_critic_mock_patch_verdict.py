# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The mock Critic's approval of a specialist patch, and the ownership a patch
needs before ``integrate_patch`` will run it."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.policy.gate import PolicyDenied
from hyperloom.orchestrator.roles import MockBackend, MockCriticBackend, ScriptedPlan
from hyperloom.orchestrator.state.task_registry import Task

SPECIALIST_ID = "spec-patch-1"


def _coordinator(session_dir: Path, *, phase: str) -> Coordinator:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "observation", "body_md": "ok"}),
    )
    coord = Coordinator(
        session_dir,
        backends={"orchestration": MockBackend(silent, name="orch"), "critic": MockCriticBackend()},
    )
    coord.shared_state.phase = phase
    coord.shared_state.baseline_tput = 1500.0
    return coord


async def _specialist_wrote_a_patch(coord: Coordinator, *, spec_params: dict) -> None:
    """Complete a specialist with one real on-disk patch and route it for review."""
    worktree = runs_dir(coord.session_dir, "specialist", SPECIALIST_ID) / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / "kernel.py").write_text("# patched\n", encoding="utf-8")
    await coord._maybe_autosubmit_specialist_patches(
        task=Task(
            task_id=SPECIALIST_ID,
            kind="specialist",
            state="running",
            params=dict(spec_params),
            idempotency_key="spec-patch-1-key",
        ),
        done_payload={"patches_written": ["kernel.py"], "proposal_set": [{"name": "fused-rmsnorm"}]},
    )


async def _integrate_tasks(coord: Coordinator) -> list[Task]:
    return [t for t in await coord.tasks.queued() if t.kind == "integrate_patch"]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["FRAMEWORK_AGENT", "KERNEL_AGENT"])
async def test_mock_critic_approval_lands_as_the_specialist_patch_verdict(session_dir: Path, phase: str) -> None:
    """``--critic-mock`` carries a specialist patch all the way through the gate."""
    coord = _coordinator(session_dir, phase=phase)
    try:
        await _specialist_wrote_a_patch(coord, spec_params={"domain": "kernel", "source_phase": "EXPLORE"})
        assert coord.shared_state.get_specialist_patch_verdict(SPECIALIST_ID) == ""

        await coord._reactor_pass("critic")

        assert coord.shared_state.get_specialist_patch_verdict(SPECIALIST_ID) == "approve"
        tasks = await _integrate_tasks(coord)
        assert [(t.params or {}).get("specialist_task_id") for t in tasks] == [SPECIALIST_ID]
        coord.policy.validate_dispatched_task("integrate_patch", dict(tasks[0].params or {}))
    finally:
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["KERNEL_AGENT", "ENABLEMENT"])
async def test_a_freeform_patch_is_integrated_and_owned_by_the_framework_agent(
    session_dir: Path,
    phase: str,
) -> None:
    """A freeform specialist names no domain, gap layer or authoring phase.

    Its patch still lands through ``integrate_patch``, which the breakdown
    attributes to the framework agent, so the round must be owned rather than
    discarded for want of a label.
    """
    coord = _coordinator(session_dir, phase=phase)
    try:
        await _specialist_wrote_a_patch(coord, spec_params={"scope": "freeform"})

        await coord._reactor_pass("critic")

        assert coord.shared_state.get_specialist_patch_verdict(SPECIALIST_ID) == "approve"
        tasks = await _integrate_tasks(coord)
        assert [(t.params or {}).get("specialist_task_id") for t in tasks] == [SPECIALIST_ID]
        assert (tasks[0].params or {}).get("source_phase") == "FRAMEWORK_AGENT"
        coord.policy.validate_dispatched_task("integrate_patch", dict(tasks[0].params or {}))
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_integrate_patch_stays_denied_without_a_verdict(session_dir: Path) -> None:
    """The gate is not weakened: an unreviewed subject is still refused."""
    coord = _coordinator(session_dir, phase="FRAMEWORK_AGENT")
    try:
        with pytest.raises(PolicyDenied) as denial:
            coord.policy.validate_dispatched_task("integrate_patch", {"specialist_task_id": SPECIALIST_ID})
        assert denial.value.rule == "integrate_patch_requires_critic_verdict"
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_integrate_patch_stays_denied_on_a_reject_verdict(session_dir: Path) -> None:
    """A recorded reject is still a refusal, not a pass-through."""
    coord = _coordinator(session_dir, phase="FRAMEWORK_AGENT")
    try:
        coord.shared_state.record_specialist_patch_verdict(SPECIALIST_ID, "reject")
        with pytest.raises(PolicyDenied) as denial:
            coord.policy.validate_dispatched_task("integrate_patch", {"specialist_task_id": SPECIALIST_ID})
        assert denial.value.rule == "integrate_patch_requires_critic_verdict"
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_an_unownable_patch_is_refused_before_it_reaches_the_critic(session_dir: Path) -> None:
    """Ownership is settled where the proposal is published, not after review.

    A patch naming a specialist that does not exist has no ownership evidence
    at all, so it never becomes a proposal and never costs a Critic turn.
    """
    coord = _coordinator(session_dir, phase="FRAMEWORK_AGENT")
    try:
        await coord._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={
                    "action_name": "integrate_patch",
                    "params": {"specialist_task_id": "missing-specialist"},
                },
            ),
        )

        assert list(coord.state.pending_proposals) == []
        assert await _integrate_tasks(coord) == []
        observations = await coord.bus.tail(topic="observation", n=20)
        assert [o.payload.get("reason") for o in observations if o.payload.get("kind") == "proposal_rejected"] == [
            "integrate_patch_owner_missing"
        ]
    finally:
        await coord.stop()
