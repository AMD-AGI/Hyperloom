# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ownership of a patch a specialist wrote, frozen where the specialist is created."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import MockBackend, MockCriticBackend, ScriptedPlan
from hyperloom.orchestrator.state.task_registry import Task

SPECIALIST_ID = "spec-patch-1"
_FREEFORM_PATCH = {"scope": "freeform", "mode": "patch", "task_description": "fix the scheduler crash"}


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
@pytest.mark.parametrize("phase", ["KERNEL_AGENT", "ENABLEMENT"])
async def test_a_patch_mode_specialist_is_owned_at_dispatch(session_dir: Path, phase: str) -> None:
    """A freeform dispatch names no domain, gap layer or authoring phase.

    In patch mode it will still author a diff that lands through
    ``integrate_patch``, which the breakdown books to the framework agent, so
    that is the owner the round is frozen with.
    """
    coord = _coordinator(session_dir, phase=phase)
    try:
        params = dict(_FREEFORM_PATCH)
        assert coord._stamp_specialist_owner(params) == "FRAMEWORK_AGENT"
        assert params["source_phase"] == "FRAMEWORK_AGENT"
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_a_research_specialist_names_no_owner(session_dir: Path) -> None:
    """A freeform dispatch defaults to research, which authors no patch.

    Owning that round would book a read-only investigation to the framework
    agent, so it keeps the gap the attribution model reports.
    """
    coord = _coordinator(session_dir, phase="KERNEL_AGENT")
    try:
        params = {"scope": "freeform", "task_description": "find out why prefill blocks decode"}
        assert coord._stamp_specialist_owner(params) == ""
        assert "source_phase" not in params
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_the_integrate_route_accepts_the_specialist_it_owned(session_dir: Path) -> None:
    """``delegate`` / ``propose_action`` read the owner off the specialist task.

    Both refuse an `integrate_patch` they cannot attribute, so a patch the
    autosubmit path would carry has to be accepted here too.
    """
    coord = _coordinator(session_dir, phase="KERNEL_AGENT")
    try:
        params = dict(_FREEFORM_PATCH)
        coord._stamp_specialist_owner(params)
        task = await coord.tasks.create(kind="specialist", params=params, idempotency_key="spec-1")

        integrate_params = {"specialist_task_id": task.task_id}
        assert await coord._stamp_integrate_patch_owner(integrate_params) == "FRAMEWORK_AGENT"
        assert integrate_params["source_phase"] == "FRAMEWORK_AGENT"
    finally:
        await coord.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["KERNEL_AGENT", "ENABLEMENT"])
async def test_the_autosubmitted_patch_is_integrated_and_owned(session_dir: Path, phase: str) -> None:
    """The round the orchestration prompt asks for, end to end.

    A patch-mode freeform specialist finishes, the Critic approves, and the
    `integrate_patch` task carries the owner frozen at dispatch.
    """
    coord = _coordinator(session_dir, phase=phase)
    try:
        params = dict(_FREEFORM_PATCH)
        coord._stamp_specialist_owner(params)
        await _specialist_wrote_a_patch(coord, spec_params=params)

        await coord._reactor_pass("critic")

        assert coord.shared_state.get_specialist_patch_verdict(SPECIALIST_ID) == "approve"
        tasks = await _integrate_tasks(coord)
        assert [(t.params or {}).get("specialist_task_id") for t in tasks] == [SPECIALIST_ID]
        assert (tasks[0].params or {}).get("source_phase") == "FRAMEWORK_AGENT"
        coord.policy.validate_dispatched_task("integrate_patch", dict(tasks[0].params or {}))
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_the_mock_critic_approval_lands_as_the_patch_verdict(session_dir: Path) -> None:
    """Evidence for #1553: `--critic-mock` is not what drops a specialist patch.

    The mock's approval reaches `specialist_patch_verdicts` under the
    specialist's own id, on the same handler the critic-agent backend uses.
    """
    coord = _coordinator(session_dir, phase="FRAMEWORK_AGENT")
    try:
        await _specialist_wrote_a_patch(coord, spec_params={"domain": "kernel", "source_phase": "EXPLORE"})
        assert coord.shared_state.get_specialist_patch_verdict(SPECIALIST_ID) == ""

        await coord._reactor_pass("critic")

        assert coord.shared_state.get_specialist_patch_verdict(SPECIALIST_ID) == "approve"
    finally:
        await coord.stop()
