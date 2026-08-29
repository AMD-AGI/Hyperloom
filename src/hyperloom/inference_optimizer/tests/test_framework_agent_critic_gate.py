# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Converged FRAMEWORK pre-screen gate.

The candidate is submitted as a normal ``framework_agent`` proposal and the
async Critic verdict drives the apply/author enqueue or the ``critic_denied``
row.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.roles import (
    Backend,
    MockBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator, PendingProposal
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends() -> dict[str, Backend]:
    return {name: MockBackend(_silent_plan(), name=name) for name in ("orchestration", "critic", "robustness")}


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return Coordinator(session_dir, backends=_build_backends())


_CANDIDATE = {
    "candidate_id": "https://github.com/sgl-project/sglang/pull/9999",
    "pr_url": "https://github.com/sgl-project/sglang/pull/9999",
    "repo": "sgl-project/sglang",
    "ref": "feature/x",
    "title": "perf: speed up moe",
    "framework": "sglang",
    "batch_id": "batch-1",
}


def _pending(payload: dict) -> PendingProposal:
    return PendingProposal(
        proposal_msg_id="m-fpr",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload=payload,
    )


# -- submit -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_submit_registers_pending_proposal(coord: Coordinator) -> None:
    await coord._submit_framework_agent_candidate_for_review(
        dict(_CANDIDATE),
        audit={"recommended_next_step": "direct_framework"},
        audit_step="direct_framework",
    )
    pendings = [p for p in coord.state.pending_proposals.values() if p.action_name == "integrate_patch"]
    assert len(pendings) == 1
    pl = pendings[0].payload
    assert pl["action_name"] == "integrate_patch"
    assert pl["framework_agent_candidate_id"] == _CANDIDATE["candidate_id"]
    assert pl["audit_step"] == "direct_framework"
    assert pl["candidate"]["pr_url"] == _CANDIDATE["pr_url"]


@pytest.mark.asyncio
async def test_submit_is_idempotent_per_candidate(coord: Coordinator) -> None:
    await coord._submit_framework_agent_candidate_for_review(dict(_CANDIDATE), audit_step="direct_framework")
    await coord._submit_framework_agent_candidate_for_review(dict(_CANDIDATE), audit_step="direct_framework")
    pendings = [p for p in coord.state.pending_proposals.values() if p.action_name == "integrate_patch"]
    assert len(pendings) == 1


# -- materialize routing ----------------------------------------------------
@pytest.mark.asyncio
async def test_materialize_direct_route_enqueues_raw_only(coord: Coordinator, monkeypatch) -> None:
    raw: list = []
    author: list = []
    monkeypatch.setattr(coord.phase_framework, "_enqueue_framework_agent_task", lambda c: _append(raw, c))
    monkeypatch.setattr(
        coord.phase_framework, "_enqueue_framework_agent_authoring_specialist", lambda c, audit=None: _append(author, c)
    )
    coord.shared_state.framework_agent_authoring_enabled = True
    await coord._materialize_framework_agent_candidate(
        _pending({"candidate": dict(_CANDIDATE), "audit_step": "direct_framework", "batch_id": "b"})
    )
    assert len(raw) == 1
    assert author == []


@pytest.mark.asyncio
async def test_materialize_author_route_enqueues_specialist_only(coord: Coordinator, monkeypatch) -> None:
    raw: list = []
    author: list = []
    monkeypatch.setattr(coord.phase_framework, "_enqueue_framework_agent_task", lambda c: _append(raw, c))
    monkeypatch.setattr(
        coord.phase_framework, "_enqueue_framework_agent_authoring_specialist", lambda c, audit=None: _append(author, c)
    )
    coord.shared_state.framework_agent_authoring_enabled = True
    await coord._materialize_framework_agent_candidate(
        _pending({"candidate": dict(_CANDIDATE), "audit_step": "author_via_specialist", "batch_id": "b"})
    )
    assert raw == []
    assert len(author) == 1


@pytest.mark.asyncio
async def test_materialize_author_route_falls_back_to_raw_when_authoring_disabled(
    coord: Coordinator, monkeypatch
) -> None:
    raw: list = []
    author: list = []
    monkeypatch.setattr(coord.phase_framework, "_enqueue_framework_agent_task", lambda c: _append(raw, c))
    monkeypatch.setattr(
        coord.phase_framework, "_enqueue_framework_agent_authoring_specialist", lambda c, audit=None: _append(author, c)
    )
    coord.shared_state.framework_agent_authoring_enabled = False
    await coord._materialize_framework_agent_candidate(
        _pending({"candidate": dict(_CANDIDATE), "audit_step": "author_via_specialist", "batch_id": "b"})
    )
    assert len(raw) == 1
    assert author == []


@pytest.mark.asyncio
async def test_materialize_unknown_route_runs_both_tracks(coord: Coordinator, monkeypatch) -> None:
    raw: list = []
    author: list = []
    monkeypatch.setattr(coord.phase_framework, "_enqueue_framework_agent_task", lambda c: _append(raw, c))
    monkeypatch.setattr(
        coord.phase_framework, "_enqueue_framework_agent_authoring_specialist", lambda c, audit=None: _append(author, c)
    )
    coord.shared_state.framework_agent_authoring_enabled = True
    await coord._materialize_framework_agent_candidate(
        _pending({"candidate": dict(_CANDIDATE), "audit_step": "", "batch_id": "b"})
    )
    assert len(raw) == 1
    assert len(author) == 1


@pytest.mark.asyncio
async def test_enqueued_task_rides_the_decaying_keep_curve(coord: Coordinator) -> None:
    """framework_agent grades against the same per-cycle bar as explore and integrate_patch."""
    from hyperloom.orchestrator.phases.machine_state import decaying_keep_threshold_pct

    coord.shared_state.macro_cycle = 2
    await coord.phase_framework._enqueue_framework_agent_task(dict(_CANDIDATE))

    queued = [t for t in await coord.tasks.queued() if t.kind == "integrate_patch"]
    assert len(queued) == 1
    assert queued[0].params["keep_threshold_pct"] == pytest.approx(decaying_keep_threshold_pct(2))


# -- verdict drives materialize/reject through _handle_single_verdict --------
@pytest.mark.asyncio
async def test_approve_verdict_materializes(coord: Coordinator, monkeypatch) -> None:
    raw: list = []
    monkeypatch.setattr(coord.phase_framework, "_enqueue_framework_agent_task", lambda c: _append(raw, c))
    pending = _pending(
        {
            "candidate": dict(_CANDIDATE),
            "audit_step": "direct_framework",
            "batch_id": "b",
            "framework_agent_candidate_id": _CANDIDATE["candidate_id"],
        }
    )
    coord.state.pending_proposals[pending.proposal_msg_id] = pending
    await coord._handle_single_verdict(source="critic", pending=pending, verdict="approve", reasoning="ok")
    assert len(raw) == 1
    # The dispatched task carries no specialist task id, so the verdict has to
    # be filed under the candidate for the executor and PolicyGate to find it.
    assert coord.shared_state.get_specialist_patch_verdict(_CANDIDATE["candidate_id"]) == "approve"


@pytest.mark.asyncio
async def test_reject_verdict_records_critic_denied(coord: Coordinator) -> None:
    pending = _pending({"framework_agent_candidate_id": _CANDIDATE["candidate_id"], "batch_id": "batch-1"})
    coord.state.pending_proposals[pending.proposal_msg_id] = pending
    await coord._handle_single_verdict(
        source="critic", pending=pending, verdict="reject", reasoning="out of scope for this gap"
    )
    prog = coord.shared_state.framework_agent_phase_progress
    denied = [r for r in prog if r.get("status") == "critic_denied"]
    assert len(denied) == 1
    assert denied[0]["candidate_id"] == _CANDIDATE["candidate_id"]
    assert "out of scope" in denied[0]["rationale"]


@pytest.mark.asyncio
async def test_reject_enablement_integrate_patch_advances_stall(coord: Coordinator) -> None:
    """A Critic-rejected ENABLEMENT integrate_patch never reaches the executor,
    so it must still advance the enablement stall accounting (bump streak, clear
    inflight_task_id) to avoid stalling before enablement_stalled fires."""
    coord.shared_state.enablement.inflight_task_id = "spec-e"
    coord.shared_state.enablement.stall_streak = 0
    pending = PendingProposal(
        proposal_msg_id="m-enable",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={"params": {"enablement": True, "specialist_task_id": "spec-e"}},
    )
    coord.state.pending_proposals[pending.proposal_msg_id] = pending
    await coord._handle_single_verdict(
        source="critic", pending=pending, verdict="reject", reasoning="empty deliverable; nothing to enable"
    )
    assert coord.shared_state.enablement.stall_streak == 1
    assert not coord.shared_state.enablement.inflight_task_id


def _append(bucket: list, candidate) -> "object":
    """Sync→awaitable shim so monkeypatched enqueue helpers stay awaitable."""

    async def _noop() -> None:
        bucket.append(candidate)

    return _noop()
