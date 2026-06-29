# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Converged FRAMEWORK_PR pre-screen gate.

The candidate is submitted as a normal ``framework_pr`` proposal (PolicyGate
bypassed, COORDINATOR_INTERNAL) and the async Critic verdict drives the
apply/author enqueue (approve/advise) or the ``critic_denied`` row (reject) via
``_handle_single_verdict`` → ``_materialize_framework_pr_candidate`` /
``_record_framework_pr_critic_denied``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    Backend,
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator, PendingProposal
from inference_optimizer.paths import make_session_dir
from inference_optimizer.protocol.intent import Intent, IntentType


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    from .conftest import seed_target_analysis_marker

    seed_target_analysis_marker(sd)
    return sd


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends() -> dict[str, Backend]:
    return {
        name: MockBackend(_silent_plan(), name=name)
        for name in ("orchestration", "kernel", "critic", "robustness")
    }


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
        action_name="framework_pr",
        predicted_gain_pct=0.0,
        payload=payload,
    )


# -- submit -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_submit_registers_pending_proposal(coord: Coordinator) -> None:
    await coord._submit_framework_pr_candidate_for_review(
        dict(_CANDIDATE),
        audit={"recommended_next_step": "direct_framework_pr"},
        audit_step="direct_framework_pr",
    )
    pendings = [p for p in coord.state.pending_proposals.values() if p.action_name == "framework_pr"]
    assert len(pendings) == 1
    pl = pendings[0].payload
    assert pl["action_name"] == "framework_pr"
    assert pl["framework_pr_candidate_id"] == _CANDIDATE["candidate_id"]
    assert pl["audit_step"] == "direct_framework_pr"
    assert pl["candidate"]["pr_url"] == _CANDIDATE["pr_url"]


@pytest.mark.asyncio
async def test_submit_is_idempotent_per_candidate(coord: Coordinator) -> None:
    await coord._submit_framework_pr_candidate_for_review(dict(_CANDIDATE), audit_step="direct_framework_pr")
    await coord._submit_framework_pr_candidate_for_review(dict(_CANDIDATE), audit_step="direct_framework_pr")
    pendings = [p for p in coord.state.pending_proposals.values() if p.action_name == "framework_pr"]
    assert len(pendings) == 1


# -- materialize routing ----------------------------------------------------
@pytest.mark.asyncio
async def test_materialize_direct_route_enqueues_raw_only(coord: Coordinator, monkeypatch) -> None:
    raw: list = []
    author: list = []
    monkeypatch.setattr(coord, "_enqueue_framework_pr_task", lambda c: _append(raw, c))
    monkeypatch.setattr(
        coord, "_enqueue_framework_pr_authoring_specialist", lambda c, audit=None: _append(author, c)
    )
    coord.shared_state.framework_pr_authoring_enabled = True
    await coord._materialize_framework_pr_candidate(
        _pending({"candidate": dict(_CANDIDATE), "audit_step": "direct_framework_pr", "batch_id": "b"})
    )
    assert len(raw) == 1
    assert author == []


@pytest.mark.asyncio
async def test_materialize_author_route_enqueues_specialist_only(coord: Coordinator, monkeypatch) -> None:
    raw: list = []
    author: list = []
    monkeypatch.setattr(coord, "_enqueue_framework_pr_task", lambda c: _append(raw, c))
    monkeypatch.setattr(
        coord, "_enqueue_framework_pr_authoring_specialist", lambda c, audit=None: _append(author, c)
    )
    coord.shared_state.framework_pr_authoring_enabled = True
    await coord._materialize_framework_pr_candidate(
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
    monkeypatch.setattr(coord, "_enqueue_framework_pr_task", lambda c: _append(raw, c))
    monkeypatch.setattr(
        coord, "_enqueue_framework_pr_authoring_specialist", lambda c, audit=None: _append(author, c)
    )
    coord.shared_state.framework_pr_authoring_enabled = False
    await coord._materialize_framework_pr_candidate(
        _pending({"candidate": dict(_CANDIDATE), "audit_step": "author_via_specialist", "batch_id": "b"})
    )
    assert len(raw) == 1
    assert author == []


@pytest.mark.asyncio
async def test_materialize_unknown_route_runs_both_tracks(coord: Coordinator, monkeypatch) -> None:
    raw: list = []
    author: list = []
    monkeypatch.setattr(coord, "_enqueue_framework_pr_task", lambda c: _append(raw, c))
    monkeypatch.setattr(
        coord, "_enqueue_framework_pr_authoring_specialist", lambda c, audit=None: _append(author, c)
    )
    coord.shared_state.framework_pr_authoring_enabled = True
    await coord._materialize_framework_pr_candidate(
        _pending({"candidate": dict(_CANDIDATE), "audit_step": "", "batch_id": "b"})
    )
    assert len(raw) == 1
    assert len(author) == 1


# -- verdict drives materialize/reject through _handle_single_verdict --------
@pytest.mark.asyncio
async def test_approve_verdict_materializes(coord: Coordinator, monkeypatch) -> None:
    raw: list = []
    monkeypatch.setattr(coord, "_enqueue_framework_pr_task", lambda c: _append(raw, c))
    pending = _pending({"candidate": dict(_CANDIDATE), "audit_step": "direct_framework_pr", "batch_id": "b", "framework_pr_candidate_id": _CANDIDATE["candidate_id"]})
    coord.state.pending_proposals[pending.proposal_msg_id] = pending
    await coord._handle_single_verdict(source="critic", pending=pending, verdict="approve", reasoning="ok")
    assert len(raw) == 1


@pytest.mark.asyncio
async def test_reject_verdict_records_critic_denied(coord: Coordinator) -> None:
    pending = _pending({"framework_pr_candidate_id": _CANDIDATE["candidate_id"], "batch_id": "batch-1"})
    coord.state.pending_proposals[pending.proposal_msg_id] = pending
    await coord._handle_single_verdict(
        source="critic", pending=pending, verdict="reject", reasoning="out of scope for this gap"
    )
    prog = coord.shared_state.framework_pr_phase_progress
    denied = [r for r in prog if r.get("status") == "critic_denied"]
    assert len(denied) == 1
    assert denied[0]["candidate_id"] == _CANDIDATE["candidate_id"]
    assert "out of scope" in denied[0]["rationale"]


def _append(bucket: list, candidate) -> "object":
    """Sync→awaitable shim so monkeypatched enqueue helpers stay awaitable."""

    async def _noop() -> None:
        bucket.append(candidate)

    return _noop()
