# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Critic-driven specialist re-author loop.

A ``needs_review`` verdict carrying non-empty ``required_evidence`` for a
framework_agent candidate / authoring proposal triggers one re-authoring round,
seeded with that evidence and dispatched under an idempotency key with a
``reauthor:{n}`` suffix. ``advise`` proceeds and never re-authors; the
per-candidate cap is the loop guard.
"""

from __future__ import annotations

from typing import Any

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


def _build_backends() -> dict[str, Backend]:
    plan = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {name: MockBackend(plan, name=name) for name in ("orchestration", "critic", "robustness")}


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return Coordinator(session_dir, backends=_build_backends())


_CANDIDATE = {
    "candidate_id": "https://github.com/sgl-project/sglang/pull/7777",
    "pr_url": "https://github.com/sgl-project/sglang/pull/7777",
    "repo": "sgl-project/sglang",
    "ref": "perf/moe",
    "title": "perf: speed up moe",
    "framework": "sglang",
    "batch_id": "batch-1",
}

_ADVISORY = {
    "required_evidence": [
        "profile showing the kernel is the bottleneck",
        "accuracy parity on the eval set",
    ],
    "advice_text": "narrow the patch to the MoE gemm path",
    "risks": ["may regress small-batch latency"],
}


def _framework_agent_pending() -> PendingProposal:
    return PendingProposal(
        proposal_msg_id="m-fpr",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={
            "candidate": dict(_CANDIDATE),
            "framework_agent_candidate_id": _CANDIDATE["candidate_id"],
            "batch_id": "batch-1",
            "audit": {"recommended_next_step": "author_via_specialist"},
            "audit_step": "author_via_specialist",
        },
    )


def _record_reauthor_calls(coord: Coordinator) -> list[dict[str, Any]]:
    """Monkeypatch the authoring-specialist enqueue to record re-author calls."""
    calls: list[dict[str, Any]] = []

    async def _fake_enqueue(
        candidate: dict[str, Any],
        audit: dict[str, Any] | None = None,
        *,
        reauthor_attempt: int = 0,
        critic_feedback: dict[str, Any] | None = None,
    ) -> str:
        calls.append(
            {
                "candidate": candidate,
                "audit": audit,
                "reauthor_attempt": reauthor_attempt,
                "critic_feedback": critic_feedback,
            }
        )
        return f"spec-{len(calls)}"

    coord.phase_framework._enqueue_framework_agent_authoring_specialist = _fake_enqueue  # type: ignore[method-assign]
    return calls


@pytest.mark.asyncio
async def test_needs_review_with_evidence_reauthors_once(coord: Coordinator) -> None:
    calls = _record_reauthor_calls(coord)
    pending = _framework_agent_pending()

    await coord._handle_single_verdict(
        source="critic",
        pending=pending,
        verdict="needs_review",
        reasoning="needs more evidence",
        advisory=dict(_ADVISORY),
    )

    assert len(calls) == 1
    assert calls[0]["reauthor_attempt"] == 1
    fb = calls[0]["critic_feedback"]
    assert fb["required_evidence"] == _ADVISORY["required_evidence"]
    assert fb["advice_text"] == _ADVISORY["advice_text"]
    assert fb["risks"] == _ADVISORY["risks"]
    assert coord.shared_state.specialist_reauthor_attempts[_CANDIDATE["candidate_id"]] == 1


@pytest.mark.asyncio
async def test_reauthor_guard_caps_and_suffixes(coord: Coordinator) -> None:
    """The first 3 needs_review verdicts re-author with incrementing
    ``reauthor:{n}`` idempotency suffixes; the 4th hits the cap and does not re-author."""
    from types import SimpleNamespace
    from hyperloom.orchestrator.loop.coordinator import _AUTHORED_LANE_MAX_ATTEMPTS

    created: list[dict[str, Any]] = []

    async def _fake_create(**kwargs: Any) -> Any:
        created.append(kwargs)
        return (
            SimpleNamespace(
                task_id=f"spec-{len(created)}",
                params=kwargs.get("params") or {},
                state="queued",
            ),
            False,
        )

    coord.tasks.create_or_return_existing = _fake_create  # type: ignore[method-assign]

    for _ in range(_AUTHORED_LANE_MAX_ATTEMPTS + 1):
        await coord._handle_single_verdict(
            source="critic",
            pending=_framework_agent_pending(),
            verdict="needs_review",
            reasoning="needs more evidence",
            advisory=dict(_ADVISORY),
        )

    assert len(created) == _AUTHORED_LANE_MAX_ATTEMPTS
    assert created[0]["idempotency_key"].endswith(":reauthor:1")
    assert created[-1]["idempotency_key"].endswith(f":reauthor:{_AUTHORED_LANE_MAX_ATTEMPTS}")
    notes = created[0]["params"]["notes"]
    assert "profile showing the kernel is the bottleneck" in notes
    assert "narrow the patch to the MoE gemm path" in notes
    assert coord.shared_state.specialist_reauthor_attempts[_CANDIDATE["candidate_id"]] == _AUTHORED_LANE_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_reauthor_skipped_when_candidate_already_materializing(
    coord: Coordinator,
) -> None:
    """A live integrate_patch task for the candidate suppresses re-author."""
    from types import SimpleNamespace

    calls = _record_reauthor_calls(coord)
    live = SimpleNamespace(
        kind="integrate_patch",
        task_id="i-live",
        params={"framework_agent_candidate_id": _CANDIDATE["candidate_id"]},
    )

    async def _queued() -> list[Any]:
        return [live]

    async def _running() -> list[Any]:
        return []

    coord.tasks.queued = _queued  # type: ignore[method-assign]
    coord.tasks.running = _running  # type: ignore[method-assign]

    await coord._handle_single_verdict(
        source="critic",
        pending=_framework_agent_pending(),
        verdict="needs_review",
        reasoning="needs more evidence",
        advisory=dict(_ADVISORY),
    )

    assert calls == []
    assert coord.shared_state.specialist_reauthor_attempts == {}


@pytest.mark.asyncio
async def test_authoring_integrate_patch_reauthors_and_records_old_task(
    coord: Coordinator,
) -> None:
    """An authored-patch integrate_patch sent back for evidence re-authors via
    the originating specialist; the observation carries old + new task ids."""
    from types import SimpleNamespace

    calls = _record_reauthor_calls(coord)
    observations: list[dict[str, Any]] = []

    async def _rec_obs(source: str, topic: str, payload: dict[str, Any]) -> None:
        observations.append(payload)

    async def _get(task_id: str) -> Any:
        return SimpleNamespace(
            params={
                "framework_agent_candidate_id": _CANDIDATE["candidate_id"],
                "framework_batch_id": "batch-1",
                "gap_symptom": "perf: speed up moe",
                "framework": "sglang",
                "gap_canonical_id": "gap.x",
                "framework_audit": {"recommended_next_step": "author_via_specialist"},
            }
        )

    coord._record_observation = _rec_obs  # type: ignore[method-assign]
    coord.tasks.get = _get  # type: ignore[method-assign]

    pending = PendingProposal(
        proposal_msg_id="m-int",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={"params": {"framework_agent_authoring": True, "specialist_task_id": "spec-old"}},
    )

    await coord._maybe_reauthor_from_critic_feedback(pending, dict(_ADVISORY))

    assert len(calls) == 1
    assert calls[0]["candidate"]["candidate_id"] == _CANDIDATE["candidate_id"]
    dispatched = [o for o in observations if o.get("kind") == "specialist_reauthor_dispatched"]
    assert len(dispatched) == 1
    assert dispatched[0]["old_specialist_task_id"] == "spec-old"
    assert dispatched[0]["new_specialist_task_id"] == "spec-1"


@pytest.mark.asyncio
async def test_advise_verdict_does_not_reauthor(coord: Coordinator) -> None:
    calls = _record_reauthor_calls(coord)
    materialized: list[Any] = []

    async def _fake_materialize(pending: Any, *, approved_variant_names: set[str] | None = None) -> None:
        materialized.append(pending)

    coord._materialize_approved_proposal = _fake_materialize  # type: ignore[method-assign]
    pending = _framework_agent_pending()

    await coord._handle_single_verdict(
        source="critic",
        pending=pending,
        verdict="advise",
        reasoning="proceed with notes",
        advisory=dict(_ADVISORY),
    )

    # advise = proceed: materialises but never re-authors.
    assert len(materialized) == 1
    assert calls == []
    assert coord.shared_state.specialist_reauthor_attempts == {}


@pytest.mark.asyncio
async def test_needs_review_without_required_evidence_no_reauthor(
    coord: Coordinator,
) -> None:
    calls = _record_reauthor_calls(coord)
    pending = _framework_agent_pending()

    await coord._handle_single_verdict(
        source="critic",
        pending=pending,
        verdict="needs_review",
        reasoning="vague",
        advisory={"advice_text": "do better", "risks": ["x"]},
    )

    assert calls == []
    assert coord.shared_state.specialist_reauthor_attempts == {}


@pytest.mark.asyncio
async def test_non_framework_agent_proposal_does_not_reauthor(coord: Coordinator) -> None:
    calls = _record_reauthor_calls(coord)
    pending = PendingProposal(
        proposal_msg_id="m-other",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={"params": {"specialist_task_id": "s-1"}},
    )

    await coord._maybe_reauthor_from_critic_feedback(pending, dict(_ADVISORY))

    assert calls == []
    assert coord.shared_state.specialist_reauthor_attempts == {}
