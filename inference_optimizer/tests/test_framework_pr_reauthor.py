# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Fix 2c — Critic-driven specialist re-author loop.

A Critic ``needs_review`` verdict carrying non-empty ``required_evidence`` for a
framework_pr candidate / authoring proposal triggers exactly one re-authoring
round, seeded with that evidence and dispatched under an idempotency key with a
``reauthor:{n}`` suffix. ``advise`` proceeds (never re-authors); the cap (=1)
plus the suffixed key are the only loop guards (re-author is not charged against
the macro-cycle budget).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _build_backends() -> dict[str, Backend]:
    plan = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        name: MockBackend(plan, name=name)
        for name in ("orchestration", "kernel", "critic", "robustness")
    }


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


def _framework_pr_pending() -> PendingProposal:
    return PendingProposal(
        proposal_msg_id="m-fpr",
        from_agent="coordinator",
        action_name="framework_pr",
        predicted_gain_pct=0.0,
        payload={
            "candidate": dict(_CANDIDATE),
            "framework_pr_candidate_id": _CANDIDATE["candidate_id"],
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

    coord._enqueue_framework_pr_authoring_specialist = _fake_enqueue  # type: ignore[method-assign]
    return calls


@pytest.mark.asyncio
async def test_needs_review_with_evidence_reauthors_once(coord: Coordinator) -> None:
    calls = _record_reauthor_calls(coord)
    pending = _framework_pr_pending()

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
    assert (
        coord.shared_state.specialist_reauthor_attempts[_CANDIDATE["candidate_id"]] == 1
    )


@pytest.mark.asyncio
async def test_reauthor_caps_at_one(coord: Coordinator) -> None:
    calls = _record_reauthor_calls(coord)

    for _ in range(3):
        pending = _framework_pr_pending()
        await coord._handle_single_verdict(
            source="critic",
            pending=pending,
            verdict="needs_review",
            reasoning="needs more evidence",
            advisory=dict(_ADVISORY),
        )

    # Cap = 1: only the first verdict re-authors; the rest are no-ops.
    assert len(calls) == 1
    assert (
        coord.shared_state.specialist_reauthor_attempts[_CANDIDATE["candidate_id"]] == 1
    )


@pytest.mark.asyncio
async def test_advise_verdict_does_not_reauthor(coord: Coordinator) -> None:
    calls = _record_reauthor_calls(coord)
    materialized: list[Any] = []

    async def _fake_materialize(pending: Any) -> None:
        materialized.append(pending)

    coord._materialize_approved_proposal = _fake_materialize  # type: ignore[method-assign]
    pending = _framework_pr_pending()

    await coord._handle_single_verdict(
        source="critic",
        pending=pending,
        verdict="advise",
        reasoning="proceed with notes",
        advisory=dict(_ADVISORY),
    )

    # advise = proceed: it materialises but never re-authors.
    assert len(materialized) == 1
    assert calls == []
    assert coord.shared_state.specialist_reauthor_attempts == {}


@pytest.mark.asyncio
async def test_needs_review_without_required_evidence_no_reauthor(
    coord: Coordinator,
) -> None:
    calls = _record_reauthor_calls(coord)
    pending = _framework_pr_pending()

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
async def test_non_framework_pr_proposal_does_not_reauthor(coord: Coordinator) -> None:
    calls = _record_reauthor_calls(coord)
    pending = PendingProposal(
        proposal_msg_id="m-other",
        from_agent="coordinator",
        action_name="integrate_patch",
        predicted_gain_pct=0.0,
        payload={"params": {"specialist_task_id": "s-1"}},  # not framework_pr authoring
    )

    await coord._maybe_reauthor_from_critic_feedback(pending, dict(_ADVISORY))

    assert calls == []
    assert coord.shared_state.specialist_reauthor_attempts == {}


@pytest.mark.asyncio
async def test_reauthor_idempotency_key_carries_suffix_and_seed(
    coord: Coordinator,
) -> None:
    """The real enqueue path stamps a ``reauthor:{n}`` idempotency suffix and
    injects the Critic feedback into the authoring seed notes."""
    captured: dict[str, Any] = {}

    async def _fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        from types import SimpleNamespace

        return SimpleNamespace(task_id="spec-9", params=kwargs.get("params") or {}, state="queued"), False

    coord.tasks.create_or_return_existing = _fake_create  # type: ignore[method-assign]

    task_id = await coord._enqueue_framework_pr_authoring_specialist(
        dict(_CANDIDATE),
        audit={"recommended_next_step": "author_via_specialist"},
        reauthor_attempt=1,
        critic_feedback=dict(_ADVISORY),
    )

    assert task_id == "spec-9"
    assert captured["idempotency_key"].endswith(":reauthor:1")
    notes = captured["params"]["notes"]
    assert "PRIOR CRITIC FEEDBACK" in notes
    assert "profile showing the kernel is the bottleneck" in notes
    assert "narrow the patch to the MoE gemm path" in notes
