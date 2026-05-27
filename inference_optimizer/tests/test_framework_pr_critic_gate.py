"""Cover the P1.b fix — Critic gate before FRAMEWORK_PR apply.

The gate lives in Coordinator._critic_review_framework_pr_candidate
and is consulted from _pump_framework_pr_phase right before
``_enqueue_framework_pr_task``. ``approve`` (or the degraded
``abstain`` returned when no Critic is wired) falls through to enqueue;
``reject`` short-circuits with a ``critic_denied`` progress row so the
candidate is never applied.

Tests bind the Coordinator method directly to a minimal stub so we
exercise the gate without spinning up the full DB/bus/backends stack.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.critic_mock import (
    MockCriticBackend,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType


class _StateStub:
    def __init__(self) -> None:
        self.phase = "FRAMEWORK_PR"
        self.framework_pr_phase_done = False
        self.framework_pr_phase_progress: list[dict[str, Any]] = []
        self.framework_pr_critic_decisions: list[dict[str, Any]] = []
        self._saves = 0

    def save(self, _session_dir: Path) -> None:
        self._saves += 1


class _CoordinatorStub:
    """Holds just enough state for the gate helper to run."""

    def __init__(self, tmp_path: Path, backend: Any | None) -> None:
        self.session_dir = tmp_path
        self.shared_state = _StateStub()
        self.backends: dict[str, Any] = {}
        if backend is not None:
            self.backends["critic"] = backend


def _call_gate(stub: _CoordinatorStub, candidate: dict[str, Any]) -> dict[str, str]:
    return asyncio.run(
        Coordinator._critic_review_framework_pr_candidate(  # type: ignore[arg-type]
            stub, candidate,
        ),
    )


# ---------------------------------------------------------------------------
# Approve / reject / abstain mapping


def test_gate_returns_approve_for_mock_critic(tmp_path: Path) -> None:
    """MockCriticBackend auto-approves every proposal it sees, so the
    gate maps that to ``approve`` and caches the decision."""
    stub = _CoordinatorStub(tmp_path, MockCriticBackend())
    candidate = {
        "candidate_id":     "https://github.com/sgl-project/sglang/pull/9999",
        "pr_url":           "https://github.com/sgl-project/sglang/pull/9999",
        "repo":             "sgl-project/sglang",
        "ref":              "feature/x",
        "title":            "perf: speed up moe",
        "framework":        "sglang",
        "gap_canonical_id": "moe_gemm_latency",
        "batch_id":         "batch-1",
    }
    result = _call_gate(stub, candidate)
    assert result["verdict"] == "approve"
    # Decision cached.
    assert len(stub.shared_state.framework_pr_critic_decisions) == 1
    row = stub.shared_state.framework_pr_critic_decisions[0]
    assert row["verdict"] == "approve"
    assert row["candidate_id"] == candidate["candidate_id"]
    assert row["batch_id"] == "batch-1"


class _RejectBackend:
    """Critic stub that returns ``verdict='reject'`` for any proposal."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> Any:
        self.calls.append(prompt)
        # Extract msg_id from prompt so the verdict targets the right
        # proposal (matches MockCriticBackend's wire format).
        import re
        m = re.search(r"msg_id=([a-f0-9]+)", prompt)
        msg_id = m.group(1) if m else "unknown"
        from types import SimpleNamespace
        return SimpleNamespace(
            intents=[Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={
                    "target_proposal_msg_id": msg_id,
                    "verdict":                "reject",
                    "reasoning":              "out of scope for this gap",
                },
            )],
            raw_text="(reject)",
        )


def test_gate_returns_reject_with_rationale(tmp_path: Path) -> None:
    stub = _CoordinatorStub(tmp_path, _RejectBackend())
    candidate = {"candidate_id": "pr-42", "batch_id": "batch-2"}
    result = _call_gate(stub, candidate)
    assert result["verdict"] == "reject"
    assert "out of scope" in result["rationale"]
    row = stub.shared_state.framework_pr_critic_decisions[0]
    assert row["verdict"] == "reject"
    assert "out of scope" in row["rationale"]


def test_gate_abstains_when_no_critic_backend(tmp_path: Path) -> None:
    """Missing Critic must NOT block the phase — the caller treats
    ``abstain`` like ``approve``. The cache still records the decision."""
    stub = _CoordinatorStub(tmp_path, backend=None)
    result = _call_gate(stub, {"candidate_id": "pr-1", "batch_id": "b-1"})
    assert result["verdict"] == "abstain"
    # No backend → no cache write (we short-circuit before the cache append).
    assert stub.shared_state.framework_pr_critic_decisions == []


class _RaisingBackend:
    """Critic stub whose .run() raises — gate must degrade to ``abstain``."""

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> Any:
        raise RuntimeError("simulated backend failure")


def test_gate_abstains_when_backend_raises(tmp_path: Path) -> None:
    stub = _CoordinatorStub(tmp_path, _RaisingBackend())
    result = _call_gate(stub, {"candidate_id": "pr-9", "batch_id": "b-9"})
    assert result["verdict"] == "abstain"
    assert "simulated backend failure" in result["rationale"]
    row = stub.shared_state.framework_pr_critic_decisions[0]
    assert row["verdict"] == "abstain"


class _NeedsReviewBackend:
    """Critic stub that returns ``needs_review`` — must map to abstain."""

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> Any:
        import re
        from types import SimpleNamespace
        m = re.search(r"msg_id=([a-f0-9]+)", prompt)
        msg_id = m.group(1) if m else "unknown"
        return SimpleNamespace(
            intents=[Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={
                    "target_proposal_msg_id": msg_id,
                    "verdict":                "needs_review",
                    "reasoning":              "insufficient context",
                },
            )],
            raw_text="(needs_review)",
        )


def test_gate_maps_needs_review_to_abstain(tmp_path: Path) -> None:
    """Critic vocab outside {approve, reject} maps to ``abstain`` so the
    phase keeps moving instead of stalling on a soft verdict."""
    stub = _CoordinatorStub(tmp_path, _NeedsReviewBackend())
    result = _call_gate(stub, {"candidate_id": "pr-7", "batch_id": "b-7"})
    assert result["verdict"] == "abstain"
    assert "insufficient context" in result["rationale"]


# ---------------------------------------------------------------------------
# Resume-safe cache lookup


class _CountingBackend:
    """Records how many .run() calls it received."""

    def __init__(self) -> None:
        self.run_count = 0

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> Any:
        self.run_count += 1
        import re
        from types import SimpleNamespace
        m = re.search(r"msg_id=([a-f0-9]+)", prompt)
        msg_id = m.group(1) if m else "unknown"
        return SimpleNamespace(
            intents=[Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={
                    "target_proposal_msg_id": msg_id,
                    "verdict":                "approve",
                    "reasoning":              "ok",
                },
            )],
            raw_text="(approve)",
        )


def test_gate_uses_cached_decision_on_repeat_call(tmp_path: Path) -> None:
    """Second call for the same candidate_id reads from
    ``framework_pr_critic_decisions`` instead of re-invoking the
    Critic — this is the resume path."""
    backend = _CountingBackend()
    stub = _CoordinatorStub(tmp_path, backend)
    cand = {"candidate_id": "pr-cached", "batch_id": "b-c"}
    first = _call_gate(stub, cand)
    second = _call_gate(stub, cand)
    assert first["verdict"] == "approve"
    assert second["verdict"] == "approve"
    assert backend.run_count == 1, "second call should hit the cache"
    # Only one decision row.
    assert len(stub.shared_state.framework_pr_critic_decisions) == 1
