# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P1.b — Critic gate before FRAMEWORK_PR apply: approve/abstain enqueue, reject writes ``critic_denied``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.critic_mock import (
    MockCriticBackend,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType


class _StateStub:
    def __init__(self) -> None:
        self.phase = "FRAMEWORK_PR"
        self.framework_pr_phase_done = False
        self.framework_pr_phase_progress: list[dict[str, Any]] = []
        self.framework_pr_critic_decisions: list[dict[str, Any]] = []
        self.phase_history: list[dict[str, Any]] = []
        self._saves = 0

    def save(self, _session_dir: Path) -> None:
        self._saves += 1


class _CoordinatorStub:
    """Holds just enough state for the gate helper to run (real ``_collect_framework_pr_priors`` bound)."""

    _CRITIC_PRIORS_DECISION_TAIL = Coordinator._CRITIC_PRIORS_DECISION_TAIL
    _CRITIC_PRIORS_OUTCOME_TAIL = Coordinator._CRITIC_PRIORS_OUTCOME_TAIL
    _collect_framework_pr_priors = Coordinator._collect_framework_pr_priors

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


# Approve / reject / abstain mapping
def test_gate_returns_approve_for_mock_critic(tmp_path: Path) -> None:
    """MockCriticBackend auto-approves, so the gate maps to ``approve`` and caches the decision."""
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


# ---------------------------------------------------------------------------
# Gap 3 — prompt enrichment (diff_url + session-local priors).


class _PromptCapturingBackend:
    """Captures the prompt body so we can assert on its contents."""

    def __init__(self) -> None:
        self.last_prompt: str = ""

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> Any:
        self.last_prompt = prompt
        import re
        from types import SimpleNamespace
        m = re.search(r"msg_id=([a-f0-9]+)", prompt)
        msg_id = m.group(1) if m else "x"
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


def test_prompt_includes_diff_url_when_present(tmp_path: Path) -> None:
    backend = _PromptCapturingBackend()
    stub = _CoordinatorStub(tmp_path, backend)
    cand = {
        "candidate_id": "pr-1",
        "batch_id":     "b-1",
        "pr_url":       "https://github.com/sgl-project/sglang/pull/1",
        "diff_url":     "https://github.com/sgl-project/sglang/pull/1.diff",
    }
    _call_gate(stub, cand)
    assert "https://github.com/sgl-project/sglang/pull/1.diff" in backend.last_prompt


# ---------------------------------------------------------------------------
# atom-candidate rendering parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "framework, diff_url",
    [
        ("sglang", "https://github.com/sgl-project/sglang/pull/100.diff"),
        ("vllm",   "https://github.com/ROCm/vllm/pull/200.diff"),
        ("atom",   "https://github.com/ROCm/ATOM/pull/123/files"),
    ],
)
def test_critic_prompt_renders_candidate_diff_url_across_frameworks(
    tmp_path: Path, framework: str, diff_url: str,
) -> None:
    """The Critic prompt must carry the candidate's ``diff_url``
    verbatim regardless of framework. atom uses the
    ``github.com/ROCm/ATOM/pull/N/files`` URL shape (variant of the
    canonical PR-diff URL); the prompt rendering must not reject or
    reshape it."""
    backend = _PromptCapturingBackend()
    stub = _CoordinatorStub(tmp_path, backend)
    cand = {
        "candidate_id": f"pr-{framework}",
        "batch_id":     f"b-{framework}",
        "pr_url":       diff_url.rsplit("/", 1)[0],
        "diff_url":     diff_url,
        "framework":    framework,
    }
    _call_gate(stub, cand)
    assert diff_url in backend.last_prompt
    # Framework name carried into the prompt verbatim too.
    assert framework in backend.last_prompt


def test_critic_prompt_no_framework_specific_rule_text_for_atom(
    tmp_path: Path,
) -> None:
    """The Critic prompt body assembled for an atom candidate must not
    contain rule text that's specific to sglang or vllm (e.g.
    ``"sglang-specific"``, ``"vllm-specific"``). Concrete examples that
    *mention* sglang or vllm are fine — the guard is on rule-flavour
    substrings that would systematically bias the verdict against atom
    by reference to the other frameworks' conventions."""
    backend = _PromptCapturingBackend()
    stub = _CoordinatorStub(tmp_path, backend)
    cand = {
        "candidate_id": "pr-atom-rules",
        "batch_id":     "b-atom-rules",
        "pr_url":       "https://github.com/ROCm/ATOM/pull/9",
        "diff_url":     "https://github.com/ROCm/ATOM/pull/9.diff",
        "framework":    "atom",
        "title":        "perf: atom MTP scheduler",
    }
    _call_gate(stub, cand)
    assert "sglang-specific" not in backend.last_prompt, (
        "Critic prompt for atom candidate contains sglang-specific "
        "rule text; rephrase the rule to be framework-neutral."
    )
    assert "vllm-specific" not in backend.last_prompt, (
        "Critic prompt for atom candidate contains vllm-specific "
        "rule text; rephrase the rule to be framework-neutral."
    )


def test_prompt_includes_session_local_priors(tmp_path: Path) -> None:
    """Recent Critic decisions + apply/bench outcomes get folded
    into the prompt so the Critic can spot patterns across the
    current FRAMEWORK_PR session."""
    backend = _PromptCapturingBackend()
    stub = _CoordinatorStub(tmp_path, backend)
    # Pre-populate the decision cache and the outcome ledger.
    stub.shared_state.framework_pr_critic_decisions.extend([
        {
            "candidate_id": "pr-prev-1",
            "verdict":      "reject",
            "rationale":    "touches kernel build",
            "ts":           "2026-05-27T00:00:00Z",
        },
        {
            "candidate_id": "pr-prev-2",
            "verdict":      "approve",
            "rationale":    "",
            "ts":           "2026-05-27T00:01:00Z",
        },
    ])
    stub.shared_state.framework_pr_phase_progress.extend([
        {
            "candidate_id": "pr-prev-2",
            "status":       "reverted",
            "gain_pct":     -1.4,
            "ts":           "2026-05-27T00:02:00Z",
        },
        {
            "candidate_id": "pr-prev-3",
            "status":       "kept",
            "gain_pct":     3.2,
            "ts":           "2026-05-27T00:03:00Z",
        },
    ])
    cand = {"candidate_id": "pr-new", "batch_id": "b-2"}
    _call_gate(stub, cand)
    # The decision cache rows should appear.
    assert "pr-prev-1" in backend.last_prompt
    assert "touches kernel build" in backend.last_prompt
    # The outcome rows should appear.
    assert "reverted" in backend.last_prompt
    assert "kept" in backend.last_prompt
    # Priors envelope key should be visible.
    assert "priors" in backend.last_prompt


def test_priors_helper_trims_to_tail_length(tmp_path: Path) -> None:
    """The helper bounds both decisions and outcomes to the
    configured tail length so the prompt does not grow unbounded."""
    from inference_optimizer.orchestrator.coordinator import Coordinator
    stub = _CoordinatorStub(tmp_path, backend=None)
    # 12 decisions, 12 terminal outcomes — both should be capped at 5.
    for i in range(12):
        stub.shared_state.framework_pr_critic_decisions.append({
            "candidate_id": f"pr-{i}",
            "verdict":      "approve",
            "rationale":    "",
        })
        stub.shared_state.framework_pr_phase_progress.append({
            "candidate_id": f"pr-{i}",
            "status":       "kept",
            "gain_pct":     1.0,
        })
    priors = Coordinator._collect_framework_pr_priors(stub)  # type: ignore[arg-type]
    assert len(priors["recent_decisions"]) == 5
    assert len(priors["recent_outcomes"]) == 5
    # Tail = the most recent 5.
    assert priors["recent_decisions"][-1]["candidate_id"] == "pr-11"
    assert priors["recent_outcomes"][-1]["candidate_id"] == "pr-11"


def test_priors_helper_skips_non_terminal_outcomes(tmp_path: Path) -> None:
    """Only rows with terminal status (kept / reverted / no_patch /
    enqueue_failed / critic_denied) feed the outcomes prior — an
    in-flight ``running`` row should not show up."""
    from inference_optimizer.orchestrator.coordinator import Coordinator
    stub = _CoordinatorStub(tmp_path, backend=None)
    stub.shared_state.framework_pr_phase_progress.extend([
        {"candidate_id": "pr-running", "status": "running"},
        {"candidate_id": "pr-kept", "status": "kept", "gain_pct": 2.0},
    ])
    priors = Coordinator._collect_framework_pr_priors(stub)  # type: ignore[arg-type]
    ids = [r["candidate_id"] for r in priors["recent_outcomes"]]
    assert "pr-kept" in ids
    assert "pr-running" not in ids
