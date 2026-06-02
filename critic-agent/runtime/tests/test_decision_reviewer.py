"""Integration tests for the prepare/commit lifecycle.

These tests use :class:`InMemoryKBClient` and a temp session-memory root,
so they cover the deterministic side of the Critic agent end-to-end (no
LLM involvement needed).
"""

from __future__ import annotations

import pytest

from runtime.decision_reviewer import DecisionReviewer
from runtime.errors import ReviewValidationError
from runtime.in_memory_kb_client import InMemoryKBClient
from runtime.kb_writer import KBWriter
from runtime.session_memory import SessionMemory


@pytest.fixture()
def reviewer(tmp_path):
    sm = SessionMemory(root=tmp_path / "sm")
    kb = InMemoryKBClient()
    writer = KBWriter(kb, session_memory=sm)
    return DecisionReviewer(session_memory=sm, kb_writer=writer), kb, sm


def _coordinator_request(prompt: str, session_id: str = "sess_a") -> dict:
    return {
        "kind": "coordinator_inbox",
        "session_id": session_id,
        "raw_prompt": prompt,
    }


_PROMPT_WITH_TWO_PROPOSALS = (
    "=== Shared session state ===\n"
    "session_id=sess_a model=Qwen3-14B framework=sglang baseline_tput=1200\n"
    "=== Inbox for critic (newest last) ===\n"
    "  seq=1 msg_id=aaa1 from=orchestration topic=proposal payload={'action_name': 'baseline'}\n"
    "  seq=2 msg_id=bbb2 from=orchestration topic=proposal payload={'action_name': 'kernel_opt', 'predicted_gain_pct': 4.2}\n"
)


def test_prepare_review_for_coordinator_inbox_extracts_proposals(reviewer):
    rev, kb, sm = reviewer
    bundle = rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS))
    assert bundle.kind == "coordinator_inbox"
    assert sorted(p["msg_id"] for p in bundle.proposals) == ["aaa1", "bbb2"]
    # Context merged from shared_state.
    assert bundle.merged_context["model"] == "Qwen3-14B"
    assert bundle.merged_context["framework"] == "sglang"
    assert bundle.kb_read_skipped_reason is None
    # _PROMPT_WITH_TWO_PROPOSALS carries baseline (framework_op) +
    # kernel_opt (evidence_producer); the strictest class wins, so the
    # bundle-level checklist mirrors evidence_producer (NOT the legacy
    # patch_landing checklist that demanded before/after benchmarks).
    constraints = bundle.review_constraints
    assert constraints["bundle_action_class"] == "evidence_producer"
    assert "comparable_before_after_benchmark" not in constraints["approve_requires"]
    assert "in_phase_allowed_action" in constraints["approve_requires"]
    assert constraints["proposal_action_classes"] == {
        "aaa1": "framework_op",
        "bbb2": "evidence_producer",
    }
    # All three class checklists are surfaced for SKILL prompt use.
    by_cls = constraints["approve_requires_by_class"]
    assert "patch_landing" in by_cls
    assert "evidence_producer" in by_cls
    assert "framework_op" in by_cls
    assert by_cls["framework_op"] == []


def test_prepare_review_propagates_known_actions(reviewer):
    rev, kb, sm = reviewer
    bundle = rev.prepare_review({
        "kind": "coordinator_inbox",
        "session_id": "sess_known",
        "raw_prompt": _PROMPT_WITH_TWO_PROPOSALS,
        "options": {"known_actions": ["sweep", "baseline", "validate_stack"]},
    })
    assert bundle.review_constraints["known_actions"] == [
        "baseline",
        "sweep",
        "validate_stack",
    ]


def test_prepare_review_omits_known_actions_when_absent(reviewer):
    rev, kb, sm = reviewer
    bundle = rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS))
    assert "known_actions" not in bundle.review_constraints


# -----------------------------------------------------------------
# Per-action review_constraints (action-class taxonomy)
# -----------------------------------------------------------------
def _explore_only_prompt() -> str:
    return (
        "=== Shared session state ===\n"
        "model=qwen3-14b framework=sglang baseline_tput=1200\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=expA from=orchestration topic=proposal payload={'action_name': 'explore'}\n"
        "  seq=2 msg_id=specB from=orchestration topic=proposal payload={'action_name': 'specialist'}\n"
    )


def _patch_landing_prompt() -> str:
    return (
        "=== Shared session state ===\n"
        "model=qwen3-14b framework=sglang baseline_tput=1200\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=intA from=orchestration topic=proposal payload={'action_name': 'integrate'}\n"
    )


def _framework_op_prompt() -> str:
    return (
        "=== Shared session state ===\n"
        "model=qwen3-14b framework=sglang baseline_tput=0\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=baseA from=orchestration topic=proposal payload={'action_name': 'baseline'}\n"
    )


def _mixed_prompt() -> str:
    return (
        "=== Shared session state ===\n"
        "model=qwen3-14b framework=sglang baseline_tput=1200\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=expA from=orchestration topic=proposal payload={'action_name': 'explore'}\n"
        "  seq=2 msg_id=intB from=orchestration topic=proposal payload={'action_name': 'integrate'}\n"
    )


def test_prepare_review_evidence_producer_relaxes_approve_requires(reviewer):
    rev, _, _ = reviewer
    bundle = rev.prepare_review(_coordinator_request(_explore_only_prompt(), "sess_evp"))
    constraints = bundle.review_constraints
    assert constraints["bundle_action_class"] == "evidence_producer"
    # No legacy strict checklist items leak through for an explore-only batch.
    assert "comparable_before_after_benchmark" not in constraints["approve_requires"]
    assert "accuracy_gate_or_waiver" not in constraints["approve_requires"]
    # Structural checks instead.
    assert "specialist_or_default_grid_provenance" in constraints["approve_requires"]
    assert "in_phase_allowed_action" in constraints["approve_requires"]
    assert "no_contradicting_kb_prior" in constraints["approve_requires"]
    assert constraints["proposal_action_classes"] == {
        "expA": "evidence_producer",
        "specB": "evidence_producer",
    }


def test_prepare_review_patch_landing_uses_strict_approve_requires(reviewer):
    rev, _, _ = reviewer
    bundle = rev.prepare_review(_coordinator_request(_patch_landing_prompt(), "sess_pl"))
    constraints = bundle.review_constraints
    assert constraints["bundle_action_class"] == "patch_landing"
    # The original 4-item strict checklist is preserved verbatim for
    # patch landing — Critic stays the last gate before
    # optimization_stack / framework_source_roots mutate.
    assert constraints["approve_requires"] == [
        "comparable_before_after_benchmark",
        "accuracy_gate_or_waiver",
        "active_path_proof_when_relevant",
        "rollback_plan",
    ]
    assert constraints["proposal_action_classes"] == {"intA": "patch_landing"}


def test_prepare_review_framework_op_emits_empty_approve_requires(reviewer):
    rev, _, _ = reviewer
    bundle = rev.prepare_review(_coordinator_request(_framework_op_prompt(), "sess_fw"))
    constraints = bundle.review_constraints
    assert constraints["bundle_action_class"] == "framework_op"
    assert constraints["approve_requires"] == []
    assert constraints["proposal_action_classes"] == {"baseA": "framework_op"}


def test_prepare_review_mixed_batch_uses_strictest_class(reviewer):
    rev, _, _ = reviewer
    bundle = rev.prepare_review(_coordinator_request(_mixed_prompt(), "sess_mix"))
    constraints = bundle.review_constraints
    # patch_landing wins over evidence_producer at the bundle level — a
    # caller that ignores proposal_action_classes still gets the most
    # conservative behaviour.
    assert constraints["bundle_action_class"] == "patch_landing"
    assert "comparable_before_after_benchmark" in constraints["approve_requires"]
    # But the per-proposal map still classifies each correctly so
    # SKILL-aware callers can apply the right per-proposal bar.
    assert constraints["proposal_action_classes"] == {
        "expA": "evidence_producer",
        "intB": "patch_landing",
    }


def test_prepare_review_unknown_action_falls_back_to_evidence_producer(reviewer):
    rev, _, _ = reviewer
    prompt = (
        "=== Shared session state ===\n"
        "model=qwen3-14b framework=sglang baseline_tput=1200\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=newA from=orchestration topic=proposal payload={'action_name': 'shiny_new_action'}\n"
    )
    bundle = rev.prepare_review(_coordinator_request(prompt, "sess_unknown"))
    constraints = bundle.review_constraints
    assert constraints["proposal_action_classes"] == {"newA": "evidence_producer"}
    assert constraints["bundle_action_class"] == "evidence_producer"


def test_prepare_review_no_proposals_uses_strict_default(reviewer):
    """``critic_decision_request`` carries no inbox proposals — the
    legacy strict checklist remains the safe default for those callers
    (kernel-patch landing dialogue review)."""
    rev, _, _ = reviewer
    bundle = rev.prepare_review({
        "kind": "critic_decision_request",
        "session_id": "sess_nop",
        "context": {"model": "qwen3-14b", "framework": "sglang"},
        "messages": [{"role": "coordinator", "content": "decide"}],
        "decision": {"summary": "land patch x"},
    })
    constraints = bundle.review_constraints
    assert constraints["approve_requires"] == [
        "comparable_before_after_benchmark",
        "accuracy_gate_or_waiver",
        "active_path_proof_when_relevant",
        "rollback_plan",
    ]
    # Decision requests don't classify per-proposal.
    assert "proposal_action_classes" not in constraints
    assert "bundle_action_class" not in constraints


def test_prepare_review_skips_kb_when_critical_context_missing(reviewer):
    rev, kb, sm = reviewer
    bundle = rev.prepare_review({
        "kind": "critic_decision_request",
        "session_id": "sess_b",
        "messages": [{"role": "coordinator", "content": "decide"}],
        "decision": {"summary": "adopt patch x"},
    })
    assert bundle.required_context == ["model", "framework"]
    assert bundle.kb_read_skipped_reason == "missing_critical_context"


def test_prepare_review_returns_kb_priors_per_proposal(reviewer):
    rev, kb, sm = reviewer
    kb.upsert({
        "scope": {
            "org": "hyperloom",
            "framework": "sglang",
            "model": "qwen3-14b",
            "model_family": "qwen",
            "workload": "decode",
            "precision": "fp8",
        },
        "kind": "pitfall",
        "slug": "active-path-unproven-pitfall",
        "importance": 0.5,
        "metadata": {"topic": "active path"},
    })
    prompt = (
        "=== Shared session state ===\n"
        "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=aaa from=orchestration topic=proposal payload={'action_name': 'kernel_opt'}\n"
    )
    bundle = rev.prepare_review(_coordinator_request(prompt, "sess_priors"))
    assert "aaa" in bundle.kb_priors_by_proposal


def test_prepare_review_filters_already_reviewed_proposals(reviewer):
    rev, kb, sm = reviewer
    sm.mark_reviewed("sess_dedup", "aaa1", "approve")
    bundle = rev.prepare_review(_coordinator_request(
        _PROMPT_WITH_TWO_PROPOSALS, "sess_dedup",
    ))
    proposal_ids = sorted(p["msg_id"] for p in bundle.proposals)
    assert proposal_ids == ["bbb2"]


def test_commit_review_for_coordinator_inbox_emits_intent_envelope(reviewer):
    rev, kb, sm = reviewer
    rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_c"))
    review = {
        "review_verdicts": [
            {
                "target_proposal_msg_id": "aaa1",
                "verdict": "approve",
                "reasoning": "matches kb-1",
                "confidence": "medium",
                "predicted_gain_pct": 0.0,
            },
            {
                "target_proposal_msg_id": "bbb2",
                "verdict": "reject",
                "reasoning": "active dispatch path unproven",
                "kb_evidence": ["kb_x"],
            },
        ]
    }
    outcome = rev.commit_review(
        _coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_c"),
        review,
    )
    assert outcome.intent_envelope is not None
    intents = outcome.intent_envelope["intents"]
    assert len(intents) == 2
    types = [i["intent_type"] for i in intents]
    verdicts = [i["payload"]["verdict"] for i in intents]
    assert types == ["review_verdict", "review_verdict"]
    assert sorted(verdicts) == ["approve", "reject"]
    # Reviewed msg_ids tracked
    assert sm.is_msg_already_reviewed("sess_c", "aaa1")
    assert sm.is_msg_already_reviewed("sess_c", "bbb2")


def test_commit_review_invalid_verdict_raises(reviewer):
    rev, kb, sm = reviewer
    rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_d"))
    with pytest.raises(ReviewValidationError, match="not valid"):
        rev.commit_review(
            _coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_d"),
            {"review_verdicts": [{"target_proposal_msg_id": "aaa1", "verdict": "lgtm"}]},
        )


def test_commit_review_no_proposals_emits_heartbeat(reviewer):
    rev, kb, sm = reviewer
    prompt = (
        "=== Shared session state ===\n"
        "model=qwen3-14b framework=sglang\n"
        "=== Inbox for critic ===\n"
        "(no new messages)\n"
    )
    rev.prepare_review(_coordinator_request(prompt, "sess_e"))
    outcome = rev.commit_review(
        _coordinator_request(prompt, "sess_e"),
        {"review_verdicts": []},
    )
    intents = outcome.intent_envelope["intents"]
    assert len(intents) == 1
    assert intents[0]["intent_type"] == "send_message"
    assert intents[0]["payload"]["topic"] == "heartbeat"


def test_commit_review_persists_to_kb_when_flagged(reviewer):
    rev, kb, sm = reviewer
    rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_f"))
    review = {
        "review_verdicts": [
            {
                "target_proposal_msg_id": "aaa1",
                "verdict": "reject",
                "reasoning": "active dispatch path unproven for this kernel",
                "packet_evidence": ["benchmark.after.gain_pct"],
                "persist_to_kb": True,
                "topic": "active dispatch path unproven",
            },
        ]
    }
    outcome = rev.commit_review(
        _coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_f"),
        review,
    )
    assert any(w["trigger"] == "review_verdict" for w in outcome.kb_writes)
    assert kb.all_rows()


def test_decision_request_prepare_with_full_context(reviewer):
    rev, kb, sm = reviewer
    bundle = rev.prepare_review({
        "kind": "critic_decision_request",
        "session_id": "sess_g",
        "messages": [{"role": "coordinator", "content": "adopt patch x?"}],
        "context": {
            "model": "deepseek-r1-0528-fp8", "framework": "sglang",
            "model_family": "deepseek", "workload": "decode", "precision": "fp8",
        },
        "decision": {"summary": "adopt patch x"},
    })
    assert bundle.required_context == []
    assert bundle.kb_read_skipped_reason is None


def test_decision_request_commit_emits_decision_review(reviewer):
    rev, kb, sm = reviewer
    request = {
        "kind": "critic_decision_request",
        "session_id": "sess_h",
        "messages": [{"role": "coordinator", "content": "adopt?"}],
        "context": {
            "model": "deepseek-r1-0528-fp8", "framework": "sglang",
            "model_family": "deepseek", "workload": "decode", "precision": "fp8",
        },
        "decision": {"summary": "adopt patch x"},
    }
    rev.prepare_review(request)
    review = {
        "verdict": "adopt",
        "reason": "no contradicting kb prior; benchmark within tolerance",
        "recommendation": "rerun final benchmark after cache clear",
        "basis": "session",
        "session_evidence": ["benchmark.after.gain_pct"],
    }
    outcome = rev.commit_review(request, review)
    assert outcome.decision_review is not None
    assert outcome.decision_review["verdict"] == "adopt"
    assert outcome.kb_writes  # review_verdict-equivalent triggered


def test_init_session_records_event(reviewer):
    rev, kb, sm = reviewer
    out = rev.init_session({
        "kind": "critic_decision_request",
        "session_id": "sess_init",
        "context": {"model": "qwen3-14b", "framework": "sglang"},
        "messages": [],
    })
    assert out["session_id"] == "sess_init"
    events = sm.list_events("sess_init")
    assert events and events[0]["kind"] == "init_session"


def test_close_session_writes_kb_drafts_when_provided(reviewer):
    rev, kb, sm = reviewer
    rev.init_session({
        "kind": "critic_decision_request",
        "session_id": "sess_close",
        "context": {
            "model": "qwen3-14b", "framework": "sglang",
            "model_family": "qwen", "workload": "decode", "precision": "fp8",
        },
        "messages": [],
    })
    outcome = rev.close_session(
        {"kind": "critic_decision_request", "session_id": "sess_close",
         "context": {"model": "qwen3-14b", "framework": "sglang",
                     "model_family": "qwen", "workload": "decode", "precision": "fp8"}},
        kb_draft={
            "kb_drafts": [
                {
                    "category": "kernel_optimization",
                    "action": "Patched the active dispatch path for Qwen3-14B.",
                    "lesson": "Active dispatch path must be kept in sync with kernel rewrite.",
                    "tags": ["dispatch"],
                    "result": {"status": "KEEP", "gain_pct": 4.2},
                    "confidence": 0.9,
                }
            ]
        },
    )
    assert outcome.kb_writes
    assert outcome.kb_writes[0]["result"]["status"] in ("ok", "skipped", "dead_lettered")


def test_kb_priors_cache_hit_avoids_second_kb_call(reviewer):
    rev, kb, sm = reviewer
    kb.upsert({
        "scope": {
            "org": "hyperloom", "framework": "sglang", "model": "qwen3-14b",
            "model_family": "qwen", "workload": "decode", "precision": "fp8",
        },
        "kind": "pitfall", "slug": "active-path-unproven-pitfall", "importance": 0.5,
        "metadata": {"topic": "active path"},
    })
    prompt = (
        "=== Shared session state ===\n"
        "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=aaa from=orchestration topic=proposal payload={'action_name': 'kernel_opt'}\n"
    )
    rev.prepare_review(_coordinator_request(prompt, "sess_cache_e2e"))
    # Now flush KB to ensure the second call uses cache.
    kb.reset()
    bundle = rev.prepare_review(_coordinator_request(prompt, "sess_cache_e2e"))
    # First call cached the result; KB is now empty but bundle still
    # contains priors because of cache hit.
    assert bundle.kb_priors_by_proposal["aaa"]
