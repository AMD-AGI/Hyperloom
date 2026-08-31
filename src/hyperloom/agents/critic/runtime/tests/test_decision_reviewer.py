# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for the prepare/commit lifecycle via :class:`InMemoryKBClient`."""

from __future__ import annotations

import json

import pytest

from hyperloom.agents.critic.runtime.decision_reviewer import DecisionReviewer
from hyperloom.agents.critic.runtime.errors import ReviewValidationError
from hyperloom.agents.critic.runtime.in_memory_kb_client import InMemoryKBClient
from hyperloom.agents.critic.runtime.kb_writer import KBWriter
from hyperloom.agents.critic.runtime.request_models import Proposal
from hyperloom.agents.critic.runtime.session_memory import SessionMemory


def test_topic_for_proposal_prefers_action_then_payload_then_fallback(reviewer):
    rev, _, _ = reviewer
    assert rev._topic_for_proposal(Proposal(msg_id="m1", from_agent="o", action_name="explore")) == "explore"
    assert (
        rev._topic_for_proposal(Proposal(msg_id="m2", from_agent="o", payload={"topic": "  tune gemm  "}))
        == "tune gemm"
    )
    assert (
        rev._topic_for_proposal(Proposal(msg_id="m3", from_agent="o", payload={"summary": "raise conc"}))
        == "raise conc"
    )
    assert rev._topic_for_proposal(Proposal(msg_id="m4", from_agent="o", payload={})) == "proposal-m4"


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
    # Mixed batch: strictest class wins at bundle level (evidence_producer).
    constraints = bundle.review_constraints
    assert constraints["bundle_action_class"] == "evidence_producer"
    assert "comparable_before_after_benchmark" not in constraints["approve_requires"]
    assert "in_phase_allowed_action" in constraints["approve_requires"]
    assert constraints["proposal_action_classes"] == {
        "aaa1": "framework_op",
        "bbb2": "evidence_producer",
    }
    by_cls = constraints["approve_requires_by_class"]
    assert "patch_landing" in by_cls
    assert "evidence_producer" in by_cls
    assert "framework_op" in by_cls
    assert by_cls["framework_op"] == []


def test_prepare_review_propagates_known_actions(reviewer):
    rev, kb, sm = reviewer
    bundle = rev.prepare_review(
        {
            "kind": "coordinator_inbox",
            "session_id": "sess_known",
            "raw_prompt": _PROMPT_WITH_TWO_PROPOSALS,
            "options": {"known_actions": ["sweep", "baseline", "validate_stack"]},
        }
    )
    assert bundle.review_constraints["known_actions"] == [
        "baseline",
        "sweep",
        "validate_stack",
    ]


def test_prepare_review_omits_known_actions_when_absent(reviewer):
    rev, kb, sm = reviewer
    bundle = rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS))
    assert "known_actions" not in bundle.review_constraints


# Per-action review_constraints (action-class taxonomy)
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
    assert "comparable_before_after_benchmark" not in constraints["approve_requires"]
    assert "accuracy_gate_or_waiver" not in constraints["approve_requires"]
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
    # The 4-item strict checklist is preserved verbatim for patch landing.
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


def test_classify_candidate_prescreen_is_framework_op():
    """A pre-screen classifies as framework_op; the same action landing a patch does not.

    Both arrive as ``integrate_patch`` -- one action lands every patch source
    now -- so the top-level candidate id is the only thing separating "is this
    candidate worth a bench" from "this patch is applied and measured". Reading
    the action name alone would drop the patch-landing evidence bar for every
    real patch.
    """
    from hyperloom.agents.critic.runtime.decision_reviewer import (
        _APPROVE_REQUIRES_BY_CLASS,
        ACTION_CLASS_FRAMEWORK_OP,
        ACTION_CLASS_PATCH_LANDING,
        classify_proposal_action,
    )

    prescreen = {"framework_agent_candidate_id": "https://github.com/o/r/pull/7"}
    assert classify_proposal_action("integrate_patch", prescreen) == ACTION_CLASS_FRAMEWORK_OP
    assert classify_proposal_action("integrate_patch", {}) == ACTION_CLASS_PATCH_LANDING
    # Nothing has been applied or measured yet, so the pre-screen carries no bar.
    assert _APPROVE_REQUIRES_BY_CLASS[ACTION_CLASS_FRAMEWORK_OP] == ()
    assert _APPROVE_REQUIRES_BY_CLASS[ACTION_CLASS_PATCH_LANDING] != ()


def test_classify_enablement_integrate_patch_is_enablement_landing():
    """A pre-boot enablement integrate_patch drops the production bar."""
    from hyperloom.agents.critic.runtime.decision_reviewer import (
        _APPROVE_REQUIRES_BY_CLASS,
        ACTION_CLASS_ENABLEMENT_LANDING,
        ACTION_CLASS_PATCH_LANDING,
        classify_proposal_action,
    )

    # Plain integrate_patch (no enablement marker) stays strict.
    assert classify_proposal_action("integrate_patch", {"params": {}}) == ACTION_CLASS_PATCH_LANDING
    assert classify_proposal_action("integrate_patch", None) == ACTION_CLASS_PATCH_LANDING
    # enablement=True or framework_agent_authoring=True downgrades the class.
    assert (
        classify_proposal_action("integrate_patch", {"params": {"enablement": True}}) == ACTION_CLASS_ENABLEMENT_LANDING
    )
    assert (
        classify_proposal_action("integrate", {"params": {"framework_agent_authoring": True}})
        == ACTION_CLASS_ENABLEMENT_LANDING
    )
    # The lighter bar excludes the pre-boot-impossible production evidence
    # and the redundant rollback restatement.
    reqs = _APPROVE_REQUIRES_BY_CLASS[ACTION_CLASS_ENABLEMENT_LANDING]
    assert "comparable_before_after_benchmark" not in reqs
    assert "accuracy_gate_or_waiver" not in reqs
    assert "rollback_plan" not in reqs
    assert "specialist_or_default_grid_provenance" in reqs
    assert "in_phase_allowed_action" in reqs
    assert "no_contradicting_kb_prior" in reqs


def test_prepare_review_enablement_integrate_relaxes_approve_requires(reviewer):
    rev, _, _ = reviewer
    prompt = (
        "=== Shared session state ===\n"
        "model=deepseek-v4 framework=vllm baseline_tput=0\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=enA from=orchestration topic=proposal payload="
        "{'action_name': 'integrate_patch', 'provenance': 'specialist', "
        "'params': {'enablement': True, 'framework_agent_authoring': True}}\n"
    )
    bundle = rev.prepare_review(_coordinator_request(prompt, "sess_enable"))
    constraints = bundle.review_constraints
    assert constraints["bundle_action_class"] == "enablement_landing"
    assert constraints["proposal_action_classes"] == {"enA": "enablement_landing"}
    assert "comparable_before_after_benchmark" not in constraints["approve_requires"]
    assert "accuracy_gate_or_waiver" not in constraints["approve_requires"]
    assert "rollback_plan" not in constraints["approve_requires"]


def test_prepare_review_enablement_mixed_with_real_patch_stays_strict(reviewer):
    """A real production integrate in the same batch still forces the strict bar."""
    rev, _, _ = reviewer
    prompt = (
        "=== Shared session state ===\n"
        "model=deepseek-v4 framework=vllm baseline_tput=1200\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=enA from=orchestration topic=proposal payload="
        "{'action_name': 'integrate_patch', 'params': {'enablement': True}}\n"
        "  seq=2 msg_id=plB from=orchestration topic=proposal payload="
        "{'action_name': 'integrate_patch', 'params': {}}\n"
    )
    bundle = rev.prepare_review(_coordinator_request(prompt, "sess_enable_mixed"))
    constraints = bundle.review_constraints
    assert constraints["bundle_action_class"] == "patch_landing"
    assert "comparable_before_after_benchmark" in constraints["approve_requires"]
    assert constraints["proposal_action_classes"] == {
        "enA": "enablement_landing",
        "plB": "patch_landing",
    }


def test_prepare_review_mixed_batch_uses_strictest_class(reviewer):
    rev, _, _ = reviewer
    bundle = rev.prepare_review(_coordinator_request(_mixed_prompt(), "sess_mix"))
    constraints = bundle.review_constraints
    # patch_landing wins over evidence_producer at the bundle level.
    assert constraints["bundle_action_class"] == "patch_landing"
    assert "comparable_before_after_benchmark" in constraints["approve_requires"]
    # Per-proposal map still classifies each correctly.
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
    """No inbox proposals: the strict checklist stays the safe default."""
    rev, _, _ = reviewer
    bundle = rev.prepare_review(
        {
            "kind": "critic_decision_request",
            "session_id": "sess_nop",
            "context": {"model": "qwen3-14b", "framework": "sglang"},
            "messages": [{"role": "coordinator", "content": "decide"}],
            "decision": {"summary": "land patch x"},
        }
    )
    constraints = bundle.review_constraints
    assert constraints["approve_requires"] == [
        "comparable_before_after_benchmark",
        "accuracy_gate_or_waiver",
        "active_path_proof_when_relevant",
        "rollback_plan",
    ]
    assert "proposal_action_classes" not in constraints
    assert "bundle_action_class" not in constraints


def test_prepare_review_skips_kb_when_critical_context_missing(reviewer):
    rev, kb, sm = reviewer
    bundle = rev.prepare_review(
        {
            "kind": "critic_decision_request",
            "session_id": "sess_b",
            "messages": [{"role": "coordinator", "content": "decide"}],
            "decision": {"summary": "adopt patch x"},
        }
    )
    assert bundle.required_context == ["model", "framework"]
    assert bundle.kb_read_skipped_reason == "missing_critical_context"


def test_prepare_review_returns_kb_priors_per_proposal(reviewer):
    rev, kb, sm = reviewer
    kb.upsert(
        {
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
        }
    )
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
    bundle = rev.prepare_review(
        _coordinator_request(
            _PROMPT_WITH_TWO_PROPOSALS,
            "sess_dedup",
        )
    )
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
    reviewed = json.loads((sm.session_dir("sess_c") / "reviewed_msg_ids.json").read_text("utf-8"))
    assert {"aaa1", "bbb2"} <= set(reviewed)


def _verdict_intent_for(intents: list[dict], target: str) -> dict:
    for intent in intents:
        if intent["intent_type"] == "review_verdict" and intent["payload"]["target_proposal_msg_id"] == target:
            return intent
    raise AssertionError(f"no review_verdict intent for {target!r}")


def test_commit_review_carries_the_cited_rule_into_the_intent(reviewer):
    """The Coordinator holds a reject to the verdict its rule declared, and it
    can only do that if the code the Critic cited survives the commit path."""
    rev, _kb, sm = reviewer
    rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_code"))
    review = {
        "review_verdicts": [
            {
                "target_proposal_msg_id": "aaa1",
                "verdict": "reject",
                "reasoning": "proposal carried a self-reported gain",
                "failure_reason_code": "specialist_quantitative_claim_violation",
            },
            {
                "target_proposal_msg_id": "bbb2",
                "verdict": "approve",
                "reasoning": "evidence is complete",
            },
        ]
    }
    outcome = rev.commit_review(
        _coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_code"),
        review,
    )
    intents = outcome.intent_envelope["intents"]

    cited = _verdict_intent_for(intents, "aaa1")["payload"]
    assert cited["failure_reason_code"] == "specialist_quantitative_claim_violation"
    uncited = _verdict_intent_for(intents, "bbb2")["payload"]
    assert uncited["failure_reason_code"] == ""
    logged = [
        json.loads(line)["decision_review"]
        for line in (sm.session_dir("sess_code") / "decisions.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    codes = {d["target_proposal_msg_id"]: d["failure_reason_code"] for d in logged}
    assert codes["aaa1"] == "specialist_quantitative_claim_violation"


def test_commit_review_backfills_advice_text_from_advice_entry(reviewer):
    rev, kb, sm = reviewer
    rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_advice"))
    review = {
        "review_verdicts": [
            {
                "target_proposal_msg_id": "aaa1",
                "verdict": "advise",
                "reasoning": "proceed with caveats",
            }
        ],
        "advice": [
            {
                "target_proposal_msg_id": "aaa1",
                "body_md": "Re-run the sweep at higher concurrency before promotion.",
            }
        ],
    }
    outcome = rev.commit_review(
        _coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_advice"),
        review,
    )
    intents = outcome.intent_envelope["intents"]
    verdict_intent = _verdict_intent_for(intents, "aaa1")
    assert verdict_intent["payload"]["advice_text"] == "Re-run the sweep at higher concurrency before promotion."
    advice_intents = [
        i for i in intents if i["intent_type"] == "send_message" and i["payload"].get("topic") == "advice"
    ]
    assert len(advice_intents) == 1
    assert advice_intents[0]["payload"]["body_md"] == "Re-run the sweep at higher concurrency before promotion."


def test_commit_review_merges_inline_and_advice_entries(reviewer):
    rev, kb, sm = reviewer
    rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_merge"))
    review = {
        "review_verdicts": [
            {
                "target_proposal_msg_id": "aaa1",
                "verdict": "advise",
                "reasoning": "proceed with caveats",
                "advice_text": "Inline advice.",
            }
        ],
        "advice": [
            {"target_proposal_msg_id": "aaa1", "body_md": "First broadcast advice."},
            {"target_proposal_msg_id": "aaa1", "body_md": "Second broadcast advice."},
        ],
    }
    outcome = rev.commit_review(
        _coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_merge"),
        review,
    )
    intents = outcome.intent_envelope["intents"]
    verdict_intent = _verdict_intent_for(intents, "aaa1")
    assert verdict_intent["payload"]["advice_text"] == (
        "Inline advice.\n\nFirst broadcast advice.\n\nSecond broadcast advice."
    )


def test_commit_review_keeps_advice_text_empty_when_no_advice(reviewer):
    rev, kb, sm = reviewer
    rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_empty"))
    review = {
        "review_verdicts": [
            {
                "target_proposal_msg_id": "aaa1",
                "verdict": "approve",
                "reasoning": "ok",
            }
        ]
    }
    outcome = rev.commit_review(
        _coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_empty"),
        review,
    )
    intents = outcome.intent_envelope["intents"]
    verdict_intent = _verdict_intent_for(intents, "aaa1")
    assert verdict_intent["payload"]["advice_text"] == ""


def test_commit_review_invalid_verdict_raises(reviewer):
    rev, kb, sm = reviewer
    rev.prepare_review(_coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_d"))
    with pytest.raises(ReviewValidationError, match="not valid"):
        rev.commit_review(
            _coordinator_request(_PROMPT_WITH_TWO_PROPOSALS, "sess_d"),
            {"review_verdicts": [{"target_proposal_msg_id": "aaa1", "verdict": "lgtm"}]},
        )


def test_commit_review_kb_draft_request_writes_drafts(reviewer):
    rev, kb, sm = reviewer
    request = {
        "kind": "kb_draft_request",
        "session_id": "sess_kbd",
        "context": {
            "model": "qwen3-14b",
            "framework": "sglang",
            "model_family": "qwen",
            "workload": "decode",
            "precision": "fp8",
        },
    }
    review = {
        "kb_drafts": [
            {
                "category": "kernel_optimization",
                "action": "Patched the active dispatch path.",
                "lesson": "Active dispatch path must be kept in sync.",
                "tags": ["dispatch"],
                "result": {"status": "KEEP", "gain_pct": 4.2},
                "confidence": 0.9,
            }
        ]
    }
    outcome = rev.commit_review(request, review)
    assert outcome.kind == "kb_draft_request"
    assert outcome.kb_writes and outcome.kb_writes[0]["trigger"] == "kb_draft"
    assert outcome.decision_review["kind"] == "critic_kb_draft"
    assert outcome.decision_review["kb_drafts_attempted"] == 1


def test_commit_review_kb_draft_non_list_raises(reviewer):
    rev, kb, sm = reviewer
    request = {
        "kind": "kb_draft_request",
        "session_id": "sess_kbd2",
        "context": {"model": "qwen3-14b", "framework": "sglang"},
    }
    with pytest.raises(ReviewValidationError, match="kb_drafts list"):
        rev.commit_review(request, {"kb_drafts": {"not": "a list"}})


def test_commit_review_no_proposals_emits_heartbeat(reviewer):
    rev, kb, sm = reviewer
    prompt = (
        "=== Shared session state ===\nmodel=qwen3-14b framework=sglang\n=== Inbox for critic ===\n(no new messages)\n"
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
    bundle = rev.prepare_review(
        {
            "kind": "critic_decision_request",
            "session_id": "sess_g",
            "messages": [{"role": "coordinator", "content": "adopt patch x?"}],
            "context": {
                "model": "deepseek-r1-0528-fp8",
                "framework": "sglang",
                "model_family": "deepseek",
                "workload": "decode",
                "precision": "fp8",
            },
            "decision": {"summary": "adopt patch x"},
        }
    )
    assert bundle.required_context == []
    assert bundle.kb_read_skipped_reason is None


def test_decision_request_commit_emits_decision_review(reviewer):
    rev, kb, sm = reviewer
    request = {
        "kind": "critic_decision_request",
        "session_id": "sess_h",
        "messages": [{"role": "coordinator", "content": "adopt?"}],
        "context": {
            "model": "deepseek-r1-0528-fp8",
            "framework": "sglang",
            "model_family": "deepseek",
            "workload": "decode",
            "precision": "fp8",
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
    assert outcome.kb_writes


def test_init_session_merges_context(reviewer):
    rev, kb, sm = reviewer
    out = rev.init_session(
        {
            "kind": "critic_decision_request",
            "session_id": "sess_init",
            "context": {"model": "qwen3-14b", "framework": "sglang"},
            "messages": [],
        }
    )
    assert out["session_id"] == "sess_init"


def test_close_session_writes_kb_drafts_when_provided(reviewer):
    rev, kb, sm = reviewer
    rev.init_session(
        {
            "kind": "critic_decision_request",
            "session_id": "sess_close",
            "context": {
                "model": "qwen3-14b",
                "framework": "sglang",
                "model_family": "qwen",
                "workload": "decode",
                "precision": "fp8",
            },
            "messages": [],
        }
    )
    outcome = rev.close_session(
        {
            "kind": "critic_decision_request",
            "session_id": "sess_close",
            "context": {
                "model": "qwen3-14b",
                "framework": "sglang",
                "model_family": "qwen",
                "workload": "decode",
                "precision": "fp8",
            },
        },
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
    kb.upsert(
        {
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
        }
    )
    prompt = (
        "=== Shared session state ===\n"
        "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=aaa from=orchestration topic=proposal payload={'action_name': 'kernel_opt'}\n"
    )
    rev.prepare_review(_coordinator_request(prompt, "sess_cache_e2e"))
    kb.reset()
    bundle = rev.prepare_review(_coordinator_request(prompt, "sess_cache_e2e"))
    # KB is now empty, but the bundle still has priors from the cache hit.
    assert bundle.kb_priors_by_proposal["aaa"]


# KB trace audit fields (kb_priors_trace)
def test_kb_priors_trace_records_scope_and_requests(tmp_path):
    sm = SessionMemory(root=tmp_path / "sm")
    kb = InMemoryKBClient()
    kb.upsert(
        {
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
        }
    )
    writer = KBWriter(kb, session_memory=sm)
    rev = DecisionReviewer(session_memory=sm, kb_writer=writer)
    prompt = (
        "=== Shared session state ===\n"
        "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
        "=== Inbox for critic ===\n"
        "  seq=1 msg_id=aaa from=orchestration topic=proposal payload={'action_name': 'kernel_opt'}\n"
    )
    bundle = rev.prepare_review(_coordinator_request(prompt, "sess_pt"))
    trace = bundle.kb_priors_trace
    assert trace["configured"] is True
    assert trace["mode"] == "per_proposal"
    assert trace["scope_filter"].get("model") == "qwen3-14b"
    assert any(r["msg_id"] == "aaa" for r in trace["requests"])
