"""P0-2 agent role + PolicyGate tests.

Covers:

* default_role_registry returns the 4 v0.6 PascalCase-capable roles
* Orchestration permission matrix (DELEGATE / REQUEST / no kernel-owned)
* Kernel permission matrix (RESPONSE only / no PROPOSE_ACTION / no REQUEST)
* Critic permission matrix (REVIEW_VERDICT only; no DELEGATE / REQUEST)
* Robustness permission matrix (KILL_TASK + scheduling police; no propose)
* PolicyGate REVIEW_VERDICT validation (verdict allowlist + critic-only source)
* PolicyGate kernel-owned action delegate rejection
* PolicyGate REQUEST routing (only orchestration→kernel)
* PolicyGate kill_task source allowlist + scope guard
* PolicyGate prune_branch / force_dispatch / escalate_strategy_change source guard
* PolicyGate state field guard (no role can mutate CORE_STATE_FIELDS)
* allowed_tools_for_agent semantics (Codex no-tools / Claude → ["emit_intent"])
* All 4 system_prompts/*.md files exist and are non-empty
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import (
    BackendType,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    default_role_registry,
    roles_for_run,
)
from inference_optimizer.protocol.intent import (
    Intent,
    IntentType,
)
from inference_optimizer.orchestrator.policy import (
    CORE_STATE_FIELDS,
    DELEGATE_ACTION_REQUIRED_PAYLOAD,
    DELEGATE_ACTION_SOURCE_ALLOWLIST,
    KERNEL_OWNED_ACTIONS,
    KILL_TASK_SOURCE_ALLOWLIST,
    PolicyDenied,
    PolicyGate,
    REQUEST_ROUTING,
    REVIEW_VERDICTS,
    REVIEW_VERDICT_SOURCE_ALLOWLIST,
    ROBUSTNESS_ONLY_INTENTS,
    ROBUSTNESS_ONLY_SOURCE_ALLOWLIST,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import asset_system_prompts_dir


# ===========================================================================
# agent_role
# ===========================================================================
def test_default_role_registry_has_4_v06_agents():
    reg = default_role_registry()
    assert set(reg.keys()) == {"orchestration", "kernel", "critic", "robustness"}


def test_roles_for_run_deterministic_order():
    assert roles_for_run() == ("orchestration", "kernel", "critic", "robustness")


def test_orchestration_permissions():
    role = default_role_registry()["orchestration"]
    assert role.backend_type == BackendType.CLAUDE
    assert role.model == DEFAULT_CLAUDE_MODEL
    assert role.can_delegate_side_effects is True
    assert role.can_mutate_core_state is False
    assert IntentType.PROPOSE_ACTION in role.allowed_intents
    assert IntentType.DELEGATE in role.allowed_intents
    assert IntentType.REQUEST in role.allowed_intents
    assert IntentType.UPDATE_STATE in role.allowed_intents
    # Orchestration may forward roofline-driven prune advice and request
    # phase-advance hints directly (skip_to_kernel / sweep / close).
    assert IntentType.PRUNE_BRANCH in role.allowed_intents
    assert IntentType.ESCALATE_STRATEGY_CHANGE in role.allowed_intents
    # Cannot review / kill / force_dispatch / response
    assert IntentType.REVIEW_VERDICT not in role.allowed_intents
    assert IntentType.KILL_TASK not in role.allowed_intents
    assert IntentType.FORCE_DISPATCH not in role.allowed_intents
    assert IntentType.RESPONSE not in role.allowed_intents


def test_kernel_responder_only():
    role = default_role_registry()["kernel"]
    assert role.backend_type == BackendType.CLAUDE
    assert role.can_delegate_side_effects is False
    assert IntentType.RESPONSE in role.allowed_intents
    # Cannot initiate
    assert IntentType.PROPOSE_ACTION not in role.allowed_intents
    assert IntentType.DELEGATE not in role.allowed_intents
    assert IntentType.REQUEST not in role.allowed_intents


def test_critic_review_only_codex_no_tools():
    role = default_role_registry()["critic"]
    assert role.backend_type == BackendType.CODEX
    assert role.model == DEFAULT_CODEX_MODEL
    assert role.no_tools is True
    assert IntentType.REVIEW_VERDICT in role.allowed_intents
    # Cannot delegate / request / propose
    assert IntentType.DELEGATE not in role.allowed_intents
    assert IntentType.REQUEST not in role.allowed_intents
    assert IntentType.PROPOSE_ACTION not in role.allowed_intents
    # Cannot kill / scheduling-police
    assert IntentType.KILL_TASK not in role.allowed_intents
    assert IntentType.FORCE_DISPATCH not in role.allowed_intents


def test_robustness_scheduling_police():
    role = default_role_registry()["robustness"]
    assert role.backend_type == BackendType.CLAUDE
    assert IntentType.KILL_TASK in role.allowed_intents
    assert IntentType.FORCE_DISPATCH in role.allowed_intents
    assert IntentType.PRUNE_BRANCH in role.allowed_intents
    assert IntentType.ESCALATE_STRATEGY_CHANGE in role.allowed_intents
    # Cannot propose / request / review_verdict
    assert IntentType.PROPOSE_ACTION not in role.allowed_intents
    assert IntentType.REQUEST not in role.allowed_intents
    assert IntentType.REVIEW_VERDICT not in role.allowed_intents


# ===========================================================================
# PolicyGate constants
# ===========================================================================
def test_kernel_owned_actions_include_gemm_tuning():
    assert KERNEL_OWNED_ACTIONS == frozenset({
        "kernel_opt", "integrate", "deep_kernel_analysis",
        "operator_tuning", "vendor_kernel_config", "gemm_tuning",
    })


def test_request_routing_v06_only_orchestration_to_kernel():
    assert set(REQUEST_ROUTING.keys()) == {"orchestration"}
    assert REQUEST_ROUTING["orchestration"] == frozenset({"kernel"})


def test_review_verdict_critic_only():
    assert REVIEW_VERDICT_SOURCE_ALLOWLIST == frozenset({"critic"})
    assert "approve" in REVIEW_VERDICTS
    assert "needs_review" in REVIEW_VERDICTS
    assert "objection" not in REVIEW_VERDICTS  # parliament removed


def test_kill_and_robustness_only_renamed():
    assert KILL_TASK_SOURCE_ALLOWLIST == frozenset({"robustness"})
    assert ROBUSTNESS_ONLY_SOURCE_ALLOWLIST == frozenset({"robustness"})
    assert ROBUSTNESS_ONLY_INTENTS == frozenset({
        IntentType.FORCE_DISPATCH,
        IntentType.PRUNE_BRANCH,
        IntentType.ESCALATE_STRATEGY_CHANGE,
    })


def test_core_state_fields_includes_current_best():
    assert "current_best" in CORE_STATE_FIELDS
    assert "stop_reason" in CORE_STATE_FIELDS


# ===========================================================================
# PolicyGate validation
# ===========================================================================
@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry())


def test_gate_unknown_agent_rejected(gate):
    with pytest.raises(PolicyDenied, match="unknown agent"):
        gate.validate_intent("ghost", Intent(type=IntentType.SEND_MESSAGE,
                                              payload={"topic": "heartbeat"}))


def test_gate_orchestration_propose_action_ok(gate):
    gate.validate_intent("orchestration", Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
    ))


def test_gate_orchestration_delegate_kernel_owned_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": "kernel_opt"},
        ))
    assert exc.value.rule == "kernel_owned_by_kernel_agent"


def test_gate_gemm_tuning_rejected_for_non_fp8_proposal():
    state = SharedState(phase="KERNEL", precision="bf16", framework="sglang")
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "gemm_tuning", "predicted_gain_pct": 10.0},
        ))
    assert exc.value.rule == "fp8_only_action"


def test_gate_run_gemm_tuning_request_allowed_for_fp8():
    state = SharedState(phase="KERNEL", precision="fp8", framework="sglang")
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    gate.validate_intent("orchestration", Intent(
        type=IntentType.REQUEST,
        payload={"target_agent": "kernel", "kind": "run_gemm_tuning", "params": {}},
    ))


def test_gate_orchestration_delegate_normal_action_ok(gate):
    gate.validate_intent("orchestration", Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "baseline"},
    ))


# ---------------------------------------------------------------------------
# Per-action delegate source allowlist (DELEGATE_ACTION_SOURCE_ALLOWLIST)
#
# ``recover`` walks SIGTERM/SIGKILL against matching processes and is
# env-gated to optionally invoke ``rocm-smi --gpureset``. The
# robustness-agent path emits it as the tail of the gpu_memory_leaked
# action ladder; any other source must be rejected so PolicyGate is the
# single chokepoint between an LLM-generated intent and a kill spree.
# ---------------------------------------------------------------------------
def test_delegate_action_source_allowlist_constant_shape():
    """``recover`` is the only entry today; if more side-effecting
    actions need source gating the test should be extended deliberately."""
    assert DELEGATE_ACTION_SOURCE_ALLOWLIST == {
        "recover": frozenset({"robustness"}),
    }


def test_delegate_action_required_payload_constant_shape():
    assert DELEGATE_ACTION_REQUIRED_PAYLOAD == {
        "recover": ("reason", "evidence"),
    }


def test_gate_robustness_delegate_recover_with_evidence_ok(gate):
    """Robustness with full evidence at top of payload passes the gate."""
    gate.validate_intent("robustness", Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "recover",
            "reason": "gpu_memory_leaked",
            "force_gpu_cleanup": True,
            "evidence": {
                "consecutive_hits": 2,
                "per_gpu": [{"gpu_id": 0, "free_mb": 12.0}],
            },
        },
    ))


def test_gate_robustness_delegate_recover_with_nested_params_ok(gate):
    """Real-world shape: ``build_delegate`` nests ``reason`` / ``evidence``
    inside ``payload["params"]`` so the executor reads them via
    ``ctx.task.params``. The gate must accept that shape too."""
    gate.validate_intent("robustness", Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "recover",
            "params": {
                "reason": "gpu_memory_leaked",
                "force_gpu_cleanup": True,
                "evidence": {
                    "consecutive_hits": 2,
                    "per_gpu": [{"gpu_id": 0, "free_mb": 12.0}],
                },
            },
            "idempotency_key": "recover-gpu-leak-tick-1",
        },
    ))


def test_gate_orchestration_delegate_recover_rejected_by_source(gate):
    """Orchestration must NOT initiate ``recover`` even with full payload —
    the only valid path is robustness escalation via the action ladder."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "recover",
                "reason": "gpu_memory_leaked",
                "evidence": {"per_gpu": [{"gpu_id": 0, "free_mb": 0.0}]},
            },
        ))
    assert exc.value.rule == "delegate_action_source"
    assert "robustness" in str(exc.value)


def test_gate_robustness_delegate_recover_missing_evidence_rejected(gate):
    """Even from robustness, ``recover`` without evidence is denied so the
    audit trail always captures the symptom that justified the kill."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("robustness", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "recover",
                "reason": "gpu_memory_leaked",
                # no `evidence`
            },
        ))
    assert exc.value.rule == "delegate_action_evidence"
    assert "evidence" in str(exc.value)


def test_gate_robustness_delegate_recover_missing_reason_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("robustness", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "recover",
                # no `reason`
                "evidence": {"per_gpu": [{"gpu_id": 0, "free_mb": 0.0}]},
            },
        ))
    assert exc.value.rule == "delegate_action_evidence"
    assert "reason" in str(exc.value)


def test_gate_robustness_delegate_recover_empty_evidence_rejected(gate):
    """Empty dict / empty string count as missing — the gate is asserting
    *information presence*, not just key existence."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("robustness", Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "recover",
                "reason": "   ",  # whitespace-only string
                "evidence": {},   # empty dict
            },
        ))
    assert exc.value.rule == "delegate_action_evidence"


def test_gate_orchestration_request_to_kernel_ok(gate):
    gate.validate_intent("orchestration", Intent(
        type=IntentType.REQUEST,
        payload={"target_agent": "kernel", "kind": "trace_analyze"},
    ))


def test_gate_orchestration_request_to_critic_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "critic", "kind": "review"},
        ))
    assert exc.value.rule == "request_target"


def test_gate_kernel_response_ok(gate):
    gate.validate_intent("kernel", Intent(
        type=IntentType.RESPONSE,
        payload={"in_reply_to": "msg-abc", "kind": "trace_analyze_done"},
    ))


def test_gate_kernel_request_rejected_by_role(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("kernel", Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "orchestration", "kind": "x"},
        ))
    assert exc.value.rule == "role"  # kernel.allowed_intents lacks REQUEST


def test_gate_kernel_propose_rejected_by_role(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("kernel", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "x", "predicted_gain_pct": 0},
        ))
    assert exc.value.rule == "role"


def test_gate_critic_review_verdict_ok(gate):
    gate.validate_intent("critic", Intent(
        type=IntentType.REVIEW_VERDICT,
        payload={
            "target_proposal_msg_id": "p1",
            "verdict": "approve",
            "reasoning": "matches kb-7",
        },
    ))


def test_gate_orchestration_review_verdict_rejected(gate):
    """Only Critic may emit review_verdict."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={"target_proposal_msg_id": "p1", "verdict": "approve"},
        ))
    assert exc.value.rule == "role"


def test_gate_critic_review_verdict_unknown_verdict_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("critic", Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={"target_proposal_msg_id": "p1", "verdict": "objection"},
        ))
    assert exc.value.rule == "payload"


def test_gate_critic_delegate_rejected_by_role(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("critic", Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": "baseline"},
        ))
    assert exc.value.rule == "role"


def test_gate_robustness_kill_task_ok(gate):
    gate.validate_intent("robustness", Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": "t1", "reason": "stalled", "scope": "task"},
    ))


def test_gate_orchestration_kill_task_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.KILL_TASK,
            payload={"task_id": "t1", "reason": "stalled"},
        ))
    assert exc.value.rule == "role"


def test_gate_robustness_kill_task_process_scope_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("robustness", Intent(
            type=IntentType.KILL_TASK,
            payload={"task_id": "t1", "reason": "stalled", "scope": "process"},
        ))
    assert exc.value.rule == "kill_scope"


def test_gate_robustness_force_dispatch_ok(gate):
    gate.validate_intent("robustness", Intent(
        type=IntentType.FORCE_DISPATCH,
        payload={"task_id": "t1", "reason": "high value"},
    ))


def test_gate_robustness_prune_branch_requires_family(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("robustness", Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"reason": "3 fails"},
        ))
    assert exc.value.rule == "payload"


def test_gate_orchestration_prune_branch_allowed_with_family(gate):
    """Roofline-v2 C3: Orchestration was granted PRUNE_BRANCH so it can
    forward the structured ``suggested_prunes`` advice from the ``roofline``
    action to the Coordinator. FORCE_DISPATCH / ESCALATE_STRATEGY_CHANGE
    remain robustness-only; see
    ``test_orchestration_prune_branch_permission.py`` for the boundary tests.
    """
    gate.validate_intent("orchestration", Intent(
        type=IntentType.PRUNE_BRANCH,
        payload={"family": "deep_kernel", "reason": "x"},
    ))


def test_gate_orchestration_update_state_non_core_ok(gate):
    gate.validate_intent("orchestration", Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"current_action": "baseline"}},
    ))


def test_gate_orchestration_update_state_core_field_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.UPDATE_STATE,
            payload={"changes": {"current_best": {"foo": 1}}},
        ))
    assert exc.value.rule == "state_field"


def test_core_state_fields_includes_model_arch_tags():
    """``model_architectures`` / ``model_type`` are fact-layer tags lifted
    from the model weights' config.json (launcher / CLI owned). Locking
    them keeps an LLM ``update_state`` from polluting the recipe-snapshot
    tags that ``_kb_amend_recipe`` stamps into the KB."""
    assert "model_architectures" in CORE_STATE_FIELDS
    assert "model_type" in CORE_STATE_FIELDS


def test_gate_update_state_model_arch_tags_rejected(gate):
    """A non-core-mutating role must not overwrite the config.json
    architecture tags via ``update_state`` (else the next
    ``_kb_amend_recipe`` writes the polluted value into the recipe row)."""
    for field_name in ("model_architectures", "model_type"):
        with pytest.raises(PolicyDenied) as exc:
            gate.validate_intent("orchestration", Intent(
                type=IntentType.UPDATE_STATE,
                payload={"changes": {field_name: ["X"]}},
            ))
        assert exc.value.rule == "state_field", field_name


# ===========================================================================
# allowed_tools_for_agent
# ===========================================================================
def test_allowed_tools_claude_returns_emit_intent(gate):
    assert gate.allowed_tools_for_agent("orchestration") == ["emit_intent"]
    assert gate.allowed_tools_for_agent("kernel") == ["emit_intent"]
    assert gate.allowed_tools_for_agent("robustness") == ["emit_intent"]


def test_allowed_tools_codex_returns_empty(gate):
    """Critic = Codex no-tools (KB Bash exception lives in SubAgentRunner)."""
    assert gate.allowed_tools_for_agent("critic") == []


def test_allowed_tools_unknown_agent_returns_empty(gate):
    assert gate.allowed_tools_for_agent("ghost") == []


# ===========================================================================
# system_prompts assets
# ===========================================================================
@pytest.mark.parametrize("name", ["orchestration", "kernel", "critic", "robustness"])
def test_system_prompt_files_exist_and_nonempty(name):
    p = asset_system_prompts_dir() / f"{name}.md"
    assert p.is_file(), f"missing system prompt: {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > 200, f"system prompt too short: {p}"
    assert name.capitalize() in text or name in text.lower()
