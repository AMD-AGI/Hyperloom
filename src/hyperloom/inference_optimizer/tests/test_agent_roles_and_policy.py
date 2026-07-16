# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent role + PolicyGate tests."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.roles.agent_role import (
    BackendType,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    default_role_registry,
    roles_for_run,
)
from hyperloom.inference_optimizer.protocol.intent import (
    Intent,
    IntentType,
)
from hyperloom.orchestrator.policy.gate import (
    CORE_STATE_FIELDS,
    DELEGATE_ACTION_REQUIRED_PAYLOAD,
    DELEGATE_ACTION_SOURCE_ALLOWLIST,
    KERNEL_AGENT_OWNED_ACTIONS,
    KILL_TASK_SOURCE_ALLOWLIST,
    PolicyDenied,
    PolicyGate,
    REQUEST_ROUTING,
    REVIEW_VERDICTS,
    REVIEW_VERDICT_SOURCE_ALLOWLIST,
    ROBUSTNESS_ONLY_INTENTS,
    ROBUSTNESS_ONLY_SOURCE_ALLOWLIST,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir


# agent_role
def test_default_role_registry_has_4_v06_agents():
    reg = default_role_registry()
    assert set(reg.keys()) == {"orchestration", "kernel_agent", "critic", "robustness"}


def test_roles_for_run_deterministic_order():
    assert roles_for_run() == ("orchestration", "kernel_agent", "critic", "robustness")


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
    assert IntentType.PRUNE_BRANCH in role.allowed_intents
    assert IntentType.ESCALATE_STRATEGY_CHANGE in role.allowed_intents
    assert IntentType.REVIEW_VERDICT not in role.allowed_intents
    assert IntentType.KILL_TASK not in role.allowed_intents
    assert IntentType.RESPONSE not in role.allowed_intents


def test_kernel_responder_only():
    role = default_role_registry()["kernel_agent"]
    assert role.backend_type == BackendType.CLAUDE
    assert role.can_delegate_side_effects is False
    assert IntentType.RESPONSE in role.allowed_intents
    assert IntentType.PROPOSE_ACTION not in role.allowed_intents
    assert IntentType.DELEGATE not in role.allowed_intents
    assert IntentType.REQUEST not in role.allowed_intents


def test_critic_review_only_codex_no_tools():
    role = default_role_registry()["critic"]
    assert role.backend_type == BackendType.CODEX
    assert role.model == DEFAULT_CODEX_MODEL
    assert role.no_tools is True
    assert IntentType.REVIEW_VERDICT in role.allowed_intents
    assert IntentType.DELEGATE not in role.allowed_intents
    assert IntentType.REQUEST not in role.allowed_intents
    assert IntentType.PROPOSE_ACTION not in role.allowed_intents
    assert IntentType.KILL_TASK not in role.allowed_intents


def test_robustness_scheduling_police():
    role = default_role_registry()["robustness"]
    assert role.backend_type == BackendType.CLAUDE
    assert IntentType.KILL_TASK in role.allowed_intents
    assert IntentType.PRUNE_BRANCH in role.allowed_intents
    assert IntentType.ESCALATE_STRATEGY_CHANGE in role.allowed_intents
    assert IntentType.PROPOSE_ACTION not in role.allowed_intents
    assert IntentType.REQUEST not in role.allowed_intents
    assert IntentType.REVIEW_VERDICT not in role.allowed_intents


# PolicyGate constants
def test_kernel_owned_actions_include_gemm_tuning():
    assert KERNEL_AGENT_OWNED_ACTIONS == frozenset(
        {
            "kernel_opt",
            "integrate",
            "deep_kernel_analysis",
            "operator_tuning",
            "vendor_kernel_config",
            "gemm_tuning",
        }
    )


def test_request_routing_v06_only_orchestration_to_kernel():
    assert set(REQUEST_ROUTING.keys()) == {"orchestration"}
    assert REQUEST_ROUTING["orchestration"] == frozenset({"kernel_agent"})


def test_review_verdict_critic_only():
    assert REVIEW_VERDICT_SOURCE_ALLOWLIST == frozenset({"critic"})
    assert "approve" in REVIEW_VERDICTS
    assert "needs_review" in REVIEW_VERDICTS
    assert "objection" not in REVIEW_VERDICTS


def test_kill_and_robustness_only_renamed():
    assert KILL_TASK_SOURCE_ALLOWLIST == frozenset({"robustness"})
    assert ROBUSTNESS_ONLY_SOURCE_ALLOWLIST == frozenset({"robustness"})
    assert ROBUSTNESS_ONLY_INTENTS == frozenset(
        {
            IntentType.PRUNE_BRANCH,
            IntentType.ESCALATE_STRATEGY_CHANGE,
        }
    )


def test_core_state_fields_includes_current_best():
    assert "current_best" in CORE_STATE_FIELDS
    assert "stop_reason" in CORE_STATE_FIELDS


# PolicyGate validation
@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry())


def test_gate_unknown_agent_rejected(gate):
    with pytest.raises(PolicyDenied, match="unknown agent"):
        gate.validate_intent("ghost", Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat"}))


def test_gate_orchestration_propose_action_ok(gate):
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
        ),
    )


def test_gate_orchestration_delegate_kernel_owned_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.DELEGATE,
                payload={"action_name": "kernel_opt"},
            ),
        )
    assert exc.value.rule == "kernel_owned_by_kernel_agent"


def test_gate_orchestration_propose_kernel_owned_rejected():
    """Kernel-owned actions are REQUEST-only on both channels: propose_action is denied like delegate."""
    state = SharedState(phase="KERNEL", precision="bf16", framework="sglang")
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    for action in ("kernel_opt", "gemm_tuning", "integrate", "deep_kernel_analysis"):
        with pytest.raises(PolicyDenied) as exc:
            gate.validate_intent(
                "orchestration",
                Intent(
                    type=IntentType.PROPOSE_ACTION,
                    payload={"action_name": action, "predicted_gain_pct": 10.0},
                ),
            )
        assert exc.value.rule == "kernel_owned_by_kernel_agent", action


def test_gate_run_gemm_tuning_request_rejected_for_non_fp8_geak(monkeypatch):
    """GEAK GEMM tuning remains FP8-only on the REQUEST channel."""
    monkeypatch.setenv("GEMM_TUNING_BACKEND", "geak")
    state = SharedState(phase="KERNEL", precision="bf16", framework="sglang")
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "kernel_agent", "kind": "run_gemm_tuning", "params": {}},
            ),
        )
    assert exc.value.rule == "fp8_only_action"


def test_gate_run_gemm_tuning_request_allowed_for_fp8(monkeypatch):
    monkeypatch.setenv("GEMM_TUNING_BACKEND", "geak")
    state = SharedState(phase="KERNEL", precision="fp8", framework="sglang")
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel_agent", "kind": "run_gemm_tuning", "params": {}},
        ),
    )


def test_gate_run_gemm_tuning_request_allowed_for_bf16_forge(monkeypatch):
    monkeypatch.setenv("GEMM_TUNING_BACKEND", "forge")
    state = SharedState(phase="KERNEL", precision="bf16", framework="sglang")
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel_agent", "kind": "run_gemm_tuning", "params": {}},
        ),
    )


def test_gate_orchestration_delegate_normal_action_ok(gate):
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": "baseline"},
        ),
    )


# Per-action delegate source allowlist: recover is robustness-only.
def test_delegate_action_source_allowlist_constant_shape():
    """``recover`` is the only entry today."""
    assert DELEGATE_ACTION_SOURCE_ALLOWLIST == {
        "recover": frozenset({"robustness"}),
    }


def test_delegate_action_required_payload_constant_shape():
    assert DELEGATE_ACTION_REQUIRED_PAYLOAD == {
        "recover": ("reason", "evidence"),
    }


def test_gate_robustness_delegate_recover_with_evidence_ok(gate):
    """Robustness with full evidence at top of payload passes the gate."""
    gate.validate_intent(
        "robustness",
        Intent(
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
        ),
    )


def test_gate_robustness_delegate_recover_with_nested_params_ok(gate):
    """The gate must accept the nested ``payload["params"]`` shape from ``build_delegate``."""
    gate.validate_intent(
        "robustness",
        Intent(
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
        ),
    )


def test_gate_orchestration_delegate_recover_rejected_by_source(gate):
    """Orchestration must NOT initiate ``recover`` even with full payload."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "recover",
                    "reason": "gpu_memory_leaked",
                    "evidence": {"per_gpu": [{"gpu_id": 0, "free_mb": 0.0}]},
                },
            ),
        )
    assert exc.value.rule == "delegate_action_source"
    assert "robustness" in str(exc.value)


def test_gate_orchestration_propose_recover_rejected_by_source(gate):
    """Orchestration must NOT reach ``recover`` through propose_action either; the source allowlist gates both intent kinds."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "recover"},
            ),
        )
    assert exc.value.rule == "propose_action_source"
    assert "robustness" in str(exc.value)


def test_gate_robustness_delegate_recover_in_phase_ok():
    """The robustness ``gpu_memory_leaked`` ladder delegates ``recover`` with a live phase set."""
    state = SharedState(phase="EXPLORE", framework="sglang")
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    gate.validate_intent(
        "robustness",
        Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "recover",
                "params": {
                    "reason": "gpu_memory_leaked",
                    "force_gpu_cleanup": True,
                    "evidence": {"per_gpu": [{"gpu_id": 0, "free_mb": 0.0}]},
                },
            },
        ),
    )


def test_gate_orchestration_propose_recover_in_phase_rejected():
    """With a live phase set, Orchestration's propose(recover) is denied (the source gate fires first)."""
    state = SharedState(phase="EXPLORE", framework="sglang")
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "recover"},
            ),
        )
    assert exc.value.rule == "propose_action_source"


def test_gate_robustness_delegate_recover_missing_evidence_rejected(gate):
    """Even from robustness, ``recover`` without evidence is denied."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "robustness",
            Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "recover",
                    "reason": "gpu_memory_leaked",
                },
            ),
        )
    assert exc.value.rule == "delegate_action_evidence"
    assert "evidence" in str(exc.value)


def test_gate_robustness_delegate_recover_missing_reason_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "robustness",
            Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "recover",
                    "evidence": {"per_gpu": [{"gpu_id": 0, "free_mb": 0.0}]},
                },
            ),
        )
    assert exc.value.rule == "delegate_action_evidence"
    assert "reason" in str(exc.value)


def test_gate_robustness_delegate_recover_empty_evidence_rejected(gate):
    """Empty dict / empty string count as missing (gate asserts information presence)."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "robustness",
            Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "recover",
                    "reason": "   ",
                    "evidence": {},
                },
            ),
        )
    assert exc.value.rule == "delegate_action_evidence"


def test_gate_orchestration_request_to_kernel_ok(gate):
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel_agent", "kind": "trace_analyze"},
        ),
    )


def test_gate_orchestration_request_to_critic_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "critic", "kind": "review"},
            ),
        )
    assert exc.value.rule == "request_target"


def test_gate_kernel_response_ok(gate):
    gate.validate_intent(
        "kernel_agent",
        Intent(
            type=IntentType.RESPONSE,
            payload={"in_reply_to": "msg-abc", "kind": "trace_analyze_done"},
        ),
    )


def test_gate_kernel_request_rejected_by_role(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "kernel_agent",
            Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "orchestration", "kind": "x"},
            ),
        )
    assert exc.value.rule == "role"


def test_gate_kernel_propose_rejected_by_role(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "kernel_agent",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "x", "predicted_gain_pct": 0},
            ),
        )
    assert exc.value.rule == "role"


def test_gate_critic_review_verdict_ok(gate):
    gate.validate_intent(
        "critic",
        Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={
                "target_proposal_msg_id": "p1",
                "verdict": "approve",
                "reasoning": "matches kb-7",
            },
        ),
    )


def test_gate_orchestration_review_verdict_rejected(gate):
    """Only Critic may emit review_verdict."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={"target_proposal_msg_id": "p1", "verdict": "approve"},
            ),
        )
    assert exc.value.rule == "role"


def test_gate_critic_review_verdict_unknown_verdict_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic",
            Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={"target_proposal_msg_id": "p1", "verdict": "objection"},
            ),
        )
    assert exc.value.rule == "payload"


def test_gate_critic_delegate_rejected_by_role(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic",
            Intent(
                type=IntentType.DELEGATE,
                payload={"action_name": "baseline"},
            ),
        )
    assert exc.value.rule == "role"


def test_gate_robustness_kill_task_ok(gate):
    gate.validate_intent(
        "robustness",
        Intent(
            type=IntentType.KILL_TASK,
            payload={"task_id": "t1", "reason": "stalled", "scope": "task"},
        ),
    )


def test_gate_orchestration_kill_task_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.KILL_TASK,
                payload={"task_id": "t1", "reason": "stalled"},
            ),
        )
    assert exc.value.rule == "role"


def test_gate_robustness_kill_task_process_scope_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "robustness",
            Intent(
                type=IntentType.KILL_TASK,
                payload={"task_id": "t1", "reason": "stalled", "scope": "process"},
            ),
        )
    assert exc.value.rule == "kill_scope"


def test_gate_robustness_prune_branch_requires_family(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "robustness",
            Intent(
                type=IntentType.PRUNE_BRANCH,
                payload={"reason": "3 fails"},
            ),
        )
    assert exc.value.rule == "payload"


def test_gate_orchestration_prune_branch_allowed_with_family(gate):
    """Orchestration has PRUNE_BRANCH so it can forward ``suggested_prunes`` advice to the Coordinator."""
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"family": "deep_kernel", "reason": "x"},
        ),
    )


def test_gate_orchestration_update_state_non_core_ok(gate):
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.UPDATE_STATE,
            payload={"changes": {"current_action": "baseline"}},
        ),
    )


def test_gate_orchestration_update_state_core_field_rejected(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.UPDATE_STATE,
                payload={"changes": {"current_best": {"foo": 1}}},
            ),
        )
    assert exc.value.rule == "state_field"


def test_core_state_fields_includes_model_arch_tags():
    """``model_architectures`` / ``model_type`` are config.json fact-layer tags that stay locked."""
    assert "model_architectures" in CORE_STATE_FIELDS
    assert "model_type" in CORE_STATE_FIELDS


def test_gate_update_state_model_arch_tags_rejected(gate):
    """A non-core-mutating role must not overwrite the config.json architecture tags via ``update_state``."""
    for field_name in ("model_architectures", "model_type"):
        with pytest.raises(PolicyDenied) as exc:
            gate.validate_intent(
                "orchestration",
                Intent(
                    type=IntentType.UPDATE_STATE,
                    payload={"changes": {field_name: ["X"]}},
                ),
            )
        assert exc.value.rule == "state_field", field_name


def test_core_state_fields_includes_degraded_markers():
    """``degraded_mode`` / ``model_warnings`` are preflight-authored facts that stay locked."""
    assert "degraded_mode" in CORE_STATE_FIELDS
    assert "model_warnings" in CORE_STATE_FIELDS


def test_gate_update_state_degraded_markers_rejected(gate):
    """A non-core-mutating role must not forge/clear the degraded-run markers."""
    for field_name, value in (("degraded_mode", False), ("model_warnings", [])):
        with pytest.raises(PolicyDenied) as exc:
            gate.validate_intent(
                "orchestration",
                Intent(
                    type=IntentType.UPDATE_STATE,
                    payload={"changes": {field_name: value}},
                ),
            )
        assert exc.value.rule == "state_field", field_name


# allowed_tools_for_agent
def test_allowed_tools_claude_returns_emit_intent(gate):
    assert gate.allowed_tools_for_agent("kernel_agent") == ["emit_intent"]
    assert gate.allowed_tools_for_agent("robustness") == ["emit_intent"]
    from hyperloom.orchestrator.roles.mcp_context_tools import (
        CONTEXT_TOOL_NAMES,
    )

    orch = gate.allowed_tools_for_agent("orchestration")
    assert orch[0] == "emit_intent"
    assert "Read" in orch
    for name in CONTEXT_TOOL_NAMES:
        assert name in orch
    assert "get_recent_outcomes" in orch
    assert "run_action_now" in orch
    assert "WebSearch" in orch
    assert "WebFetch" in orch


def test_allowed_tools_codex_returns_empty(gate):
    """Critic = Codex no-tools (KB Bash exception lives in SubAgentRunner)."""
    assert gate.allowed_tools_for_agent("critic") == []


def test_allowed_tools_unknown_agent_returns_empty(gate):
    assert gate.allowed_tools_for_agent("ghost") == []


# system_prompts assets
@pytest.mark.parametrize("name", ["orchestration", "kernel_agent", "critic", "robustness"])
def test_system_prompt_files_exist_and_nonempty(name):
    p = asset_system_prompts_dir() / f"{name}.md"
    assert p.is_file(), f"missing system prompt: {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > 200, f"system prompt too short: {p}"
    assert name.capitalize() in text or name in text.lower()


def test_core_state_fields_includes_closing_phase_and_baseline_config():
    # closing_phase and baseline_config_path are Coordinator-only fact fields
    # locked against non-coordinator update_state.
    assert "closing_phase" in CORE_STATE_FIELDS
    assert "baseline_config_path" in CORE_STATE_FIELDS


def test_gate_update_state_closing_phase_and_baseline_config_rejected(gate):
    # A non-core-mutating role must not force wind-down or inject a launch
    # config path via update_state.
    for field_name, value in (("closing_phase", True), ("baseline_config_path", "/etc/evil.yaml")):
        with pytest.raises(PolicyDenied) as exc:
            gate.validate_intent(
                "orchestration",
                Intent(
                    type=IntentType.UPDATE_STATE,
                    payload={"changes": {field_name: value}},
                ),
            )
        assert exc.value.rule == "state_field", field_name


def test_core_state_fields_synced_with_robustness_envelope():
    # gate.CORE_STATE_FIELDS and the robustness
    # envelope copy must stay byte-identical. This direct assertion never skips
    # (unlike tests/test_role_contract.py, which needs the optimizer on sys.path).
    from hyperloom.agents.robustness.role.envelope import (
        CORE_STATE_FIELDS as ENVELOPE_CORE_STATE_FIELDS,
    )

    assert CORE_STATE_FIELDS == ENVELOPE_CORE_STATE_FIELDS
