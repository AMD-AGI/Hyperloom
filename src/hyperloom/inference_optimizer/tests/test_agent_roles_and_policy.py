# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent role + PolicyGate tests."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.mcp_tool_names import qualified_context_tool_name
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.inference_optimizer.protocol.intent import (
    EXTEND_LEASE_MAX_SEC,
    Intent,
    IntentType,
    IntentValidationError,
    validate_envelope,
)
from hyperloom.inference_optimizer.protocol.action_surfaces import COORDINATOR_OWNED_KERNEL_REQUEST_KINDS
from hyperloom.orchestrator.policy.denial import PolicyDenied
from hyperloom.orchestrator.policy.gate import PolicyGate
from hyperloom.orchestrator.state.shared_state import SharedState


# Intent vocabulary
@pytest.mark.parametrize(
    ("intent_type", "payload"),
    [
        pytest.param("kill_task", {"task_id": "t1", "reason": "stalled"}, id="kill_task"),
        # Requests are answered inline on the ``response`` topic.
        pytest.param("response", {"in_reply_to": "m1", "kind": "profile_done"}, id="response"),
    ],
)
def test_retired_intent_type_is_rejected(intent_type, payload):
    """A retired intent type left the vocabulary; an envelope carrying it must be rejected."""
    assert intent_type not in {member.value for member in IntentType}
    envelope = {"intents": [{"intent_type": intent_type, "payload": payload}]}
    with pytest.raises(IntentValidationError, match="not in allowed set"):
        validate_envelope(envelope)


# PolicyGate validation
@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry())


def test_legacy_watchdog_options_do_not_change_resumed_measurements():
    legacy_options = {
        "robustness_options": {"auto_probe_inference_server": False},
        "explore_overtime_kill_ratio": 2.0,
        "explore_variant_timeout_sec_override": 123,
        "explore_variant_timeout_safety_margin": 0.5,
        "conc_sweep_variant_timeout_sec": 456,
    }
    persisted = {
        "session_id": "existing",
        "benchmark_mode": "agentx",
        "baseline_tput": 1200.0,
        "baseline_perf": {
            "output_throughput": 1200.0,
            "total_throughput": 2400.0,
            "e2e_norm_intvty_p90": 10.0,
        },
        "current_best": {
            "tput": 1320.0,
            "output_throughput": 1320.0,
            "total_throughput": 2640.0,
            "e2e_norm_intvty_p90": 11.0,
        },
        "current_best_measurement": {
            "output_throughput": 1320.0,
            "total_throughput": 2640.0,
            "e2e_norm_intvty_p90": 11.0,
        },
        "operator_server_args": "--max-num-seqs 512",
        "operator_extra_env": {"SGLANG_USE_AITER": "0"},
    }
    restored = SharedState.from_dict({**persisted, **legacy_options}).to_dict()
    assert all(name not in restored for name in legacy_options)
    assert {name: restored[name] for name in persisted} == persisted


def test_gate_unknown_agent_rejected(gate):
    with pytest.raises(PolicyDenied, match="unknown agent"):
        gate.validate_intent("ghost", Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat"}))


@pytest.mark.parametrize(
    ("agent", "intent_type", "payload"),
    [
        pytest.param(
            "orchestration",
            IntentType.PROPOSE_ACTION,
            {"action_name": "baseline", "predicted_gain_pct": 0.0},
            id="orchestration-propose_action",
        ),
        pytest.param("orchestration", IntentType.DELEGATE, {"action_name": "baseline"}, id="orchestration-delegate"),
        pytest.param(
            "orchestration",
            IntentType.REQUEST,
            {"target_agent": "kernel_agent", "kind": "trace_analyze"},
            id="orchestration-request-to-kernel_agent",
        ),
        pytest.param(
            "critic",
            IntentType.REVIEW_VERDICT,
            {"target_proposal_msg_id": "p1", "verdict": "approve", "reasoning": "matches kb-7"},
            id="critic-review_verdict",
        ),
        # Orchestration forwards ``suggested_prunes`` advice to the Coordinator.
        pytest.param(
            "orchestration",
            IntentType.PRUNE_BRANCH,
            {"family": "deep_kernel", "reason": "x"},
            id="orchestration-prune_branch-with-family",
        ),
    ],
)
def test_gate_allows_an_intent_the_role_owns(gate, agent, intent_type, payload):
    gate.validate_intent(agent, Intent(type=intent_type, payload=payload))


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
@pytest.mark.parametrize("kernel_optimizer", ["geak", "forge"])
def test_gate_refuses_a_model_requested_gemm_tuning_run(precision, kernel_optimizer):
    """Refused by channel, not by applicability."""
    state = SharedState(
        phase="KERNEL_AGENT", precision=precision, framework="sglang", kernel_optimizer=kernel_optimizer
    )
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "kernel_agent", "kind": "run_gemm_tuning", "params": {}},
            ),
        )
    assert exc.value.rule == "request_kind"


def test_a_model_requested_kernel_optimization_has_no_handler_to_reach():
    """Source rewrite is the controller's, so the kind is unregistered."""
    from hyperloom.orchestrator.kernel.request_handlers import KERNEL_REQUEST_HANDLERS

    assert "run_optimization" not in KERNEL_REQUEST_HANDLERS
    assert "run_optimization" not in COORDINATOR_OWNED_KERNEL_REQUEST_KINDS


def test_gate_still_allows_the_model_to_drain_the_keep_queue(monkeypatch):
    """Closing the lanes must not close integrate; draining KEEPs stays its job."""
    state = SharedState(phase="KERNEL_AGENT", precision="bf16", framework="sglang")
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel_agent", "kind": "integrate", "params": {"kernel_id": "k1"}},
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


@pytest.mark.parametrize(
    "intent_type",
    [IntentType.DELEGATE, IntentType.REQUEST, IntentType.PROPOSE_ACTION],
    ids=lambda intent_type: intent_type.value,
)
def test_gate_critic_non_review_intent_rejected_by_role(gate, intent_type):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic",
            Intent(
                type=intent_type,
                payload={"action_name": "baseline"},
            ),
        )
    assert exc.value.rule == "role"


def test_gate_role_denial_names_the_intents_the_role_may_emit(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={"target_proposal_msg_id": "p1", "verdict": "approve"},
            ),
        )
    assert exc.value.rule == "role"
    assert exc.value.hint
    for intent_type in default_role_registry()["orchestration"].allowed_intents:
        assert intent_type.value in exc.value.hint
    assert "review_verdict" not in exc.value.hint


def test_gate_extend_lease_bounds_hint_states_the_step_cap(gate):
    """The hint says how far one call reaches and how to go further."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.EXTEND_LEASE,
                payload={"task_id": "t1", "extra_sec": EXTEND_LEASE_MAX_SEC + 1},
            ),
        )
    assert exc.value.rule == "extend_lease_bounds"
    assert f"at most {EXTEND_LEASE_MAX_SEC}s per call" in exc.value.hint
    assert "while the task runs" in exc.value.hint
    # The tool the hint sends the agent back to is named the way the runtime registers it.
    assert qualified_context_tool_name("get_running_tasks") in exc.value.hint


def test_gate_orchestration_prune_branch_requires_family(gate):
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.PRUNE_BRANCH,
                payload={"reason": "3 fails"},
            ),
        )
    assert exc.value.rule == "payload"


# allowed_tools_for_agent
def test_allowed_tools_claude_returns_emit_intent(gate):
    assert gate.allowed_tools_for_agent("robustness") == []
    from hyperloom.orchestrator.roles.mcp_context_tools import (
        CONTEXT_TOOL_NAMES,
        CONTEXT_TOOL_QUALIFIED_NAMES,
    )
    from hyperloom.orchestrator.roles.mcp_emit_intent import EMIT_INTENT_TOOL_QUALIFIED

    orch = gate.allowed_tools_for_agent("orchestration")
    # The grant is both the SDK allow-list and what the prompt advertises, so
    # every MCP tool is named the way the runtime registers it.
    assert orch[0] == EMIT_INTENT_TOOL_QUALIFIED
    assert "Read" in orch
    for name in CONTEXT_TOOL_QUALIFIED_NAMES:
        assert name in orch
    for name in CONTEXT_TOOL_NAMES:
        assert name not in orch
    assert "mcp__inference_optimizer_context__get_recent_outcomes" in orch
    assert "run_action_now" not in orch
    assert "WebSearch" in orch
    assert "WebFetch" in orch


def test_allowed_tools_codex_returns_empty(gate):
    """Critic = Codex no-tools (KB Bash exception lives in SubAgentRunner)."""
    assert gate.allowed_tools_for_agent("critic") == []
