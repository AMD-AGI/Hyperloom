"""Tests for orchestrator/policy.py — DESIGN §10.5.7 / §10.5.8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pytest

from inference_optimizer.orchestrator.agent_role import (
    ROLE_CRITIC,
    ROLE_EXECUTOR,
    ROLE_KERNEL,
    ROLE_TRIAGE,
    default_role_registry,
)
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.feature_flags import build_feature_flags
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import (
    CORE_STATE_FIELDS,
    DEFAULT_QUICK_ACTION_ALLOWLIST,
    PolicyDenied,
    PolicyGate,
    QUICK_BASH_DENYLIST,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_gate(
    mode: ExecutionMode = ExecutionMode.GUIDED_KERNEL_OPT,
    *,
    action_registry=None,
) -> PolicyGate:
    return PolicyGate(
        flags=build_feature_flags(mode),
        mode=mode,
        role_registry=default_role_registry(),
        action_registry=action_registry,
    )


def intent(t: IntentType, **payload) -> Intent:
    return Intent(type=t, payload=payload)


# ---------------------------------------------------------------------------
# Role gate
# ---------------------------------------------------------------------------
def test_unknown_agent_denied():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("nobody", intent(IntentType.SEND_MESSAGE, topic="event"))
    assert exc.value.rule == "role"


def test_role_cannot_emit_forbidden_intent_type():
    gate = make_gate()
    # Critic cannot delegate by spec.
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic", intent(IntentType.DELEGATE, action_name="kernel_opt")
        )
    assert exc.value.rule == "role"
    assert "critic" in str(exc.value)


def test_executor_can_send_message():
    gate = make_gate()
    gate.validate_intent(
        "executor",
        intent(IntentType.SEND_MESSAGE, topic="event", body_md="hi"),
    )  # no exception


def test_critic_can_send_message_verdict():
    """v0.4 — critic emits KEEP/REVERT verdicts via send_message
    (OBJECTION/VOTE removed with parliament)."""
    gate = make_gate()
    gate.validate_intent(
        "critic",
        intent(IntentType.SEND_MESSAGE, topic="observation",
               body_md="verdict: keep\ntarget_decision_seq: 5\nreason: ok"),
    )


# ---------------------------------------------------------------------------
# Delegate gate
# ---------------------------------------------------------------------------
def test_delegate_blocked_when_feature_flag_disabled():
    """Post-Phase 8a: quick mode now ALLOWS delegate (the action_executor
    bridge is the canonical way to run actions). To verify the
    flag-driven denial path still works, construct a PolicyGate with
    ``enable_subagent_delegate=False`` explicitly. This guards against
    accidentally re-enabling delegate in a future minimal mode."""
    from dataclasses import replace
    from inference_optimizer.orchestrator.feature_flags import build_feature_flags
    base_flags = build_feature_flags(ExecutionMode.QUICK_PARAM_SWEEP)
    flags = replace(base_flags, enable_subagent_delegate=False)
    gate = PolicyGate(
        flags=flags,
        mode=ExecutionMode.QUICK_PARAM_SWEEP,
        role_registry=default_role_registry(),
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.DELEGATE, action_name="kernel_opt"),
        )
    assert exc.value.rule == "mode"
    assert "delegation disabled" in str(exc.value)


def test_delegate_allowed_in_quick_mode_post_phase_8a():
    """The default quick-mode gate now permits delegate; the action_executor
    bridge needs queued delegate tasks to run shell scripts."""
    # Bare gate (no action_registry) — falls back to QUICK_ACTION_ALLOWLIST.
    # ``baseline`` is on it (see DEFAULT_QUICK_ACTION_ALLOWLIST).
    gate = make_gate(ExecutionMode.QUICK_PARAM_SWEEP)
    gate.validate_intent(
        "executor",
        intent(IntentType.DELEGATE, action_name="baseline"),
    )


def test_delegate_critic_role_blocked_by_role():
    """Critic doesn't have DELEGATE in allowed_intents -> role rule fires."""
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic", intent(IntentType.DELEGATE, action_name="kernel_opt"),
        )
    assert exc.value.rule == "role"


def test_delegate_triage_role_blocked_by_role():
    """v0.4 — triage cannot delegate; role rule rejects."""
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "triage", intent(IntentType.DELEGATE, action_name="bench_runner"),
        )
    assert exc.value.rule == "role"


def test_delegate_missing_action_name():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("executor", intent(IntentType.DELEGATE))
    assert exc.value.rule == "payload"


def test_delegate_in_guided_with_no_registry_accepts():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    gate.validate_intent(
        "executor",
        intent(IntentType.DELEGATE, action_name="anything_goes_in_guided"),
    )  # no exception


def test_delegate_quick_action_allowlist_enforced_when_mode_allows():
    # Force a fake mode where delegate flag is on but mode is quick-like.
    # We do this by overriding the gate directly.
    gate = PolicyGate(
        flags=build_feature_flags(ExecutionMode.GUIDED_KERNEL_OPT),
        mode=ExecutionMode.QUICK_PARAM_SWEEP,  # mismatch on purpose
        role_registry=default_role_registry(),
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.DELEGATE, action_name="not_in_allowlist"),
        )
    assert exc.value.rule == "mode"


def test_delegate_quick_action_allowlist_passes_for_known_action():
    gate = PolicyGate(
        flags=build_feature_flags(ExecutionMode.GUIDED_KERNEL_OPT),
        mode=ExecutionMode.QUICK_PARAM_SWEEP,
        role_registry=default_role_registry(),
    )
    gate.validate_intent(
        "executor",
        intent(IntentType.DELEGATE, action_name="bench_runner"),
    )


# ---------------------------------------------------------------------------
# Action registry integration (Phase 4 will provide a real registry)
# ---------------------------------------------------------------------------
@dataclass
class _StubAction:
    name: str
    allowed_modes: tuple[ExecutionMode, ...]
    allowed_tools: tuple[str, ...] = ("emit_intent", "Read", "Bash")


class _StubRegistry:
    def __init__(self, actions: Iterable[_StubAction]):
        self._by_name = {a.name: a for a in actions}

    def get(self, name: str):
        return self._by_name.get(name)


def test_propose_action_rejected_when_registry_says_mode_unsupported():
    reg = _StubRegistry(
        [_StubAction("kernel_opt", (ExecutionMode.MARATHON_MULTI_AGENT,))]
    )
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT, action_registry=reg)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.PROPOSE_ACTION, action_name="kernel_opt", predicted_gain_pct=3.0),
        )
    assert exc.value.rule == "mode"


def test_propose_action_unknown_action_with_registry_passes_in_v06():
    # PolicyGate doesn't reject unknown actions for *propose*; only delegate.
    reg = _StubRegistry([])
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT, action_registry=reg)
    gate.validate_intent(
        "executor",
        intent(
            IntentType.PROPOSE_ACTION,
            action_name="brand_new_action",
            predicted_gain_pct=2.0,
        ),
    )


def test_delegate_unknown_action_with_registry_rejected():
    reg = _StubRegistry([])
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT, action_registry=reg)
    with pytest.raises(PolicyDenied):
        gate.validate_intent(
            "executor",
            intent(IntentType.DELEGATE, action_name="never_heard_of"),
        )


def test_allowed_tools_for_action_uses_registry():
    reg = _StubRegistry(
        [_StubAction("kernel_opt", (ExecutionMode.MARATHON_MULTI_AGENT,),
                     allowed_tools=("emit_intent", "Read", "Bash", "Edit"))]
    )
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT, action_registry=reg)
    tools = gate.allowed_tools_for_action(
        ExecutionMode.MARATHON_MULTI_AGENT, "kernel_opt"
    )
    assert "Edit" in tools


# ---------------------------------------------------------------------------
# update_state gate
# ---------------------------------------------------------------------------
def test_update_state_blocks_core_field_for_executor():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(
                IntentType.UPDATE_STATE,
                changes={"current_best": "v2", "current_action": "x"},
            ),
        )
    assert exc.value.rule == "state_field"
    assert "current_best" in str(exc.value)


def test_update_state_blocks_core_field_for_triage():
    """v0.4 — triage may emit update_state but not for CORE_STATE_FIELDS."""
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "triage",
            intent(IntentType.UPDATE_STATE, changes={"stop_reason": "x"}),
        )
    assert exc.value.rule == "state_field"


def test_update_state_passes_for_non_core_fields():
    gate = make_gate()
    gate.validate_intent(
        "executor",
        intent(
            IntentType.UPDATE_STATE,
            changes={"current_action": "kernel_opt", "current_tput": 1234.5},
        ),
    )


def test_update_state_rejects_empty_changes():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor", intent(IntentType.UPDATE_STATE, changes={})
        )
    assert exc.value.rule == "payload"


def test_update_state_codex_role_denied_by_role_rule():
    gate = make_gate()
    # Critic doesn't have UPDATE_STATE allowed at all.
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "critic",
            intent(IntentType.UPDATE_STATE, changes={"current_action": "x"}),
        )
    assert exc.value.rule == "role"


# ---------------------------------------------------------------------------
# send_message topic gate
# ---------------------------------------------------------------------------
def test_send_message_downgrades_unknown_topic_to_observation():
    """Post-v0.8: PolicyGate no longer denies unknown topics. Instead it
    mutates ``payload.topic`` to ``observation`` and stashes the original
    name under ``original_topic``. Rationale: see SKILL.md L6 (the v0.8
    marathon run wasted ~80% of LLM tokens on policy_denied chatter for
    invented topic names like ``rca_finding`` / ``kb_status``)."""
    gate = make_gate()
    payload = {"topic": "not-on-bus", "body_md": "x"}
    msg_intent = intent(IntentType.SEND_MESSAGE, **payload)
    gate.validate_intent("executor", msg_intent)
    # The intent's payload should be mutated in place.
    assert msg_intent.payload["topic"] == "observation"
    assert msg_intent.payload["original_topic"] == "not-on-bus"


def test_send_message_still_rejects_empty_topic():
    """Empty topic is a real bug (caller forgot to set it) — PolicyGate
    should still catch this."""
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.SEND_MESSAGE, topic="", body_md="x"),
        )
    assert exc.value.rule == "payload"


def test_send_message_accepts_known_topic():
    gate = make_gate()
    msg_intent = intent(IntentType.SEND_MESSAGE, topic="event", body_md="hello")
    gate.validate_intent("executor", msg_intent)
    # Allowlisted topic must NOT be downgraded.
    assert msg_intent.payload["topic"] == "event"
    assert "original_topic" not in msg_intent.payload


# ---------------------------------------------------------------------------
# Bash allowlist / denylist
# ---------------------------------------------------------------------------
def test_quick_bash_denylist_always_fires():
    gate = make_gate(ExecutionMode.MARATHON_MULTI_AGENT)
    with pytest.raises(PolicyDenied) as exc:
        gate.check_bash_denylist("rm -rf /")
    assert exc.value.rule == "bash"


def test_quick_bash_allowlist_in_quick_mode():
    gate = make_gate(ExecutionMode.QUICK_PARAM_SWEEP)
    # Should pass (rocm-smi is on the allowlist).
    gate.validate_quick_bash("rocm-smi --showuse")
    # Should fail (random unknown command).
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_quick_bash("./scripts/random_thing.sh")
    assert exc.value.rule == "bash"


def test_quick_bash_outside_quick_is_noop_for_allowlist():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    gate.validate_quick_bash("./scripts/random_thing.sh")  # noop


# ---------------------------------------------------------------------------
# allowed_tools_for_agent
# ---------------------------------------------------------------------------
def test_allowed_tools_for_claude_role_has_emit_intent():
    """v0.4 — all 4 roles are Claude-backed; each gets emit_intent."""
    gate = make_gate()
    for agent in ("executor", "critic", "triage", "kernel"):
        assert "emit_intent" in gate.allowed_tools_for_agent(agent), (
            f"{agent} should have emit_intent"
        )


def test_allowed_tools_for_unknown_agent_empty():
    gate = make_gate()
    assert gate.allowed_tools_for_agent("nobody") == []


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------
def test_core_state_fields_includes_current_best_and_stop_reason():
    assert "current_best" in CORE_STATE_FIELDS
    assert "stop_reason" in CORE_STATE_FIELDS


def test_default_quick_allowlist_non_empty():
    assert "bench_runner" in DEFAULT_QUICK_ACTION_ALLOWLIST


# ---------------------------------------------------------------------------
# Plan A — REQUEST / RESPONSE gating + kernel-owned action denial
# ---------------------------------------------------------------------------
from inference_optimizer.orchestrator.policy import (  # noqa: E402
    KERNEL_OWNED_ACTIONS,
    REQUEST_ROUTING,
)


def test_executor_request_to_kernel_passes():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel",
            "kind": "select_kernels",
            "params": {"trace_path": "/tmp/x.json.gz"},
        },
    )
    gate.validate_intent("executor", intent, state=None)


def test_critic_cannot_request():
    """Only executor may emit REQUEST today (REQUEST_ROUTING)."""
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={"target_agent": "kernel", "kind": "select_kernels"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("critic", intent, state=None)
    # Critic's role allow-list should reject REQUEST entirely.
    assert exc.value.rule == "role"


def test_executor_request_to_unknown_target_rejected():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={"target_agent": "watchdog", "kind": "rca"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("executor", intent, state=None)
    assert exc.value.rule == "request_target"


def test_executor_request_missing_target_rejected():
    gate = make_gate()
    intent = Intent(type=IntentType.REQUEST, payload={"kind": "select_kernels"})
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("executor", intent, state=None)
    assert exc.value.rule == "payload"


def test_executor_request_missing_kind_rejected():
    gate = make_gate()
    intent = Intent(type=IntentType.REQUEST, payload={"target_agent": "kernel"})
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("executor", intent, state=None)
    assert exc.value.rule == "payload"


def test_kernel_response_passes():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent = Intent(
        type=IntentType.RESPONSE,
        payload={
            "in_reply_to": "abc123",
            "kind": "select_kernels_done",
            "status": "succeeded",
            "result": {"candidates": ["a.py"]},
        },
    )
    gate.validate_intent("kernel", intent, state=None)


def test_executor_cannot_emit_response():
    """RESPONSE is reserved for kernel agent; role gate rejects."""
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent = Intent(
        type=IntentType.RESPONSE,
        payload={"in_reply_to": "x", "kind": "ack"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("executor", intent, state=None)
    assert exc.value.rule == "role"


def test_kernel_response_missing_in_reply_to_rejected():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent = Intent(
        type=IntentType.RESPONSE,
        payload={"kind": "select_kernels_done"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("kernel", intent, state=None)
    assert exc.value.rule == "payload"


def test_executor_delegate_kernel_opt_denied():
    """Plan A — kernel_opt is owned by kernel agent; executor must REQUEST."""
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "kernel_opt", "params": {}},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("executor", intent, state=None)
    assert exc.value.rule == "kernel_owned_by_kernel_agent"


def test_executor_delegate_integrate_denied():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "integrate", "params": {}},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("executor", intent, state=None)
    assert exc.value.rule == "kernel_owned_by_kernel_agent"


def test_executor_delegate_baseline_still_works():
    """Sanity: only kernel-owned actions are blocked, not all actions."""
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "baseline", "params": {}},
    )
    # No exception expected.
    gate.validate_intent("executor", intent, state=None)


def test_kernel_owned_actions_constants():
    assert KERNEL_OWNED_ACTIONS == frozenset({"kernel_opt", "integrate"})
    assert "kernel" in REQUEST_ROUTING["executor"]


# ---------------------------------------------------------------------------
# v0.4 MVP — KILL_TASK gate (triage-only, scope-restricted)
# ---------------------------------------------------------------------------
from inference_optimizer.orchestrator.policy import (  # noqa: E402
    KILL_TASK_ALLOWED_SCOPES,
    KILL_TASK_SOURCE_ALLOWLIST,
)


def test_triage_can_kill_task():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent_obj = Intent(
        type=IntentType.KILL_TASK,
        payload={
            "task_id": "abc12345",
            "reason": "stuck for 4x lease_ttl",
            "scope": "task",
        },
    )
    gate.validate_intent("triage", intent_obj, state=None)


def test_triage_can_kill_task_default_scope():
    """scope defaults to 'task' when omitted."""
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent_obj = Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": "abc12345", "reason": "stuck"},
    )
    gate.validate_intent("triage", intent_obj, state=None)


def test_executor_cannot_kill_task_role_gate():
    """Executor's allowed_intents doesn't include KILL_TASK -> role rule."""
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent_obj = Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": "abc12345", "reason": "stuck"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("executor", intent_obj, state=None)
    assert exc.value.rule == "role"


def test_critic_cannot_kill_task_role_gate():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent_obj = Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": "abc12345", "reason": "stuck"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("critic", intent_obj, state=None)
    assert exc.value.rule == "role"


def test_kernel_cannot_kill_task_role_gate():
    gate = make_gate(ExecutionMode.GUIDED_KERNEL_OPT)
    intent_obj = Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": "abc12345", "reason": "stuck"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("kernel", intent_obj, state=None)
    assert exc.value.rule == "role"


def test_kill_task_missing_task_id_rejected():
    gate = make_gate()
    intent_obj = Intent(
        type=IntentType.KILL_TASK,
        payload={"reason": "stuck"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("triage", intent_obj, state=None)
    assert exc.value.rule == "payload"


def test_kill_task_missing_reason_rejected():
    gate = make_gate()
    intent_obj = Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": "abc12345"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("triage", intent_obj, state=None)
    assert exc.value.rule == "payload"


def test_kill_task_scope_process_denied():
    """v0.4 MVP — only scope='task' is allowed (IR-5 owns server lifecycle)."""
    gate = make_gate()
    intent_obj = Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": "x", "reason": "stuck", "scope": "process"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("triage", intent_obj, state=None)
    assert exc.value.rule == "kill_scope"


def test_kill_task_scope_server_denied():
    gate = make_gate()
    intent_obj = Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": "x", "reason": "stuck", "scope": "server"},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("triage", intent_obj, state=None)
    assert exc.value.rule == "kill_scope"


def test_kill_task_constants():
    assert KILL_TASK_SOURCE_ALLOWLIST == frozenset({"triage"})
    assert KILL_TASK_ALLOWED_SCOPES == frozenset({"task"})


# ---------------------------------------------------------------------------
# PolicyDenied hint surface (Phase D — agent self-correction signal)
# ---------------------------------------------------------------------------
def test_update_state_empty_changes_carries_hint():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor", intent(IntentType.UPDATE_STATE, changes={})
        )
    assert exc.value.rule == "payload"
    assert exc.value.hint
    assert "current_action" in exc.value.hint


def test_update_state_core_field_violation_carries_hint():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.UPDATE_STATE,
                   changes={"cumulative_gain": 5.0}),
        )
    assert exc.value.rule == "state_field"
    assert exc.value.hint and "cumulative_gain" in exc.value.hint


def test_send_message_empty_topic_carries_hint():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.SEND_MESSAGE, topic="", body_md="x"),
        )
    assert exc.value.hint and "heartbeat" in exc.value.hint


def test_triage_can_emit_force_dispatch():
    gate = make_gate()
    gate.validate_intent(
        "triage",
        intent(IntentType.FORCE_DISPATCH,
               task_id="abcd1234", reason="bump validation bench"),
    )


def test_executor_cannot_emit_force_dispatch_role_gate():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.FORCE_DISPATCH,
                   task_id="abcd1234", reason="bump"),
        )
    assert exc.value.rule == "role"


def test_triage_can_emit_prune_branch():
    gate = make_gate()
    gate.validate_intent(
        "triage",
        intent(IntentType.PRUNE_BRANCH,
               family="long", reason="3 consecutive failures"),
    )


def test_prune_branch_requires_family():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "triage",
            intent(IntentType.PRUNE_BRANCH, family="", reason="x"),
        )
    assert exc.value.rule == "payload"
    assert exc.value.hint and "family" in exc.value.hint


def test_triage_can_emit_escalate_strategy_change():
    gate = make_gate()
    gate.validate_intent(
        "triage",
        intent(IntentType.ESCALATE_STRATEGY_CHANGE,
               reason="GPU 0% for 12min",
               next_action_hint="switch to triton prefill"),
    )


def test_escalate_requires_next_action_hint():
    gate = make_gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "triage",
            intent(IntentType.ESCALATE_STRATEGY_CHANGE,
                   reason="stuck", next_action_hint=""),
        )
    assert exc.value.rule == "payload"


def test_unknown_action_hint_lists_candidates():
    """When an ActionRegistry is wired, an unknown action name should
    surface up to 8 valid candidates so the agent can recover."""
    from inference_optimizer.paths import asset_actions_dir
    from inference_optimizer.orchestrator.action_registry import ActionRegistry

    registry = ActionRegistry(asset_actions_dir()).load()
    gate = make_gate(action_registry=registry)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "executor",
            intent(IntentType.DELEGATE, action_name="totally_not_an_action"),
        )
    assert exc.value.rule == "mode"
    assert exc.value.hint and "valid examples" in exc.value.hint
