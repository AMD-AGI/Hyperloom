# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Intent envelope contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


class IntentType(str, Enum):
    """Intent vocabulary mirrored from upstream ``intent_parser.IntentType``."""

    SEND_MESSAGE = "send_message"
    DELEGATE = "delegate"
    PROPOSE_ACTION = "propose_action"
    UPDATE_STATE = "update_state"
    ALERT = "alert"
    REQUEST = "request"
    RESPONSE = "response"
    REVIEW_VERDICT = "review_verdict"
    # Robustness never emits this; kept in the mirror for the contract test.
    EXTEND_LEASE = "extend_lease"
    PRUNE_BRANCH = "prune_branch"
    ESCALATE_STRATEGY_CHANGE = "escalate_strategy_change"
    # Robustness never emits this; kept in the mirror for the contract test.
    SPECIALIST_DONE = "specialist_done"


# Intents PolicyGate restricts to ``source == "robustness"``; guarded locally to fail fast, still enforced server-side
# by the gate.
ROBUSTNESS_ONLY_INTENTS: frozenset[IntentType] = frozenset(
    {
        IntentType.PRUNE_BRANCH,
        IntentType.ESCALATE_STRATEGY_CHANGE,
    }
)


# Intents the robustness role may emit; other roles' intents are excluded to fail fast.
ROBUSTNESS_ALLOWED_INTENTS: frozenset[IntentType] = frozenset(
    {
        IntentType.SEND_MESSAGE,
        IntentType.ALERT,
        IntentType.UPDATE_STATE,
        IntentType.DELEGATE,
        IntentType.PRUNE_BRANCH,
        IntentType.ESCALATE_STRATEGY_CHANGE,
    }
)


# Severities accepted by ``alert`` and ``escalate_strategy_change``; ``high`` raises priority 0 broadcasts.
ALERT_SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high"})


# Mirrors upstream ``ROBUSTNESS_DELEGATE_ONLY_ACTIONS``; every other remediation rides an alert for Orchestration to
# act on.
ROBUSTNESS_DELEGATE_ACTIONS: frozenset[str] = frozenset({"recover"})


# Core SharedState fields the robustness role must not write via ``update_state``; kept in lock-step by
# ``tests/test_role_contract.py``.
CORE_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "current_best",
        "stop_reason",
        "stop_ts",
        "last_tick_exception",
        "cumulative_gain_validated",
        "cumulative_gain_validated_ts",
        "cumulative_gain_validated_stack_len",
        "pending_integrate",
        "resume_pending_revalidation",
        "baseline_tput",
        "baseline_accuracy",
        "session_id",
        "model_path",
        "model_name",
        "model_class",
        "compute_partition",
        "start_ts",
        "resumed_ts",
        "max_minutes",
        "deadline_unix",
        "closing_grace_sec",
        "optimization_stack",
        "gain_per_stack_entry",
        "schema_version",
        # Recipe KB integration.
        "recipe_kb_session_id",
        "warm_start_recipe",
        "warm_start_pitfalls",
        "warm_start_lessons",
        "warm_start_ts",
        "warm_start_context",
        "kb_stage_outbox",
        "kb_stage_dead_letter",
        "recipe_finalize_status",
        "recipe_finalize_attempts",
        "recipe_finalize_outcome",
        "stack_fingerprint_meta",
        "baseline_workload_extra",
        "last_profile_workload",
        "last_profile_workload_action",
        # warm-recipe replay.
        "warm_replay_attempted",
        "warm_replay_outcome",
        "warm_history_injected",
        # phase state machine.
        "phase",
        "phase_started_ts",
        "phase_started_unix",
        "phase_history",
        "phase_budget_pct",
        "explore_elapsed_accum_s",
        "phase_elapsed_totals",
        # KERNEL idle-streak bookkeeping; measured by the Coordinator from observed facts, never proposable.
        "kernel_idle_ticks",
        "kernel_progress_fingerprint",
        "kernel_idle_since_unix",
        # Cyclic phase-machine state; locked so an LLM update_state cannot forge macro-cycle / convergence / per-cycle
        # budget state.
        "macro_cycle",
        "cycle_minutes",
        "gain_at_cycle_start",
        "no_gain_cycle_streak",
        "pending_bottleneck_switch",
        "last_cycle_bottleneck",
        "saturated_directions",
        "bottleneck_shift",
        "cycle_strategy_log",
        # operator-facing lifecycle event log.
        "lifecycle",
        # specialist sub-agent ledger.
        "specialist_rounds",
        # per-kb_anchor coverage counters.
        "rounds_since_last_specialist",
        "rounds_since_last_keep",
        "last_specialist",
        "research_lane_capacity",
        "gpu_specialist_capacity",
        # phase-machine escalation plumbing.
        "pending_escalate_hint",
        "last_consumed_escalate_hint",
        "last_consumed_escalate_hint_ts",
        "last_discarded_escalate_hint",
        "last_discarded_escalate_hint_ts",
        "plateau_overrides",
        # CLOSE phase sequencer flag.
        "close_sequence_done",
        # Objective-met marker; the Coordinator is its only writer.
        "target_reached_at",
        # unified explore search ledger.
        "explore_search",
        # structured gaps ledger.
        "gaps",
        # Orchestration working-memory checkpoint (Coordinator-authored).
        "orchestration_memory",
        # Bounded rollback ring of prior good orchestration_memory records.
        "orchestration_memory_history",
        # Advisory model-architecture profile.
        "model_arch",
        # Architecture-identity tags from config.json.
        "model_architectures",
        "model_type",
        # Multimodal text-fallback degraded-run markers; locked so an LLM update_state can't forge/clear the degraded
        # verdict.
        "degraded_mode",
        "model_warnings",
        # Kernel-opt ledgers + Critic patch-verdict store; locked against LLM update_state.
        "specialist_patch_verdicts",
        "last_trace_analyze",
        "last_kernel_opt",
        "kernel_opt_task_attempts",
        "pending_kernel_integrations",
        # kept in lock-step with upstream policy.CORE_STATE_FIELDS (see tests/test_role_contract.py).
        "closing_phase",
        "baseline_config_path",
        "failures",
    }
)


# Robustness may only mutate these state fields directly.
ROBUSTNESS_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "crash_count",
        "current_action",
    }
)


@dataclass
class Intent:
    """One validated intent from the reactor."""

    type: IntentType
    payload: dict[str, Any] = field(default_factory=dict)

    def to_envelope_item(self) -> dict[str, Any]:
        """Return the dict shape used inside an ``intents`` envelope."""
        return {"intent_type": self.type.value, "payload": dict(self.payload)}


class PolicyViolation(ValueError):
    """Raised when an intent fails the local PolicyGate-equivalent checks."""

    def __init__(self, reason: str, *, rule: str, hint: str | None = None):
        """Initialise the violation with a reason, rule id, and optional hint."""
        super().__init__(reason)
        self.rule = rule
        self.hint = hint


# Intent builders


def build_send_message(
    topic: str,
    *,
    body_md: str | None = None,
    to: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> Intent:
    """Generic send_message builder."""
    payload: dict[str, Any] = {"topic": topic}
    if body_md is not None:
        payload["body_md"] = body_md
    if to:
        payload["to"] = to
    if extras:
        for k, v in extras.items():
            if k == "topic":
                continue
            payload[k] = v
    return Intent(type=IntentType.SEND_MESSAGE, payload=payload)


def build_alert(
    severity: str,
    summary: str,
    *,
    detail: Mapping[str, Any] | None = None,
) -> Intent:
    """Construct an ``alert`` intent."""
    if severity not in ALERT_SEVERITIES:
        raise ValueError(f"alert severity {severity!r} not in {sorted(ALERT_SEVERITIES)!r}")
    if not summary:
        raise ValueError("alert summary must be non-empty")
    payload: dict[str, Any] = {"severity": severity, "summary": summary}
    if detail is not None:
        payload["detail"] = dict(detail)
    return Intent(type=IntentType.ALERT, payload=payload)


def build_escalate(
    reason: str,
    next_action_hint: str,
    *,
    severity: str = "medium",
) -> Intent:
    """Construct an ``escalate_strategy_change`` intent."""
    if not reason:
        raise ValueError("escalate reason must be non-empty")
    if not next_action_hint:
        raise ValueError("escalate next_action_hint must be non-empty")
    if severity not in ALERT_SEVERITIES:
        raise ValueError(f"escalate severity {severity!r} not in {sorted(ALERT_SEVERITIES)!r}")
    return Intent(
        type=IntentType.ESCALATE_STRATEGY_CHANGE,
        payload={
            "reason": reason,
            "next_action_hint": next_action_hint,
            "severity": severity,
        },
    )


def build_prune_branch(family: str, reason: str) -> Intent:
    """Construct a ``prune_branch`` intent. Robustness-only."""
    if not family:
        raise ValueError("prune_branch family must be non-empty")
    if not reason:
        raise ValueError("prune_branch reason must be non-empty")
    return Intent(
        type=IntentType.PRUNE_BRANCH,
        payload={"family": family, "reason": reason},
    )


def build_delegate(
    action_name: str,
    *,
    params: Mapping[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> Intent:
    """Construct a ``delegate`` intent."""
    if action_name not in ROBUSTNESS_DELEGATE_ACTIONS:
        raise ValueError(
            f"delegate action_name {action_name!r} not allowed for robustness "
            f"(allowed: {sorted(ROBUSTNESS_DELEGATE_ACTIONS)!r})"
        )
    payload: dict[str, Any] = {"action_name": action_name}
    if params is not None:
        payload["params"] = dict(params)
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    return Intent(type=IntentType.DELEGATE, payload=payload)


def build_update_state(changes: Mapping[str, Any]) -> Intent:
    """Construct an ``update_state`` intent."""
    if not changes:
        raise ValueError("update_state changes must be a non-empty mapping")
    illegal = sorted(set(changes.keys()) - ROBUSTNESS_STATE_FIELDS)
    if illegal:
        raise ValueError(
            "update_state contains fields outside robustness allowlist: "
            f"{illegal!r}; allowed: {sorted(ROBUSTNESS_STATE_FIELDS)!r}"
        )
    return Intent(type=IntentType.UPDATE_STATE, payload={"changes": dict(changes)})


# Per-intent payload validators (mirror upstream ``PolicyGate.validate_intent``)


def _validate_alert_payload(payload: dict[str, Any]) -> None:
    """Validate an ``alert`` payload's severity and summary."""
    severity = str(payload.get("severity", "")).strip()
    if severity not in ALERT_SEVERITIES:
        raise PolicyViolation(
            f"alert.severity={severity!r} not in {sorted(ALERT_SEVERITIES)!r}",
            rule="payload",
        )
    summary = str(payload.get("summary", "")).strip()
    if not summary:
        raise PolicyViolation(
            "alert.summary must be a non-empty string",
            rule="payload",
        )


def _validate_escalate_payload(payload: dict[str, Any]) -> None:
    """Validate an ``escalate_strategy_change`` payload."""
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise PolicyViolation(
            "escalate_strategy_change.reason must be non-empty",
            rule="payload",
        )
    hint = str(payload.get("next_action_hint", "")).strip()
    if not hint:
        raise PolicyViolation(
            "escalate_strategy_change.next_action_hint must be non-empty",
            rule="payload",
        )
    severity = payload.get("severity")
    if severity is not None and severity not in ALERT_SEVERITIES:
        raise PolicyViolation(
            f"escalate severity={severity!r} not in {sorted(ALERT_SEVERITIES)!r}",
            rule="payload",
        )


def _validate_prune_branch_payload(payload: dict[str, Any]) -> None:
    """Validate a ``prune_branch`` payload."""
    family = str(payload.get("family", "")).strip()
    if not family:
        raise PolicyViolation("prune_branch.family must be non-empty", rule="payload")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise PolicyViolation("prune_branch.reason must be non-empty", rule="payload")


def _validate_delegate_payload(payload: dict[str, Any]) -> None:
    """Validate a ``delegate`` payload's action name against the allowlist."""
    action_name = str(payload.get("action_name", "")).strip()
    if not action_name:
        raise PolicyViolation("delegate.action_name must be non-empty", rule="payload")
    if action_name not in ROBUSTNESS_DELEGATE_ACTIONS:
        raise PolicyViolation(
            f"delegate.action_name={action_name!r} not allowed for "
            f"robustness; allowed: "
            f"{sorted(ROBUSTNESS_DELEGATE_ACTIONS)!r}",
            rule="delegate_action",
            hint="raise an alert and let Orchestration own the remediation",
        )


def _validate_update_state_payload(payload: dict[str, Any]) -> None:
    """Validate an ``update_state`` payload's field allowlist."""
    changes = payload.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise PolicyViolation(
            "update_state.changes must be a non-empty dict",
            rule="payload",
        )
    core_fields = sorted(set(changes.keys()) & CORE_STATE_FIELDS)
    if core_fields:
        raise PolicyViolation(
            f"update_state cannot mutate core state fields: {core_fields!r}",
            rule="state_field",
        )
    non_robust = sorted(set(changes.keys()) - ROBUSTNESS_STATE_FIELDS)
    if non_robust:
        raise PolicyViolation(
            f"update_state contains fields outside robustness allowlist: {non_robust!r}",
            rule="state_field",
            hint=f"allowed: {sorted(ROBUSTNESS_STATE_FIELDS)!r}",
        )


def _validate_send_message_payload(payload: dict[str, Any]) -> None:
    """Validate a ``send_message`` payload's topic."""
    topic = str(payload.get("topic", "")).strip()
    if not topic:
        raise PolicyViolation("send_message.topic must be non-empty", rule="payload")
    # Unknown topics are not rejected (upstream soft-degrades to observation).


# Intent spec table — single source for required fields + builder + validator


@dataclass(frozen=True)
class IntentSpec:
    """Contract for one robustness-emittable intent type."""

    required: tuple[str, ...]
    builder: Callable[..., Intent]
    validator: Callable[[dict[str, Any]], None]


# The 6 intents the robustness role may actually emit; each carries its builder + validator so the required-field map
# and the validator dispatch stay in lock-step.
INTENT_SPEC: Mapping[IntentType, IntentSpec] = {
    IntentType.SEND_MESSAGE: IntentSpec(
        required=("topic",),
        builder=build_send_message,
        validator=_validate_send_message_payload,
    ),
    IntentType.DELEGATE: IntentSpec(
        required=("action_name",),
        builder=build_delegate,
        validator=_validate_delegate_payload,
    ),
    IntentType.UPDATE_STATE: IntentSpec(
        required=("changes",),
        builder=build_update_state,
        validator=_validate_update_state_payload,
    ),
    IntentType.ALERT: IntentSpec(
        required=("severity", "summary"),
        builder=build_alert,
        validator=_validate_alert_payload,
    ),
    IntentType.PRUNE_BRANCH: IntentSpec(
        required=("family", "reason"),
        builder=build_prune_branch,
        validator=_validate_prune_branch_payload,
    ),
    IntentType.ESCALATE_STRATEGY_CHANGE: IntentSpec(
        required=("reason", "next_action_hint"),
        builder=build_escalate,
        validator=_validate_escalate_payload,
    ),
}


# Required-field map for intents robustness never emits but the upstream contract test still diffs against.
_REQUIRED_ONLY: Mapping[IntentType, tuple[str, ...]] = {
    IntentType.PROPOSE_ACTION: ("action_name", "predicted_gain_pct"),
    IntentType.REQUEST: ("target_agent", "kind"),
    IntentType.RESPONSE: ("in_reply_to", "kind"),
    IntentType.REVIEW_VERDICT: ("target_proposal_msg_id",),
    IntentType.EXTEND_LEASE: ("task_id", "extra_sec"),
    IntentType.SPECIALIST_DONE: (
        "gap_canonical_id",
        "domain",
        "proposal_set",
        "summary",
    ),
}


# Per-intent required payload fields, derived from the single spec table so it cannot drift from the validator
# dispatch.
PAYLOAD_REQUIRED: Mapping[IntentType, tuple[str, ...]] = {
    intent_type: (INTENT_SPEC[intent_type].required if intent_type in INTENT_SPEC else _REQUIRED_ONLY[intent_type])
    for intent_type in IntentType
}


# Envelope serialisation (multi-cli outbox, jsonl rows)


def build_envelope_dict(intents: list[Intent]) -> dict[str, Any]:
    """Serialise a list of intents into a single envelope dict."""
    return {"intents": [i.to_envelope_item() for i in intents]}
