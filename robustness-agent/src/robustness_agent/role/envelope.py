# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Intent envelope contract.

This module mirrors the wire shape defined by
``inference_optimizer/protocol/intent.py`` so that the
robustness reactor can construct intents that the Coordinator's
``PolicyGate`` accepts without change.

Hosts drive the reactor through the subprocess CLI in
:mod:`robustness_agent.runtime.cli`, which writes the resulting
intents through :func:`build_envelope_dict` into ``emit.json`` —
identical to how ``critic-agent`` ships its commit-review output.
This file is *transport-agnostic*: same shape works for the Coordinator
subprocess bridge today and for a long-running CLI writing JSONL rows
to ``$SESSION_DIR/agents/robustness/outbox.jsonl`` in the future.

The class layout intentionally avoids importing from inference_optimizer
to keep this package independent. A contract test cross-checks the
``IntentType`` / ``_PAYLOAD_REQUIRED`` table against the upstream module
when both packages are importable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class IntentType(str, Enum):
    """Intent vocabulary mirrored from upstream ``intent_parser.IntentType``.

    Values are the exact strings the Coordinator persists; do not rename
    without coordinating with inference_optimizer.
    """

    SEND_MESSAGE = "send_message"
    DELEGATE = "delegate"
    PROPOSE_ACTION = "propose_action"
    UPDATE_STATE = "update_state"
    UPDATE_PERSONA = "update_persona"
    ASK_QUESTION = "ask_question"
    ANSWER = "answer"
    ALERT = "alert"
    REQUEST = "request"
    RESPONSE = "response"
    REVIEW_VERDICT = "review_verdict"
    KILL_TASK = "kill_task"
    FORCE_DISPATCH = "force_dispatch"
    PRUNE_BRANCH = "prune_branch"
    ESCALATE_STRATEGY_CHANGE = "escalate_strategy_change"
    # specialist sub-agent exit protocol mirror.
    # Robustness never emits this intent (PolicyGate restricts the
    # source to specialist sub-agents), but the value belongs in the
    # mirror so the upstream-contract test stays green and any tooling
    # that round-trips envelopes does not lose the symbol.
    SPECIALIST_DONE = "specialist_done"


# Per-intent required payload fields. Identical to upstream
# ``policy._PAYLOAD_REQUIRED``; ``decision.policy_aware`` reuses the same
# table to validate intents before they leave the reactor.
PAYLOAD_REQUIRED: Mapping[IntentType, tuple[str, ...]] = {
    IntentType.SEND_MESSAGE: ("topic",),
    IntentType.DELEGATE: ("action_name",),
    IntentType.PROPOSE_ACTION: ("action_name", "predicted_gain_pct"),
    IntentType.UPDATE_STATE: ("changes",),
    IntentType.UPDATE_PERSONA: ("body_md",),
    IntentType.ASK_QUESTION: ("topic", "question"),
    IntentType.ANSWER: ("in_reply_to", "answer"),
    IntentType.ALERT: ("severity", "summary"),
    IntentType.REQUEST: ("target_agent", "kind"),
    IntentType.RESPONSE: ("in_reply_to", "kind"),
    # The ``verdict``/``verdict_map`` choice is mutually exclusive but
    # at least one of them must be
    # present. intent_parser only enforces the structural
    # ``target_proposal_msg_id`` here; the verdict-payload mutual
    # exclusion lives in ``policy._validate_review_verdict_payload``.
    # Mirror the same shape so the upstream-sync contract test stays
    # green.
    IntentType.REVIEW_VERDICT: ("target_proposal_msg_id",),
    IntentType.KILL_TASK: ("task_id", "reason"),
    IntentType.FORCE_DISPATCH: ("task_id", "reason"),
    IntentType.PRUNE_BRANCH: ("family", "reason"),
    IntentType.ESCALATE_STRATEGY_CHANGE: ("reason", "next_action_hint"),
    # specialist exit envelope; payload validated by
    # PolicyGate R3 (``policy._validate_specialist_done``).
    IntentType.SPECIALIST_DONE: (
        "gap_canonical_id", "domain",
        "proposal_set", "empty", "summary",
    ),
}


# Intents that PolicyGate restricts to ``source == "robustness"``. The
# reactor guards these locally to surface configuration / programming
# bugs early; the gate still enforces them server-side.
ROBUSTNESS_ONLY_INTENTS: frozenset[IntentType] = frozenset({
    IntentType.KILL_TASK,
    IntentType.FORCE_DISPATCH,
    IntentType.PRUNE_BRANCH,
    IntentType.ESCALATE_STRATEGY_CHANGE,
})


# Intents the robustness role may emit. Mirrors ``_ROBUSTNESS_INTENTS``
# in upstream agent_role.py. PROPOSE_ACTION / REQUEST / RESPONSE /
# REVIEW_VERDICT / ASK_QUESTION continuations handled by other roles are
# excluded so a programming mistake is caught at construction time.
ROBUSTNESS_ALLOWED_INTENTS: frozenset[IntentType] = frozenset({
    IntentType.SEND_MESSAGE,
    IntentType.ASK_QUESTION,
    IntentType.ANSWER,
    IntentType.ALERT,
    IntentType.UPDATE_PERSONA,
    IntentType.UPDATE_STATE,
    IntentType.DELEGATE,
    IntentType.KILL_TASK,
    IntentType.FORCE_DISPATCH,
    IntentType.PRUNE_BRANCH,
    IntentType.ESCALATE_STRATEGY_CHANGE,
})


# Severities accepted by ``alert`` and ``escalate_strategy_change``.
# ``high`` raises priority 0 broadcasts.
ALERT_SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high"})


# Allowed kill_task scopes per upstream ``KILL_TASK_ALLOWED_SCOPES``.
KILL_TASK_ALLOWED_SCOPES: frozenset[str] = frozenset({"task"})


# Handle actions the robustness role is allowed to delegate. Upstream
# documents the quartet in ``system_prompts/robustness.md``.
#
# ``report`` is an exception to the "handle action" pattern of the other
# three: it is the deterministic session-finalize action owned by
# Orchestration. Robustness is allowed to delegate it ONLY as a
# last-resort wind-down lever, when the evidence shows the session is
# locked out from making further progress on the remaining time budget
# (deadline_imminent with zero validated gain, or recover_failed_finalize
# after a GPU leak recovery returned needs_review and the leak re-fires).
# Action-ladder ``_recommend`` is the single source-of-truth for those
# guard conditions; ``build_delegate`` here only enforces the allowlist.
ROBUSTNESS_DELEGATE_ACTIONS: frozenset[str] = frozenset({
    "accuracy_gate",
    "recover",
    "report",
    "server_lifecycle",
})


# Core SharedState fields the robustness role must not write via
# ``update_state``. Mirrors upstream ``policy.CORE_STATE_FIELDS``;
# kept in lock-step by ``tests/test_role_contract.py``.
CORE_STATE_FIELDS: frozenset[str] = frozenset({
    "current_best",
    "stop_reason",
    "last_tick_exception",
    "cumulative_gain",
    # Coordinator-owned validated cumulative gain trio.
    "cumulative_gain_validated",
    "cumulative_gain_validated_ts",
    "cumulative_gain_validated_stack_len",
    "baseline_tput",
    "baseline_accuracy",
    "session_id",
    "model_path",
    "model_name",
    "model_class",
    "start_ts",
    "max_minutes",
    # fact-layer KEEP ledger (Coordinator-only writer).
    "optimization_stack",
    "gain_per_stack_entry",
    # schema migration breadcrumb.
    "schema_version",
    # Cortex KB integration.
    "cortex_session_id",
    "cortex_session_summary",
    "warm_start_recipe",
    "warm_start_pitfalls",
    "warm_start_lessons",
    "warm_start_ts",
    # KB tag completeness.
    "stack_fingerprint_meta",
    "baseline_workload_extra",
    # warm-recipe replay.
    "warm_replay_attempted",
    "warm_replay_outcome",
    "warm_history_injected",
    # phase state machine (Coordinator-only writer).
    "phase",
    "phase_started_ts",
    "phase_started_unix",
    "phase_history",
    "phase_budget_pct",
    # specialist sub-agent ledger.
    "specialist_rounds",
    "specialist_domain_empty_streak",
    "last_specialist",
    "research_lane_capacity",
    "gpu_specialist_capacity",
    # phase-machine escalation plumbing.
    "pending_escalate_hint",
    "last_consumed_escalate_hint",
    "last_consumed_escalate_hint_ts",
    "plateau_overrides",
    # CLOSE phase sequencer flag.
    "close_sequence_done",
    # unified explore search ledger.
    "explore_search",
    # structured gaps ledger.
    "gaps",
    # Orchestration working-memory checkpoint (Coordinator-authored).
    "orchestration_memory",
    # FRAMEWORK_PR per-repo discovery budget (Coordinator-controlled
    # search depth knob).
    "framework_pr_max_candidates",
    # Advisory model-architecture profile (launcher / state.json owned).
    "model_arch",
    # Architecture-identity tags lifted from config.json (recipe-snapshot
    # KB tags). Fact-layer; locked to mirror upstream
    # ``policy.CORE_STATE_FIELDS``.
    "model_architectures",
    "model_type",
})


# Robustness may only mutate these state fields directly.
ROBUSTNESS_STATE_FIELDS: frozenset[str] = frozenset({
    "crash_count",
    "current_action",
})


@dataclass
class Intent:
    """One validated intent from the reactor.

    The shape matches upstream ``intent_parser.Intent``: a typed enum
    plus a free-form ``payload`` dict. Construction does not validate
    payload contents; use :func:`assert_payload_valid` (see
    ``decision.policy_aware``) before emitting.
    """

    type: IntentType
    payload: dict[str, Any] = field(default_factory=dict)

    def to_envelope_item(self) -> dict[str, Any]:
        """Return the dict shape used inside an ``intents`` envelope."""
        return {"intent_type": self.type.value, "payload": dict(self.payload)}


@dataclass
class BackendTurnResult:
    """Mirror of upstream ``backends.base.BackendTurnResult``.

    The Coordinator inspects ``intents`` and ignores the rest in
    P0-3 / P0-4. ``raw_text`` is recorded for debugging only.
    """

    intents: list[Intent] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Intent builders
# ---------------------------------------------------------------------------

def build_heartbeat(body_md: str = "ok (robustness-agent)") -> Intent:
    """Default tick-end fallback when no symptom warrants an emit."""
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": body_md},
    )


def build_send_message(
    topic: str,
    *,
    body_md: str | None = None,
    to: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> Intent:
    """Generic send_message builder.

    The Coordinator soft-degrades unknown topics to ``observation`` per
    DESIGN v0.6 13.2; callers should still use a known topic.
    """
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
    """Construct an ``alert`` intent.

    severity must be one of :data:`ALERT_SEVERITIES`. ``summary`` is the
    one-line message PolicyGate sees; ``detail`` carries structured
    evidence the Coordinator persists verbatim.
    """
    if severity not in ALERT_SEVERITIES:
        raise ValueError(
            f"alert severity {severity!r} not in {sorted(ALERT_SEVERITIES)!r}"
        )
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
    """Construct an ``escalate_strategy_change`` intent.

    Robustness-only. Non-destructive priority-0 broadcast hint per
    DESIGN v0.6 19.3.4.
    """
    if not reason:
        raise ValueError("escalate reason must be non-empty")
    if not next_action_hint:
        raise ValueError("escalate next_action_hint must be non-empty")
    if severity not in ALERT_SEVERITIES:
        raise ValueError(
            f"escalate severity {severity!r} not in {sorted(ALERT_SEVERITIES)!r}"
        )
    return Intent(
        type=IntentType.ESCALATE_STRATEGY_CHANGE,
        payload={
            "reason": reason,
            "next_action_hint": next_action_hint,
            "severity": severity,
        },
    )


def build_kill_task(task_id: str, reason: str) -> Intent:
    """Construct a ``kill_task`` intent.

    Robustness-only. ``scope`` is hardcoded to ``"task"`` because the
    upstream PolicyGate v0.6 rejects any other value (server / process
    kills go through delegate(server_lifecycle) under IR-5).
    """
    if not task_id:
        raise ValueError("kill_task task_id must be non-empty")
    if not reason:
        raise ValueError("kill_task reason must be non-empty")
    return Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": task_id, "reason": reason, "scope": "task"},
    )


def build_force_dispatch(task_id: str, reason: str) -> Intent:
    """Construct a ``force_dispatch`` intent. Robustness-only."""
    if not task_id:
        raise ValueError("force_dispatch task_id must be non-empty")
    if not reason:
        raise ValueError("force_dispatch reason must be non-empty")
    return Intent(
        type=IntentType.FORCE_DISPATCH,
        payload={"task_id": task_id, "reason": reason},
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
    """Construct a ``delegate`` intent.

    Robustness may only delegate handle actions listed in
    :data:`ROBUSTNESS_DELEGATE_ACTIONS`. Other action names will be
    rejected by PolicyGate's ``KERNEL_OWNED_ACTIONS`` / role check; we
    fail fast locally to keep error context.
    """
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
    """Construct an ``update_state`` intent.

    Restricted to fields in :data:`ROBUSTNESS_STATE_FIELDS`;
    fields in :data:`CORE_STATE_FIELDS` are rejected upstream.
    """
    if not changes:
        raise ValueError("update_state changes must be a non-empty mapping")
    illegal = sorted(set(changes.keys()) - ROBUSTNESS_STATE_FIELDS)
    if illegal:
        raise ValueError(
            "update_state contains fields outside robustness allowlist: "
            f"{illegal!r}; allowed: {sorted(ROBUSTNESS_STATE_FIELDS)!r}"
        )
    return Intent(type=IntentType.UPDATE_STATE, payload={"changes": dict(changes)})


# ---------------------------------------------------------------------------
# Envelope serialisation (multi-cli outbox, jsonl rows)
# ---------------------------------------------------------------------------

def build_envelope_dict(intents: list[Intent]) -> dict[str, Any]:
    """Serialise a list of intents into a single envelope dict.

    Matches upstream ``INTENT_ENVELOPE_SCHEMA``. Used by the runtime
    CLI's ``tick`` command to populate ``emit.json.intent_envelope`` —
    identical to ``critic-agent``'s ``commit-review`` output, so the
    same ``validate_envelope`` host-side check accepts both.
    """
    return {"intents": [i.to_envelope_item() for i in intents]}
