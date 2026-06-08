# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Structured-intent transport schema (protocol layer).

Two transports share the same envelope schema:

* **Claude transport** (§14.2) — ``emit_intent`` MCP tool_call
* **Codex transport** (§14.3) — ``validated_json_output`` (single JSON object)

Both validate via :func:`validate_envelope`; downstream consumers see a
uniform list of :class:`Intent` objects.

This module is the bottom-layer protocol definition: it depends on nothing
inside the package and must never import ``orchestrator`` / ``shared_state``
(doing so would create an import cycle).

v0.6 changes vs v0.5:

* ``OBJECTION`` / ``VOTE`` removed (parliament gone — DESIGN ADR-38).
* ``REVIEW_VERDICT`` added (Critic Review Protocol — DESIGN §18.2).
* Triage-only intents kept but renamed: source allowlist now
  ``{"robustness"}`` (see ``policy.py``); intent literal strings unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
class IntentType(str, Enum):
    """Enumeration of every structured intent an agent may emit.

    String-valued so the literal wire token equals the member value.
    PolicyGate restricts which sources may emit which members; this enum
    only defines the vocabulary shared by both transports.
    """

    SEND_MESSAGE = "send_message"
    DELEGATE = "delegate"
    PROPOSE_ACTION = "propose_action"
    UPDATE_STATE = "update_state"
    UPDATE_PERSONA = "update_persona"
    ASK_QUESTION = "ask_question"
    ANSWER = "answer"
    ALERT = "alert"
    # Bidirectional agent-to-agent RPC. PolicyGate restricts (source,
    # target_agent) pairs and `kind` per pair.
    REQUEST = "request"
    RESPONSE = "response"
    # Critic Review Protocol (DESIGN §18.2). Critic-only.
    REVIEW_VERDICT = "review_verdict"
    # Robustness-only — task cancellation. Payload.scope must be "task".
    KILL_TASK = "kill_task"
    # Robustness-only — scheduling police (DESIGN §19.3).
    FORCE_DISPATCH = "force_dispatch"
    PRUNE_BRANCH = "prune_branch"
    ESCALATE_STRATEGY_CHANGE = "escalate_strategy_change"
    # specialist sub-agent exit protocol.
    # specialist tasks (LLM sub-agents on the research_lane) close their
    # lifecycle with exactly one SPECIALIST_DONE intent. PolicyGate R3
    # checks from_agent prefix + gap/domain match + payload schema; see
    # ``policy._validate_specialist_done`` for the full contract.
    SPECIALIST_DONE = "specialist_done"


_ALL_INTENT_VALUES: tuple[str, ...] = tuple(t.value for t in IntentType)


@dataclass
class Intent:
    """One validated intent from any transport."""

    type: IntentType
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_envelope_item(cls, item: dict[str, Any]) -> "Intent":
        """Build an :class:`Intent` from one raw envelope item.

        Args:
            item (dict[str, Any]): Raw item with an ``intent_type`` key and
                an optional ``payload`` mapping.

        Returns:
            Intent: The parsed intent with a copied payload dict.

        Raises:
            ValueError: If ``intent_type`` is not a valid :class:`IntentType`.
        """
        intent_type = IntentType(item["intent_type"])
        return cls(type=intent_type, payload=dict(item.get("payload") or {}))


# ---------------------------------------------------------------------------
# Envelope schema (DESIGN §14.1) — kept inline so we don't need jsonschema
# ---------------------------------------------------------------------------
INTENT_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "intent_type": {
                        "type": "string",
                        "enum": list(_ALL_INTENT_VALUES),
                    },
                    "payload": {
                        "type": "object",
                        "description": "Schema depends on intent_type.",
                    },
                },
                "required": ["intent_type", "payload"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["intents"],
    "additionalProperties": False,
}


# Per-intent payload required-field map (DESIGN §14.1 / §15 dispatch / §18.2)
_PAYLOAD_REQUIRED: dict[IntentType, tuple[str, ...]] = {
    IntentType.SEND_MESSAGE:    ("topic",),
    IntentType.DELEGATE:        ("action_name",),
    IntentType.PROPOSE_ACTION:  ("action_name", "predicted_gain_pct"),
    IntentType.UPDATE_STATE:    ("changes",),
    IntentType.UPDATE_PERSONA:  ("body_md",),
    IntentType.ASK_QUESTION:    ("topic", "question"),
    IntentType.ANSWER:          ("in_reply_to", "answer"),
    IntentType.ALERT:           ("severity", "summary"),
    IntentType.REQUEST:         ("target_agent", "kind"),
    IntentType.RESPONSE:        ("in_reply_to", "kind"),
    # The ``verdict`` / ``verdict_map`` choice is mutually exclusive but
    # one of them must be present. We require only the structural
    # ``target_proposal_msg_id`` field here; the
    # ``verdict``/``verdict_map`` mutual exclusion is enforced by
    # :func:`_validate_review_verdict_payload` below (called from
    # :func:`validate_envelope`) so the per-variant variant_name
    # schema can be validated in one place alongside the legacy single-
    # verdict shape.
    IntentType.REVIEW_VERDICT:  ("target_proposal_msg_id",),
    IntentType.KILL_TASK:       ("task_id", "reason"),
    IntentType.FORCE_DISPATCH:  ("task_id", "reason"),
    IntentType.PRUNE_BRANCH:    ("family", "reason"),
    IntentType.ESCALATE_STRATEGY_CHANGE: ("reason", "next_action_hint"),
    # specialist exit envelope. ``proposal_set`` is a list (may
    # be empty when ``empty=true``); per-variant schema is enforced by
    # PolicyGate R3 (``policy._validate_specialist_done``) since the
    # intent parser sees only the envelope skeleton.
    IntentType.SPECIALIST_DONE: ("gap_canonical_id", "domain",
                                  "proposal_set", "empty", "summary"),
}


# ---------------------------------------------------------------------------
# Tool schema for the Claude SDK (DESIGN §14.2)
# ---------------------------------------------------------------------------
EMIT_INTENT_TOOL_SCHEMA: dict[str, Any] = {
    "name": "emit_intent",
    "description": (
        "The ONLY way to communicate decisions, messages, or actions to "
        "the system. Free-text responses are ignored. Call this tool one "
        "or more times per turn; each call carries exactly one intent."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent_type": {
                "type": "string",
                "enum": list(_ALL_INTENT_VALUES),
            },
            "payload": {
                "type": "object",
                "description": (
                    "Per-intent payload. send_message: {topic, body_md, to?}; "
                    "propose_action: {action_name, predicted_gain_pct, "
                    "reason?}; delegate: {action_name, params?, "
                    "idempotency_key?}; alert: {severity, summary, detail?}; "
                    "update_state: {changes}; update_persona: {body_md}; "
                    "ask_question: {topic, question}; answer: {in_reply_to, "
                    "answer}; request: {target_agent, kind, params?, "
                    "reason?}; response: {in_reply_to, kind, status?, "
                    "result?}; review_verdict: {target_proposal_msg_id, "
                    "verdict ∈ approve/reject/redirect/advise/needs_review, "
                    "reasoning, kb_evidence?} for single-proposal review, "
                    "OR {target_proposal_msg_id, verdict_map: "
                    "{variant_name: {verdict, rationale?}}} for batch "
                    "explore review (v0.8 KB_gaps/Gap-11); the two "
                    "shapes are mutually exclusive — Critic-only; "
                    "kill_task / force_dispatch / prune_branch / "
                    "escalate_strategy_change — Robustness-only (PolicyGate); "
                    "specialist_done: {gap_canonical_id, domain, "
                    "proposal_set: [variant...], empty, summary, "
                    "confidence?, new_findings?, residual_questions?} — "
                    "specialist-only, exactly one per task."
                ),
            },
        },
        "required": ["intent_type", "payload"],
    },
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class NoIntentEmitted(RuntimeError):
    """Backend produced no parseable envelope and no tool_use blocks."""


class IntentValidationError(RuntimeError):
    """Envelope present but schema invalid (raw + reason captured)."""

    def __init__(self, reason: str, raw: str | None = None):
        """Initialise the validation error.

        Args:
            reason (str): Human-readable description of the schema problem.
            raw (str | None): The raw envelope text, captured for repair
                prompts / diagnostics.
        """
        super().__init__(reason)
        self.raw = raw


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_envelope(envelope: dict[str, Any]) -> list[Intent]:
    """Validate the top-level envelope shape + per-intent payloads.

    Args:
        envelope (dict[str, Any]): The decoded envelope, expected to carry
            an ``intents`` list of ``{intent_type, payload}`` items.

    Returns:
        list[Intent]: The validated intents in envelope order.

    Raises:
        IntentValidationError: On any structural issue so the caller can
            surface a single repair-prompt path (DESIGN §14.4).
    """
    if not isinstance(envelope, dict):
        raise IntentValidationError(f"envelope must be object, got {type(envelope).__name__}")
    if "intents" not in envelope:
        raise IntentValidationError("envelope missing required 'intents' key")
    items = envelope["intents"]
    if not isinstance(items, list):
        raise IntentValidationError("envelope.intents must be a list")

    validated: list[Intent] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise IntentValidationError(f"intents[{i}] must be object, got {type(item).__name__}")
        if "intent_type" not in item or "payload" not in item:
            raise IntentValidationError(
                f"intents[{i}] missing intent_type or payload"
            )
        try:
            it = IntentType(item["intent_type"])
        except ValueError:
            raise IntentValidationError(
                f"intents[{i}].intent_type {item['intent_type']!r} not in allowed set"
            )
        payload = item["payload"]
        if not isinstance(payload, dict):
            raise IntentValidationError(
                f"intents[{i}].payload must be object, got {type(payload).__name__}"
            )
        for required in _PAYLOAD_REQUIRED[it]:
            if required not in payload:
                raise IntentValidationError(
                    f"intents[{i}] (type={it.value}) missing required "
                    f"payload field: {required!r}"
                )
        # REVIEW_VERDICT shape branch.
        if it is IntentType.REVIEW_VERDICT:
            _validate_review_verdict_payload(payload, index=i)
        validated.append(Intent(type=it, payload=dict(payload)))
    return validated


def _validate_review_verdict_payload(
    payload: dict[str, Any], *, index: int,
) -> None:
    """Enforce the legacy REVIEW_VERDICT envelope schema.

    KB_design §3.5 §5 / M5 §5 step 5 — Critic Review returns one of:

    * ``verdict``: legacy single-verdict shape for one-proposal
      reviews (kernel_opt / integrate / report / specialist dispatch
      etc.). Free-form string; PolicyGate later checks it against
      :data:`policy.REVIEW_VERDICTS`.
    * ``verdict_map``: v0.8 batch shape — ``{variant_name:
      {verdict: ..., rationale?: ...}}``. Required when the parent
      proposal is a multi-variant ``explore`` grid (KB_gaps/Gap-11
      §5.2). Per-variant verdict strings are also vetted by
      PolicyGate.

    The two fields are mutually exclusive; exactly one must be
    present. PolicyGate handles the *content* validation (verdict
    string vocab, variant_name vs original grid). This validator
    only enforces structural shape so callers downstream (resume
    rebuild + policy) can rely on at-most-one-present.

    Args:
        payload (dict[str, Any]): The ``review_verdict`` payload to check.
        index (int): Position of the intent in the envelope, used for
            error messages.

    Raises:
        IntentValidationError: If neither or both of ``verdict`` /
            ``verdict_map`` are present, or ``verdict_map`` is malformed.
    """
    has_single = "verdict" in payload
    has_map = "verdict_map" in payload
    if not has_single and not has_map:
        raise IntentValidationError(
            f"intents[{index}] (type=review_verdict) must include either "
            f"'verdict' (single) or 'verdict_map' (per-variant); both missing"
        )
    if has_single and has_map:
        raise IntentValidationError(
            f"intents[{index}] (type=review_verdict): 'verdict' and "
            f"'verdict_map' are mutually exclusive"
        )
    if has_map:
        vm = payload["verdict_map"]
        if not isinstance(vm, dict) or not vm:
            raise IntentValidationError(
                f"intents[{index}] (type=review_verdict).verdict_map must "
                f"be a non-empty object keyed by variant_name"
            )
        for vname, entry in vm.items():
            if not isinstance(vname, str) or not vname.strip():
                raise IntentValidationError(
                    f"intents[{index}] (type=review_verdict).verdict_map "
                    f"keys must be non-empty variant names, got {vname!r}"
                )
            if not isinstance(entry, dict):
                raise IntentValidationError(
                    f"intents[{index}] (type=review_verdict).verdict_map"
                    f"[{vname!r}] must be an object with at least a "
                    f"'verdict' key, got {type(entry).__name__}"
                )
            if "verdict" not in entry:
                raise IntentValidationError(
                    f"intents[{index}] (type=review_verdict).verdict_map"
                    f"[{vname!r}] missing required 'verdict' key"
                )


__all__ = [
    "EMIT_INTENT_TOOL_SCHEMA",
    "INTENT_ENVELOPE_SCHEMA",
    "Intent",
    "IntentType",
    "IntentValidationError",
    "NoIntentEmitted",
    "validate_envelope",
]
