# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structured-intent transport validation (protocol layer).

Claude (``emit_intent`` MCP tool_call) and Codex (JSON-in-text envelope)
transports share one envelope shape, validated via :func:`validate_envelope`.
Must never import ``orchestrator`` / ``shared_state`` (import cycle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hyperloom.common.coerce import to_int


# ---------------------------------------------------------------------------
# Valid payload value sets used by _validate_*_payload helpers below.
# These must stay consistent with the agent-side frozensets that enforce the
# same constraints at emit time (agents/robustness/role/envelope.ALERT_SEVERITIES
# and agents/critic/runtime/intent_envelope.ALLOWED_VERDICTS).
_ALERT_SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high"})
_ALLOWED_VERDICTS: frozenset[str] = frozenset(
    {"approve", "reject", "redirect", "advise", "needs_review"}
)

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
    ALERT = "alert"
    # Bidirectional agent-to-agent RPC.
    REQUEST = "request"
    RESPONSE = "response"
    REVIEW_VERDICT = "review_verdict"  # Critic-only
    EXTEND_LEASE = "extend_lease"  # refresh a live task's lease TTL
    # Robustness-only scheduling police.
    PRUNE_BRANCH = "prune_branch"
    ESCALATE_STRATEGY_CHANGE = "escalate_strategy_change"
    # specialist exit: one per task.
    SPECIALIST_DONE = "specialist_done"


@dataclass
class Intent:
    """One validated intent from any transport."""

    type: IntentType
    payload: dict[str, Any] = field(default_factory=dict)


# Per-intent payload required-field map
_PAYLOAD_REQUIRED: dict[IntentType, tuple[str, ...]] = {
    IntentType.SEND_MESSAGE: ("topic",),
    IntentType.DELEGATE: ("action_name",),
    IntentType.PROPOSE_ACTION: ("action_name", "predicted_gain_pct"),
    IntentType.UPDATE_STATE: ("changes",),
    IntentType.ALERT: ("severity", "summary"),
    IntentType.REQUEST: ("target_agent", "kind"),
    IntentType.RESPONSE: ("in_reply_to", "kind"),
    # verdict/verdict_map mutual exclusion enforced by _validate_review_verdict_payload.
    IntentType.REVIEW_VERDICT: ("target_proposal_msg_id",),
    IntentType.EXTEND_LEASE: ("task_id", "extra_sec"),
    IntentType.PRUNE_BRANCH: ("family", "reason"),
    IntentType.ESCALATE_STRATEGY_CHANGE: ("reason", "next_action_hint"),
    # specialist exit envelope; the runner re-stamps and defaults the payload.
    IntentType.SPECIALIST_DONE: ("gap_canonical_id", "domain", "proposal_set", "empty", "summary"),
}


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


def _validate_alert_payload(payload: dict[str, Any], *, index: int) -> None:
    """Check ALERT payload value constraints.

    Args:
        payload: The ALERT intent payload.
        index: Position in the envelope (for error messages).

    Raises:
        IntentValidationError: If severity is not a recognised string or
            summary is absent/empty.
    """
    severity = payload.get("severity")
    if not isinstance(severity, str) or severity not in _ALERT_SEVERITIES:
        raise IntentValidationError(
            f"intents[{index}] (type=alert).severity must be one of "
            f"{sorted(_ALERT_SEVERITIES)!r}, got {severity!r}"
        )
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise IntentValidationError(
            f"intents[{index}] (type=alert).summary must be a non-empty string"
        )


def _validate_extend_lease_payload(payload: dict[str, Any], *, index: int) -> None:
    """Check EXTEND_LEASE payload value constraints.

    Args:
        payload: The EXTEND_LEASE intent payload.
        index: Position in the envelope (for error messages).

    Raises:
        IntentValidationError: If task_id is empty or extra_sec is not a
            positive integer.
    """
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise IntentValidationError(
            f"intents[{index}] (type=extend_lease).task_id must be a non-empty string"
        )
    extra_sec = to_int(payload.get("extra_sec"))
    if extra_sec is None or extra_sec <= 0:
        raise IntentValidationError(
            f"intents[{index}] (type=extend_lease).extra_sec must be a positive integer, "
            f"got {payload.get('extra_sec')!r}"
        )


def validate_envelope(envelope: dict[str, Any]) -> list[Intent]:
    """Validate the top-level envelope shape + per-intent payloads.

    Checks both structural constraints (required keys, correct types) and
    value constraints (enum membership, numeric ranges, non-empty strings).

    Args:
        envelope (dict[str, Any]): The decoded envelope, expected to carry
            an ``intents`` list of ``{intent_type, payload}`` items.

    Returns:
        list[Intent]: The validated intents in envelope order.

    Raises:
        IntentValidationError: On any structural or value issue so the caller
            can surface a single repair-prompt path.
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
            raise IntentValidationError(f"intents[{i}] missing intent_type or payload")
        try:
            it = IntentType(item["intent_type"])
        except ValueError:
            raise IntentValidationError(f"intents[{i}].intent_type {item['intent_type']!r} not in allowed set")
        payload = item["payload"]
        if not isinstance(payload, dict):
            raise IntentValidationError(f"intents[{i}].payload must be object, got {type(payload).__name__}")
        for required in _PAYLOAD_REQUIRED[it]:
            if required not in payload:
                raise IntentValidationError(
                    f"intents[{i}] (type={it.value}) missing required payload field: {required!r}"
                )
        if it is IntentType.REVIEW_VERDICT:
            _validate_review_verdict_payload(payload, index=i)
        elif it is IntentType.ALERT:
            _validate_alert_payload(payload, index=i)
        elif it is IntentType.EXTEND_LEASE:
            _validate_extend_lease_payload(payload, index=i)
        validated.append(Intent(type=it, payload=dict(payload)))
    return validated


def _validate_review_verdict_payload(
    payload: dict[str, Any],
    *,
    index: int,
) -> None:
    """Enforce REVIEW_VERDICT structural shape: exactly one of ``verdict``
    (single) or ``verdict_map`` (per-variant batch) must be present.

    Args:
        payload: The REVIEW_VERDICT intent payload to validate.
        index: Position of the intent in the envelope (for error messages).

    Raises:
        IntentValidationError: If neither or both of ``verdict`` and
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
            f"intents[{index}] (type=review_verdict): 'verdict' and 'verdict_map' are mutually exclusive"
        )
    if has_single:
        v = payload["verdict"]
        if not isinstance(v, str) or v not in _ALLOWED_VERDICTS:
            raise IntentValidationError(
                f"intents[{index}] (type=review_verdict).verdict must be one of "
                f"{sorted(_ALLOWED_VERDICTS)!r}, got {v!r}"
            )
    if has_map:
        vm = payload["verdict_map"]
        if not isinstance(vm, dict) or not vm:
            raise IntentValidationError(
                f"intents[{index}] (type=review_verdict).verdict_map must be a non-empty object keyed by variant_name"
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
                    f"intents[{index}] (type=review_verdict).verdict_map[{vname!r}] missing required 'verdict' key"
                )
            ev = entry["verdict"]
            if not isinstance(ev, str) or ev not in _ALLOWED_VERDICTS:
                raise IntentValidationError(
                    f"intents[{index}] (type=review_verdict).verdict_map[{vname!r}].verdict "
                    f"must be one of {sorted(_ALLOWED_VERDICTS)!r}, got {ev!r}"
                )


__all__ = [
    "Intent",
    "IntentType",
    "IntentValidationError",
    "NoIntentEmitted",
    "validate_envelope",
    "_ALERT_SEVERITIES",
    "_ALLOWED_VERDICTS",
]
