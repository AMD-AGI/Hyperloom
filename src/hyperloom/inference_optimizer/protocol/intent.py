# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structured-intent transport validation (protocol layer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hyperloom_common.coerce import to_int


_ALERT_SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high"})

# Ceiling on a single extend_lease step; repeated extensions are allowed. The PolicyGate enforces it; it is declared
# here so the payload contract each backend renders for the agents states the same number.
EXTEND_LEASE_MAX_SEC: int = 3600


# ---------------------------------------------------------------------------
class IntentType(str, Enum):
    """Enumeration of every structured intent an agent may emit."""

    SEND_MESSAGE = "send_message"
    DELEGATE = "delegate"
    ALERT = "alert"
    # Agent-to-agent request; the handler answers inline on the ``response`` topic.
    REQUEST = "request"
    EXTEND_LEASE = "extend_lease"  # refresh a live task's lease TTL
    # Orchestration scheduling and phase control.
    PRUNE_BRANCH = "prune_branch"
    ESCALATE_STRATEGY_CHANGE = "escalate_strategy_change"


@dataclass
class Intent:
    """One validated intent from any transport."""

    type: IntentType
    payload: dict[str, Any] = field(default_factory=dict)


# Per-intent payload required-field map
PAYLOAD_REQUIRED: dict[IntentType, tuple[str, ...]] = {
    IntentType.SEND_MESSAGE: ("topic",),
    IntentType.DELEGATE: ("action_name",),
    IntentType.ALERT: ("severity", "summary"),
    IntentType.REQUEST: ("target_agent", "kind"),
    IntentType.EXTEND_LEASE: ("task_id", "extra_sec"),
    IntentType.PRUNE_BRANCH: ("family", "reason"),
    IntentType.ESCALATE_STRATEGY_CHANGE: ("reason", "next_action_hint"),
}


class NoIntentEmitted(RuntimeError):
    """Backend produced no parseable envelope and no tool_use blocks."""


class IntentValidationError(RuntimeError):
    """Envelope present but schema invalid."""


def _without_unset_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop the payload fields an agent wrote as JSON ``null``, one level into ``params``.

    Readers downstream resolve an omitted field through ``get(name, default)``,
    which returns the stored ``None`` when the key is present, so a field the
    agent declined to set reaches them as a value rather than as its default.
    An explore round proposing ``keep_threshold_pct: null`` died on
    ``float(None)`` and took its whole measurement with it. Null is the
    transport's way of spelling "not supplied", so it is removed here, where
    the envelope enters, rather than guarded for at each reader.
    """
    cleaned = {name: value for name, value in payload.items() if value is not None}
    params = cleaned.get("params")
    if isinstance(params, dict):
        cleaned["params"] = {name: value for name, value in params.items() if value is not None}
    return cleaned


def validate_envelope(envelope: dict[str, Any]) -> list[Intent]:
    """Validate the top-level envelope shape + per-intent payloads."""
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
        payload = _without_unset_fields(payload)
        for required in PAYLOAD_REQUIRED[it]:
            if required not in payload:
                raise IntentValidationError(
                    f"intents[{i}] (type={it.value}) missing required payload field: {required!r}"
                )
        if it is IntentType.ALERT:
            _validate_alert_payload(payload, index=i)
        elif it is IntentType.EXTEND_LEASE:
            _validate_extend_lease_payload(payload, index=i)
        validated.append(Intent(type=it, payload=dict(payload)))
    return validated


def _validate_alert_payload(payload: dict[str, Any], *, index: int) -> None:
    """Enforce ALERT value constraints: severity enum, non-empty summary."""
    severity = payload.get("severity")
    if not isinstance(severity, str) or severity not in _ALERT_SEVERITIES:
        raise IntentValidationError(
            f"intents[{index}] (type=alert).severity must be one of {sorted(_ALERT_SEVERITIES)!r}, got {severity!r}"
        )
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise IntentValidationError(f"intents[{index}] (type=alert).summary must be a non-empty string")


def _validate_extend_lease_payload(payload: dict[str, Any], *, index: int) -> None:
    """Enforce EXTEND_LEASE value constraints: non-empty task, positive seconds."""
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise IntentValidationError(f"intents[{index}] (type=extend_lease).task_id must be a non-empty string")
    extra_sec = to_int(payload.get("extra_sec"))
    if extra_sec is None or extra_sec <= 0:
        raise IntentValidationError(
            f"intents[{index}] (type=extend_lease).extra_sec must be a positive integer, "
            f"got {payload.get('extra_sec')!r}"
        )


__all__ = [
    "EXTEND_LEASE_MAX_SEC",
    "PAYLOAD_REQUIRED",
    "Intent",
    "IntentType",
    "IntentValidationError",
    "NoIntentEmitted",
    "validate_envelope",
]
