# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Internal request / context models for the Critic runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import RequestValidationError


# The only request kind the Critic accepts.
COORDINATOR_INBOX = "coordinator_inbox"


CRITICAL_CONTEXT_KEYS: tuple[str, ...] = ("model", "framework")


# Proposal extracted from a Coordinator inbox row
@dataclass
class Proposal:
    """One ``topic=proposal`` row extracted from a Coordinator inbox."""

    msg_id: str
    from_agent: str
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int | None = None
    action_name: str | None = None
    predicted_gain_pct: float | None = None

    @classmethod
    def from_payload(cls, *, msg_id: str, from_agent: str, payload: dict[str, Any], seq: int | None) -> Proposal:
        """Build a proposal, lifting a string ``action_name`` and a numeric ``predicted_gain_pct`` out of ``payload``."""
        action_name = payload.get("action_name") if isinstance(payload.get("action_name"), str) else None
        gain_raw = payload.get("predicted_gain_pct")
        predicted_gain_pct = float(gain_raw) if isinstance(gain_raw, (int, float)) else None
        return cls(
            msg_id=msg_id,
            from_agent=from_agent,
            payload=payload,
            seq=seq,
            action_name=action_name,
            predicted_gain_pct=predicted_gain_pct,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the proposal as a plain dict via :func:`dataclasses.asdict`."""
        return asdict(self)


# CriticRequest
@dataclass
class CriticRequest:
    """Normalised Critic input."""

    kind: str
    session_id: str
    decision_id: str | None = None
    raw_prompt: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    proposals: list[Proposal] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


# Parsing
def _require_str(d: dict[str, Any], key: str, *, where: str) -> str:
    """Extract a required non-empty string field."""
    if key not in d:
        raise RequestValidationError(f"{where}: missing required field {key!r}")
    value = d[key]
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(f"{where}: field {key!r} must be a non-empty string, got {type(value).__name__}")
    return value


def _optional_str(d: dict[str, Any], key: str, *, where: str) -> str | None:
    """Extract an optional string field."""
    if key not in d or d[key] is None:
        return None
    value = d[key]
    if not isinstance(value, str):
        raise RequestValidationError(
            f"{where}: field {key!r} must be a string when present, got {type(value).__name__}"
        )
    return value


def _optional_dict(d: dict[str, Any], key: str, *, where: str) -> dict[str, Any]:
    """Extract an optional object field."""
    if key not in d or d[key] is None:
        return {}
    value = d[key]
    if not isinstance(value, dict):
        raise RequestValidationError(
            f"{where}: field {key!r} must be an object when present, got {type(value).__name__}"
        )
    return dict(value)


def _optional_list(d: dict[str, Any], key: str, *, where: str) -> list[Any]:
    """Extract an optional list field."""
    if key not in d or d[key] is None:
        return []
    value = d[key]
    if not isinstance(value, list):
        raise RequestValidationError(f"{where}: field {key!r} must be a list when present, got {type(value).__name__}")
    return list(value)


def parse_request(raw: dict[str, Any]) -> CriticRequest:
    """Validate the input dict and return a :class:`CriticRequest`."""
    if not isinstance(raw, dict):
        raise RequestValidationError(f"top-level must be an object, got {type(raw).__name__}")

    kind = _require_str(raw, "kind", where="request")
    if kind != COORDINATOR_INBOX:
        raise RequestValidationError(f"request: kind {kind!r} is not {COORDINATOR_INBOX!r}")
    session_id = _require_str(raw, "session_id", where="request")

    decision_id = _optional_str(raw, "decision_id", where="request")

    raw_prompt = _optional_str(raw, "raw_prompt", where="request")
    if not (raw_prompt and raw_prompt.strip()):
        raise RequestValidationError("coordinator_inbox: raw_prompt must be a non-empty string")

    context = _optional_dict(raw, "context", where="request")
    options = _optional_dict(raw, "options", where="request")

    proposals_raw = _optional_list(raw, "proposals", where="request")
    proposals: list[Proposal] = []
    for i, prop in enumerate(proposals_raw):
        if not isinstance(prop, dict):
            raise RequestValidationError(f"request.proposals[{i}] must be an object, got {type(prop).__name__}")
        msg_id = _require_str(prop, "msg_id", where=f"request.proposals[{i}]")
        from_agent = _require_str(prop, "from_agent", where=f"request.proposals[{i}]")
        payload = _optional_dict(prop, "payload", where=f"request.proposals[{i}]")
        seq = prop.get("seq")
        if seq is not None and not isinstance(seq, int):
            raise RequestValidationError(f"request.proposals[{i}].seq must be int when present")
        proposals.append(Proposal.from_payload(msg_id=msg_id, from_agent=from_agent, payload=payload, seq=seq))

    return CriticRequest(
        kind=kind,
        session_id=session_id,
        decision_id=decision_id,
        raw_prompt=raw_prompt,
        context=context,
        proposals=proposals,
        options=options,
    )


__all__ = [
    "COORDINATOR_INBOX",
    "CRITICAL_CONTEXT_KEYS",
    "CriticRequest",
    "Proposal",
    "parse_request",
]
