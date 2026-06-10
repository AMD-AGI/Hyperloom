# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Internal request / context models for the Critic runtime.

Two entry shapes — ``coordinator_inbox`` (textual prompt, must emit an
intent envelope) and ``critic_decision_request`` (decision review whose
incremental turns merge against session memory) — both converge on
:class:`CriticRequest` (session id, context, parsed proposals, raw payload).
Import-light: no KB / inbox-parser / LLM dependency, only structural
validation other modules can rely on.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import RequestValidationError


# ---------------------------------------------------------------------------
# Allowed request kinds
# ---------------------------------------------------------------------------
COORDINATOR_INBOX = "coordinator_inbox"
DECISION_REQUEST = "critic_decision_request"
KB_DRAFT_REQUEST = "kb_draft_request"
KB_HINT_REQUEST = "kb_hint_request"
OBJECTION_REQUEST = "objection_signal"

REQUEST_KINDS: frozenset[str] = frozenset({
    COORDINATOR_INBOX,
    DECISION_REQUEST,
    KB_DRAFT_REQUEST,
    KB_HINT_REQUEST,
    OBJECTION_REQUEST,
})


# Context dimensions that the KB scope is built from. ``model`` and
# ``framework`` are treated as critical: when either is missing after
# memory merge, KB reads are skipped and the verdict downgrades to
# ``needs_review`` rather than guess.
CONTEXT_DIMENSIONS: tuple[str, ...] = (
    "model",
    "framework",
    "model_family",
    "workload",
    "precision",
    "scale",
    "objective",
)
CRITICAL_CONTEXT_KEYS: tuple[str, ...] = ("model", "framework")


# ---------------------------------------------------------------------------
# Proposal extracted from a Coordinator inbox row
# ---------------------------------------------------------------------------
@dataclass
class Proposal:
    """One ``topic=proposal`` row extracted from a Coordinator inbox.

    Attributes:
        msg_id: The Coordinator-issued ``msg_id`` (hex32 in the legacy release).
        from_agent: Originating agent name (e.g. ``orchestration``).
        seq: Optional bus sequence number (kept for ordering / replay).
        action_name: Convenience copy of ``payload.action_name``.
        predicted_gain_pct: Convenience copy of
            ``payload.predicted_gain_pct``; ``None`` when absent.
        payload: The raw proposal payload as parsed.
    """

    msg_id: str
    from_agent: str
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int | None = None
    action_name: str | None = None
    predicted_gain_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the proposal as a plain dict via :func:`dataclasses.asdict`.

        Returns:
            dict[str, Any]: All proposal fields keyed by name.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# CriticRequest
# ---------------------------------------------------------------------------
@dataclass
class CriticRequest:
    """Normalised Critic input.

    Both Coordinator-driven and dialogue-driven entry points produce one
    of these. Downstream modules (``decision_reviewer``,
    ``intent_envelope``, ``session_memory``) only see this shape.
    """

    kind: str
    session_id: str
    decision_id: str | None = None
    raw_prompt: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    proposals: list[Proposal] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the request as a plain JSON-serialisable dict.

        The raw payload (``raw``) is intentionally omitted; proposals are
        converted via :meth:`Proposal.to_dict`.

        Returns:
            dict[str, Any]: The request fields keyed by name.
        """
        out: dict[str, Any] = {
            "kind": self.kind,
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "raw_prompt": self.raw_prompt,
            "messages": list(self.messages),
            "context": dict(self.context),
            "decision": dict(self.decision),
            "proposals": [p.to_dict() for p in self.proposals],
            "options": dict(self.options),
        }
        return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _require_str(d: dict[str, Any], key: str, *, where: str) -> str:
    """Extract a required non-empty string field.

    Args:
        d (dict[str, Any]): The source mapping.
        key (str): The field name to read.
        where (str): Context label used in error messages.

    Returns:
        str: The field value.

    Raises:
        RequestValidationError: If the key is missing or not a non-empty
            string.
    """
    if key not in d:
        raise RequestValidationError(f"{where}: missing required field {key!r}")
    value = d[key]
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(
            f"{where}: field {key!r} must be a non-empty string, got {type(value).__name__}"
        )
    return value


def _optional_str(d: dict[str, Any], key: str, *, where: str) -> str | None:
    """Extract an optional string field.

    Args:
        d (dict[str, Any]): The source mapping.
        key (str): The field name to read.
        where (str): Context label used in error messages.

    Returns:
        str | None: The string value, or ``None`` when absent/``None``.

    Raises:
        RequestValidationError: If present but not a string.
    """
    if key not in d or d[key] is None:
        return None
    value = d[key]
    if not isinstance(value, str):
        raise RequestValidationError(
            f"{where}: field {key!r} must be a string when present, got {type(value).__name__}"
        )
    return value


def _optional_dict(d: dict[str, Any], key: str, *, where: str) -> dict[str, Any]:
    """Extract an optional object field.

    Args:
        d (dict[str, Any]): The source mapping.
        key (str): The field name to read.
        where (str): Context label used in error messages.

    Returns:
        dict[str, Any]: A copy of the dict value, or ``{}`` when
        absent/``None``.

    Raises:
        RequestValidationError: If present but not a dict.
    """
    if key not in d or d[key] is None:
        return {}
    value = d[key]
    if not isinstance(value, dict):
        raise RequestValidationError(
            f"{where}: field {key!r} must be an object when present, got {type(value).__name__}"
        )
    return dict(value)


def _optional_list(d: dict[str, Any], key: str, *, where: str) -> list[Any]:
    """Extract an optional list field.

    Args:
        d (dict[str, Any]): The source mapping.
        key (str): The field name to read.
        where (str): Context label used in error messages.

    Returns:
        list[Any]: A copy of the list value, or ``[]`` when absent/``None``.

    Raises:
        RequestValidationError: If present but not a list.
    """
    if key not in d or d[key] is None:
        return []
    value = d[key]
    if not isinstance(value, list):
        raise RequestValidationError(
            f"{where}: field {key!r} must be a list when present, got {type(value).__name__}"
        )
    return list(value)


def parse_request(raw: dict[str, Any]) -> CriticRequest:
    """Validate the input dict and return a :class:`CriticRequest`.

    The function is permissive about extra keys (they go into ``raw``)
    but strict about required ones, mirroring contract §14.4 — invalid
    requests should fail fast so the SKILL never produces a bogus
    verdict.

    Args:
        raw (dict[str, Any]): The raw request payload.

    Returns:
        CriticRequest: The validated, normalised request.

    Raises:
        RequestValidationError: If the payload shape, ``kind``, required
            fields, or nested proposals/messages are invalid.
    """
    if not isinstance(raw, dict):
        raise RequestValidationError(
            f"top-level must be an object, got {type(raw).__name__}"
        )

    kind = _require_str(raw, "kind", where="request")
    if kind not in REQUEST_KINDS:
        raise RequestValidationError(
            f"request: kind {kind!r} is not in {sorted(REQUEST_KINDS)!r}"
        )
    session_id = _require_str(raw, "session_id", where="request")

    decision_id = _optional_str(raw, "decision_id", where="request")
    if decision_id is None and kind in (DECISION_REQUEST, OBJECTION_REQUEST):
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"

    raw_prompt = _optional_str(raw, "raw_prompt", where="request")
    if kind == COORDINATOR_INBOX and not (raw_prompt and raw_prompt.strip()):
        raise RequestValidationError(
            "coordinator_inbox: raw_prompt must be a non-empty string"
        )

    messages = _optional_list(raw, "messages", where="request")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise RequestValidationError(
                f"request.messages[{i}] must be an object, got {type(msg).__name__}"
            )

    context = _optional_dict(raw, "context", where="request")
    decision = _optional_dict(raw, "decision", where="request")
    options = _optional_dict(raw, "options", where="request")

    proposals_raw = _optional_list(raw, "proposals", where="request")
    proposals: list[Proposal] = []
    for i, prop in enumerate(proposals_raw):
        if not isinstance(prop, dict):
            raise RequestValidationError(
                f"request.proposals[{i}] must be an object, got {type(prop).__name__}"
            )
        msg_id = _require_str(prop, "msg_id", where=f"request.proposals[{i}]")
        from_agent = _require_str(prop, "from_agent", where=f"request.proposals[{i}]")
        payload = _optional_dict(prop, "payload", where=f"request.proposals[{i}]")
        seq = prop.get("seq")
        if seq is not None and not isinstance(seq, int):
            raise RequestValidationError(
                f"request.proposals[{i}].seq must be int when present"
            )
        action_name = payload.get("action_name") if isinstance(payload.get("action_name"), str) else None
        gain_raw = payload.get("predicted_gain_pct")
        if isinstance(gain_raw, (int, float)):
            predicted_gain_pct: float | None = float(gain_raw)
        else:
            predicted_gain_pct = None
        proposals.append(Proposal(
            msg_id=msg_id,
            from_agent=from_agent,
            payload=payload,
            seq=seq,
            action_name=action_name,
            predicted_gain_pct=predicted_gain_pct,
        ))

    return CriticRequest(
        kind=kind,
        session_id=session_id,
        decision_id=decision_id,
        raw_prompt=raw_prompt,
        messages=messages,
        context=context,
        decision=decision,
        proposals=proposals,
        options=options,
        raw=dict(raw),
    )


__all__ = [
    "COORDINATOR_INBOX",
    "CONTEXT_DIMENSIONS",
    "CRITICAL_CONTEXT_KEYS",
    "CriticRequest",
    "DECISION_REQUEST",
    "KB_DRAFT_REQUEST",
    "KB_HINT_REQUEST",
    "OBJECTION_REQUEST",
    "Proposal",
    "REQUEST_KINDS",
    "parse_request",
]
