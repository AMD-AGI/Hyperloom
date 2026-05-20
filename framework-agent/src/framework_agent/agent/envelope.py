"""RESPONSE payload envelopes for the framework sibling-skill.

Single source of truth for the four envelope shapes the framework agent
emits per ``hyperloom-framework-agent-design.md`` §4.6:

* :class:`OptimizeSuccess`  -- ``framework_optimize`` returns a patch
  (or pure flag-discovery: empty ``patch_path`` + non-empty
  ``discovered_flags``).
* :class:`OptimizeFailure`  -- gave up early (AST scan empty, source
  not found, LLM hit max turns without a usable patch, ...).
* :class:`IntegrateSuccess` -- ``framework_integrate`` reached a
  verdict (KEEP / REVERT / NEEDS_REVIEW); always paired with metrics.
* :class:`IntegrateFailure` -- hard failure before any verdict (patch
  apply rejected, server failed to restart, bench timed out, ...).

Each envelope is validated by :func:`validate_envelope` at both ends:

* ``fa agent commit-result`` -- before writing to disk.
* ``FrameworkAgentBackend``  -- before forwarding to the coordinator.

The schemas live as jsonschema dicts so they can be reused by callers
that don't import this Python module (e.g. tests that lint a sample
envelope from a fixture).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import jsonschema


# ---------------------------------------------------------------------------
# TypedDict definitions (type-checker friendly; jsonschema is the runtime
# enforcement boundary)
# ---------------------------------------------------------------------------
class OptimizeSuccess(TypedDict, total=False):
    """framework_optimize returned a candidate patch (or flags-only)."""

    payload_kind: Literal["OptimizeSuccess"]
    patch_path: str  # absolute; "" allowed when only discovered_flags is set
    predicted_gain_pct: float
    rationale: str
    discovered_flags: dict[str, list[str]]
    target_framework: Literal["vllm", "sglang"]
    stage_a_elapsed_ms: int


class OptimizeFailure(TypedDict):
    """framework_optimize aborted before producing a usable patch."""

    payload_kind: Literal["OptimizeFailure"]
    reason: str
    stage_a_elapsed_ms: int


class IntegrateSuccess(TypedDict, total=False):
    """framework_integrate produced a verdict + metrics."""

    payload_kind: Literal["IntegrateSuccess"]
    verdict: Literal["KEEP", "REVERT", "NEEDS_REVIEW"]
    patch_id: str
    tput_before: float
    tput_after: float
    accuracy_before: float
    accuracy_after: float
    accuracy_drop: float
    stage_b_elapsed_ms: int


class IntegrateFailure(TypedDict):
    """framework_integrate aborted (apply / restart / bench fail)."""

    payload_kind: Literal["IntegrateFailure"]
    reason: str
    patch_id: str
    stage_b_elapsed_ms: int


# ---------------------------------------------------------------------------
# jsonschema dicts (runtime enforcement)
# ---------------------------------------------------------------------------
OPTIMIZE_SUCCESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "payload_kind", "patch_path", "predicted_gain_pct",
        "rationale", "stage_a_elapsed_ms",
    ],
    "properties": {
        "payload_kind":         {"const": "OptimizeSuccess"},
        "patch_path":           {"type": "string"},
        "predicted_gain_pct":   {"type": "number", "minimum": 0.0},
        "rationale":            {"type": "string"},
        "discovered_flags":     {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "target_framework":     {"enum": ["vllm", "sglang"]},
        "stage_a_elapsed_ms":   {"type": "integer", "minimum": 0},
    },
}


OPTIMIZE_FAILURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["payload_kind", "reason", "stage_a_elapsed_ms"],
    "properties": {
        "payload_kind":       {"const": "OptimizeFailure"},
        "reason":             {"type": "string", "minLength": 1},
        "stage_a_elapsed_ms": {"type": "integer", "minimum": 0},
    },
}


INTEGRATE_SUCCESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "payload_kind", "verdict", "patch_id", "stage_b_elapsed_ms",
    ],
    "properties": {
        "payload_kind":       {"const": "IntegrateSuccess"},
        "verdict":            {"enum": ["KEEP", "REVERT", "NEEDS_REVIEW"]},
        "patch_id":           {"type": "string", "minLength": 1},
        "tput_before":        {"type": "number"},
        "tput_after":         {"type": "number"},
        "accuracy_before":    {"type": "number"},
        "accuracy_after":     {"type": "number"},
        "accuracy_drop":      {"type": "number"},
        "stage_b_elapsed_ms": {"type": "integer", "minimum": 0},
    },
}


INTEGRATE_FAILURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "payload_kind", "reason", "patch_id", "stage_b_elapsed_ms",
    ],
    "properties": {
        "payload_kind":       {"const": "IntegrateFailure"},
        "reason":             {"type": "string", "minLength": 1},
        "patch_id":           {"type": "string", "minLength": 1},
        "stage_b_elapsed_ms": {"type": "integer", "minimum": 0},
    },
}


ENVELOPE_SCHEMAS: dict[str, dict[str, Any]] = {
    "OptimizeSuccess":  OPTIMIZE_SUCCESS_SCHEMA,
    "OptimizeFailure":  OPTIMIZE_FAILURE_SCHEMA,
    "IntegrateSuccess": INTEGRATE_SUCCESS_SCHEMA,
    "IntegrateFailure": INTEGRATE_FAILURE_SCHEMA,
}


class EnvelopeValidationError(ValueError):
    """Raised when an envelope fails jsonschema validation."""


def validate_envelope(envelope: dict[str, Any]) -> str:
    """Validate ``envelope`` against the matching schema.

    Returns the resolved ``payload_kind`` on success. Raises
    :class:`EnvelopeValidationError` with a precise message on any
    failure (missing payload_kind, unknown payload_kind, schema
    violation).
    """
    if not isinstance(envelope, dict):
        raise EnvelopeValidationError(
            f"envelope must be a dict, got {type(envelope).__name__}"
        )
    kind = envelope.get("payload_kind")
    if not isinstance(kind, str):
        raise EnvelopeValidationError(
            f"envelope.payload_kind is required (str); got {kind!r}"
        )
    schema = ENVELOPE_SCHEMAS.get(kind)
    if schema is None:
        raise EnvelopeValidationError(
            f"unknown envelope payload_kind={kind!r}; "
            f"expected one of {sorted(ENVELOPE_SCHEMAS)!r}"
        )
    try:
        jsonschema.validate(envelope, schema)
    except jsonschema.ValidationError as exc:
        raise EnvelopeValidationError(
            f"envelope payload_kind={kind!r} failed schema: "
            f"{exc.message} (path={list(exc.absolute_path)})"
        ) from exc
    return kind


__all__ = [
    "ENVELOPE_SCHEMAS",
    "EnvelopeValidationError",
    "IntegrateFailure",
    "IntegrateSuccess",
    "INTEGRATE_FAILURE_SCHEMA",
    "INTEGRATE_SUCCESS_SCHEMA",
    "OPTIMIZE_FAILURE_SCHEMA",
    "OPTIMIZE_SUCCESS_SCHEMA",
    "OptimizeFailure",
    "OptimizeSuccess",
    "validate_envelope",
]
