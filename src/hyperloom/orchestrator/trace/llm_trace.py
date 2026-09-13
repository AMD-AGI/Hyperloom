# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Closed-schema writer for ``reports/trace/llm_calls.jsonl``."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from hyperloom.common.io import append_jsonl
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import llm_calls_path
from ._row_utils import (
    coerce_optional_int as _coerce_optional_int,
    coerce_optional_str as _coerce_optional_str,
    validate_closed_row,
)

log = logging.getLogger(__name__)


# Closed vocabulary of components that may appear in a trace row, so a typo'd ``component=`` is caught instead of
# fragmenting the per-component rollup.
VALID_COMPONENTS: frozenset[str] = frozenset(
    {
        "orchestration",
        "kernel_agent",
        "dynamic_action",
        "specialist",
        "critic",
        "robustness",
        "proposal_scorer",
        "geak",
        "forge",
        "tracelens",
        "breakdown",
        # Framework-side reasoning (agent ranker, audit refinement, KB synthesis) and the quantization agent.
        "framework",
        "quantization",
    }
)


# Closed vocabulary for a row's terminal status, so a typo'd status cannot make a failed call silently rejoin the
# success rollups.
LLM_STATUS_OK = "ok"
LLM_STATUS_ERROR = "error"
VALID_STATUSES: frozenset[str] = frozenset({LLM_STATUS_OK, LLM_STATUS_ERROR})

# A gateway error body can embed an entire upstream payload (litellm wraps the provider response verbatim), so cap
# what reaches the ledger.
_ERROR_MESSAGE_MAX = 500


# Canonical field contract for one ``llm_calls.jsonl`` row; the closed-schema check compares serialized keys against
# this set exactly.
_ROW_FIELDS: frozenset[str] = frozenset(
    {
        "session_id",
        "ts",
        "component",
        "call_id",
        "role",
        "task_id",
        "dyn_id",
        "tick",
        "phase",
        "turn",
        "model",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "reasoning_output_tokens",
        "latency_ms",
        "reviewed_msg_ids",
        "status",
        "error_type",
        "error_message",
    }
)


class LLMTraceRowError(ValueError):
    """Raised when an LLM-call row violates the closed schema."""


def new_call_id() -> str:
    """Mint a per-call id for the two halves of one LLM call to share."""
    return uuid.uuid4().hex


# Canonical timestamp helper; kept importable for callers.
_now_iso = now_iso


@dataclass
class LLMCallRecord:
    """One LLM call's worth of token accounting + join keys."""

    session_id: str
    component: str
    # Per-call identity shared with the conversation half of the same call, so the two streams pair on the call itself
    # instead of on a ts-second bucket (which splits a call across a second boundary and marries two calls made inside
    # one second).
    call_id: str | None = None
    role: str | None = None
    task_id: str | None = None
    dyn_id: str | None = None
    tick: int | None = None
    phase: str | None = None
    turn: int | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    # Reasoning models bill hidden reasoning output separately; kept next to the canonical four rather than folded
    # into ``output_tokens``, which counts only the visible reply.
    reasoning_output_tokens: int | None = None
    # Wall-clock latency of the model call in ms, measured at the call site (None = not measured); the Langfuse
    # generation is placed at ``[ts - latency_ms, ts]``.
    latency_ms: int | None = None
    # Proposal ``msg_id``s this call reviewed (critic only), so the call can be attributed to the decision it served.
    reviewed_msg_ids: list[str] | None = None
    # An ``error`` row records a call that never produced a usable response, so its token counters stay ``None``;
    # spend rollups must exclude it.
    status: str = LLM_STATUS_OK
    error_type: str | None = None
    error_message: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Serialize to the on-disk row dict, stamping ``ts`` (UTC µs)."""
        return {
            "session_id": str(self.session_id),
            "ts": _now_iso(),
            "component": str(self.component),
            "call_id": _coerce_optional_str(self.call_id),
            "role": _coerce_optional_str(self.role),
            "task_id": _coerce_optional_str(self.task_id),
            "dyn_id": _coerce_optional_str(self.dyn_id),
            "tick": _coerce_optional_int(self.tick),
            "phase": _coerce_optional_str(self.phase),
            "turn": _coerce_optional_int(self.turn),
            "model": _coerce_optional_str(self.model),
            "input_tokens": _coerce_optional_int(self.input_tokens),
            "output_tokens": _coerce_optional_int(self.output_tokens),
            "cache_creation_input_tokens": _coerce_optional_int(self.cache_creation_input_tokens),
            "cache_read_input_tokens": _coerce_optional_int(self.cache_read_input_tokens),
            "reasoning_output_tokens": _coerce_optional_int(self.reasoning_output_tokens),
            "latency_ms": _coerce_optional_int(self.latency_ms),
            "reviewed_msg_ids": _coerce_optional_str_list(self.reviewed_msg_ids),
            "status": str(self.status),
            "error_type": _coerce_optional_str(self.error_type),
            "error_message": _coerce_optional_str(self.error_message),
        }

    @classmethod
    def from_metadata(
        cls,
        *,
        session_id: str,
        component: str,
        metadata: dict[str, Any] | None,
        role: str | None = None,
        task_id: str | None = None,
        dyn_id: str | None = None,
        tick: int | None = None,
        phase: str | None = None,
        turn: int | None = None,
        latency_ms: int | None = None,
    ) -> "LLMCallRecord":
        """Build a record from a ``BackendTurnResult.metadata`` dict."""
        md = metadata or {}
        return cls(
            session_id=session_id,
            component=component,
            # Stamped by the backend that produced the metadata, so the conversation half of the same call reads the
            # same id.
            call_id=md.get("call_id"),
            role=role,
            task_id=task_id,
            dyn_id=dyn_id,
            tick=tick,
            phase=phase,
            turn=turn,
            model=md.get("model"),
            input_tokens=md.get("input_tokens"),
            output_tokens=md.get("output_tokens"),
            cache_creation_input_tokens=md.get("cache_creation_input_tokens"),
            cache_read_input_tokens=md.get("cache_read_input_tokens"),
            reasoning_output_tokens=md.get("reasoning_output_tokens"),
            latency_ms=latency_ms if latency_ms is not None else md.get("latency_ms"),
        )

    @classmethod
    def for_failure(
        cls,
        *,
        session_id: str,
        component: str,
        error: BaseException | str,
        model: str | None = None,
        call_id: str | None = None,
        role: str | None = None,
        task_id: str | None = None,
        dyn_id: str | None = None,
        tick: int | None = None,
        phase: str | None = None,
        turn: int | None = None,
        latency_ms: int | None = None,
    ) -> "LLMCallRecord":
        """Build an ``error`` record for a call that produced no usable response."""
        return cls(
            session_id=session_id,
            component=component,
            call_id=call_id,
            role=role,
            task_id=task_id,
            dyn_id=dyn_id,
            tick=tick,
            phase=phase,
            turn=turn,
            model=model,
            latency_ms=latency_ms,
            status=LLM_STATUS_ERROR,
            error_type=type(error).__name__ if isinstance(error, BaseException) else None,
            error_message=str(error)[:_ERROR_MESSAGE_MAX],
        )


def _coerce_optional_str_list(value: Any) -> list[str] | None:
    """Coerce an iterable of ids to a list of non-empty strings, or ``None``."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            return None
    out = [s for s in (str(v).strip() for v in items) if s]
    return out or None


def append_llm_call(
    *,
    session_dir: Path,
    record: LLMCallRecord,
) -> None:
    """Append one validated LLM-call row to the trace ledger."""
    row = record.to_row()
    validate_closed_row(
        row,
        fields=_ROW_FIELDS,
        valid_components=VALID_COMPONENTS,
        error_cls=LLMTraceRowError,
        label="llm_calls",
    )
    status = row.get("status")
    if status not in VALID_STATUSES:
        raise LLMTraceRowError(f"llm_calls row 'status'={status!r} is not one of {sorted(VALID_STATUSES)!r}")
    dest = llm_calls_path(session_dir)
    try:
        append_jsonl(dest, row, make_parents=True, sort_keys=True)
    except OSError as exc:
        log.warning(
            "llm_trace: append failed for component=%s session_id=%s: %r",
            record.component,
            record.session_id,
            exc,
        )

    # Second sink (opt-in): mirror the call to Langfuse live. Best-effort.
    try:
        from .langfuse_emitter import get_emitter

        get_emitter(session_dir).record_llm_call(row)
    except Exception:  # noqa: BLE001 — Langfuse must never break the ledger
        log.debug("llm_trace: langfuse mirror failed", exc_info=True)


# Sanity guard: the dataclass fields (minus the write-time ``ts``) must stay in lockstep with the on-disk row schema,
# caught at import.
_DATACLASS_FIELDS: frozenset[str] = frozenset(f.name for f in fields(LLMCallRecord))
assert _DATACLASS_FIELDS | {"ts"} == _ROW_FIELDS, (
    f"LLMCallRecord fields drifted from _ROW_FIELDS: dataclass={sorted(_DATACLASS_FIELDS)} row={sorted(_ROW_FIELDS)}"
)


__all__ = [
    "LLM_STATUS_ERROR",
    "LLM_STATUS_OK",
    "LLMCallRecord",
    "LLMTraceRowError",
    "VALID_COMPONENTS",
    "VALID_STATUSES",
    "append_llm_call",
    "new_call_id",
]
