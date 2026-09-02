# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Closed-schema writer for ``reports/trace/llm_calls.jsonl``.

Owns the canonical field contract for a single LLM call so every producer
emits rows the collector can join without guessing.

Design contract:

* **Closed schema**: a row carrying an unknown field — or missing a
  required one — fails fast (:class:`LLMTraceRowError`).
* **Best-effort I/O**: disk failures while appending are logged and
  swallowed; trace writes must never break the optimization loop.
* **Token shape**: the counters mirror the keys both
  :class:`ClaudeBackend` and :class:`CodexBackend` put on
  ``BackendTurnResult.metadata``. Backends without a prompt-cache split
  (OpenAI / GEAK) report ``None`` for the two ``cache_*`` counters so the
  collector can tell "no cache concept" from "zero cache hits";
  ``reasoning_output_tokens`` is the same story for reasoning models, and is
  kept out of ``output_tokens`` because that counts the visible reply only.
* **Pairing**: ``call_id`` is the join key against the conversation ledger's
  row for the same call. Both halves carry it when the producing backend
  stamped one; otherwise the emitter falls back to its identity+second key.
* **Success and failure**: a row carries a terminal ``status``. Only
  ``status="ok"`` rows have token accounting; ``status="error"`` rows record a
  call that never produced a usable response (built via
  :meth:`LLMCallRecord.for_failure`). Without them a failed call is simply
  absent from the ledger, which reads as "never happened" rather than "failed"
  — the reason Langfuse showed a 0% error rate while calls were demonstrably
  failing. Spend rollups must therefore filter on ``status``.

The record is a dataclass so call sites get constructor-time field checking
and a single :meth:`to_row` serialization path; the closed-schema check in
:func:`append_llm_call` is a second guard covering rows rebuilt from dicts.
"""

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


# Closed vocabulary of components that may appear in a trace row, so a typo'd
# ``component=`` is caught instead of fragmenting the per-component rollup.
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
        "kernel_rewrite_controller",
        "tracelens",
        "breakdown",
        # Framework-side reasoning (agent ranker, audit refinement, KB synthesis)
        # and the quantization agent. Both spend against the gateway; neither has
        # an append_llm_call producer yet, so today they appear only in gateway
        # attribution.
        "framework",
        "quantization",
    }
)


# Closed vocabulary for a row's terminal status, so a typo'd status cannot make
# a failed call silently rejoin the success rollups.
LLM_STATUS_OK = "ok"
LLM_STATUS_ERROR = "error"
VALID_STATUSES: frozenset[str] = frozenset({LLM_STATUS_OK, LLM_STATUS_ERROR})

# A gateway error body can embed an entire upstream payload (litellm wraps the
# provider response verbatim), so cap what reaches the ledger.
_ERROR_MESSAGE_MAX = 500


# Canonical field contract for one ``llm_calls.jsonl`` row; the closed-schema
# check compares serialized keys against this set exactly.
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
    """Mint a per-call id for the two halves of one LLM call to share.

    Stamped by the backend that produced the turn (the only place that knows
    the two halves describe the same call) and carried on
    ``BackendTurnResult.metadata`` so both writers pick it up.

    Returns:
        A fresh hex id.
    """
    return uuid.uuid4().hex


# Canonical timestamp helper; kept importable for callers.
_now_iso = now_iso


@dataclass
class LLMCallRecord:
    """One LLM call's worth of token accounting + join keys.

    Required identity / classification:

    * ``session_id`` — cross-process aggregation primary key.
    * ``component`` — producer label; must be in :data:`VALID_COMPONENTS`.

    Optional join keys (filled when the call site has them):

    * ``role`` — reactor role name (in-process reactors only).
    * ``task_id`` / ``dyn_id`` — decision-association keys; carried by
      sub-agents and out-of-process children so the collector can attach a
      call to the decision it served.
    * ``tick`` / ``phase`` — timeline grouping; from ``shared_state``
      in-process, env-passthrough for children, ``ts``-window fallback in
      the collector.
    * ``turn`` — multi-turn sub-agent sequence index.
    * ``model`` — backend model id.

    Token counters (``None`` = not measured / not applicable):

    * ``input_tokens`` / ``output_tokens``
    * ``cache_creation_input_tokens`` / ``cache_read_input_tokens`` —
      ``None`` for backends with no prompt-cache split.

    Terminal status:

    * ``status`` — ``ok`` or ``error`` (:data:`VALID_STATUSES`).
    * ``error_type`` / ``error_message`` — set on ``error`` rows only.

    ``ts`` is filled by :meth:`to_row` at serialization time so a record
    built ahead of the actual write still timestamps the write.
    """

    session_id: str
    component: str
    # Per-call identity shared with the conversation half of the same call, so
    # the two streams pair on the call itself instead of on a ts-second bucket
    # (which splits a call across a second boundary and marries two calls made
    # inside one second). ``None`` when the call site has no id to thread.
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
    # Reasoning models bill hidden reasoning output separately; kept next to the
    # canonical four rather than folded into ``output_tokens``, which counts only
    # the visible reply. ``None`` for backends with no reasoning split.
    reasoning_output_tokens: int | None = None
    # Wall-clock latency of the model call in ms, measured at the call site
    # (None = not measured); the Langfuse generation is placed at
    # ``[ts - latency_ms, ts]``.
    latency_ms: int | None = None
    # Proposal ``msg_id``s this call reviewed (critic only), so the call can be
    # attributed to the decision it served. ``None`` for non-critic producers.
    reviewed_msg_ids: list[str] | None = None
    # An ``error`` row records a call that never produced a usable response, so
    # its token counters stay ``None``; spend rollups must exclude it.
    status: str = LLM_STATUS_OK
    error_type: str | None = None
    error_message: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Serialize to the on-disk row dict, stamping ``ts`` (UTC µs).

        Normalizes identity fields to ``str`` and the four token counters via
        :func:`_coerce_optional_int` so a stray float/numpy scalar never lands
        raw in the ledger.

        Returns:
            The on-disk LLM-call row dict.
        """
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
        """Build a record from a ``BackendTurnResult.metadata`` dict.

        Both :class:`ClaudeBackend` and :class:`CodexBackend` put ``model``,
        ``call_id`` and the token counters on ``metadata`` under identical keys,
        so this one constructor covers every in-process backend call site.
        Missing token keys degrade to ``None`` rather than ``0``.

        Args:
            session_id: Cross-process aggregation primary key.
            component: Producer label; must be in :data:`VALID_COMPONENTS`.
            metadata: Backend turn metadata carrying model + token counters.
            role: Reactor role name, when known.
            task_id: Decision-association task id, when known.
            dyn_id: Dynamic-action id, when known.
            tick: Timeline tick, when known.
            phase: Phase name, when known.
            turn: Multi-turn sub-agent sequence index, when known.
            latency_ms: Measured call latency in ms; overrides
                ``metadata["latency_ms"]`` when not ``None``.

        Returns:
            A populated :class:`LLMCallRecord`.
        """
        md = metadata or {}
        return cls(
            session_id=session_id,
            component=component,
            # Stamped by the backend that produced the metadata, so the
            # conversation half of the same call reads the same id.
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
        """Build an ``error`` record for a call that produced no usable response.

        A failed call has no ``BackendTurnResult``, so :meth:`from_metadata`
        cannot describe it and the token counters stay ``None``. The join keys
        are still known at the call site, which is what lets the failure land on
        the same trace / phase / agent as the successful calls around it.

        Args:
            session_id: Cross-process aggregation primary key.
            component: Producer label; must be in :data:`VALID_COMPONENTS`.
            error: The raised exception, or a pre-formatted message.
            model: Backend model id, when the call site knows it.
            call_id: Per-call id, when the call site minted one.
            role: Reactor role name, when known.
            task_id: Decision-association task id, when known.
            dyn_id: Dynamic-action id, when known.
            tick: Timeline tick, when known.
            phase: Phase name, when known.
            turn: Multi-turn sub-agent sequence index, when known.
            latency_ms: Time spent before failing, when measured.

        Returns:
            A populated :class:`LLMCallRecord` with ``status="error"``.
        """
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
    """Coerce an iterable of ids to a list of non-empty strings, or ``None``.

    Returns ``None`` (stripped from the row) when the input is ``None`` or
    yields no usable ids, so a non-critic row stays free of the field and the
    closed schema only ever sees ``list[str]`` or ``None``.
    """
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
    """Append one validated LLM-call row to the trace ledger.

    The row is serialized via :meth:`LLMCallRecord.to_row` (which stamps
    ``ts``), checked against the closed schema, then appended to
    ``<session_dir>/reports/trace/llm_calls.jsonl``. ``OSError`` while writing
    is logged and swallowed so disk faults never break the optimization loop.

    :class:`LLMTraceRowError` (schema violation) is *not* swallowed: a malformed
    row is a programming error at the call site and must surface in tests.

    In-process producers append directly into the parent's ``llm_calls.jsonl``;
    out-of-process children instead write their own
    ``reports/trace/ext/<component>-<pid>.jsonl`` shard, which the collector and
    Langfuse emitter backfill at read time.

    Args:
        session_dir: Session directory used to resolve the ledger path.
        record: The LLM-call record to serialize and append.

    Raises:
        LLMTraceRowError: If the serialized row violates the closed schema.
    """
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


# Sanity guard: the dataclass fields (minus the write-time ``ts``) must stay in
# lockstep with the on-disk row schema, caught at import.
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
