# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Closed-schema writer for ``reports/trace/llm_calls_detail.jsonl``.

The sidecar to :mod:`llm_trace`. That ledger stays one row per agentic *turn*,
because the breakdown rollup, the CI session report and the Langfuse mirror all
count its rows; changing what a row means would silently restate every published
total. A turn, however, is often several real API calls -- an agentic backend
loops tool-use round trips inside one ``run`` -- and the questions this
instrumentation exists to answer (what did each call cost, how long was its
first token, how much of it was thinking) are per call, not per turn.

So the per-call rows live here, joined to their turn on ``call_id`` and ordered
within it by ``api_call_index``. Nothing reads this file to compute a total that
already has a published value.

``isl`` and ``osl`` are derived once, here, so every consumer means the same
thing by them: ISL counts everything the model read (fresh input plus both cache
halves), OSL everything it wrote (visible reply plus hidden reasoning).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from hyperloom.common.io import append_jsonl
from hyperloom.common.llm_attribution import TASK_PATH_SEPARATOR
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import llm_call_detail_path
from .llm_trace import (
    LLM_STATUS_OK,
    VALID_COMPONENTS,
    VALID_STATUSES,
    resolve_task_path,
)
from .pricing import resolve_cost
from ._row_utils import (
    coerce_optional_float as _coerce_optional_float,
    coerce_optional_int as _coerce_optional_int,
    coerce_optional_str as _coerce_optional_str,
    validate_closed_row,
)

log = logging.getLogger(__name__)

# A transcript can carry a pathological number of tool blocks in one reply; cap
# what reaches the ledger so a single row cannot dominate the file.
_TOOL_CALLS_MAX = 200
_TOOL_ARG_MAX = 200

#: Values for a row's ``timing_source``; see :class:`CallDetailRecord`.
TIMING_MEASURED = "measured"
TIMING_APPORTIONED = "apportioned"


_ROW_FIELDS: frozenset[str] = frozenset(
    {
        "session_id",
        "ts",
        "started_ts",
        "component",
        "call_id",
        "api_call_index",
        "parent_call_id",
        "role",
        "task_id",
        "dyn_id",
        "tick",
        "phase",
        "turn",
        "task_path",
        "task_depth",
        "model",
        "isl",
        "osl",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "reasoning_output_tokens",
        "latency_ms",
        "ttft_ms",
        "thinking_ms",
        "output_ms",
        "timing_source",
        "cost_usd",
        "cost_input_usd",
        "cost_output_usd",
        "cost_thinking_usd",
        "cost_cache_usd",
        "cost_source",
        "tool_calls",
        "tool_call_count",
        "stop_reason",
        "status",
        "error_type",
        "error_message",
    }
)


class CallDetailRowError(ValueError):
    """Raised when a per-API-call row violates the closed schema."""


def _tool_call_entry(entry: Any) -> dict[str, Any] | None:
    """Project one tool-call record onto the fields the ledger keeps.

    Args:
        entry: A tool-call mapping from a backend or a transcript parser.

    Returns:
        The projected entry, or ``None`` when it names no tool.
    """
    if not isinstance(entry, Mapping):
        name = _coerce_optional_str(entry)
        return {"tool": name} if name else None
    tool = _coerce_optional_str(entry.get("tool") or entry.get("name"))
    if not tool:
        return None
    out: dict[str, Any] = {"tool": tool}
    query = _coerce_optional_str(entry.get("query"))
    if query:
        out["query"] = query[:_TOOL_ARG_MAX]
    for key in ("tool_use_id", "ts"):
        value = _coerce_optional_str(entry.get(key))
        if value:
            out[key] = value
    for key in ("turn_index", "duration_ms"):
        value = _coerce_optional_int(entry.get(key))
        if value is not None:
            out[key] = value
    return out


def _coerce_tool_calls(value: Any) -> list[dict[str, Any]] | None:
    """Normalize a backend's tool-call list for the ledger, or ``None``."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, Mapping)):
        items: Sequence[Any] = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            return None
    out = [e for e in (_tool_call_entry(i) for i in items[:_TOOL_CALLS_MAX]) if e]
    return out or None


def _sum_or_none(*values: int | None) -> int | None:
    """Sum the counters that were measured, or ``None`` when none were.

    ``None`` means "not measured" throughout the trace layer, so a sum of
    nothing must stay ``None`` rather than becoming a confident zero.
    """
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def split_turn_timing(
    *,
    latency_ms: int | None,
    output_tokens: int | None,
    reasoning_output_tokens: int | None,
    thinking_ms: int | None = None,
    output_ms: int | None = None,
) -> tuple[int | None, int | None, str | None]:
    """Resolve a call's thinking/output time split and say where it came from.

    A backend that saw partial stream events can time the two halves directly;
    pass them and they are returned as ``TIMING_MEASURED``. Without them the
    span is divided in proportion to the reasoning and visible output token
    counts -- the model writes both at roughly one rate, so the proportion is a
    fair estimate, but it is an estimate and is labelled ``TIMING_APPORTIONED``
    so a report never presents it as a stopwatch reading.

    Args:
        latency_ms: The measured span of the call.
        output_tokens: Visible output tokens the call produced.
        reasoning_output_tokens: Hidden reasoning tokens the call produced.
        thinking_ms: Directly measured reasoning time, when available.
        output_ms: Directly measured visible-output time, when available.

    Returns:
        ``(thinking_ms, output_ms, timing_source)``; the source is ``None``
        when neither half could be established.
    """
    measured_think = _coerce_optional_int(thinking_ms)
    measured_out = _coerce_optional_int(output_ms)
    if measured_think is not None or measured_out is not None:
        return measured_think, measured_out, TIMING_MEASURED
    span = _coerce_optional_int(latency_ms)
    think_tok = _coerce_optional_int(reasoning_output_tokens)
    out_tok = _coerce_optional_int(output_tokens)
    if span is None or span < 0 or think_tok is None or out_tok is None:
        return None, None, None
    total_tok = think_tok + out_tok
    if total_tok <= 0:
        return None, None, None
    think_span = int(round(span * think_tok / total_tok))
    return think_span, span - think_span, TIMING_APPORTIONED


@dataclass
class CallDetailRecord:
    """One real LLM API call.

    Carries the same join keys as its turn's :class:`llm_trace.LLMCallRecord`
    plus ``api_call_index``, its position within that turn.
    """

    session_id: str
    component: str
    call_id: str | None = None
    api_call_index: int | None = None
    # The turn-level ``call_id`` when this row was produced by a nested agent
    # whose own calls carry a different id -- a GEAK sub-agent, say. ``None``
    # when the call is a direct child of its turn.
    parent_call_id: str | None = None
    role: str | None = None
    task_id: str | None = None
    dyn_id: str | None = None
    tick: int | None = None
    phase: str | None = None
    turn: int | None = None
    task_path: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    latency_ms: int | None = None
    ttft_ms: int | None = None
    thinking_ms: int | None = None
    output_ms: int | None = None
    # How the thinking/output split above was arrived at. ``latency_ms`` is
    # always a measurement; the split is only one when the backend saw partial
    # stream events. ``TIMING_APPORTIONED`` means the measured span was divided
    # in proportion to the reasoning and visible output token counts, which is
    # an estimate and must not be reported as a stopwatch reading.
    timing_source: str | None = None
    # The provider's own charge for this call when it reported one; it wins
    # over the rate card and is never mixed with a derived figure.
    total_cost_usd: float | None = None
    tool_calls: list[dict[str, Any]] | None = None
    stop_reason: str | None = None
    # When the call started. Detail rows are usually written in a batch once
    # the turn ends, so the write-time ``ts`` cannot order them.
    started_ts: str | None = None
    status: str = LLM_STATUS_OK
    error_type: str | None = None
    error_message: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Serialize to the on-disk row dict, deriving ISL/OSL and cost.

        Returns:
            The on-disk per-API-call row dict.
        """
        in_tok = _coerce_optional_int(self.input_tokens)
        out_tok = _coerce_optional_int(self.output_tokens)
        write_tok = _coerce_optional_int(self.cache_creation_input_tokens)
        read_tok = _coerce_optional_int(self.cache_read_input_tokens)
        think_tok = _coerce_optional_int(self.reasoning_output_tokens)
        tokens = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_creation_input_tokens": write_tok,
            "cache_read_input_tokens": read_tok,
            "reasoning_output_tokens": think_tok,
        }
        cost = resolve_cost(
            model=self.model,
            tokens=tokens,
            provider_usd=_coerce_optional_float(self.total_cost_usd),
        )
        path = resolve_task_path(self.task_path)
        tool_calls = _coerce_tool_calls(self.tool_calls)
        return {
            "session_id": str(self.session_id),
            "ts": now_iso(),
            "started_ts": _coerce_optional_str(self.started_ts),
            "component": str(self.component),
            "call_id": _coerce_optional_str(self.call_id),
            "api_call_index": _coerce_optional_int(self.api_call_index),
            "parent_call_id": _coerce_optional_str(self.parent_call_id),
            "role": _coerce_optional_str(self.role),
            "task_id": _coerce_optional_str(self.task_id),
            "dyn_id": _coerce_optional_str(self.dyn_id),
            "tick": _coerce_optional_int(self.tick),
            "phase": _coerce_optional_str(self.phase),
            "turn": _coerce_optional_int(self.turn),
            "task_path": path or None,
            "task_depth": path.count(TASK_PATH_SEPARATOR) + 1 if path else None,
            "model": _coerce_optional_str(self.model),
            "isl": _sum_or_none(in_tok, read_tok, write_tok),
            "osl": _sum_or_none(out_tok, think_tok),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_creation_input_tokens": write_tok,
            "cache_read_input_tokens": read_tok,
            "reasoning_output_tokens": think_tok,
            "latency_ms": _coerce_optional_int(self.latency_ms),
            "ttft_ms": _coerce_optional_int(self.ttft_ms),
            "thinking_ms": _coerce_optional_int(self.thinking_ms),
            "output_ms": _coerce_optional_int(self.output_ms),
            "timing_source": _coerce_optional_str(self.timing_source),
            "cost_usd": cost.total_usd,
            "cost_input_usd": cost.input_usd,
            "cost_output_usd": cost.output_usd,
            "cost_thinking_usd": cost.thinking_usd,
            "cost_cache_usd": cost.cache_usd,
            "cost_source": cost.source,
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls) if tool_calls is not None else None,
            "stop_reason": _coerce_optional_str(self.stop_reason),
            "status": str(self.status),
            "error_type": _coerce_optional_str(self.error_type),
            "error_message": _coerce_optional_str(self.error_message),
        }


def append_call_detail(
    *,
    session_dir: Path,
    record: CallDetailRecord,
    dest: Path | None = None,
) -> None:
    """Append one validated per-API-call row to the sidecar ledger.

    Mirrors :func:`llm_trace.append_llm_call`: the row is validated against the
    closed schema (a violation raises, because it is a call-site bug), and
    ``OSError`` while writing is logged and swallowed, because instrumentation
    must never break the optimization loop.

    Args:
        session_dir: Session directory used to resolve the ledger path.
        record: The per-API-call record to serialize and append.
        dest: Write here instead of the session's sidecar -- used by
            out-of-process producers, which own their own ``ext/`` shard
            because appending into the shared file is not atomic across
            processes.

    Raises:
        CallDetailRowError: If the serialized row violates the closed schema.
    """
    row = record.to_row()
    validate_closed_row(
        row,
        fields=_ROW_FIELDS,
        valid_components=VALID_COMPONENTS,
        error_cls=CallDetailRowError,
        label="llm_calls_detail",
    )
    status = row.get("status")
    if status not in VALID_STATUSES:
        raise CallDetailRowError(f"llm_calls_detail row 'status'={status!r} is not one of {sorted(VALID_STATUSES)!r}")
    target = dest if dest is not None else llm_call_detail_path(session_dir)
    try:
        append_jsonl(target, row, make_parents=True, sort_keys=True)
    except OSError as exc:
        log.warning(
            "call_detail: append failed for component=%s session_id=%s: %r",
            record.component,
            record.session_id,
            exc,
        )


#: Backend metadata key holding the per-API-call entries of one turn.
METADATA_DETAIL_KEY = "api_call_details"

_TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_output_tokens",
)


def _has_token_counts(metadata: Mapping[str, Any]) -> bool:
    """Whether turn metadata carries any token counter worth a stand-in row."""
    return any(metadata.get(key) is not None for key in _TOKEN_KEYS)


def _resolve_call_index(entry: Mapping[str, Any], position: int) -> int | None:
    """Position of a call within its turn, or ``None`` for a whole-turn row.

    An entry that names no index but sets the key to ``None`` is the stand-in
    for a backend that reports per-turn totals only; one that omits the key
    entirely is a real call that simply did not number itself.
    """
    if "api_call_index" in entry and entry.get("api_call_index") is None:
        return None
    stated = _coerce_optional_int(entry.get("api_call_index"))
    return position if stated is None else stated


def append_turn_call_details(
    *,
    session_dir: Path,
    session_id: str,
    component: str,
    metadata: Mapping[str, Any] | None,
    role: str | None = None,
    task_id: str | None = None,
    dyn_id: str | None = None,
    tick: int | None = None,
    phase: str | None = None,
    turn: int | None = None,
    task_path: str | None = None,
    turn_latency_ms: int | None = None,
    dest: Path | None = None,
) -> int:
    """Write the sidecar rows for one turn from its backend metadata.

    The backend collects the per-call entries but knows nothing about the
    session; the caller that writes the turn row holds that context and so
    writes these too, with the same join keys.

    A backend whose SDK reports only per-turn totals gets one stand-in row
    carrying those totals, with ``api_call_index`` left unset to say the row
    covers a whole turn rather than a positioned call inside one. That keeps
    the sidecar a complete account of spend; the turn row's ``api_calls``
    column stays ``None``, which is how a report tells "one call" from "call
    count not reported".

    Args:
        session_dir: Session directory used to resolve the sidecar path.
        session_id: Session id stamped on every row.
        component: Trace component the turn belongs to.
        metadata: The turn's ``BackendTurnResult.metadata``.
        role: Role that ran the turn.
        task_id: Task id in scope, when the caller has one.
        dyn_id: Dynamic-agent id in scope, when the caller has one.
        tick: Loop tick in scope.
        phase: Phase in scope.
        turn: Turn index within the task.
        task_path: Explicit task path; defaults to the ambient scope.
        turn_latency_ms: The turn's measured wall-clock, used for any entry
            that reports none of its own.
        dest: Write here instead of the session's sidecar (shard writers).

    Returns:
        The number of rows written.
    """
    md = metadata or {}
    entries = md.get(METADATA_DETAIL_KEY)
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        entries = [dict(md, api_call_index=None)] if _has_token_counts(md) else []
    call_id = _coerce_optional_str(md.get("call_id"))
    written = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            continue
        latency_ms = _coerce_optional_int(entry.get("latency_ms"))
        if latency_ms is None:
            latency_ms = _coerce_optional_int(turn_latency_ms)
        out_tok = _coerce_optional_int(entry.get("output_tokens"))
        think_tok = _coerce_optional_int(entry.get("reasoning_output_tokens"))
        think_ms, out_ms, timing_source = split_turn_timing(
            latency_ms=latency_ms,
            output_tokens=out_tok,
            reasoning_output_tokens=think_tok,
            thinking_ms=entry.get("thinking_ms"),
            output_ms=entry.get("output_ms"),
        )
        record = CallDetailRecord(
            session_id=session_id,
            component=component,
            call_id=call_id,
            api_call_index=_resolve_call_index(entry, index),
            role=role,
            task_id=task_id,
            dyn_id=dyn_id,
            tick=tick,
            phase=phase,
            turn=turn,
            task_path=task_path,
            model=entry.get("model"),
            input_tokens=entry.get("input_tokens"),
            output_tokens=out_tok,
            cache_creation_input_tokens=entry.get("cache_creation_input_tokens"),
            cache_read_input_tokens=entry.get("cache_read_input_tokens"),
            reasoning_output_tokens=think_tok,
            latency_ms=latency_ms,
            ttft_ms=entry.get("ttft_ms"),
            thinking_ms=think_ms,
            output_ms=out_ms,
            timing_source=timing_source,
            total_cost_usd=entry.get("total_cost_usd"),
            tool_calls=entry.get("tool_calls"),
            stop_reason=entry.get("stop_reason"),
            started_ts=entry.get("started_ts"),
        )
        append_call_detail(session_dir=session_dir, record=record, dest=dest)
        written += 1
    return written


# The dataclass carries the raw inputs; the row carries what is derived from
# them. Guard the derived-only keys explicitly so a field added to one and not
# the other is caught at import, the way llm_trace's drift assert does.
_DERIVED_ONLY: frozenset[str] = frozenset(
    {
        "ts",
        "task_depth",
        "isl",
        "osl",
        "tool_call_count",
        "cost_usd",
        "cost_input_usd",
        "cost_output_usd",
        "cost_thinking_usd",
        "cost_cache_usd",
        "cost_source",
    }
)
_INPUT_ONLY: frozenset[str] = frozenset({"total_cost_usd"})
_DATACLASS_FIELDS: frozenset[str] = frozenset(f.name for f in fields(CallDetailRecord))
assert (_DATACLASS_FIELDS - _INPUT_ONLY) | _DERIVED_ONLY == _ROW_FIELDS, (
    f"CallDetailRecord fields drifted from _ROW_FIELDS: dataclass={sorted(_DATACLASS_FIELDS)} row={sorted(_ROW_FIELDS)}"
)


__all__ = [
    "METADATA_DETAIL_KEY",
    "TIMING_APPORTIONED",
    "TIMING_MEASURED",
    "CallDetailRecord",
    "CallDetailRowError",
    "append_call_detail",
    "append_turn_call_details",
    "split_turn_timing",
]
