# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Append-only session trajectory ledger under ``reports/trace/trajectory/``.

Each writing process appends to its own ``<pid>-<nonce>.jsonl`` shard, so two processes never interleave lines on a
shared session dir. Rows form spans: an open row (``queued`` / ``started``) and its terminal row share a ``span_id``;
``point`` rows are instantaneous. The join keys (session dir, component, agent, phase, tick, task, call, parent span)
ride a context variable, so asyncio tasks and ``to_thread`` workers inherit the scope that spawned them.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from hyperloom.common.io import append_jsonl
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import trajectory_dir
from ._row_utils import coerce_optional_int, coerce_optional_str
from .llm_trace import VALID_COMPONENTS as _LLM_COMPONENTS

log = logging.getLogger(__name__)

SCHEMA_VERSION = "hyperloom.trajectory.v1"

STATUS_QUEUED = "queued"
STATUS_STARTED = "started"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_POINT = "point"
OPEN_STATUSES: frozenset[str] = frozenset({STATUS_QUEUED, STATUS_STARTED})
TERMINAL_STATUSES: frozenset[str] = frozenset({STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED})
VALID_STATUSES: frozenset[str] = OPEN_STATUSES | TERMINAL_STATUSES | {STATUS_POINT}

EVENT_SESSION = "session"
EVENT_LLM_CALL = "llm.call"
# One model request inside an LLM call (a tool round trip); its usage is informational and never re-summed into
# spend, which ``llm_calls.jsonl`` owns.
EVENT_LLM_REQUEST = "llm.request"
EVENT_PHASE = "phase"
EVENT_INTENT = "intent"
# span_id is the proposal's bus msg_id: queued at propose time, started when the Critic's verdict is applied.
EVENT_PROPOSAL = "proposal"
# span_id is the task_id: queued / started / terminal follow the TaskRegistry state machine.
EVENT_TASK = "task"
EVENT_TASK_RETRY = "task.retry"
EVENT_TOOL = "tool"
VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EVENT_SESSION,
        EVENT_LLM_CALL,
        EVENT_LLM_REQUEST,
        EVENT_PHASE,
        EVENT_INTENT,
        EVENT_PROPOSAL,
        EVENT_TASK,
        EVENT_TASK_RETRY,
        EVENT_TOOL,
    }
)

VALID_COMPONENTS: frozenset[str] = _LLM_COMPONENTS | {"coordinator"}

_ERROR_MESSAGE_MAX = 500

_ROW_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "event_id",
        "event_type",
        "status",
        "ts",
        "start_ts",
        "writer",
        "seq",
        "session_id",
        "span_id",
        "parent_span_id",
        "component",
        "agent",
        "phase",
        "tick",
        "task_id",
        "call_id",
        "attributes",
    }
)

PhaseTickSource = Callable[[], tuple[str | None, int | None]]


class TrajectoryRowError(ValueError):
    """Raised when a trajectory row violates the closed schema."""


def new_span_id() -> str:
    """Mint an id for one trajectory span."""
    return uuid.uuid4().hex


@dataclass(frozen=True)
class TrajectoryContext:
    """Ambient join keys stamped onto every event recorded inside a scope."""

    session_dir: Path | None = None
    component: str | None = None
    agent: str | None = None
    phase: str | None = None
    tick: int | None = None
    task_id: str | None = None
    call_id: str | None = None
    parent_span_id: str | None = None
    # Read at record time when ``phase`` / ``tick`` are unset, so a long-lived scope follows the live phase machine.
    phase_tick_source: PhaseTickSource | None = field(default=None, compare=False)


_CONTEXT: ContextVar[TrajectoryContext] = ContextVar(
    "hyperloom_trajectory_context",
    default=TrajectoryContext(),
)


def current_context() -> TrajectoryContext:
    """Return the ambient trajectory context."""
    return _CONTEXT.get()


def _overlay(base: TrajectoryContext, fields: dict[str, Any]) -> TrajectoryContext:
    """Return ``base`` with ``fields`` replaced, normalizing ``session_dir`` to a Path."""
    if fields.get("session_dir") is not None:
        fields = {**fields, "session_dir": Path(fields["session_dir"])}
    return replace(base, **fields) if fields else base


@contextmanager
def trajectory_scope(**fields: Any) -> Iterator[TrajectoryContext]:
    """Overlay ``fields`` onto the ambient trajectory context for the enclosed work."""
    ctx = _overlay(_CONTEXT.get(), fields)
    token = _CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CONTEXT.reset(token)


class _ShardWriter:
    """Per-process shard identity and sequence; re-minted in a forked child."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.writer = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._seq = itertools.count()
        self.lock = threading.Lock()

    def next_seq(self) -> int:
        return next(self._seq)


_WRITER = _ShardWriter()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_WRITER.reset)


def _jsonable_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Round-trip ``attributes`` through JSON so the row is always serializable."""
    if attributes is None:
        return {}
    if not isinstance(attributes, dict):
        raise TrajectoryRowError(f"trajectory attributes must be a dict; got {type(attributes).__name__}")
    return json.loads(json.dumps(attributes, default=str, sort_keys=True))


def _resolved_phase_tick(ctx: TrajectoryContext) -> tuple[str | None, int | None]:
    """Return the context's phase/tick, falling back to its live source."""
    phase, tick = ctx.phase, ctx.tick
    if (phase is None or tick is None) and ctx.phase_tick_source is not None:
        live_phase, live_tick = ctx.phase_tick_source()
        phase = live_phase if phase is None else phase
        tick = live_tick if tick is None else tick
    return coerce_optional_str(phase), coerce_optional_int(tick)


def _validate_row(row: dict[str, Any]) -> None:
    """Fail fast if ``row`` deviates from the closed trajectory schema."""
    keys = set(row)
    if keys != _ROW_FIELDS:
        raise TrajectoryRowError(
            f"trajectory row violates closed schema: extra={sorted(keys - _ROW_FIELDS)!r} "
            f"missing={sorted(_ROW_FIELDS - keys)!r}"
        )
    for key, vocabulary in (
        ("event_type", VALID_EVENT_TYPES),
        ("status", VALID_STATUSES),
        ("component", VALID_COMPONENTS),
    ):
        value = row[key]
        if key == "component" and value is None:
            continue
        if value not in vocabulary:
            raise TrajectoryRowError(f"trajectory {key}={value!r} is not one of {sorted(vocabulary)!r}")


def _build_row(
    ctx: TrajectoryContext,
    *,
    event_type: str,
    status: str,
    span_id: str,
    start_ts: str | None,
    ts: str | None,
    attributes: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble one closed-schema row from the resolved context."""
    phase, tick = _resolved_phase_tick(ctx)
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "event_type": event_type,
        "status": status,
        "ts": ts or now_iso(),
        "start_ts": coerce_optional_str(start_ts),
        "writer": _WRITER.writer,
        "seq": _WRITER.next_seq(),
        "session_id": ctx.session_dir.name if ctx.session_dir is not None else "",
        "span_id": span_id,
        "parent_span_id": coerce_optional_str(ctx.parent_span_id),
        "component": coerce_optional_str(ctx.component),
        "agent": coerce_optional_str(ctx.agent),
        "phase": phase,
        "tick": tick,
        "task_id": coerce_optional_str(ctx.task_id),
        "call_id": coerce_optional_str(ctx.call_id),
        "attributes": _jsonable_attributes(attributes),
    }


def record_event(
    event_type: str,
    *,
    status: str = STATUS_POINT,
    span_id: str | None = None,
    start_ts: str | None = None,
    ts: str | None = None,
    attributes: dict[str, Any] | None = None,
    **context: Any,
) -> str | None:
    """Append one event to this process's trajectory shard.

    ``ts`` defaults to now; pass it when the event is written after it happened. ``context`` overrides
    :class:`TrajectoryContext` fields for this row only. A no-op (returning ``None``) when no ``session_dir`` is in
    scope; otherwise returns the row's ``span_id``, minted when not given.
    """
    ctx = _overlay(_CONTEXT.get(), context)
    if ctx.session_dir is None:
        return None
    span_id = span_id or new_span_id()
    with _WRITER.lock:
        row = _build_row(
            ctx,
            event_type=event_type,
            status=status,
            span_id=span_id,
            start_ts=start_ts,
            ts=ts,
            attributes=attributes,
        )
        _validate_row(row)
        dest = trajectory_dir(ctx.session_dir) / f"{row['writer']}.jsonl"
        try:
            append_jsonl(dest, row, make_parents=True, sort_keys=True)
        except OSError as exc:
            log.warning("trajectory: append failed for event_type=%s: %r", event_type, exc)
    return span_id


class TrajectorySpan:
    """Handle to an open span; :meth:`finish` sets the terminal status and attributes."""

    def __init__(self, span_id: str) -> None:
        self.span_id = span_id
        self.status: str | None = None
        self.attributes: dict[str, Any] = {}

    def finish(self, status: str = STATUS_COMPLETED, **attributes: Any) -> None:
        """Record the terminal status the span closes with (default ``completed``)."""
        if status not in TERMINAL_STATUSES:
            raise TrajectoryRowError(f"span terminal status={status!r} is not one of {sorted(TERMINAL_STATUSES)!r}")
        self.status = status
        self.attributes.update(attributes)


_LLM_CALL_SUMMARY_KEYS: tuple[str, ...] = (
    "model",
    "stop_reason",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_output_tokens",
    "context_tokens_peak",
)


def llm_call_summary(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """The ``llm.call`` terminal attributes read off a backend turn's metadata (text fields excluded)."""
    md = metadata or {}
    return {key: md[key] for key in _LLM_CALL_SUMMARY_KEYS if md.get(key) is not None}


def scalar_attributes(mapping: dict[str, Any] | None, *, max_chars: int = 200) -> dict[str, Any]:
    """The scalar entries of ``mapping`` with strings clipped, for attributes read off free-form evidence."""
    out: dict[str, Any] = {}
    for key, value in (mapping or {}).items():
        if isinstance(value, str):
            out[str(key)] = value[:max_chars]
        elif value is None or isinstance(value, (bool, int, float)):
            out[str(key)] = value
    return out


def _error_attributes(exc: BaseException) -> dict[str, Any]:
    return {"error_type": type(exc).__name__, "error_message": str(exc)[:_ERROR_MESSAGE_MAX]}


@contextmanager
def trajectory_span(
    event_type: str,
    *,
    attributes: dict[str, Any] | None = None,
    span_id: str | None = None,
    **context: Any,
) -> Iterator[TrajectorySpan]:
    """Record a ``started`` row, scope the body as its child, then record the terminal row.

    The body raising closes the span ``failed`` (``cancelled`` for :class:`asyncio.CancelledError`) and re-raises.
    """
    span = TrajectorySpan(span_id or new_span_id())
    start_ts = now_iso()
    record_event(event_type, status=STATUS_STARTED, span_id=span.span_id, attributes=attributes, **context)
    status = STATUS_FAILED
    error: dict[str, Any] = {}
    try:
        with trajectory_scope(**{**context, "parent_span_id": span.span_id}):
            yield span
    except asyncio.CancelledError as exc:
        status, error = STATUS_CANCELLED, _error_attributes(exc)
        raise
    except BaseException as exc:
        status, error = STATUS_FAILED, _error_attributes(exc)
        raise
    finally:
        record_event(
            event_type,
            status=status if error else (span.status or STATUS_COMPLETED),
            span_id=span.span_id,
            start_ts=start_ts,
            attributes={**span.attributes, **error},
            **context,
        )


def trajectory_shards(session_dir: Path) -> list[Path]:
    """Return every trajectory shard of a session, sorted by name."""
    directory = trajectory_dir(Path(session_dir))
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.jsonl"))


def load_shard(path: Path) -> list[dict[str, Any]]:
    """Read one shard, skipping a torn trailing line from a live writer."""
    from hyperloom.common.jsonio import read_jsonl

    return read_jsonl(path, require_dict=True, skip_malformed=True, skip_non_dict=True)


def load_events(session_dir: Path) -> list[dict[str, Any]]:
    """Return every trajectory row of a session in ``(ts, writer, seq)`` order."""
    rows = [row for shard in trajectory_shards(session_dir) for row in load_shard(shard)]
    return sorted(rows, key=lambda r: (str(r.get("ts") or ""), str(r.get("writer") or ""), int(r.get("seq") or 0)))


__all__ = [
    "EVENT_INTENT",
    "EVENT_LLM_CALL",
    "EVENT_LLM_REQUEST",
    "EVENT_PHASE",
    "EVENT_PROPOSAL",
    "EVENT_SESSION",
    "EVENT_TASK",
    "EVENT_TASK_RETRY",
    "EVENT_TOOL",
    "OPEN_STATUSES",
    "SCHEMA_VERSION",
    "STATUS_CANCELLED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_POINT",
    "STATUS_QUEUED",
    "STATUS_STARTED",
    "TERMINAL_STATUSES",
    "TrajectoryContext",
    "TrajectoryRowError",
    "TrajectorySpan",
    "VALID_COMPONENTS",
    "VALID_EVENT_TYPES",
    "VALID_STATUSES",
    "current_context",
    "llm_call_summary",
    "load_events",
    "load_shard",
    "new_span_id",
    "record_event",
    "scalar_attributes",
    "trajectory_scope",
    "trajectory_shards",
    "trajectory_span",
]
