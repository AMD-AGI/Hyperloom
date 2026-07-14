# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Closed-schema writer for ``reports/trace/llm_calls.jsonl``.

One module owns the canonical field contract for a single LLM call so
every producer — the orchestration/kernel reactor, specialist sub-agents,
and the Codex/critic/scorer inference steps — emits rows that the collector
can join without guessing.

Design contract:

* **Closed schema**: a row carrying an unknown field — or missing a
  required one — fails fast (:class:`LLMTraceRowError`) so a buggy call
  site cannot silently pollute the audit stream.
* **Best-effort I/O**: disk failures while appending are logged and
  swallowed; trace writes must never break the optimization loop.
* **Token shape**: the four counters mirror the keys both
  :class:`ClaudeBackend` and :class:`CodexBackend` put on
  ``BackendTurnResult.metadata``. Backends without a prompt-cache split
  (OpenAI / GEAK) report ``None`` for the two ``cache_*`` counters so the
  collector can tell "no cache concept" from "zero cache hits".

The record is intentionally a dataclass (not a TypedDict) so call sites
get constructor-time field checking and a single :meth:`to_row`
serialization path; the closed-schema check in :func:`append_llm_call`
is a second guard that also covers rows rebuilt from raw dicts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import llm_calls_path
from ._row_utils import (
    coerce_optional_int as _coerce_optional_int,
    coerce_optional_str as _coerce_optional_str,
    validate_closed_row,
)

log = logging.getLogger(__name__)


# Components that may legitimately appear in a trace row. Kept as a closed
# vocabulary so a typo'd ``component=`` at a call site is caught instead of
# fragmenting the per-component rollup. Add a new producer label here
# deliberately when one lands.
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
        "geak_v3",
        "oob",
        "forge",
        "tracelens",
        "breakdown",
    }
)


# Canonical, ordered field contract for one ``llm_calls.jsonl`` row. The
# closed-schema check compares serialized keys against this set exactly.
_ROW_FIELDS: frozenset[str] = frozenset(
    {
        "session_id",
        "ts",
        "component",
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
        "latency_ms",
        "reviewed_msg_ids",
        "resume_downgraded",
    }
)


class LLMTraceRowError(ValueError):
    """Raised when an LLM-call row violates the closed schema."""


# microseconds + ``+00:00`` (canonical helper; kept importable for callers).
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

    ``ts`` is filled by :meth:`to_row` at serialization time so a record
    built ahead of the actual write still timestamps the write.
    """

    session_id: str
    component: str
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
    # Wall-clock latency of the model call in milliseconds, measured at the
    # call site (None = not measured). Lets the trace report a real per-call
    # duration instead of a zero-width point: the Langfuse generation is
    # placed at ``[ts - latency_ms, ts]`` (``ts`` is the post-call write time).
    latency_ms: int | None = None
    # Proposal ``msg_id``s this call reviewed (critic only). The proposal that
    # gets approved is materialized into a task whose ``task_id`` the collector
    # recovers via ``proposal_task_map.jsonl``, so a critic call that reviewed a
    # single materialized proposal can be attributed to that decision instead of
    # falling into the overhead bucket. ``None`` for every non-critic producer.
    reviewed_msg_ids: list[str] | None = None
    # True when a conversational turn's ``resume=`` was rejected by the SDK and
    # dropped to a stateless call (Claude only); ``None`` for every other case.
    resume_downgraded: bool | None = None

    def to_row(self) -> dict[str, Any]:
        """Serialize to the on-disk row dict, stamping ``ts`` (UTC µs).

        Normalizes the identity fields to ``str`` and the four token
        counters via :func:`_coerce_optional_int` so a stray float / numpy
        scalar from an SDK ``usage`` object never lands raw in the ledger.

        Returns:
            The on-disk LLM-call row dict.
        """
        return {
            "session_id": str(self.session_id),
            "ts": _now_iso(),
            "component": str(self.component),
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
            "latency_ms": _coerce_optional_int(self.latency_ms),
            "reviewed_msg_ids": _coerce_optional_str_list(self.reviewed_msg_ids),
            "resume_downgraded": self.resume_downgraded,
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

        Both :class:`ClaudeBackend` and :class:`CodexBackend` put ``model``
        and the four token counters on ``metadata`` under identical keys, so
        this one constructor covers every in-process backend call site
        (orchestration/kernel, dynamic_action, specialist fallback).
        Missing token keys degrade to ``None`` rather than ``0`` so the
        collector can distinguish "unreported" from "reported zero".

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

        Returns:
            A populated :class:`LLMCallRecord`.
        """
        md = metadata or {}
        return cls(
            session_id=session_id,
            component=component,
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
            latency_ms=latency_ms if latency_ms is not None else md.get("latency_ms"),
            resume_downgraded=md.get("resume_downgraded"),
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
    ``ts``), checked against the closed schema, then atomically appended to
    ``<session_dir>/reports/trace/llm_calls.jsonl``. ``OSError`` while writing
    is logged and swallowed so a full disk or a permissions glitch never breaks
    the optimization loop — exactly the fault posture of
    :func:`..dynamic_action_history.append_dispatch_history_row`.

    :class:`LLMTraceRowError` (schema violation) is *not* swallowed: a
    malformed row is a programming error at the call site, not a runtime
    disk condition, and must surface in tests.

    In-process producers append directly into the parent's
    ``llm_calls.jsonl``. Out-of-process children instead write their own
    ``reports/trace/ext/<component>-<pid>.jsonl`` shard;
    the collector and Langfuse emitter backfill those shards at read time, so
    this function intentionally has no shard-target override.

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
    dest = llm_calls_path(session_dir)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as exc:
        log.warning(
            "llm_trace: append failed for component=%s session_id=%s: %r",
            record.component,
            record.session_id,
            exc,
        )

    # Second sink (opt-in): mirror the call to Langfuse live. Best-effort;
    # never raises into the ledger path.
    try:
        from .langfuse_emitter import get_emitter

        get_emitter(session_dir).record_llm_call(row)
    except Exception:  # noqa: BLE001 — Langfuse must never break the ledger
        log.debug("llm_trace: langfuse mirror failed", exc_info=True)


# Sanity guard: the dataclass fields (minus the write-time ``ts``) must
# stay in lockstep with the on-disk row schema. A drift here means a new
# field was added to one side only — caught at import, not at runtime.
_DATACLASS_FIELDS: frozenset[str] = frozenset(f.name for f in fields(LLMCallRecord))
assert _DATACLASS_FIELDS | {"ts"} == _ROW_FIELDS, (
    f"LLMCallRecord fields drifted from _ROW_FIELDS: dataclass={sorted(_DATACLASS_FIELDS)} row={sorted(_ROW_FIELDS)}"
)


__all__ = [
    "LLMCallRecord",
    "LLMTraceRowError",
    "VALID_COMPONENTS",
    "append_llm_call",
]
