# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Closed-schema writer for ``reports/trace/llm_calls.jsonl``.

One module owns the canonical field contract for a single LLM call so
every producer — the orchestration/kernel reactor, dynamic_action and
specialist sub-agents, the Codex/critic/scorer inference steps, and the
out-of-process children that write their own ``ext/*.jsonl`` shards — emits
rows that the collector can join without guessing.

Design contract (FULL_TRACE_DESIGN §3.1, §4):

* **Closed schema**: a row carrying an unknown field — or missing a
  required one — fails fast (:class:`LLMTraceRowError`) so a buggy call
  site cannot silently pollute the audit stream.
* **Best-effort I/O**: disk failures while appending are logged and
  swallowed; trace writes must never break the optimization loop. This
  mirrors :mod:`..dynamic_action_history`.
* **Token shape**: the four counters mirror the keys both
  :class:`ClaudeBackend` and :class:`CodexBackend` put on
  ``BackendTurnResult.metadata``. Backends without a prompt-cache split
  (OpenAI / GEAK) report ``None`` for the two ``cache_*`` counters so the
  collector can tell "no cache concept" from "zero cache hits".

The record is intentionally a dataclass (not a TypedDict) so call sites
get constructor-time field checking and a single :meth:`to_row`
serialization path; the closed-schema check in :func:`append_llm_call`
is a second guard that also covers rows rebuilt from raw dicts (e.g. the
out-of-process ``ext/*.jsonl`` ingestion path).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...session_paths import llm_calls_path

log = logging.getLogger(__name__)


# Components that may legitimately appear in a trace row. Kept as a closed
# vocabulary so a typo'd ``component=`` at a call site is caught instead of
# fragmenting the per-component rollup. Extend this set deliberately when a
# new producer lands (P1/P2 add specialist subprocess parsing, geak, oob,
# robustness, tracelens, breakdown).
VALID_COMPONENTS: frozenset[str] = frozenset({
    "orchestration",
    "kernel",
    "dynamic_action",
    "specialist",
    "critic",
    "robustness",
    "proposal_scorer",
    "geak",
    "oob",
    "tracelens",
    "breakdown",
})


# Canonical, ordered field contract for one ``llm_calls.jsonl`` row. The
# closed-schema check compares serialized keys against this set exactly.
_ROW_FIELDS: frozenset[str] = frozenset({
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
})


class LLMTraceRowError(ValueError):
    """Raised when an LLM-call row violates the closed schema."""


def _now_iso() -> str:
    """Return the current UTC time as a microsecond-precision ISO string.

    Returns:
        ISO 8601 timestamp in UTC.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


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
            "cache_creation_input_tokens": _coerce_optional_int(
                self.cache_creation_input_tokens
            ),
            "cache_read_input_tokens": _coerce_optional_int(
                self.cache_read_input_tokens
            ),
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
    ) -> "LLMCallRecord":
        """Build a record from a ``BackendTurnResult.metadata`` dict.

        Both :class:`ClaudeBackend` and :class:`CodexBackend` put ``model``
        and the four token counters on ``metadata`` under identical keys, so
        this one constructor covers every in-process backend call site (A1
        orchestration/kernel, A2 dynamic_action, A3 specialist fallback).
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
        )


def _coerce_optional_str(value: Any) -> str | None:
    """Coerce a value to a non-empty stripped string or ``None``.

    Args:
        value: Arbitrary value to normalize.

    Returns:
        The stripped string, or ``None`` if it is empty or ``None``.
    """
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _coerce_optional_int(value: Any) -> int | None:
    """Coerce a token / index value to int, or ``None`` on miss/bad type.

    Unlike the backends' ``_safe_int`` (which floors to 0), this keeps
    ``None`` distinct from ``0``: a backend that does not report a counter
    must not be indistinguishable from one that genuinely spent zero.

    Args:
        value: Arbitrary token / index value.

    Returns:
        The integer value, or ``None`` on a miss or bad type.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_row(row: dict[str, Any]) -> None:
    """Fail fast if ``row`` deviates from the closed schema.

    Args:
        row: A serialized LLM-call row dict.

    Raises:
        LLMTraceRowError: If the row has extra/missing fields, an empty
            ``session_id``, or an unknown ``component``.
    """
    keys = set(row.keys())
    extra = sorted(keys - _ROW_FIELDS)
    missing = sorted(_ROW_FIELDS - keys)
    if extra or missing:
        raise LLMTraceRowError(
            f"llm_calls row violates closed schema: "
            f"extra={extra!r} missing={missing!r}"
        )
    session_id = row.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise LLMTraceRowError(
            f"llm_calls row requires a non-empty 'session_id'; got "
            f"{session_id!r}"
        )
    component = row.get("component")
    if component not in VALID_COMPONENTS:
        raise LLMTraceRowError(
            f"llm_calls row 'component'={component!r} is not one of "
            f"{sorted(VALID_COMPONENTS)!r}"
        )


def append_llm_call(
    *,
    session_dir: Path,
    record: LLMCallRecord,
    target: Path | None = None,
) -> None:
    """Append one validated LLM-call row to the trace ledger.

    The row is serialized via :meth:`LLMCallRecord.to_row` (which stamps
    ``ts``), checked against the closed schema, then atomically appended.
    ``OSError`` while writing is logged and swallowed so a full disk or a
    permissions glitch never breaks the optimization loop — exactly the
    fault posture of :func:`..dynamic_action_history.append_dispatch_history_row`.

    ``target`` overrides the destination file; it defaults to
    ``<session_dir>/reports/trace/llm_calls.jsonl``. Out-of-process
    children pass an ``ext/<component>-<pid>.jsonl`` path here so the
    schema/serialization logic is shared with the in-process ledger.

    :class:`LLMTraceRowError` (schema violation) is *not* swallowed: a
    malformed row is a programming error at the call site, not a runtime
    disk condition, and must surface in tests.

    Args:
        session_dir: Session directory used to resolve the ledger path.
        record: The LLM-call record to serialize and append.
        target: Optional override destination (e.g. an ext shard path);
            defaults to the session's ``llm_calls.jsonl``.

    Raises:
        LLMTraceRowError: If the serialized row violates the closed schema.
    """
    row = record.to_row()
    _validate_row(row)
    dest = target if target is not None else llm_calls_path(session_dir)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as exc:
        log.warning(
            "llm_trace: append failed for component=%s session_id=%s: %r",
            record.component, record.session_id, exc,
        )

    # Second sink (opt-in): mirror in-process calls to Langfuse live. Skipped
    # for ext/ shards (target set) — those are out-of-process children that
    # the parent backfills at flush_session. Best-effort; never raises.
    if target is None:
        try:
            from .langfuse_emitter import get_emitter

            get_emitter(session_dir).record_llm_call(row)
        except Exception:  # noqa: BLE001 — Langfuse must never break the ledger
            log.debug("llm_trace: langfuse mirror failed", exc_info=True)


def row_field_set() -> frozenset[str]:
    """Public accessor for the closed row schema (used by the collector).

    Returns:
        The frozenset of canonical row field names.
    """
    return _ROW_FIELDS


# Sanity guard: the dataclass fields (minus the write-time ``ts``) must
# stay in lockstep with the on-disk row schema. A drift here means a new
# field was added to one side only — caught at import, not at runtime.
_DATACLASS_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(LLMCallRecord)
)
assert _DATACLASS_FIELDS | {"ts"} == _ROW_FIELDS, (
    "LLMCallRecord fields drifted from _ROW_FIELDS: "
    f"dataclass={sorted(_DATACLASS_FIELDS)} row={sorted(_ROW_FIELDS)}"
)


__all__ = [
    "LLMCallRecord",
    "LLMTraceRowError",
    "VALID_COMPONENTS",
    "append_llm_call",
    "row_field_set",
]
