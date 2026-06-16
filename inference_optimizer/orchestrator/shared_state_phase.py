# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Phase-transition / lifecycle write-owner functions extracted from SharedState.

Part of the SharedState behavior-offload (phase 2). Recording phase transitions
and lifecycle events belongs to the phase-state domain; they live here as free
functions taking ``state`` first. ``SharedState`` keeps forwarding shims so
existing callers are unchanged.
"""

from __future__ import annotations

from typing import Any

from .shared_state import _LIFECYCLE_CAP, _PHASE_HISTORY_CAP, _now_iso

def record_phase_transition(
    state,
    *,
    to_phase: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
    ts: str | None = None,
    ts_unix: float | None = None,
) -> dict[str, Any]:
    """Append a phase_history row and atomically update ``phase`` fields; ``phase``/``phase_history`` are CORE_STATE_FIELDS so LLM update_state is rejected. Returns the inserted row."""
    from datetime import datetime as _dt, timezone as _tz
    import time as _time
    # Lazy import to avoid an import-time cycle.
    from .phase_state import make_history_row

    now_ts = ts or _dt.now(_tz.utc).isoformat(timespec="seconds")
    now_unix = float(ts_unix if ts_unix is not None else _time.time())
    row = make_history_row(
        from_phase=state.phase or "",
        to_phase=to_phase,
        reason=reason,
        evidence=evidence,
        ts=now_ts,
        ts_unix=now_unix,
        cycle=int(getattr(state, "macro_cycle", 0) or 0),
    )
    history = list(state.phase_history or [])
    history.append(row)
    if len(history) > _PHASE_HISTORY_CAP:
        history = history[-_PHASE_HISTORY_CAP:]
    state.phase_history = history
    state.phase = row["to_phase"]
    state.phase_started_ts = now_ts
    state.phase_started_unix = now_unix
    return row


def record_lifecycle_event(
    state,
    *,
    step: str,
    status: str,
    phase: str | None = None,
    label: str | None = None,
    artifacts: dict[str, str] | None = None,
    detail: str = "",
    duration_s: float | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Append a structured lifecycle event (#266, method 1).

    Each event marks a phase/step boundary so operators can see — in
    state.json, and via the launcher in chat — that a phase ran, where
    its outputs went, and which artifact feeds the next phase.

    ``step`` is the machine step/handler name (e.g. ``trace_analyze``);
    ``label`` defaults to the human-friendly name from
    :data:`phase_state.LIFECYCLE_STEP_LABELS` so both naming dimensions
    are carried. ``phase`` defaults to the current coordinator phase.
    ``seq`` is monotonic across the cap so consumers can order events
    even after the oldest rows are trimmed.

    Coordinator-only writer (``policy.CORE_STATE_FIELDS`` guards
    ``lifecycle`` so an LLM ``update_state`` cannot forge events).
    Returns the inserted row.
    """
    # Lazy import to avoid an import-time cycle with the orchestrator
    # package (phase_state imports nothing from SharedState).
    from .phase_state import make_lifecycle_event

    events = state.lifecycle
    if events is None:
        events = state.lifecycle = []
    next_seq = (int(events[-1].get("seq", -1)) + 1) if events else 0
    event = make_lifecycle_event(
        step=step,
        status=status,
        phase=(phase if phase is not None else (state.phase or "")),
        label=label,
        artifacts=artifacts,
        detail=detail,
        duration_s=duration_s,
        seq=next_seq,
        ts=ts or _now_iso(),
    )
    # Append in place and trim only when over the cap, so the common
    # per-step-boundary path is an O(1) append rather than copying the
    # whole list on every call.
    events.append(event)
    if len(events) > _LIFECYCLE_CAP:
        del events[: -_LIFECYCLE_CAP]
    return event

