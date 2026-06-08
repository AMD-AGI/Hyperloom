# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Closed-schema writer for
``agents/orchestration/dynamic_actions/<dyn_id>/dispatch_history.jsonl``
plus the per-``dyn_id`` ``telemetry.json`` rollup.

One module owns the canonical event vocabulary and the field
contract for every audit row. Writes carrying an unknown field — or
missing a required field — fail fast so a buggy hook cannot silently
corrupt the audit trail.

Lifecycle coverage (Coordinator hook → event):

* ``_prepare_dynamic_action_dispatch``           → ``DISPATCHED``
* ``_handle_dynamic_action_runner_result``       → ``SUB_AGENT_DONE``
                                                    / ``SUB_AGENT_TERMINATED``
* ``_mirror_critic_verdict_to_dynamic_action``   → ``CRITIC_VERDICT``
* ``_maybe_update_dynamic_action_after_integrate``
                                                 → ``INTEGRATE_RESULT``
* ``resume_abandon_dynamic_actions``             → ``ABANDONED_ON_RESUME``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ..session_paths import (
    dynamic_action_dispatch_history_path,
    dynamic_action_telemetry_path,
)
from .dynamic_action_proposal import (
    DynamicActionStatus,
    TERMINAL_LIFECYCLE_STATUSES,
)


log = logging.getLogger(__name__)


class DispatchHistoryEvent(str, Enum):
    """Canonical event labels for dispatch_history rows."""

    DISPATCHED = "dispatched"
    SUB_AGENT_DONE = "sub_agent_done"
    SUB_AGENT_TERMINATED = "sub_agent_terminated"
    CRITIC_VERDICT = "critic_verdict"
    INTEGRATE_RESULT = "integrate_result"
    ABANDONED_ON_RESUME = "abandoned_on_resume"


# Common header fields shared by every event row.
_COMMON_FIELDS: frozenset[str] = frozenset({"event", "ts"})


DISPATCHED_FIELDS: frozenset[str] = _COMMON_FIELDS | {
    "round_index",
    "scope_domains",
    "side_effects_declared",
    "budget_hint",
    "degraded_dispatch",
    "seed_kit_tokens",
}


# SUB_AGENT_DONE and SUB_AGENT_TERMINATED share the same payload shape
# (the event label carries the success / non-success distinction).
SUB_AGENT_DONE_FIELDS: frozenset[str] = _COMMON_FIELDS | {
    "terminal_state",
    "reason",
    "turns_used",
    "journal_path",
    "proposal_count",
}
SUB_AGENT_TERMINATED_FIELDS: frozenset[str] = SUB_AGENT_DONE_FIELDS


CRITIC_VERDICT_FIELDS: frozenset[str] = _COMMON_FIELDS | {
    "verdict",
    "reason_codes",
    "applied_rules",
    "cross_domain_flag",
    "mechanical_floor_blocked",
}


INTEGRATE_RESULT_FIELDS: frozenset[str] = _COMMON_FIELDS | {
    "integrate_status",
    "lifecycle",
    "delta_pct",
    "task_id",
    "patches_applied",
    "patches_reverted",
}


# Aliased as ``dynamic_action_resume.ABANDONED_HISTORY_FIELDS`` for
# backwards-compatible imports.
ABANDONED_FIELDS: frozenset[str] = _COMMON_FIELDS | {
    "previous_status",
    "coordinator_session_id",
    "worktree_cleanup_outcome",
    "artifact_missing",
}


_EVENT_FIELD_SETS: dict[DispatchHistoryEvent, frozenset[str]] = {
    DispatchHistoryEvent.DISPATCHED: DISPATCHED_FIELDS,
    DispatchHistoryEvent.SUB_AGENT_DONE: SUB_AGENT_DONE_FIELDS,
    DispatchHistoryEvent.SUB_AGENT_TERMINATED: SUB_AGENT_TERMINATED_FIELDS,
    DispatchHistoryEvent.CRITIC_VERDICT: CRITIC_VERDICT_FIELDS,
    DispatchHistoryEvent.INTEGRATE_RESULT: INTEGRATE_RESULT_FIELDS,
    DispatchHistoryEvent.ABANDONED_ON_RESUME: ABANDONED_FIELDS,
}


def event_field_set(event: DispatchHistoryEvent | str) -> frozenset[str]:
    """Return the closed field schema for one event type.

    Args:
        event (DispatchHistoryEvent | str): Event as an enum member or
            its string value.

    Returns:
        frozenset[str]: The allowed field names for that event's rows.
    """
    e = event if isinstance(event, DispatchHistoryEvent) else DispatchHistoryEvent(event)
    return _EVENT_FIELD_SETS[e]


class DispatchHistoryRowError(ValueError):
    """Raised when a row violates the closed schema."""


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        str: Timestamp with microsecond precision in UTC.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def append_dispatch_history_row(
    *,
    session_dir: Path,
    dyn_id: str,
    event: DispatchHistoryEvent | str,
    payload: dict[str, Any],
) -> None:
    """Append one row to ``dispatch_history.jsonl`` after schema check.

    ``payload`` MUST carry every non-header field for the event and no
    extras. ``event`` + ``ts`` are filled in by the writer; OSError on
    disk is logged + swallowed so audit-write failures never break the
    lifecycle.

    Args:
        session_dir (Path): Session directory holding the history file.
        dyn_id (str): Dynamic-action identifier for the history path.
        event (DispatchHistoryEvent | str): Event type whose schema the
            payload must satisfy.
        payload (dict[str, Any]): Non-header fields for the row; must
            not include ``event`` or ``ts``.

    Raises:
        DispatchHistoryRowError: If the payload includes header fields,
            or has extra/missing fields against the event schema.
    """
    event_enum = (
        event if isinstance(event, DispatchHistoryEvent)
        else DispatchHistoryEvent(str(event))
    )
    field_set = _EVENT_FIELD_SETS[event_enum]
    if {"event", "ts"} & set(payload.keys()):
        raise DispatchHistoryRowError(
            f"payload for event={event_enum.value!r} must not include "
            "'event' / 'ts' — these are header fields filled in by the "
            "writer."
        )
    row: dict[str, Any] = {
        "event": event_enum.value,
        "ts": _now_iso(),
        **payload,
    }
    keys = set(row.keys())
    extra = sorted(keys - field_set)
    missing = sorted(field_set - keys)
    if extra or missing:
        raise DispatchHistoryRowError(
            f"dispatch_history row for event={event_enum.value!r} "
            f"violates closed schema: extra={extra!r} missing={missing!r}"
        )
    target = dynamic_action_dispatch_history_path(session_dir, dyn_id)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as exc:
        log.warning(
            "dynamic_action history: append failed for dyn_id=%s "
            "event=%s: %r",
            dyn_id, event_enum.value, exc,
        )


# ---------------------------------------------------------------------------
# telemetry.json — per-dyn_id terminal-state rollup
# ---------------------------------------------------------------------------
TELEMETRY_FIELDS: frozenset[str] = frozenset({
    "dyn_id",
    "rolled_up_at",
    "lifecycle",
    "kept",
    "reverted",
    "integrate_failed",
    "critic_rejected",
    "timed_out",
    "failed",
    "completed_empty",
    "abandoned",
    "gain_pct",
    "round_index",
})

# Map of lifecycle terminal → one-of-eight counter field set to 1.
_TELEMETRY_COUNTER_FOR: dict[DynamicActionStatus, str] = {
    DynamicActionStatus.KEPT: "kept",
    DynamicActionStatus.REVERTED: "reverted",
    DynamicActionStatus.INTEGRATE_FAILED: "integrate_failed",
    DynamicActionStatus.CRITIC_REJECTED: "critic_rejected",
    DynamicActionStatus.TIMED_OUT: "timed_out",
    DynamicActionStatus.FAILED: "failed",
    DynamicActionStatus.COMPLETED_EMPTY: "completed_empty",
    DynamicActionStatus.ABANDONED: "abandoned",
}


class TelemetryRowError(ValueError):
    """Raised when a telemetry payload violates the closed schema."""


def write_dynamic_action_telemetry(
    *,
    session_dir: Path,
    dyn_id: str,
    lifecycle: DynamicActionStatus | str,
    gain_pct: float | None = None,
    round_index: int | None = None,
) -> None:
    """Overwrite ``telemetry.json`` for one dyn_id on terminal transition.

    Idempotent: a later write replaces an earlier one (the resume
    abandoned sweep is the canonical second-pass writer).

    Args:
        session_dir (Path): Session directory holding the telemetry file.
        dyn_id (str): Dynamic-action identifier for the telemetry path.
        lifecycle (DynamicActionStatus | str): Terminal lifecycle status
            whose counter is set to 1.
        gain_pct (float | None): Optional integrate gain percentage.
        round_index (int | None): Optional dispatch round index.

    Raises:
        TelemetryRowError: If ``lifecycle`` is non-terminal or the
            payload violates :data:`TELEMETRY_FIELDS`.
    """
    lifecycle_enum = (
        lifecycle if isinstance(lifecycle, DynamicActionStatus)
        else DynamicActionStatus(str(lifecycle))
    )
    if lifecycle_enum not in TERMINAL_LIFECYCLE_STATUSES:
        raise TelemetryRowError(
            f"telemetry write requires a terminal lifecycle; got "
            f"{lifecycle_enum.value!r}"
        )
    counters: dict[str, int] = {
        v: 0 for v in _TELEMETRY_COUNTER_FOR.values()
    }
    counters[_TELEMETRY_COUNTER_FOR[lifecycle_enum]] = 1
    payload: dict[str, Any] = {
        "dyn_id": str(dyn_id),
        "rolled_up_at": _now_iso(),
        "lifecycle": lifecycle_enum.value,
        "gain_pct": float(gain_pct) if gain_pct is not None else None,
        "round_index": (
            int(round_index) if round_index is not None else None
        ),
        **counters,
    }
    extra = sorted(set(payload.keys()) - TELEMETRY_FIELDS)
    missing = sorted(TELEMETRY_FIELDS - set(payload.keys()))
    if extra or missing:
        raise TelemetryRowError(
            f"telemetry payload violates closed schema for dyn_id="
            f"{dyn_id!r}: extra={extra!r} missing={missing!r}"
        )
    target = dynamic_action_telemetry_path(session_dir, dyn_id)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning(
            "dynamic_action telemetry: write failed for dyn_id=%s: %r",
            dyn_id, exc,
        )


__all__ = [
    "ABANDONED_FIELDS",
    "CRITIC_VERDICT_FIELDS",
    "DISPATCHED_FIELDS",
    "DispatchHistoryEvent",
    "DispatchHistoryRowError",
    "INTEGRATE_RESULT_FIELDS",
    "SUB_AGENT_DONE_FIELDS",
    "SUB_AGENT_TERMINATED_FIELDS",
    "TELEMETRY_FIELDS",
    "TelemetryRowError",
    "append_dispatch_history_row",
    "event_field_set",
    "write_dynamic_action_telemetry",
]
