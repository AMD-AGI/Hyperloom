"""dynamic_action.MD §1.5 + dynamic_action_gaps.md G2 / G13 —
dispatch_history.jsonl writer with closed per-event schemas.

One module owns the canonical event vocabulary and the field
contract for every row written into
``agents/orchestration/dynamic_actions/<dyn_id>/dispatch_history.jsonl``.

The schema is **closed** (per P2 §5.2 b): a row carrying an unknown
field — or missing a required field — fails fast at write time so a
buggy Coordinator hook cannot silently corrupt the audit trail.

Lifecycle coverage (Coordinator hook → event):

* ``_prepare_dynamic_action_dispatch``         → ``DISPATCHED``
* ``_handle_dynamic_action_runner_result``     → ``SUB_AGENT_DONE``
                                                  (COMPLETED) or
                                                  ``SUB_AGENT_TERMINATED``
                                                  (TIMED_OUT / FAILED /
                                                  COMPLETED_EMPTY)
* ``_mirror_critic_verdict_to_dynamic_action`` → ``CRITIC_VERDICT``
* ``_maybe_update_dynamic_action_after_integrate``
                                               → ``INTEGRATE_RESULT``
* ``resume_abandon_dynamic_actions``           → ``ABANDONED_ON_RESUME``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ..session_paths import dynamic_action_dispatch_history_path


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


# Mirrors ``dynamic_action_resume.ABANDONED_HISTORY_FIELDS`` so the
# resume sweep + general writer share one source of truth.
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
    """Public accessor for the closed schema of one event type."""
    e = event if isinstance(event, DispatchHistoryEvent) else DispatchHistoryEvent(event)
    return _EVENT_FIELD_SETS[e]


class DispatchHistoryRowError(ValueError):
    """Raised when a row violates the closed schema."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def append_dispatch_history_row(
    *,
    session_dir: Path,
    dyn_id: str,
    event: DispatchHistoryEvent | str,
    payload: dict[str, Any],
) -> None:
    """Append one row to ``dispatch_history.jsonl`` after schema check.

    ``payload`` MUST contain every non-header field for the event and
    no extras. ``event`` + ``ts`` are filled in by the writer and must
    not appear in ``payload``.
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
        # Non-fatal: an audit-trail write failure must not stop the
        # lifecycle. Coordinator hooks log and continue.
        log.warning(
            "dynamic_action history: append failed for dyn_id=%s "
            "event=%s: %r",
            dyn_id, event_enum.value, exc,
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
    "append_dispatch_history_row",
    "event_field_set",
]
