# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""In-memory Coordinator state, and the trajectory close of the proposals it still holds when a session ends."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.inference_optimizer.trace.trajectory_trace import (
    EVENT_PROPOSAL,
    STATUS_CANCELLED,
    TERMINAL_STATUSES,
    EventSpan,
    load_events,
    record_event,
)

if TYPE_CHECKING:
    from ..state.pending_proposal import PendingProposal

log = logging.getLogger(__name__)


@dataclass
class CoordinatorState:
    """In-memory ephemeral state for the reactor + dispatcher."""

    pending_proposals: dict[str, PendingProposal] = field(default_factory=dict)


def close_undecided_proposals(
    session_dir: Path, pending_proposals: Mapping[str, PendingProposal], stop_reason: str
) -> None:
    """Close the trajectory span of every proposal the session ends without a verdict on.

    A supervisor restart ends a leg, not the session: the next leg's replay restores these proposals. A resumed
    session may end again over proposals an earlier leg already closed, so those are skipped.
    """
    from hyperloom.inference_optimizer.breakdown.stop_reasons import SUPERVISOR_RESTART_REASON

    if stop_reason == SUPERVISOR_RESTART_REASON:
        return
    # A decided proposal is popped from the registry, so everything still pending is undecided.
    undecided = list(pending_proposals.values())
    if not undecided:
        return
    try:
        closed = {
            row.get("span_id")
            for row in load_events(session_dir)
            if row.get("event_type") == EVENT_PROPOSAL and row.get("status") in TERMINAL_STATUSES
        }
        for pending in undecided:
            if pending.proposal_msg_id in closed:
                continue
            record_event(
                EVENT_PROPOSAL,
                span=EventSpan(span_id=pending.proposal_msg_id, status=STATUS_CANCELLED),
                attributes={
                    "name": pending.action_name,
                    "action_name": pending.action_name,
                    "from_agent": pending.from_agent,
                    "reason": "session_ended_undecided",
                    "stop_reason": stop_reason,
                },
            )
    except (OSError, ValueError):  # the trace must never mask the stop reason the session is closing on
        log.warning("trajectory: closing undecided proposals failed", exc_info=True)
