# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who owns a unit of work, decided the same way on both sides of the record.

The write side stamps an owner when work settles; the read side has to name
one for sessions recorded before it did. Both were answering the same question
about patch application with their own copy of the same rule, and a copy that
drifts moves gain between agents on a leaderboard without anything failing.

Kept free of recorder and collector imports so both can depend on it.
"""

from __future__ import annotations

from typing import Any, Mapping


#: Phase label -> owning agent. A phase is the weakest evidence of ownership
#: available, used only when the producer recorded nothing better.
AGENT_BY_PHASE = {
    "FRAMEWORK": "framework_agent",
    "FRAMEWORK_AGENT": "framework_agent",
    "EXPLORE": "explore",
    "KERNEL": "kernel_agent",
    "KERNEL_AGENT": "kernel_agent",
}

#: Returned when there is no evidence of an owner. A reportable gap, not a
#: guess: crediting the phase that happened to be active is how a delayed
#: patch ends up on the wrong agent's total.
UNATTRIBUTED = "unattributed"


def agent_from_phase(value: Any) -> str:
    """Map a phase label to its owning agent, or ``""`` when unknown."""
    return AGENT_BY_PHASE.get(str(value or "").strip().upper(), "")


def patch_owner_phase(evidence: Mapping[str, Any] | None) -> str:
    """Resolve the immutable authoring phase from recorded ownership evidence."""
    evidence = evidence or {}
    if evidence.get("framework_agent_authoring") or evidence.get("framework_agent_candidate_id"):
        return "FRAMEWORK_AGENT"
    phase = str(evidence.get("source_phase") or "").strip().upper()
    if phase in {"FRAMEWORK", "FRAMEWORK_AGENT"}:
        return "FRAMEWORK_AGENT"
    if phase == "EXPLORE":
        return "EXPLORE"
    return ""


def patch_author(evidence: Mapping[str, Any] | None) -> str:
    """Name who wrote a patch, from the markers its applier left behind.

    Patch application is the one action whose owner cannot be read off the
    active phase: by the time a patch lands the run has usually moved on, so
    the phase names whoever happens to be running, not whoever wrote it. The
    executor records its own markers, and this is the order they are trusted
    in.

    Args:
        evidence: The applier's result on the write side, or the recorded
            operation's ``outputs`` on the read side. Same keys either way.

    Returns:
        The owning agent, or :data:`UNATTRIBUTED`.
    """
    return agent_from_phase(patch_owner_phase(evidence)) or UNATTRIBUTED


__all__ = [
    "AGENT_BY_PHASE",
    "UNATTRIBUTED",
    "agent_from_phase",
    "patch_author",
    "patch_owner_phase",
]
