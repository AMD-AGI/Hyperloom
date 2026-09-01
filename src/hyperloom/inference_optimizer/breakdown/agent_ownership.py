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


#: What kind of lever a unit of work moved. This is the attribution key that
#: survives the phase machine: a phase says *when* work ran, which stops being
#: evidence the moment two lanes share one phase, while the lever says *what
#: was changed*, which is what a report is actually about.
LEVER_CONFIG = "config"  # server args / envs only; nothing on disk is touched
LEVER_SOURCE_PATCH = "source_patch"  # a diff a specialist authored
LEVER_UPSTREAM_PR = "upstream_pr"  # a diff fetched from an upstream PR
LEVER_ENABLEMENT = "enablement"  # graded on runnability + accuracy, not throughput
LEVER_KERNEL = "kernel"  # a tuned or authored kernel, graded on the e2e bench

LEVER_KINDS = (
    LEVER_CONFIG,
    LEVER_SOURCE_PATCH,
    LEVER_UPSTREAM_PR,
    LEVER_ENABLEMENT,
    LEVER_KERNEL,
)

#: Lever -> owning agent. Stronger evidence than the phase: once both arms run
#: inside one phase, what a unit of work delivered is the only thing that still
#: separates their owners.
AGENT_BY_LEVER = {
    LEVER_CONFIG: "explore",
    LEVER_SOURCE_PATCH: "framework_agent",
    LEVER_UPSTREAM_PR: "framework_agent",
    LEVER_ENABLEMENT: "framework_agent",
    LEVER_KERNEL: "kernel_agent",
}

#: Lever kinds whose phase is not in doubt. ``source_patch`` and ``config`` are
#: absent on purpose: either can be dispatched from more than one phase, so the
#: lever alone does not name one and the older evidence still decides.
_PHASE_BY_LEVER = {
    LEVER_UPSTREAM_PR: "FRAMEWORK_AGENT",
    LEVER_ENABLEMENT: "FRAMEWORK_AGENT",
}


def patch_lever_kind(evidence: Mapping[str, Any] | None) -> str:
    """Name the lever a unit of work moved, or ``""`` when nothing recorded one.

    Reads the stamp the dispatcher left. Falls back to deriving one from the
    markers that predate the stamp so a session recorded before it, or a task
    a caller forgot to stamp, still attributes rather than silently landing in
    :data:`UNATTRIBUTED`.

    Args:
        evidence: Task params, an executor result, or a recorded operation's
            outputs. The same keys either way.

    Returns:
        One of :data:`LEVER_KINDS`, or ``""``.
    """
    evidence = evidence or {}
    explicit = str(evidence.get("lever_kind") or "").strip().lower()
    if explicit in LEVER_KINDS:
        return explicit
    # Derivation order mirrors how the gates differ: enablement grades on
    # runnability, an upstream PR carries a fetched diff, an authored patch
    # carries a written one, and anything left changed only configuration.
    if evidence.get("enablement"):
        return LEVER_ENABLEMENT
    if evidence.get("pr_url") or evidence.get("pr_lead"):
        return LEVER_UPSTREAM_PR
    # A candidate id names a PR unless it is the candidate-free local arm, which
    # authors against the live source with no upstream lead to attribute to.
    candidate_id = str(evidence.get("framework_agent_candidate_id") or "")
    if candidate_id:
        if not candidate_id.startswith("local_explore:"):
            return LEVER_UPSTREAM_PR
        # That arm is told which gap to close, not which lever to move, so it
        # returns server args about as often as a diff.
        wrote_a_patch = evidence.get("patch_name") or evidence.get("patches_applied") or evidence.get("patch_path")
        return LEVER_SOURCE_PATCH if wrote_a_patch else LEVER_CONFIG
    # A task id says a specialist ran, not that it wrote anything; the arm
    # returns server args about as often as a diff, and the Coordinator names
    # the diff it resolved before the patch reaches an applier.
    if evidence.get("patch_name") or evidence.get("patches_applied"):
        return LEVER_SOURCE_PATCH
    return ""


def agent_from_phase(value: Any) -> str:
    """Map a phase label to its owning agent, or ``""`` when unknown."""
    return AGENT_BY_PHASE.get(str(value or "").strip().upper(), "")


def agent_from_lever(value: Any) -> str:
    """Map a lever kind to its owning agent, or ``""`` when unknown."""
    return AGENT_BY_LEVER.get(str(value or "").strip().lower(), "")


def patch_owner_phase(evidence: Mapping[str, Any] | None) -> str:
    """Resolve the immutable authoring phase from recorded ownership evidence."""
    evidence = evidence or {}
    # The lever is the stronger evidence where it names a phase at all.
    phase_from_lever = _PHASE_BY_LEVER.get(patch_lever_kind(evidence))
    if phase_from_lever:
        return phase_from_lever
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
    "AGENT_BY_LEVER",
    "AGENT_BY_PHASE",
    "LEVER_CONFIG",
    "LEVER_ENABLEMENT",
    "LEVER_KERNEL",
    "LEVER_KINDS",
    "LEVER_SOURCE_PATCH",
    "LEVER_UPSTREAM_PR",
    "UNATTRIBUTED",
    "agent_from_lever",
    "agent_from_phase",
    "patch_author",
    "patch_lever_kind",
    "patch_owner_phase",
]
