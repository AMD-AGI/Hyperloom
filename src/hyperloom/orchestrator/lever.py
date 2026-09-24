# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lever vocabulary: what a unit of work changed and which phase authored it.

The lever is the attribution key that survives the phase machine: a phase says
*when* work ran (which stops being evidence the moment two lanes share one
phase), while the lever says *what was changed*.

Depends on common.framework_arm for is_local_explore_candidate so that the
dependency points arm → lever (dispatched identity → settled result), never
the other direction.
"""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.common.framework_arm import is_local_explore_candidate

LEVER_CONFIG = "config"
LEVER_SOURCE_PATCH = "source_patch"
LEVER_UPSTREAM_PR = "upstream_pr"
LEVER_ENABLEMENT = "enablement"
LEVER_KERNEL = "kernel"

LEVER_KINDS: tuple[str, ...] = (
    LEVER_CONFIG,
    LEVER_SOURCE_PATCH,
    LEVER_UPSTREAM_PR,
    LEVER_ENABLEMENT,
    LEVER_KERNEL,
)

_PHASE_BY_LEVER: dict[str, str] = {
    LEVER_UPSTREAM_PR: "FRAMEWORK_AGENT",
    LEVER_ENABLEMENT: "FRAMEWORK_AGENT",
}


def patch_lever_kind(evidence: Mapping[str, Any] | None) -> str:
    """Name the lever a unit of work moved, or '' when nothing recorded one."""
    evidence = evidence or {}
    explicit = str(evidence.get("lever_kind") or "").strip().lower()
    if explicit in LEVER_KINDS:
        return explicit
    if evidence.get("enablement"):
        return LEVER_ENABLEMENT
    if evidence.get("pr_url") or evidence.get("pr_lead"):
        return LEVER_UPSTREAM_PR
    candidate_id_val = str(evidence.get("framework_agent_candidate_id") or "")
    if candidate_id_val:
        if not is_local_explore_candidate(candidate_id_val):
            return LEVER_UPSTREAM_PR
        wrote_a_patch = evidence.get("patch_name") or evidence.get("patches_applied") or evidence.get("patch_path")
        return LEVER_SOURCE_PATCH if wrote_a_patch else LEVER_CONFIG
    if evidence.get("patch_name") or evidence.get("patches_applied"):
        return LEVER_SOURCE_PATCH
    return ""


def patch_owner_phase(evidence: Mapping[str, Any] | None) -> str:
    """Resolve the immutable authoring phase from recorded ownership evidence."""
    evidence = evidence or {}
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


__all__ = [
    "LEVER_CONFIG",
    "LEVER_ENABLEMENT",
    "LEVER_KERNEL",
    "LEVER_KINDS",
    "LEVER_SOURCE_PATCH",
    "LEVER_UPSTREAM_PR",
    "patch_lever_kind",
    "patch_owner_phase",
]
