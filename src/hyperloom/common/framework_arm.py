# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework arm identity: predicates and helpers that answer questions about dispatched work.

The framework arm has no runtime type.  It is identified by string markers in
task params/payload dicts.  This module provides the canonical predicates so
every consumer reads the same answer from the same source.

IMPORTANT read order rules — do NOT break these:

1. ``is_upstream_pr_prescreen`` reads ``framework_agent_candidate_id`` at the
   **payload top level only**.  Using ``candidate_id()`` (which also checks
   params) would misclassify every authored patch as a pre-screen.

2. ``verdict_subject`` and ``review_row_id`` answer different questions and
   must not be merged.  Their disagreement on the authored-patch payload is
   what makes the two-review design work.

3. Excluded sites (must not use the canonical helpers):
   - ``writeback.py`` line reading ``framework_agent_authoring`` via key
     membership (``"framework_agent_authoring" in bv``) — preserves False.
   - ``dispatch.py`` specialist-round entry builder — reads done_payload OR
     task_params union.
   - ``critic_reviews.py`` — uses ``_optional_bool`` strict tri-state coercion.
"""

from __future__ import annotations

from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Task-kind vocabulary
# ---------------------------------------------------------------------------

TASK_KIND_FRAMEWORK_AUTHORING = "framework_authoring"
TASK_KIND_LOCAL_EXPLORE = "framework_local_explore"
TASK_KIND_APPLY_RETRY = "explore_apply_retry"
TASK_KIND_CANDIDATE_DISCOVERY = "candidate_discovery"

#: Task kinds that belong to the source arm and are treated as authoring work
#: by the breakdown collector.  ``explore_apply_retry`` is listed here because
#: ``agent_ownership.patch_lever_kind`` derives ``LEVER_SOURCE_PATCH`` from the
#: delivered patch rather than the dispatch marker, so the arm-identity question
#: and the lever question have different correct answers for it.
AUTHORING_TASK_KINDS: frozenset[str] = frozenset(
    {TASK_KIND_FRAMEWORK_AUTHORING, TASK_KIND_LOCAL_EXPLORE, TASK_KIND_APPLY_RETRY}
)

# ---------------------------------------------------------------------------
# Candidate-id vocabulary
# ---------------------------------------------------------------------------

LOCAL_EXPLORE_CANDIDATE_PREFIX: str = "local_explore:"


def is_local_explore_candidate(candidate_id: str) -> bool:
    """True when the candidate id was minted by the candidate-free local arm."""
    return str(candidate_id).startswith(LOCAL_EXPLORE_CANDIDATE_PREFIX)


# ---------------------------------------------------------------------------
# Ownership key set
# ---------------------------------------------------------------------------

#: Keys that travel from an authoring specialist to the integrate_patch it spawns.
FRAMEWORK_OWNERSHIP_KEYS: tuple[str, ...] = (
    "domain",
    "source_domain",
    "provenance",
    "gap_canonical_id",
    "gap_layer",
    "lever_kind",
    "reauthor_attempt",
    "apply_retry_attempt",
    "framework_agent_authoring",
    "framework_agent_candidate_id",
    "framework_batch_id",
)


# ---------------------------------------------------------------------------
# Arm-identity predicates
# ---------------------------------------------------------------------------


def is_framework_arm(evidence: Mapping[str, Any]) -> bool:
    """True when this params/payload/result dict belongs to the framework arm.

    Uses OR-of-both-markers so a candidate-id-only payload (the pre-screen
    shape) is correctly attributed when the authoring flag is absent.
    """
    return bool(evidence.get("framework_agent_authoring")) or bool(
        evidence.get("framework_agent_candidate_id")
    )


def candidate_id(evidence: Mapping[str, Any]) -> str:
    """The framework-arm candidate this unit of work belongs to, or ''.

    Resolution order: payload top level → payload['params'] → candidate map →
    ''.  The params descent handles the authored-patch proposal shape where
    the id is nested.
    """
    top = evidence.get("framework_agent_candidate_id")
    if top:
        return str(top).strip()
    params = evidence.get("params")
    if isinstance(params, Mapping):
        nested = params.get("framework_agent_candidate_id")
        if nested:
            return str(nested).strip()
    cand = evidence.get("candidate")
    if isinstance(cand, Mapping):
        for key in ("candidate_id", "pr_url", "ref"):
            val = cand.get(key)
            if val:
                return str(val).strip()
    return ""


def verdict_subject(params: Mapping[str, Any]) -> str:
    """The key under which an integrate_patch's Critic verdict is filed.

    An authored patch is reviewed as the specialist that wrote it
    (``specialist_task_id``); an upstream-PR pre-screen uses the candidate id
    because no specialist exists yet.  These two answers are intentionally
    different — the two-review design depends on the distinction.
    """
    sid = str(params.get("specialist_task_id") or "").strip()
    if sid:
        return sid
    return str(params.get("framework_agent_candidate_id") or "").strip()


def review_row_id(payload: Mapping[str, Any], *, fallback_msg_id: str = "") -> str:
    """The timeline row a Critic ruling is filed on.

    The candidate id (from the payload top level or its params) is preferred
    so the ruling lands on the candidate the phase already recorded.  Falls
    back to the bus message id when no candidate id is available.
    """
    cid = candidate_id(payload)
    return cid or fallback_msg_id


def is_upstream_pr_prescreen(action_name: str, payload: Mapping[str, Any] | None) -> bool:
    """True when this integrate_patch proposal only decides whether to bench a PR.

    CRITICAL: reads framework_agent_candidate_id at the payload TOP LEVEL only.
    Do NOT replace this with ``candidate_id()``—that would also match the nested
    authored-patch shape and classify every authored patch as a pre-screen,
    dropping the Critic's patch-landing requirements.
    """
    if action_name != "integrate_patch" or not isinstance(payload, Mapping):
        return False
    if payload.get("patches") or (payload.get("params") or {}).get("patches"):
        return False
    return bool(payload.get("framework_agent_candidate_id"))


__all__ = [
    "AUTHORING_TASK_KINDS",
    "FRAMEWORK_OWNERSHIP_KEYS",
    "LOCAL_EXPLORE_CANDIDATE_PREFIX",
    "TASK_KIND_APPLY_RETRY",
    "TASK_KIND_CANDIDATE_DISCOVERY",
    "TASK_KIND_FRAMEWORK_AUTHORING",
    "TASK_KIND_LOCAL_EXPLORE",
    "candidate_id",
    "is_framework_arm",
    "is_local_explore_candidate",
    "is_upstream_pr_prescreen",
    "review_row_id",
    "verdict_subject",
]
