# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework-arm identity read off task params and proposal payloads.

The arm has no runtime type; it is recognised by string markers, and these helpers are the one reading of them.
"""

from __future__ import annotations

from typing import Any, Mapping

LOCAL_EXPLORE_CANDIDATE_PREFIX: str = "local_explore:"


def is_local_explore_candidate(candidate_id: str) -> bool:
    """True when the candidate id was minted by the candidate-free local arm."""
    return str(candidate_id).startswith(LOCAL_EXPLORE_CANDIDATE_PREFIX)


def verdict_subject(params: Mapping[str, Any]) -> str:
    """The id an ``integrate_patch``'s Critic verdict is filed under.

    An authored patch is reviewed as the specialist that wrote it; an upstream-PR pre-screen has no specialist yet, so
    it is filed under the candidate id. This differs from :func:`review_row_id` on an authored patch by design.
    """
    sid = str(params.get("specialist_task_id") or "").strip()
    return sid or str(params.get("framework_agent_candidate_id") or "").strip()


def review_row_id(payload: Mapping[str, Any], *, fallback_msg_id: str = "") -> str:
    """The timeline row a Critic ruling is filed on: the proposal's candidate id, else its bus message id."""
    params = payload.get("params")
    nested = params.get("framework_agent_candidate_id") if isinstance(params, Mapping) else None
    return str(payload.get("framework_agent_candidate_id") or nested or "").strip() or fallback_msg_id


def is_upstream_pr_prescreen(action_name: str, payload: Mapping[str, Any] | None) -> bool:
    """True when this ``integrate_patch`` proposal only decides whether to bench an upstream PR.

    The candidate id is read at the payload top level only: an authored patch carries it in ``params``.
    """
    if action_name != "integrate_patch" or not isinstance(payload, Mapping):
        return False
    if payload.get("patches") or (payload.get("params") or {}).get("patches"):
        return False
    return bool(payload.get("framework_agent_candidate_id"))


__all__ = [
    "LOCAL_EXPLORE_CANDIDATE_PREFIX",
    "is_local_explore_candidate",
    "is_upstream_pr_prescreen",
    "review_row_id",
    "verdict_subject",
]
