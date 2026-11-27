# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stateless PolicyGate checks on intent payloads and specialist dispatch params."""

from __future__ import annotations

import logging
from typing import Any

from hyperloom.inference_optimizer import framework_registry
from ..specialists.domains import (
    KNOWLEDGE_DOMAIN_TAG_SET,
    SPECIALIST_MAX_TURNS_HARD_CAP,
)
from .denial import PolicyDenied
from ..specialists.profile import (
    SCOPE_DOMAIN as SPECIALIST_SCOPE_DOMAIN,
    SCOPE_DOMAINS as SPECIALIST_SCOPE_DOMAINS,
    SCOPE_VALUES as SPECIALIST_SCOPE_VALUES,
)

log = logging.getLogger(__name__)


# The GEMM tuning action, refused as a REQUEST kind.
GEMM_TUNING_ACTION_NAME: str = "gemm_tuning"

# Free-form (``scope='freeform'``) sanity-gate limits; the real ceiling is the
# research_lane capacity and the GPU specialist pool.
SPECIALIST_FREEFORM_WAVE_MAX: int = 16
SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS: int = 8000

# prune_branch scopes. ``family`` retires the action for the rest of the run;
# ``queued`` only drains the backlog and leaves the family usable.
PRUNE_BRANCH_SCOPE_FAMILY: str = "family"
PRUNE_BRANCH_SCOPE_QUEUED: str = "queued"
PRUNE_BRANCH_ALLOWED_SCOPES: frozenset[str] = frozenset(
    {
        PRUNE_BRANCH_SCOPE_FAMILY,
        PRUNE_BRANCH_SCOPE_QUEUED,
    }
)


def validate_gemm_tuning_request(kind: str) -> None:
    """Refuse the GEMM tuning action as a REQUEST kind; the Coordinator owns the lane.

    The lane is dispatched once at phase entry from a lane budget, so a
    per-tick re-issue would spend time the allocation never granted.

    Args:
        kind (str): the request kind being checked.

    Raises:
        PolicyDenied: When ``kind`` is the GEMM tuning action.
    """
    if kind != GEMM_TUNING_ACTION_NAME:
        return
    raise PolicyDenied(
        f"{kind!r} is a Coordinator-owned kernel lane and not model-requestable",
        rule="phase_incompatible",
        hint=(
            "GEMM tuning is dispatched by the Coordinator at KERNEL entry once its "
            "deterministic gate passes; it draws on a lane budget rather than a "
            "per-request one, so it cannot be re-issued per tick."
        ),
    )


def validate_review_verdict(payload: dict[str, Any]) -> None:
    """Refuse a ``REVIEW_VERDICT`` whose ``target_proposal_msg_id`` is blank."""
    if not str(payload.get("target_proposal_msg_id", "")).strip():
        raise PolicyDenied("review_verdict missing target_proposal_msg_id", rule="payload")


def validate_prune_branch(payload: dict[str, Any]) -> None:
    """Validate the family and scope of an orchestration prune request."""
    family = str(payload.get("family", "")).strip()
    if not family:
        raise PolicyDenied("prune_branch missing family", rule="payload")
    scope = str(payload.get("scope") or PRUNE_BRANCH_SCOPE_FAMILY).strip()
    if scope not in PRUNE_BRANCH_ALLOWED_SCOPES:
        raise PolicyDenied(
            f"prune_branch scope={scope!r} not allowed (allowed: {sorted(PRUNE_BRANCH_ALLOWED_SCOPES)!r})",
            rule="prune_scope",
            hint=(
                f"{PRUNE_BRANCH_SCOPE_FAMILY!r} retires the action for "
                f"the rest of the run; {PRUNE_BRANCH_SCOPE_QUEUED!r} "
                f"only cancels the queued backlog."
            ),
        )


def validate_specialist_framework(params: dict[str, Any]) -> None:
    """Refuse a specialist dispatch whose ``params.framework`` the registry does not know.

    The specialist resolves the source tree it patches from that name, and a
    dispatch that omits it inherits the session's.

    Args:
        params: The specialist dispatch ``params``.

    Raises:
        PolicyDenied: When ``params.framework`` is present and not registered.
    """
    if "framework" in params and not framework_registry.is_supported(params["framework"]):
        raise PolicyDenied(
            f"delegate{{action='specialist'}}: params.framework={params['framework']!r} is not a registered framework",
            rule="specialist_framework_unregistered",
            hint=(
                "Omit params.framework to dispatch against the session's framework, "
                f"or pass one of {list(framework_registry.names())!r}."
            ),
        )


def validate_specialist_max_turns_raw(
    max_turns_raw: Any,
    *,
    where: str,
) -> None:
    """Validate an optional specialist ``max_turns`` dial.

    Args:
        max_turns_raw: Raw ``max_turns`` value from dispatch params, or ``None``.
        where: Label used in error messages (e.g. ``params.max_turns``).

    Raises:
        PolicyDenied: When the value is not an int, is negative, or exceeds
            :data:`SPECIALIST_MAX_TURNS_HARD_CAP`.
    """
    if max_turns_raw is None:
        return
    try:
        max_turns = int(max_turns_raw)
    except (TypeError, ValueError) as exc:
        raise PolicyDenied(
            f"delegate{{action='specialist'}}: {where} max_turns must be int, got {max_turns_raw!r}",
            rule="specialist_dispatch_source",
        ) from exc
    if max_turns < 0:
        raise PolicyDenied(
            f"delegate{{action='specialist'}}: {where} max_turns={max_turns} must be >= 0",
            rule="specialist_dispatch_source",
            hint=(
                "Use a non-negative integer. "
                "0 = unbounded (bounded by the wall-clock budget); "
                "omit max_turns to use the default turn cap."
            ),
        )
    if max_turns > SPECIALIST_MAX_TURNS_HARD_CAP:
        raise PolicyDenied(
            f"delegate{{action='specialist'}}: {where} max_turns={max_turns} "
            f"exceeds the hard cap {SPECIALIST_MAX_TURNS_HARD_CAP}",
            rule="specialist_dispatch_source",
            hint=(
                f"max_turns must be <= {SPECIALIST_MAX_TURNS_HARD_CAP} "
                "(0 = unbounded; depth is bounded by the wall-clock "
                "budget, so omit max_turns unless capping a probe early)."
            ),
        )


def log_specialist_dispatch_coherence(scope_raw: str, tags: list[str]) -> None:
    """Log a tag / scope combination the specialist profile will re-infer."""
    if not tags:
        log.info("specialist dispatch declares a scope but no tags; profile will re-infer")
    unknown_tags = [t for t in tags if t not in KNOWLEDGE_DOMAIN_TAG_SET]
    if unknown_tags:
        log.info(
            "specialist dispatch carries out-of-vocabulary tag(s)=%r (known: %r)",
            unknown_tags,
            sorted(KNOWLEDGE_DOMAIN_TAG_SET),
        )

    if scope_raw and scope_raw not in SPECIALIST_SCOPE_VALUES:
        log.info(
            "specialist dispatch scope=%r not in %r; re-inferred from tags",
            scope_raw,
            sorted(SPECIALIST_SCOPE_VALUES),
        )
    elif scope_raw == SPECIALIST_SCOPE_DOMAINS and len(tags) < 2:
        log.info("specialist dispatch scope='domains' with %d tag(s)=%r; profile will re-infer", len(tags), tags)
    elif scope_raw == SPECIALIST_SCOPE_DOMAIN and len(tags) > 1:
        log.info("specialist dispatch scope='domain' with %d tags=%r", len(tags), tags)


def check_freeform_task_description(desc: str, *, where: str) -> None:
    """Per-task structural checks for a free-form ``task_description``: non-empty and length-bounded.

    Args:
        desc (str): the freeform task description to validate.
        where (str): a label identifying the source location, used in error
            messages.

    Raises:
        PolicyDenied: when ``desc`` is empty or exceeds the length cap.
    """
    if not desc:
        raise PolicyDenied(
            f"delegate{{action='specialist',scope='freeform'}}: {where} task_description must be non-empty",
            rule="specialist_freeform_empty_description",
            hint=("Each freeform task needs a natural-language task_description (the whole mandate)."),
        )
    if len(desc) > SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS:
        raise PolicyDenied(
            f"delegate{{action='specialist',scope='freeform'}}: "
            f"{where} task_description is {len(desc)} chars > cap "
            f"{SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS}",
            rule="specialist_freeform_description_too_long",
        )
