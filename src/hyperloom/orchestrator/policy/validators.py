# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stateless PolicyGate checks on intent payloads and specialist dispatch params."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.protocol.intent import ALLOWED_VERDICTS
from hyperloom.inference_optimizer.protocol.action_surfaces import COORDINATOR_INTERNAL_ACTIONS
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

if TYPE_CHECKING:  # pragma: no cover — type-only
    from ..roles.agent_role import AgentRole


log = logging.getLogger(__name__)


# GEMM tuning action; the hook that guards it is called for every action, so it
# needs its own name to answer only for itself.
GEMM_TUNING_ACTION_NAME: str = "gemm_tuning"

# Free-form (``scope='freeform'``) sanity-gate limits; the real ceiling is the
# research_lane capacity and the GPU specialist pool.
SPECIALIST_FREEFORM_WAVE_MAX: int = 16
SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS: int = 8000

# Critic-only: REVIEW_VERDICT
REVIEW_VERDICT_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"critic"})

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


def validate_coordinator_managed_action(action_name: str, *, intent_kind: str) -> None:
    """Deny an agent-initiated copy of an action the Coordinator dispatches itself.

    Phase-independent: these actions carry their own entry conditions and
    SharedState accounting, which a second run would skip.

    Args:
        action_name: The proposed/delegated action name.
        intent_kind: The channel it arrived on, for the message.

    Raises:
        PolicyDenied: when ``action_name`` is Coordinator-managed.
    """
    if action_name not in COORDINATOR_INTERNAL_ACTIONS:
        return
    raise PolicyDenied(
        f"action {action_name!r} is Coordinator-managed and not LLM-proposable ({intent_kind})",
        rule="coordinator_managed_action",
        hint="read the outcome in SharedState rather than ordering a second run.",
    )


# GEMM tuning ownership
def validate_gemm_tuning_action(
    action_name: str,
    *,
    intent_kind: str,
) -> None:
    """Refuse a model-proposed GEMM tuning run; the Coordinator owns the lane.

    Applicability is still not pre-filtered here -- the producer decides
    internally whether tuning applies to the workload. What this now refuses
    is the *channel*: the lane is dispatched once at phase entry from a lane
    budget, so a per-tick re-issue would spend time the allocation never
    granted. Mirrors how the fusion lane is already closed.

    Args:
        action_name (str): the action name being checked.
        intent_kind (str): the channel the action arrived on, used in the
            error hint.

    Raises:
        PolicyDenied: When ``action_name`` is the GEMM tuning action.
    """
    # Called unconditionally for every action, so it answers only for its own.
    if action_name != GEMM_TUNING_ACTION_NAME:
        return
    raise PolicyDenied(
        f"{action_name!r} is a Coordinator-owned kernel lane and not model-requestable ({intent_kind})",
        rule="phase_incompatible",
        hint=(
            "GEMM tuning is dispatched by the Coordinator at KERNEL entry once its "
            "deterministic gate passes; it draws on a lane budget rather than a "
            "per-request one, so it cannot be re-issued per tick."
        ),
    )


def validate_review_verdict(role: "AgentRole", payload: dict[str, Any]) -> None:
    """Validate a ``REVIEW_VERDICT`` intent (Critic-only).

    Enforces that the source role is on
    :data:`REVIEW_VERDICT_SOURCE_ALLOWLIST`, that
    ``target_proposal_msg_id`` is present, and that exactly one of the
    single ``verdict`` field or the per-variant ``verdict_map`` is
    supplied. Every verdict string (single or per-variant) must belong
    to the closed :data:`ALLOWED_VERDICTS` vocabulary.

    Args:
        role (AgentRole): the resolved role of the emitting agent.
        payload (dict[str, Any]): the review_verdict payload, carrying
            ``target_proposal_msg_id`` and either ``verdict`` or
            ``verdict_map``.

    Returns:
        None: returns silently when the verdict is well-formed.

    Raises:
        PolicyDenied: if the role is not a Critic, the target id is
            missing, neither/both verdict forms are present, or a
            verdict string is outside ``ALLOWED_VERDICTS``.
    """
    if role.name not in REVIEW_VERDICT_SOURCE_ALLOWLIST:
        raise PolicyDenied(
            f"role={role.name!r} cannot emit review_verdict (allowed: {sorted(REVIEW_VERDICT_SOURCE_ALLOWLIST)!r})",
            rule="review_verdict_source",
        )
    target = str(payload.get("target_proposal_msg_id", "")).strip()
    if not target:
        raise PolicyDenied(
            "review_verdict missing target_proposal_msg_id",
            rule="payload",
        )
    # Accept the single ``verdict`` or the per-variant ``verdict_map``.
    has_single = "verdict" in payload
    verdict_map = payload.get("verdict_map")
    has_map = isinstance(verdict_map, dict) and bool(verdict_map)
    if has_single == has_map:
        raise PolicyDenied(
            "review_verdict: exactly one of 'verdict' or 'verdict_map' must be present",
            rule="payload",
            hint=(
                "single-proposal review: emit {target_proposal_msg_id, "
                "verdict, reasoning, failure_reason_code?}. Explore batch "
                "review: emit {target_proposal_msg_id, verdict_map: "
                "{variant_name: {verdict, rationale?, "
                "failure_reason_code?}}}"
            ),
        )
    if has_single:
        verdict = str(payload.get("verdict", "")).strip()
        if verdict not in ALLOWED_VERDICTS:
            raise PolicyDenied(
                f"review_verdict.verdict={verdict!r} not in allowed set {sorted(ALLOWED_VERDICTS)!r}",
                rule="payload",
                hint=f"use one of {'/'.join(sorted(ALLOWED_VERDICTS))}",
            )
        return
    # verdict_map path — every entry's verdict must be in the closed vocab.
    for vname, entry in verdict_map.items():
        v = str((entry or {}).get("verdict") or "").strip()
        if v not in ALLOWED_VERDICTS:
            raise PolicyDenied(
                f"review_verdict.verdict_map[{vname!r}].verdict={v!r} not in allowed set {sorted(ALLOWED_VERDICTS)!r}",
                rule="payload",
                hint=f"every per-variant verdict must be one of {'/'.join(sorted(ALLOWED_VERDICTS))}",
            )


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
