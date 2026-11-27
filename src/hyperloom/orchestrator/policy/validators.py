# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stateless PolicyGate checks on intent payloads and specialist dispatch params."""

from __future__ import annotations

import logging
from typing import Any

from hyperloom.inference_optimizer import framework_registry
from ..specialists.domains import KNOWLEDGE_DOMAIN_TAG_SET
from .denial import PolicyDenied
from ..specialists.profile import (
    SCOPE_DOMAIN as SPECIALIST_SCOPE_DOMAIN,
    SCOPE_DOMAINS as SPECIALIST_SCOPE_DOMAINS,
    SCOPE_VALUES as SPECIALIST_SCOPE_VALUES,
)

log = logging.getLogger(__name__)


# The GEMM tuning action, refused as a REQUEST kind.
GEMM_TUNING_ACTION_NAME: str = "gemm_tuning"


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


def validate_prune_branch(payload: dict[str, Any]) -> None:
    """Refuse an orchestration prune request that names no family."""
    family = str(payload.get("family", "")).strip()
    if not family:
        raise PolicyDenied("prune_branch missing family", rule="payload")


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


def validate_specialist_max_turns_raw(max_turns_raw: Any) -> None:
    """Validate an optional specialist ``max_turns`` dial.

    Args:
        max_turns_raw: Raw ``max_turns`` value from dispatch params, or ``None``.

    Raises:
        PolicyDenied: When the value is not an int or is negative.
    """
    if max_turns_raw is None:
        return
    try:
        max_turns = int(max_turns_raw)
    except (TypeError, ValueError) as exc:
        raise PolicyDenied(
            f"delegate{{action='specialist'}}: params.max_turns max_turns must be int, got {max_turns_raw!r}",
            rule="specialist_dispatch_source",
        ) from exc
    if max_turns < 0:
        raise PolicyDenied(
            f"delegate{{action='specialist'}}: params.max_turns max_turns={max_turns} must be >= 0",
            rule="specialist_dispatch_source",
            hint=(
                "Use a non-negative integer. "
                "0 = unbounded (bounded by the wall-clock budget); "
                "omit max_turns to use the default turn cap."
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


def validate_freeform_task_description(params: dict[str, Any]) -> None:
    """Refuse a free-form specialist dispatch that carries no ``task_description``: the description is its whole mandate.

    Args:
        params: The specialist dispatch ``params``.

    Raises:
        PolicyDenied: When ``params.task_description`` is empty.
    """
    if not str(params.get("task_description") or "").strip():
        raise PolicyDenied(
            "delegate{action='specialist',scope='freeform'}: params task_description must be non-empty",
            rule="specialist_freeform_empty_description",
            hint=("A freeform specialist needs a natural-language task_description (the whole mandate)."),
        )
