# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Translate Critic ``kb_drafts[].category`` into KB ``kind`` (contract §2.1).

The KB contract has exactly four ``kind`` values; the richer Critic
category vocabulary maps onto them surjectively. Categories with no KB
equivalent are ``rejected`` rather than silently dropped.
"""

from __future__ import annotations

from typing import Iterable

from .errors import RuntimeAdapterError


# KB-side ``kind`` enum (kb-critic-integration-contract §2.1).
KB_KINDS: frozenset[str] = frozenset({
    "pitfall",
    "technique",
    "params_catalog",
    "model_profile",
})


# Critic categories → KB kinds. Categories without an entry are treated as
# "no KB equivalent" — caller may surface them in ``rejected_candidates``.
CATEGORY_TO_KIND: dict[str, str] = {
    "pitfall": "pitfall",
    "crash_recovery": "pitfall",
    "benchmark_methodology": "pitfall",
    "architecture_constraint": "pitfall",
    "kernel_optimization": "technique",
    "call_stack_optimization": "technique",
    "backend_exploration": "technique",
    "framework_comparison": "technique",
    "target_comparison": "technique",
    "lesson": "technique",
    "dream_consolidation": "technique",
    "server_params": "params_catalog",
}


def map_category_to_kind(category: str) -> str:
    """Return the KB ``kind`` for a Critic ``category``.

    Raises:
        RuntimeAdapterError: If ``category`` is not in the catalogue.
    """
    if not isinstance(category, str):
        raise RuntimeAdapterError(
            f"category must be str, got {type(category).__name__}"
        )
    kind = CATEGORY_TO_KIND.get(category)
    if not kind:
        raise RuntimeAdapterError(
            f"unsupported Critic category {category!r}; mapping table only "
            f"covers {sorted(CATEGORY_TO_KIND.keys())!r}"
        )
    return kind


def filter_supported_categories(
    categories: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Split an iterable into ``(supported, rejected)`` category lists."""
    supported: list[str] = []
    rejected: list[str] = []
    for c in categories:
        if c in CATEGORY_TO_KIND:
            supported.append(c)
        else:
            rejected.append(c)
    return supported, rejected


__all__ = [
    "CATEGORY_TO_KIND",
    "KB_KINDS",
    "filter_supported_categories",
    "map_category_to_kind",
]
