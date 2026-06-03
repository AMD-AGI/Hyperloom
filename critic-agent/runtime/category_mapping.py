"""Translate Critic ``kb_drafts[].category`` into KB ``kind`` (contract §2.1).

The KB contract enumerates exactly four ``kind`` values; the Critic
SKILL operates on a richer category vocabulary (see
``actions/draft_kb.md`` and ``references/verdict_schema.md``). This
module provides a deterministic surjective mapping plus a list of
categories that intentionally have no KB equivalent and should be
``rejected`` rather than silently dropped.
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

    Args:
        category (str): The Critic category to translate.

    Returns:
        str: The mapped KB ``kind``.

    Raises:
        RuntimeAdapterError: If ``category`` is not a string or is not in the
            catalogue.
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
    """Split an iterable into ``(supported, rejected)`` category lists.

    Args:
        categories (Iterable[str]): Critic categories to partition.

    Returns:
        tuple[list[str], list[str]]: ``(supported, rejected)`` where supported
        categories appear in the mapping table and rejected ones do not.
    """
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
