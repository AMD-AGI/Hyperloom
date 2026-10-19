# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Translate Critic ``kb_drafts[].category`` into KB ``kind`` (contract §2.1)."""

from __future__ import annotations


from .errors import RuntimeAdapterError


# Critic categories → KB kinds; categories without an entry have no KB equivalent.
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
    """Return the KB ``kind`` for a Critic ``category``."""
    if not isinstance(category, str):
        raise RuntimeAdapterError(f"category must be str, got {type(category).__name__}")
    kind = CATEGORY_TO_KIND.get(category)
    if not kind:
        raise RuntimeAdapterError(
            f"unsupported Critic category {category!r}; mapping table only covers {sorted(CATEGORY_TO_KIND.keys())!r}"
        )
    return kind


__all__ = [
    "CATEGORY_TO_KIND",
    "map_category_to_kind",
]
