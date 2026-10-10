###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""The one routable/skipped splitter shared by the compute and bypass paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def partition_kernels(
    hot_kernels: list[dict[str, Any]],
    is_routable: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a hot-kernel list into ``(routable, skipped)`` one way, everywhere.

    The two callers pass different predicates -- compute's coarse
    ``reusable_native_kernel`` reusability and bypass's stricter routable gate --
    but the partition itself is one rule, so both import it here.
    """
    routable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in hot_kernels:
        if not isinstance(row, dict):
            continue
        (routable if is_routable(row) else skipped).append(row)
    return routable, skipped


def build_kernel_candidates_document(
    header: dict[str, Any],
    candidates: list[dict[str, Any]],
    task_groups: list[dict[str, Any]],
    *,
    is_routable: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Assemble the ``kernel_candidates.json`` document one way, for both routes.

    Both routes call this with their own header, task_groups, and routability predicate;
    the four-list shape (hot_kernels, routable_kernels, skipped_kernels, task_groups) is the contract.
    """
    routable_kernels, skipped_kernels = partition_kernels(candidates, is_routable)
    return {
        **header,
        "hot_kernels": candidates,
        "routable_kernels": routable_kernels,
        "skipped_kernels": skipped_kernels,
        "task_groups": task_groups,
    }
