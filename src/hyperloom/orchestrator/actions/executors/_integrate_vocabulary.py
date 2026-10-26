# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The words an integration result and a patch source are named with.

A leaf on purpose: the phases and the writeback loop read this vocabulary to
grade a result they were handed, and must not pull the executor that produced
it in behind the words.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


#: Verdicts that accept the candidate, so the attempt leaves it in the tree.
KEEP_STATUSES: frozenset[str] = frozenset({"kept", "advanced", "kept_inert"})

#: Set on every result the bench-and-gate stage returned, whatever its verdict:
#: that attempt reached the bench, and no result an earlier stage returned did.
RESULT_BENCHED = "benched"

#: Task param naming the screen a switched patch goes through instead of a replay of its own; carried on the result.
SCREEN_PARAM = "screen"

#: The cycle bench screens the patch's levers, so integration commits the patch dormant without benching it.
SCREEN_CYCLE = "cycle"

#: Attribution source of a lever admitted for the cycle screen and not measured yet.
ATTRIBUTION_CYCLE_PENDING = "cycle_bench_pending"

#: Attribution sources a cycle batch writes: a graded gain, an arm whose server failed, a batch whose bases disagreed.
ATTRIBUTION_CYCLE_BENCH = "cycle_bench"
ATTRIBUTION_CYCLE_BROKEN = "cycle_bench_broken"
ATTRIBUTION_CYCLE_INCONCLUSIVE = "cycle_bench_inconclusive"

#: Where the patches this action applies came from. Every source lands through
#: the same apply / vet / bench / KEEP-REVERT pipeline below; they differ only
#: in how the diff is obtained and, for ``upstream_pr``, in which gate admitted
#: it (see :meth:`IntegratePatchExecutor._stage_resolve`).
PATCH_SOURCE_SPECIALIST = "specialist_authored"

PATCH_SOURCE_UPSTREAM_PR = "upstream_pr"

PATCH_SOURCES = (PATCH_SOURCE_SPECIALIST, PATCH_SOURCE_UPSTREAM_PR)


def resolve_patch_source(params: Mapping[str, Any]) -> str:
    """Return the declared patch source, defaulting to the specialist lane.

    Args:
        params: Task params.

    Returns:
        One of :data:`PATCH_SOURCES`.
    """
    declared = str(params.get("patch_source") or "").strip().lower()
    return declared if declared in PATCH_SOURCES else PATCH_SOURCE_SPECIALIST
