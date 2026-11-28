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
#: False on a variant a pre-bench filter dropped, which reached the bench and was never launched.
RESULT_BENCHED = "benched"

#: True on a result whose verdict the performance gate reached on a round the session could grade against its base:
#: the candidate was judged, on that comparison or on the accuracy or switch-off parity it then owed. Absent or
#: false on every other result, which is no verdict on the candidate: among them a parity leg that returned no
#: throughput or had no base to compare with.
RESULT_GRADED = "graded"

#: Set on every result of an attempt that reached the apply stage: whether it put changed framework code where its
#: server imports from (:attr:`IntegrateAttempt.changes_framework`), so a reader of the result can attribute a
#: framework the server could not import.
RESULT_FRAMEWORK_CHANGED = "framework_changed"

#: Error class of a bench the environment kept from measuring: no verdict on the patch.
BENCH_UNMEASURED_CLASS = "bench_unmeasured"

#: Error class of a variant a pre-bench filter dropped: it was refused, not measured.
FILTERED_BEFORE_BENCH_CLASS = "filtered_before_bench"

#: Where the patches this action applies came from. Every source lands through
#: the same apply / vet / bench / KEEP-REVERT pipeline below; they differ only
#: in how the diff is obtained (see :meth:`IntegratePatchExecutor._stage_resolve`).
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
