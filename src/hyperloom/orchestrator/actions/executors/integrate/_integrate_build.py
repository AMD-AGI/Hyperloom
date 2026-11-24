# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Building an applied candidate whose framework server is built from source, before anything measures it.

A session that builds its server (:class:`~..candidate_builds.CandidateBuilds`) builds the framework checkout as the
candidate left it; the bench then runs that build, first on the server's ``PATH``. A candidate that does not compile,
or changes what no build may take, is reverted without a bench; a build the machine failed is no verdict on it.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ....state.failure_evidence import FAILURE_ATTRIBUTION_ENVIRONMENT
from .._integrate_attempt import IntegrateAttempt
from ..candidate_builds import CandidateBuilds, TreeBuildStatus
from ..patching._patch_snapshot import _patch_touched_paths

log = logging.getLogger(__name__)

#: Error class of a candidate whose tree did not build: the candidate's failure, reverted without a bench.
_NATIVE_COMPILE_ERROR_CLASS = "native_compile_error"
#: Error class of a build the machine failed: no verdict on the candidate.
_NATIVE_BUILD_INFRA_CLASS = "native_build_infra"


class BuildStageMixin:
    """Build the applied candidate's tree when the session builds its framework's server."""

    native: CandidateBuilds | None

    async def _stage_build(self, attempt: IntegrateAttempt, params: dict[str, Any]) -> dict[str, Any] | None:
        """Build the candidate the apply stage left in the tree; the verdict of one that does not build, or ``None`` to
        go on (the bench then runs the build, :meth:`IntegrateAttempt.benchmark_params`)."""
        native = self.native
        root = attempt.framework_root
        if native is None or not attempt.changes_framework:
            return None
        if root is None or Path(root).resolve() != Path(native.checkout).resolve():
            return _build_failed(
                attempt,
                f"the candidate changed {root or 'no tree'}, and the session builds its server from {native.checkout}",
            )
        paths = _patch_touched_paths(root, attempt.applied) + [
            str(entry["rel_target"]) for entry in attempt.applied_artifacts if entry.get("rel_target")
        ]
        build = await asyncio.to_thread(native.build_tree, paths)
        attempt.native_build = build
        log.info(
            "integrate_patch: native build of task=%s: %s key=%s reused=%s",
            attempt.specialist_task_id,
            build.status.value,
            build.key,
            build.reused,
        )
        if build.status is TreeBuildStatus.BUILT:
            return None
        if build.status is TreeBuildStatus.INFRA:
            return _build_failed(attempt, build.detail)
        return {
            "status": "reverted",
            "error_class": _NATIVE_COMPILE_ERROR_CLASS,
            "error": build.detail,
            "reason": f"the candidate does not build: {build.detail[-500:]}",
            "specialist_task_id": attempt.specialist_task_id,
            "patches_applied": [],
            "workspace": str(attempt.output_root),
        }

    def _accept_build(self, attempt: IntegrateAttempt) -> None:
        """Make the build of an attempt the session keeps its accepted build, the one later rounds run."""
        if attempt.native_build is not None and self.native is not None:
            self.native.accept(attempt.native_build)


def _build_failed(attempt: IntegrateAttempt, detail: str) -> dict[str, Any]:
    """The result of a candidate the machine could not build: no verdict on it."""
    return {
        "status": "failed",
        "error_class": _NATIVE_BUILD_INFRA_CLASS,
        "failure_attribution": FAILURE_ATTRIBUTION_ENVIRONMENT,
        "error": detail,
        "reason": f"the candidate could not be built here: {detail[-500:]}",
        "specialist_task_id": attempt.specialist_task_id,
        "patches_applied": [],
        "workspace": str(attempt.output_root),
    }
