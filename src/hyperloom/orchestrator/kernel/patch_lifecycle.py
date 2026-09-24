# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared patch-lifecycle completion contract for integrate lanes."""

from __future__ import annotations

from typing import Any

CLEANUP_COMPLETE = "complete"
CLEANUP_RECOVERY_REQUIRED = "recovery_required"

CLEANUP_ACTION_NONE = ""
CLEANUP_ACTION_FINALIZE = "finalize"
CLEANUP_ACTION_REVERT = "revert"


def lifecycle_complete(result: Any) -> bool:
    """Return True when a finalize or revert left nothing owed."""
    return isinstance(result, dict) and result.get("status") in {"ok", "skipped"}


def revert_owed(result: Any) -> bool:
    """Return True when a finished attempt left the target tree holding its patches.

    An owed finalize is a backup that outlived its patch, which costs disk and
    nothing else; an owed revert means the sources on disk are not the sources
    the session believes it is measuring, so no later step may build on them.
    """
    return isinstance(result, dict) and result.get("patch_cleanup_action") == CLEANUP_ACTION_REVERT


def cleanup_verdict(
    *,
    decision: str,
    revert_result: dict[str, Any],
    finalize_result: dict[str, Any],
    revert_required: bool,
) -> tuple[str, str, str]:
    """Return (top_status, patch_cleanup_status, patch_cleanup_action)."""
    if decision == "KEEP":
        if lifecycle_complete(finalize_result):
            return "ok", CLEANUP_COMPLETE, CLEANUP_ACTION_NONE
        return "ok", CLEANUP_RECOVERY_REQUIRED, CLEANUP_ACTION_FINALIZE

    if not revert_required or lifecycle_complete(revert_result):
        return "ok", CLEANUP_COMPLETE, CLEANUP_ACTION_NONE

    return "failed", CLEANUP_RECOVERY_REQUIRED, CLEANUP_ACTION_REVERT
