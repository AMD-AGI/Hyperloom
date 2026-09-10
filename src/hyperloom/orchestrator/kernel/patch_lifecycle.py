# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared patch-lifecycle completion contract for integrate lanes."""

from __future__ import annotations

from typing import Any

CLEANUP_COMPLETE = "complete"
CLEANUP_RECOVERY_REQUIRED = "recovery_required"


def lifecycle_complete(result: Any) -> bool:
    """Return True when a finalize or revert left nothing owed."""
    return isinstance(result, dict) and result.get("status") in {"ok", "skipped"}


def finalize_settled(result: Any) -> bool:
    """Return True when finalize will not run again for this apply."""
    return isinstance(result, dict) and (lifecycle_complete(result) or bool(result.get("settled")))


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
            return "ok", CLEANUP_COMPLETE, ""
        return "ok", CLEANUP_RECOVERY_REQUIRED, "finalize"

    if not revert_required or lifecycle_complete(revert_result):
        return "ok", CLEANUP_COMPLETE, ""

    return "failed", CLEANUP_RECOVERY_REQUIRED, "revert"
