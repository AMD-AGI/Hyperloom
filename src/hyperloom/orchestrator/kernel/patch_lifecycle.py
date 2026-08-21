# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared patch-lifecycle completion contract for integrate lanes.

Leaf module (zero intra-package imports). Three result axes:
- ``status``             — did this integrate produce a valid verdict?
- ``decision``           — KEEP / REVERT / NEEDS_REVIEW
- ``patch_cleanup_status`` — did the owed cleanup (finalize/revert) complete?

KEEP + finalize failure: status stays ``ok``; the patch is correctly on tree,
only backup deletion is outstanding (``recovery_required``).
Non-KEEP + revert partial/failed: status becomes ``failed``; patch may still
be live on a remote pod.
"""

from __future__ import annotations

from typing import Any

CLEANUP_COMPLETE = "complete"
CLEANUP_RECOVERY_REQUIRED = "recovery_required"


def cleanup_complete(result: Any) -> bool:
    """Return True when a KEEP finalize reached a terminal state (ok or partial)."""
    return isinstance(result, dict) and result.get("status") in {"ok", "partial"}


def revert_complete(result: Any) -> bool:
    """Return True when a non-KEEP revert fully restored all patch sites.

    Rejects ``"partial"`` — remote-pod revert failure means the patch is still live.
    """
    return isinstance(result, dict) and result.get("status") in {"ok", "skipped"}


def cleanup_verdict(
    *,
    decision: str,
    revert_result: dict[str, Any],
    finalize_result: dict[str, Any],
    revert_required: bool,
) -> tuple[str, str, str]:
    """Return (top_status, patch_cleanup_status, patch_cleanup_action).

    top_status is ``"failed"`` only when a required non-KEEP revert did not
    fully complete (patch may still be on tree). KEEP + finalize failure stays
    ``"ok"``; cleanup_status tracks the outstanding backup deletion.
    """
    if decision == "KEEP":
        if cleanup_complete(finalize_result):
            return "ok", CLEANUP_COMPLETE, ""
        return "ok", CLEANUP_RECOVERY_REQUIRED, "finalize"

    if not revert_required or revert_complete(revert_result):
        return "ok", CLEANUP_COMPLETE, ""

    return "failed", CLEANUP_RECOVERY_REQUIRED, "revert"
