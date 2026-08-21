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


def lifecycle_complete(result: Any) -> bool:
    """Return True when a finalize or revert left nothing owed.

    ``"partial"`` is owed work for either: a remote pod can be left with the
    patch still live, or with its backups undeleted. What that costs differs per
    branch, which is :func:`cleanup_verdict`'s call to make.
    """
    return isinstance(result, dict) and result.get("status") in {"ok", "skipped"}


def finalize_settled(result: Any) -> bool:
    """Return True when finalize will not run again for this apply.

    Distinct from :func:`lifecycle_complete`, which grades what a cleanup still
    owes. A finalize that reached a terminal manifest state is *settled* even
    when it only removed part of the backups: the manifest cannot be finalized
    again, so re-entering finalize repeats work that is done and leaves the
    apply checkpoint on disk forever. Revert has no equivalent — a partial
    revert may still leave the patch live on a remote pod, which is exactly why
    :func:`lifecycle_complete` refuses to call it complete.

    Callers mark such a result with ``settled=True``; anything
    :func:`lifecycle_complete` already accepts is settled too.
    """
    return isinstance(result, dict) and (lifecycle_complete(result) or bool(result.get("settled")))


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
        if lifecycle_complete(finalize_result):
            return "ok", CLEANUP_COMPLETE, ""
        return "ok", CLEANUP_RECOVERY_REQUIRED, "finalize"

    if not revert_required or lifecycle_complete(revert_result):
        return "ok", CLEANUP_COMPLETE, ""

    return "failed", CLEANUP_RECOVERY_REQUIRED, "revert"
