# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Result codes and audit rows for publishing the final Recipe to the remote KB Store."""

from __future__ import annotations

import logging as _logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom_common.io import JsonLineFormat, append_jsonl
from hyperloom.inference_optimizer.breakdown.recorder import close_out as _close_out

log = _logging.getLogger(__name__)


# Stable ``result_type`` codes for the reasons the remote KB Store returns.
# Use an exact lookup so a reason token cannot collide with the same text inside
# an exception name or explanatory message.
_REMOTE_RESULT_TYPES: dict[str, str] = {
    "no_new_keep_or_pure_warm_replay": _close_out.RESULT_NO_NEW_KEEP,
    "nonfinite_optimized_throughput": _close_out.RESULT_INVALID_THROUGHPUT,
    "missing_optimized_throughput": _close_out.RESULT_MISSING_THROUGHPUT,
    "invalid_recipe_scope": _close_out.RESULT_INVALID_SCOPE,
    "invalid_recipe_selection_profile": _close_out.RESULT_INVALID_SELECTION_PROFILE,
    "empty_replay_material": _close_out.RESULT_EMPTY_REPLAY_MATERIAL,
    "not_better_than_champion": _close_out.RESULT_NOT_BETTER,
    "champion_not_promoted": _close_out.RESULT_CHAMPION_NOT_PROMOTED,
}

# The one exception class meaning the store rejected the bundle, rather than
# that we never reached the store at all.
REMOTE_VALIDATION_ERROR = "RemoteRecipeValidationError"


@dataclass(frozen=True)
class RemoteRecipeAudit:
    """The outcome of one remote Recipe KB publish attempt, as an audit row records it."""

    status: str
    canonical_id: str
    session_id: str
    primary_metric: str = ""
    primary_value: float = 0.0
    reason: str = ""
    error: Exception | None = None


def remote_result_type(status: str, reason: str) -> str:
    """The ``close_out.RESULT_*`` code for a remote KB Store write result.

    ``status`` is consulted only when the store's own reason token is
    unrecognized.
    """
    known = _REMOTE_RESULT_TYPES.get(str(reason or "").strip())
    if known:
        return known
    verdict = str(status or "").strip().lower()
    if verdict == "written":
        return _close_out.RESULT_WRITTEN
    if verdict in {"skipped", "disabled"}:
        return _close_out.RESULT_SKIPPED_OTHER
    return _close_out.RESULT_TRANSPORT_FAILED


def record_remote_recipe_audit(session_dir: Path, source: str, audit: RemoteRecipeAudit) -> None:
    """Append one best-effort, secret-free KB Store publish audit row."""
    try:
        from .remote_recipe.audit import publish_error_fields
        from hyperloom.inference_optimizer.session.session_paths import (
            recipe_snapshot_audit_jsonl,
        )

        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "op": "write",
            "method": "write",
            "mode": "remote",
            "backend": "kb-store",
            "remote": "kb-store",
            "resolution": audit.status,
            "success": audit.status in {"written", "skipped"},
            "generator": source,
            "phase": "CLOSE",
            "status": audit.status,
            "reason": audit.reason,
            "request": {
                "canonical_id": audit.canonical_id or None,
                "session_id": audit.session_id or None,
            },
            "result": {
                "canonical_id": audit.canonical_id,
                "session_id": audit.session_id,
                "created": audit.status == "written",
            },
            "provenance": {
                "component": "remote_recipe",
                "source": source,
            },
        }
        if audit.primary_metric:
            row["result"].update(
                {
                    "primary_metric": audit.primary_metric,
                    "primary_value": audit.primary_value,
                }
            )
        if audit.primary_metric == "optimized_throughput":
            row["result"]["best_throughput"] = audit.primary_value
        if audit.error is not None:
            row["error"] = publish_error_fields(audit.error)
        append_jsonl(
            recipe_snapshot_audit_jsonl(session_dir),
            row,
            make_parents=True,
            fmt=JsonLineFormat(sort_keys=True),
        )
    except OSError:
        log.debug("Remote Recipe KB audit append failed", exc_info=True)
