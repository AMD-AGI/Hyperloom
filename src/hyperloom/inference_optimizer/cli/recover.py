# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Offline report recovery and explicit task-cleanup confirmation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RECOVER_HELP = (
    "Rebuild + push the session_breakdown for a session that exited abnormally (crash / SIGKILL) so its breakdown "
    "lands on Langfuse; or, with --confirm-scope-gone / --confirm-stopped, record the cleanup confirmation a refused "
    "--resume-from asks for."
)
RECOVER_DESCRIPTION = (
    "Without a confirmation option, rebuild and push the session_breakdown of a session that exited abnormally. "
    "With --confirm-scope-gone OWNER_SCOPE (an execution environment that no longer exists) or --confirm-stopped "
    "TASK_ID (one blocked task that names no owner scope), plus --confirmation-reason, instead record the cleanup "
    "confirmation a refused --resume-from asks for; nothing is rebuilt. Exit codes: 0 success, 1 breakdown rebuild "
    "failed, 2 invalid arguments or a refused confirmation."
)


def add_cleanup_confirmation_arguments(rec: argparse.ArgumentParser) -> None:
    """Register the operator cleanup confirmations on the ``recover`` subcommand."""
    rec.add_argument(
        "--confirm-stopped",
        metavar="TASK_ID",
        help="Attest that one task's complete process tree, remote workers and Ray actor have stopped, "
        "then cancel unfinished work and release only its unattributed execution/GPU ownership records. Requires POSIX session locking "
        "and --confirmation-reason; rejects recorded nonempty owner scopes. Does not stop processes, "
        "accept old results, rebuild reports or resume execution.",
    )
    rec.add_argument(
        "--confirm-scope-gone",
        metavar="OWNER_SCOPE",
        help="Attest that the whole execution environment recorded as OWNER_SCOPE (a boot, PID namespace and container start, "
        "as printed by a blocked resume) no longer exists, e.g. its machine or container was lost. Cancels the "
        "unfinished tasks recorded under it so they are redone rather than counted, releases their execution/GPU "
        "ownership records and appends the confirmation to runtime/pod_history.jsonl. Refuses the current "
        "environment's scope and refuses while another scope holds an unexpired lease. Requires POSIX session "
        "locking and --confirmation-reason. Does not stop processes or resume execution.",
    )
    rec.add_argument(
        "--confirmation-reason",
        metavar="TEXT",
        help="Required audit reason for --confirm-stopped or --confirm-scope-gone. Exactly one of those options "
        "and this reason must be provided together, and they cannot be combined with --force.",
    )


def add_recover_subparser(sub: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """Register the ``recover`` subcommand."""
    rec = sub.add_parser("recover", parents=[common], help=RECOVER_HELP, description=RECOVER_DESCRIPTION)
    rec.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Session directory of the crashed run (contains state.json / reports/trace/ and the recorder fragments).",
    )
    rec.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when the session already looks complete (close_sequence_done / breakdown already recorded).",
    )
    add_cleanup_confirmation_arguments(rec)


def _session_recovery_status(session_dir: Path) -> dict[str, Any]:
    """Inspect on-disk artifacts to judge whether a session finished cleanly."""

    from ..session.session_paths import BREAKDOWN_FILENAME

    state_path = session_dir / "state.json"
    close_done = False
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            close_done = bool((state or {}).get("close_sequence_done"))
        except (json.JSONDecodeError, OSError):
            close_done = False

    breakdown_exists = (session_dir / BREAKDOWN_FILENAME).exists()

    from ..trace.langfuse_emitter import read_receipt

    receipt = read_receipt(session_dir) or {}
    counts = receipt.get("counts") or {}
    breakdown_recorded = bool(counts.get("breakdown_recorded"))
    counts_final = bool(receipt.get("counts_final"))

    return {
        "close_done": close_done,
        "breakdown_exists": breakdown_exists,
        "breakdown_recorded": breakdown_recorded,
        "counts_final": counts_final,
        "looks_complete": close_done and breakdown_recorded and breakdown_exists,
    }


def _run_cleanup_confirmation(
    session_dir: Path, args: argparse.Namespace, *, task_id: str | None, scope: str | None, reason: str | None
) -> int:
    """Record one operator cleanup confirmation: a stopped task, or a vanished owner scope."""
    target = task_id if scope is None else scope
    if (task_id is None) == (scope is None) or not target or not target.strip() or not reason or not reason.strip():
        print(
            "ERROR: exactly one of --confirm-stopped / --confirm-scope-gone and a nonempty --confirmation-reason "
            "are required together.",
            file=sys.stderr,
        )
        return 2
    if getattr(args, "force", False):
        print("ERROR: cleanup confirmation cannot be combined with --force.", file=sys.stderr)
        return 2
    from ..session.resume_guard import CleanupConfirmationError, confirm_scope_gone, confirm_task_stopped

    try:
        if scope is None:
            result = confirm_task_stopped(session_dir, task_id=task_id, reason=reason)
        else:
            from hyperloom.common.owner_scope import local_owner_scope

            result = confirm_scope_gone(
                session_dir, owner_scope=scope, reason=reason, current_scope=local_owner_scope()
            )
    except CleanupConfirmationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if scope is None:
        print(
            f"cleanup confirmation: task={ascii(task_id[:80])} status={result['status']} "
            f"released_leases={result['released_leases']} released_gpu_leases={result['released_gpu_leases']}"
        )
        return 0
    cancelled = result["cancelled_tasks"]
    print(
        f"cleanup confirmation: owner_scope={ascii(scope[:80])} status={result['status']} "
        f"cancelled_tasks={len(cancelled)} released_leases={result['released_leases']} "
        f"released_gpu_leases={result['released_gpu_leases']}"
    )
    for cancelled_task in cancelled:
        print(f"  cancelled task={ascii(cancelled_task[:80])}")
    if not result["ledger_recorded"]:
        print("WARNING: the owner ledger runtime/pod_history.jsonl could not be appended.", file=sys.stderr)
    return 0


def _run_recover_session(args: argparse.Namespace) -> int:
    """Offline recovery for a session that exited abnormally."""
    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        print(f"ERROR: session dir not found: {session_dir}", file=sys.stderr)
        return 2

    task_id = getattr(args, "confirm_stopped", None)
    scope = getattr(args, "confirm_scope_gone", None)
    reason = getattr(args, "confirmation_reason", None)
    if task_id is not None or scope is not None or reason is not None:
        return _run_cleanup_confirmation(session_dir, args, task_id=task_id, scope=scope, reason=reason)

    status = _session_recovery_status(session_dir)
    print(
        f"recover           : {session_dir}\n"
        f"  close_sequence_done={status['close_done']} "
        f"breakdown_exists={status['breakdown_exists']} "
        f"breakdown_recorded={status['breakdown_recorded']} "
        f"counts_final={status['counts_final']}"
    )
    if status["looks_complete"] and not args.force:
        print("  -> already complete (breakdown built and recorded to Langfuse); pass --force to rebuild anyway.")
        return 0

    # 1) Rebuild/merge the breakdown from whatever fragments survived the crash.
    try:
        from ..breakdown import write_breakdown_json

        breakdown_path = write_breakdown_json(session_dir)
        print(f"  rebuilt breakdown : {breakdown_path}")
    except Exception:
        log.exception("recover: breakdown rebuild failed")
        return 1

    # 2) Reconcile + flush Langfuse, splice the final receipt, attach the SBD.
    try:
        from ..breakdown import patch_breakdown_langfuse
        from ..trace.langfuse_emitter import (
            flush_session,
            record_session_breakdown,
        )

        flush_session(session_dir)
        patch_breakdown_langfuse(session_dir)
        record_session_breakdown(session_dir)
        print("  langfuse          : flushed + breakdown attached")
    except (OSError, ValueError):
        log.exception("recover: langfuse push failed (non-fatal)")

    # 3) Re-package the artifact bundle so /workspace carries the recovered SBD.
    try:
        from ..breakdown import package_session_artifacts

        pkg_path = package_session_artifacts(session_dir)
        if pkg_path is not None:
            print(f"  artifact package  : {pkg_path}")
    except (OSError, ValueError):
        log.exception("recover: artifact package failed (non-fatal)")

    return 0
