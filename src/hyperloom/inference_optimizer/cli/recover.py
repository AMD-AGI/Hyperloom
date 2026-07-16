# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI entry — ``optimize`` subcommand wiring Claude+Codex backends, executors, objective, and Coordinator.run().

Env vars consumed: MODEL_PATH, OPENAI_BASE_URL + SAFE_API_KEY, ROCR_VISIBLE_DEVICES,
CLAUDE_MODEL, CODEX_MODEL, USER_DATA_PATH.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

def _session_recovery_status(session_dir: Path) -> dict[str, Any]:
    """Inspect on-disk artifacts to judge whether a session finished cleanly.

    Pure read of state.json / session_breakdown.json / langfuse_receipt.json.
    Returns flags used by :func:`_run_recover_session` to decide whether the
    session still needs a (re)build + Langfuse push.

    Args:
        session_dir (Path): The session directory to inspect.

    Returns:
        dict[str, Any]: A status mapping with ``close_done``,
            ``breakdown_exists``, ``breakdown_recorded``, ``counts_final``,
            and ``looks_complete`` flags.
    """

    from ..breakdown import BREAKDOWN_FILENAME

    state_path = session_dir / "state.json"
    close_done = False
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            close_done = bool((state or {}).get("close_sequence_done"))
        except (json.JSONDecodeError, OSError):
            close_done = False

    breakdown_exists = (session_dir / BREAKDOWN_FILENAME).exists()

    from hyperloom.orchestrator.trace.langfuse_emitter import read_receipt

    receipt = read_receipt(session_dir) or {}
    counts = receipt.get("counts") or {}
    breakdown_recorded = bool(counts.get("breakdown_recorded"))
    counts_final = bool(receipt.get("counts_final"))

    return {
        "close_done": close_done,
        "breakdown_exists": breakdown_exists,
        "breakdown_recorded": breakdown_recorded,
        "counts_final": counts_final,
        "looks_complete": close_done and breakdown_recorded,
    }

def _run_recover_session(args: argparse.Namespace) -> int:
    """Offline recovery for a session that exited abnormally.

    Rebuilds ``session_breakdown.json`` from the crash-time recorder fragments
    (the merge step), reconciles + flushes Langfuse, splices the post-flush
    receipt into the breakdown, and attaches the full breakdown JSON to the
    session's trace. Idempotent across processes (guarded by the persisted
    Langfuse receipt), so re-running is safe.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads
            ``session_dir``, ``force``, and ``backfill_trace``).

    Returns:
        int: The process exit code (``0`` on success, ``2`` when the session
            dir is missing, ``1`` on breakdown rebuild failure).
    """
    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        print(f"ERROR: session dir not found: {session_dir}", file=sys.stderr)
        return 2

    status = _session_recovery_status(session_dir)
    print(
        f"recover-session   : {session_dir}\n"
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
    except Exception:  # noqa: BLE001
        log.exception("recover-session: breakdown rebuild failed")
        return 1

    # 2) Reconcile + flush Langfuse, splice the final receipt, attach the SBD.
    try:
        from ..breakdown import patch_breakdown_langfuse
        from hyperloom.orchestrator.trace.langfuse_emitter import (
            flush_session,
            record_session_breakdown,
        )

        flush_session(session_dir)
        patch_breakdown_langfuse(session_dir)
        record_session_breakdown(session_dir)
        print("  langfuse          : flushed + breakdown attached")
    except Exception:  # noqa: BLE001
        log.exception("recover-session: langfuse push failed (non-fatal)")

    # 3) Optional full generation replay (off by default).
    if args.backfill_trace:
        try:
            from ..tools.backfill_langfuse import build_plan, ingest

            rc = ingest(build_plan(session_dir))
            print(f"  trace backfill    : rc={rc}")
        except Exception:  # noqa: BLE001
            log.exception("recover-session: trace backfill failed (non-fatal)")

    # 4) Re-package the artifact bundle so /workspace carries the recovered SBD.
    try:
        from ..breakdown import package_session_artifacts

        pkg_path = package_session_artifacts(session_dir)
        if pkg_path is not None:
            print(f"  artifact package  : {pkg_path}")
    except Exception:  # noqa: BLE001
        log.exception("recover-session: artifact package failed (non-fatal)")

    return 0
