#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Print concise optimizer state and lifecycle status.

Usage:
    python src/hyperloom/inference_optimizer/tools/read_optimizer_state.py SESSION_DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json
from hyperloom.inference_optimizer.session.session_paths import manifest_path, state_path


SUMMARY_KEYS = (
    "stop_reason",
    "baseline_tput",
    "cumulative_gain_validated",
    "current_best",
    "last_kernel_opt",
    "last_trace_analyze",
)


def _format_lifecycle_event(event: dict[str, Any]) -> str:
    """Render a single lifecycle event as a one-line human-readable summary.

    Args:
        event: A lifecycle event dict (seq, label, phase, status, optional
            duration_s, detail, and artifacts).

    Returns:
        A formatted single-line string describing the event, including any
        artifacts when present.
    """
    duration = f" {event['duration_s']}s" if event.get("duration_s") is not None else ""
    detail = f" [{event['detail']}]" if event.get("detail") else ""
    artifacts = " ".join(f"{key}={value}" for key, value in (event.get("artifacts") or {}).items())
    line = f"#{event.get('seq')} {event.get('label')} [{event.get('phase')}] {event.get('status')}{duration}{detail}"
    if artifacts:
        line += f" -> {artifacts}"
    return line


def main() -> int:
    """Print a concise optimizer state and recent lifecycle summary.

    Reads ``state.json`` (and optional ``manifest.json``) from the session
    directory and prints the summary keys, phase, and the most recent
    lifecycle events.

    Returns:
        ``0`` on success, or ``2`` when ``state.json`` is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", help="Optimizer session directory.")
    parser.add_argument(
        "--lifecycle-limit",
        type=int,
        default=12,
        help="Number of recent lifecycle events to print (default: 12).",
    )
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    state_file = state_path(session_dir)
    manifest_file = manifest_path(session_dir)
    if not state_file.is_file():
        print(f"state.json not found at {state_file}", file=sys.stderr)
        return 2

    if manifest_file.is_file():
        manifest = read_json(manifest_file, require_dict=True, strict=True)
        print("session_id:", manifest.get("session_id"))
    print("session_dir:", session_dir)

    state = read_json(state_file, require_dict=True, strict=True)
    for key in SUMMARY_KEYS:
        print(f"{key}: {state.get(key)}")
    print("explore_last_round:", state.get("explore_search", {}).get("last_round"))
    print("phase:", state.get("phase"))

    events = state.get("lifecycle") or []
    limit = max(0, int(args.lifecycle_limit))
    print(f"\n--- lifecycle (last {limit} of {len(events)}) ---")
    for event in events[-limit:] if limit else []:
        print(_format_lifecycle_event(event))
    return 0


if __name__ == "__main__":
    sys.exit(main())
