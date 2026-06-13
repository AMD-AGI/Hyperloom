#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.
"""Print concise optimizer state and lifecycle status.

Usage:
    python inference_optimizer/launcher/read_optimizer_state.py SESSION_DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


SUMMARY_KEYS = (
    "stop_reason",
    "baseline_tput",
    "cumulative_gain",
    "current_best",
    "last_kernel_opt",
    "last_trace_analyze",
    "last_sweep",
)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _format_lifecycle_event(event: dict[str, Any]) -> str:
    duration = f" {event['duration_s']}s" if event.get("duration_s") is not None else ""
    detail = f" [{event['detail']}]" if event.get("detail") else ""
    artifacts = " ".join(
        f"{key}={value}" for key, value in (event.get("artifacts") or {}).items()
    )
    line = (
        f"#{event.get('seq')} {event.get('label')} [{event.get('phase')}] "
        f"{event.get('status')}{duration}{detail}"
    )
    if artifacts:
        line += f" -> {artifacts}"
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", help="Optimizer session directory.")
    parser.add_argument(
        "--lifecycle-limit",
        type=int,
        default=12,
        help="Number of recent lifecycle events to print (default: 12).",
    )
    args = parser.parse_args()

    session_dir = pathlib.Path(args.session_dir)
    state_path = session_dir / "state.json"
    manifest_path = session_dir / "manifest.json"
    if not state_path.is_file():
        print(f"state.json not found at {state_path}", file=sys.stderr)
        return 2

    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        print("session_id:", manifest.get("session_id"))
    print("session_dir:", session_dir)

    state = _load_json(state_path)
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
