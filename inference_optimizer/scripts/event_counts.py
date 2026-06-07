#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Print action / proposal / kernel counts from a session's coordinator.db.

Usage:
    event_counts.py [SESSION_DIR] [--all] [--limit N]

SESSION_DIR defaults to $USER_DATA_PATH or /workspace/hyperloom.
By default reads the last 500 events to mirror the legacy behaviour; pass
``--all`` for full history or ``--limit N`` for a custom window. The
500-event default rotates older rounds out and silently undercounts on
long runs — use ``--all`` when comparing totals against the run report.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
from collections import Counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session_dir",
        nargs="?",
        default=None,
        help="Session directory (default: $USER_DATA_PATH or "
             "/workspace/hyperloom)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Scan the entire events table (no window).",
    )
    group.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Window size for the most-recent events (default: 500).",
    )
    args = parser.parse_args()

    if args.session_dir is not None:
        session_dir = pathlib.Path(args.session_dir)
    else:
        # Defer to inference_optimizer.paths so resolution rules
        # (env > default) stay in one place.
        from inference_optimizer.paths import session_dir as _resolve_sd
        session_dir = _resolve_sd()
    db = pathlib.Path(session_dir) / "storage" / "coordinator.db"
    if not db.exists():
        print(f"coordinator.db not found at {db}", file=sys.stderr)
        return 2

    if args.all:
        query = (
            "SELECT from_agent, to_agent, topic, payload FROM events "
            "ORDER BY seq DESC"
        )
        params: tuple = ()
    else:
        query = (
            "SELECT from_agent, to_agent, topic, payload FROM events "
            "ORDER BY seq DESC LIMIT ?"
        )
        params = (int(args.limit),)

    counts: Counter[str] = Counter()
    with sqlite3.connect(str(db)) as con:
        for fa, ta, topic, payload in con.execute(query, params):
            try:
                p = json.loads(payload)
            except Exception:
                continue
            if topic == "proposal":
                counts[f"proposal:{p.get('action_name')}"] += 1
            elif topic == "delegated_result":
                counts[f"delegated:{p.get('kind')}:{p.get('state')}"] += 1
            elif topic == "request" and ta == "kernel":
                counts[f"kernel_request:{p.get('kind')}"] += 1
            elif topic == "response" and fa == "kernel":
                counts[f"kernel_response:{p.get('kind')}:{p.get('status')}"] += 1

    print(json.dumps(dict(counts), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
