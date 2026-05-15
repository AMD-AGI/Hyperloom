#!/usr/bin/env python3
"""Print recent action / proposal / kernel counts from a session's coordinator.db.

Usage:
    event_counts.py [SESSION_DIR]

SESSION_DIR defaults to $USER_DATA_PATH or /workspace/hyperloom.
Reads at most the last 500 events from $SESSION_DIR/storage/coordinator.db and
emits a JSON object of {category: count}.
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import sys
from collections import Counter


def main() -> int:
    session_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("USER_DATA_PATH", "/workspace/hyperloom")
    )
    db = pathlib.Path(session_dir) / "storage" / "coordinator.db"
    if not db.exists():
        print(f"coordinator.db not found at {db}", file=sys.stderr)
        return 2

    counts: Counter[str] = Counter()
    with sqlite3.connect(str(db)) as con:
        rows = con.execute(
            "SELECT from_agent, to_agent, topic, payload FROM events "
            "ORDER BY seq DESC LIMIT 500"
        )
        for fa, ta, topic, payload in rows:
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
