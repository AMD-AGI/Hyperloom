#!/usr/bin/env python3
"""GEAK tracing helper — records timestamps to telemetry.jsonl and generates config.

Usage:
    python3 setup_geak_tracing.py              # Record start + output extra_headers config
    python3 setup_geak_tracing.py --record-end # Record end timestamp

Reads SESSION_ID from environment (injected by executor).

Writes to:
    /workspace/telemetry.jsonl — picked up by telemetry-watcher → backend → DB
        (backend auto-attaches message_id, so we get session + message correlation)
    /workspace/.geak_tracing.json — local state for the --record-end call
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE_PATH", "/workspace"))
TELEMETRY_FILE = WORKSPACE / "telemetry.jsonl"
STATE_FILE = WORKSPACE / ".geak_tracing.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def append_telemetry(event: dict) -> None:
    """Append a JSON line to telemetry.jsonl (picked up by executor watcher)."""
    TELEMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TELEMETRY_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_state(data: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, indent=2))


def record_start():
    session_id = os.environ.get("SESSION_ID", "")
    ts = now_iso()
    epoch = time.time()

    extra_headers = {
        "x-litellm-tags": "product:primus-claw,component:geak",
    }
    if session_id:
        extra_headers["x-litellm-spend-logs-metadata"] = json.dumps({
            "session_id": session_id,
            "component": "geak",
        })

    append_telemetry({
        "event": "action_start",
        "action": "geak_tracing",
        "ts": ts,
        "data": {
            "session_id": session_id,
            "extra_headers": extra_headers,
        },
    })

    save_state({
        "session_id": session_id,
        "start_ts": ts,
        "start_epoch": epoch,
        "extra_headers": extra_headers,
    })

    print(f"[geak-tracing] start recorded: {ts}")
    print(f"[geak-tracing] session_id: {session_id}")
    print()
    print("Add these extra_headers to model_kwargs when calling geak_set_model_config:")
    print(json.dumps(extra_headers, indent=2))


def record_end():
    ts = now_iso()
    epoch = time.time()
    state = load_state()

    start_epoch = state.get("start_epoch")
    duration = (epoch - start_epoch) if start_epoch else None

    append_telemetry({
        "event": "action_end",
        "action": "geak_tracing",
        "ts": ts,
        "data": {
            "session_id": state.get("session_id", ""),
            "start_ts": state.get("start_ts"),
            "end_ts": ts,
            "duration_s": round(duration, 1) if duration else None,
        },
    })

    state["end_ts"] = ts
    state["end_epoch"] = epoch
    state["duration_s"] = round(duration, 1) if duration else None
    save_state(state)

    print(f"[geak-tracing] end recorded: {ts}")
    if duration:
        print(f"[geak-tracing] duration: {duration:.1f}s")


def main():
    if "--record-end" in sys.argv:
        record_end()
    else:
        record_start()


if __name__ == "__main__":
    main()
