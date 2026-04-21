#!/usr/bin/env python3
"""OOB tracing helper — records action timestamps to telemetry.jsonl.

Usage:
    python3 setup_oob_tracing.py                        # Record start (component=oob)
    python3 setup_oob_tracing.py --record-end           # Record end timestamp
    python3 setup_oob_tracing.py --agent codex          # Record start with agent info
    python3 setup_oob_tracing.py --task-id <id>         # Include task_id in metadata

Reads SESSION_ID from environment (injected by executor).

Writes to:
    /workspace/telemetry.jsonl — picked up by telemetry-watcher → backend → DB
        (backend auto-attaches message_id, so we get session + message correlation)
    /workspace/.oob_tracing.json — local state for the --record-end call

NOTE: OOB header injection is handled by auth_proxy.py inside the OOB workload pod.
This script only records timing for message-level correlation.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE_PATH", "/workspace"))
TELEMETRY_FILE = WORKSPACE / "telemetry.jsonl"
STATE_FILE = WORKSPACE / ".oob_tracing.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def append_telemetry(event: dict) -> None:
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


def record_start(agent: str = "", task_id: str = ""):
    session_id = os.environ.get("SESSION_ID", "")
    ts = now_iso()
    epoch = time.time()

    data = {
        "session_id": session_id,
        "component": "oob",
    }
    if agent:
        data["agent"] = agent
    if task_id:
        data["task_id"] = task_id

    append_telemetry({
        "event": "action_start",
        "action": "oob_tracing",
        "ts": ts,
        "data": data,
    })

    save_state({
        "session_id": session_id,
        "start_ts": ts,
        "start_epoch": epoch,
        "agent": agent,
        "task_id": task_id,
    })

    print(f"[oob-tracing] start recorded: {ts}")
    print(f"[oob-tracing] session_id: {session_id}")
    if agent:
        print(f"[oob-tracing] agent: {agent}")
    if task_id:
        print(f"[oob-tracing] task_id: {task_id}")


def record_end(task_id: str = ""):
    ts = now_iso()
    epoch = time.time()
    state = load_state()

    start_epoch = state.get("start_epoch")
    duration = (epoch - start_epoch) if start_epoch else None

    data = {
        "session_id": state.get("session_id", ""),
        "component": "oob",
        "agent": state.get("agent", ""),
        "start_ts": state.get("start_ts"),
        "end_ts": ts,
        "duration_s": round(duration, 1) if duration else None,
    }
    if task_id or state.get("task_id"):
        data["task_id"] = task_id or state.get("task_id", "")

    append_telemetry({
        "event": "action_end",
        "action": "oob_tracing",
        "ts": ts,
        "data": data,
    })

    state["end_ts"] = ts
    state["end_epoch"] = epoch
    state["duration_s"] = round(duration, 1) if duration else None
    save_state(state)

    print(f"[oob-tracing] end recorded: {ts}")
    if duration:
        print(f"[oob-tracing] duration: {duration:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="OOB tracing timestamp recorder")
    parser.add_argument("--record-end", action="store_true", help="Record end timestamp")
    parser.add_argument("--agent", default="", help="Agent type (codex/claude)")
    parser.add_argument("--task-id", default="", help="OOB task ID")
    args = parser.parse_args()

    if args.record_end:
        record_end(task_id=args.task_id)
    else:
        record_start(agent=args.agent, task_id=args.task_id)


if __name__ == "__main__":
    main()
