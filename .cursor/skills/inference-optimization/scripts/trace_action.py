#!/usr/bin/env python3
"""Unified action tracing — records start/end timestamps to telemetry.jsonl.

Usage:
    python3 trace_action.py --component geak --action start
    python3 trace_action.py --component geak --action end
    python3 trace_action.py --component oob --action start --agent codex --task-id abc123
    python3 trace_action.py --component oob --action end
    python3 trace_action.py --component llm --action start
    python3 trace_action.py --component llm --action end

Environment:
    SESSION_ID      — injected by executor, used for session correlation
    WORKSPACE_PATH  — defaults to /workspace

Writes to:
    $WORKSPACE_PATH/telemetry.jsonl — polled by telemetry-watcher → backend → DB
        (backend auto-attaches message_id for per-message correlation)
    $WORKSPACE_PATH/.trace_action_<component>.json — local state for duration calc

For GEAK/OOB: records timing metadata for CLI invocations. OOB handles headers
via auth_proxy automatically when configured by bootstrap.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(os.environ.get("WORKSPACE_PATH", "/workspace"))
TELEMETRY_FILE = WORKSPACE / "telemetry.jsonl"


def _state_file(component: str) -> Path:
    return WORKSPACE / f".trace_action_{component}.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def append_telemetry(event: dict) -> None:
    TELEMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TELEMETRY_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def load_state(component: str) -> dict:
    sf = _state_file(component)
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_state(component: str, data: dict) -> None:
    sf = _state_file(component)
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps(data, indent=2))


def _build_extra_headers(session_id: str, component: str) -> dict:
    headers = {
        "x-litellm-tags": f"product:primus-claw,component:{component}",
    }
    if session_id:
        headers["x-litellm-spend-logs-metadata"] = json.dumps({
            "session_id": session_id,
            "component": component,
        })
    return headers


def record_start(component: str, agent: str = "", task_id: str = ""):
    session_id = os.environ.get("SESSION_ID", "")
    ts = now_iso()
    epoch = time.time()

    data: dict = {
        "session_id": session_id,
        "component": component,
    }
    if agent:
        data["agent"] = agent
    if task_id:
        data["task_id"] = task_id

    if component == "geak":
        extra_headers = _build_extra_headers(session_id, component)
        data["extra_headers"] = extra_headers

    append_telemetry({
        "event": "action_start",
        "action": f"{component}_tracing",
        "ts": ts,
        "data": data,
    })

    state = {
        "session_id": session_id,
        "component": component,
        "start_ts": ts,
        "start_epoch": epoch,
    }
    if agent:
        state["agent"] = agent
    if task_id:
        state["task_id"] = task_id
    save_state(component, state)

    print(f"[trace-action:{component}] start recorded: {ts}")
    print(f"[trace-action:{component}] session_id: {session_id}")
    if agent:
        print(f"[trace-action:{component}] agent: {agent}")
    if task_id:
        print(f"[trace-action:{component}] task_id: {task_id}")

    if component == "geak":
        print()
        print("GEAK extra_headers metadata recorded locally for correlation:")
        print(json.dumps(extra_headers, indent=2))


def record_end(component: str, task_id: str = ""):
    ts = now_iso()
    epoch = time.time()
    state = load_state(component)

    start_epoch = state.get("start_epoch")
    duration = (epoch - start_epoch) if start_epoch else None

    data: dict = {
        "session_id": state.get("session_id", ""),
        "component": component,
        "start_ts": state.get("start_ts"),
        "end_ts": ts,
        "duration_s": round(duration, 1) if duration else None,
    }
    if state.get("agent"):
        data["agent"] = state["agent"]
    if task_id or state.get("task_id"):
        data["task_id"] = task_id or state.get("task_id", "")

    append_telemetry({
        "event": "action_end",
        "action": f"{component}_tracing",
        "ts": ts,
        "data": data,
    })

    state["end_ts"] = ts
    state["end_epoch"] = epoch
    state["duration_s"] = round(duration, 1) if duration else None
    save_state(component, state)

    print(f"[trace-action:{component}] end recorded: {ts}")
    if duration:
        print(f"[trace-action:{component}] duration: {duration:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Record action start/end timestamps for LLM cost attribution"
    )
    parser.add_argument(
        "--component", required=True,
        help="Component name (geak, oob, llm, tracelens, etc.)"
    )
    parser.add_argument(
        "--action", required=True, choices=["start", "end"],
        help="Record start or end timestamp"
    )
    parser.add_argument("--agent", default="", help="Agent type (codex/claude)")
    parser.add_argument("--task-id", default="", help="Task ID for correlation")
    args = parser.parse_args()

    if args.action == "start":
        record_start(args.component, agent=args.agent, task_id=args.task_id)
    else:
        record_end(args.component, task_id=args.task_id)


if __name__ == "__main__":
    main()
