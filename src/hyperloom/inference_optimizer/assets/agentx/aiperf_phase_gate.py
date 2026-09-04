#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Wait for an AIPerf credit phase through its local progress API."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def pick_loopback_port() -> int:
    """Ask the kernel for an unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def process_alive(pid: int) -> bool:
    """Return whether a process still exists."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", 1)
        if len(fields) == 2 and fields[1].strip().split(maxsplit=1)[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Load a JSON object from an HTTP endpoint."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object from {url}")
    return payload


def phase_stats(api_url: str, phase: str, *, timeout_seconds: float) -> dict[str, Any] | None:
    """Return the current stats for an AIPerf credit phase."""
    payload = load_json(f"{api_url.rstrip('/')}/api/progress", timeout_seconds=timeout_seconds)
    phases = payload.get("phases")
    stats = phases.get(phase) if isinstance(phases, dict) else None
    return stats if isinstance(stats, dict) else None


def wait_for_phase(
    *,
    api_url: str,
    phase: str,
    pid: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> int:
    """Wait until AIPerf reports that ``phase`` has started."""
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    last_error = ""

    while True:
        if not process_alive(pid):
            raise RuntimeError(f"AIPerf process {pid} exited before phase {phase!r} started")
        if deadline is not None and time.monotonic() >= deadline:
            suffix = f"; last API error: {last_error}" if last_error else ""
            raise TimeoutError(f"timed out waiting for AIPerf phase {phase!r}{suffix}")

        try:
            stats = phase_stats(api_url, phase, timeout_seconds=max(1.0, poll_interval_seconds))
            start_ns = stats.get("start_ns") if stats is not None else None
            if isinstance(start_ns, int) and not isinstance(start_ns, bool) and start_ns > 0:
                return start_ns
            last_error = ""
        except (
            http.client.HTTPException,
            OSError,
            TimeoutError,
            ValueError,
            urllib.error.URLError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        time.sleep(poll_interval_seconds)


def wait_for_capture_stop(
    *,
    api_url: str,
    phase: str,
    pid: int,
    max_window_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    """Wait until the phase ends or the wall-clock safety bound is reached."""
    started = time.monotonic()
    api_error_count = 0
    last_api_error = ""

    def result(stop_reason: str, elapsed: float) -> dict[str, Any]:
        output: dict[str, Any] = {
            "stop_reason": stop_reason,
            "elapsed_seconds": round(elapsed, 3),
        }
        if api_error_count:
            output["api_error_count"] = api_error_count
            output["last_api_error"] = last_api_error
        return output

    while True:
        elapsed = time.monotonic() - started
        if elapsed >= max_window_seconds:
            return result("wall_clock_limit", elapsed)
        if not process_alive(pid):
            return result("aiperf_exited", elapsed)

        try:
            stats = phase_stats(api_url, phase, timeout_seconds=max(1.0, poll_interval_seconds))
            if stats is not None:
                if stats.get("requests_end_ns") is not None:
                    return result("phase_complete", elapsed)
        except (
            http.client.HTTPException,
            OSError,
            TimeoutError,
            ValueError,
            urllib.error.URLError,
        ) as exc:
            api_error_count += 1
            last_api_error = f"{type(exc).__name__}: {exc}"

        time.sleep(min(poll_interval_seconds, max(0.0, max_window_seconds - elapsed)))


def write_capture_status(
    *,
    output: str,
    capture_id: str,
    status: str,
    reason: str,
    phase_start_ns: int | None,
    requested_window_seconds: float,
    decision_json: str,
) -> None:
    """Atomically write the independent AgentX trace-capture result."""
    decision: dict[str, Any] = {}
    if decision_json:
        parsed = json.loads(decision_json)
        if not isinstance(parsed, dict):
            raise ValueError("capture decision must be a JSON object")
        decision = parsed
    payload = {
        "schema_version": 1,
        "capture_id": capture_id,
        "status": status,
        "reason": reason,
        "phase": "profiling",
        "phase_start_ns": phase_start_ns,
        "requested_window_seconds": requested_window_seconds,
        "decision": decision,
        "recorded_at_ns": time.time_ns(),
    }
    output_path = os.path.abspath(output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".agentx_profile_capture.", dir=os.path.dirname(output_path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pick-port", help="print an unused loopback TCP port")

    wait_parser = subparsers.add_parser("wait-phase", help="wait for an AIPerf phase")
    wait_parser.add_argument("--api-url", required=True)
    wait_parser.add_argument("--phase", default="profiling")
    wait_parser.add_argument("--pid", required=True, type=int)
    wait_parser.add_argument("--timeout-seconds", required=True, type=float)
    wait_parser.add_argument("--poll-interval-seconds", default=1.0, type=float)

    capture_parser = subparsers.add_parser(
        "wait-capture-stop",
        help="wait until capture coverage or a safety bound is reached",
    )
    capture_parser.add_argument("--api-url", required=True)
    capture_parser.add_argument("--phase", default="profiling")
    capture_parser.add_argument("--pid", required=True, type=int)
    capture_parser.add_argument("--max-window-seconds", required=True, type=float)
    capture_parser.add_argument("--poll-interval-seconds", default=0.2, type=float)

    status_parser = subparsers.add_parser(
        "write-capture-status",
        help="write the AgentX trace-capture result",
    )
    status_parser.add_argument("--output", required=True)
    status_parser.add_argument("--capture-id", required=True)
    status_parser.add_argument("--status", required=True, choices=("succeeded", "failed"))
    status_parser.add_argument("--reason", required=True)
    status_parser.add_argument("--phase-start-ns", type=int)
    status_parser.add_argument("--requested-window-seconds", required=True, type=float)
    status_parser.add_argument("--decision-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested phase-gate command."""
    args = build_parser().parse_args(argv)
    if args.command == "pick-port":
        print(pick_loopback_port())
        return 0
    if args.command == "wait-capture-stop":
        if args.max_window_seconds < 0:
            print("aiperf phase gate failed: max window must be non-negative", file=sys.stderr)
            return 1
        result = wait_for_capture_stop(
            api_url=args.api_url,
            phase=args.phase,
            pid=args.pid,
            max_window_seconds=args.max_window_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "write-capture-status":
        try:
            write_capture_status(
                output=args.output,
                capture_id=args.capture_id,
                status=args.status,
                reason=args.reason,
                phase_start_ns=args.phase_start_ns,
                requested_window_seconds=args.requested_window_seconds,
                decision_json=args.decision_json,
            )
        except (OSError, ValueError) as exc:
            print(f"aiperf phase gate failed: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        start_ns = wait_for_phase(
            api_url=args.api_url,
            phase=args.phase,
            pid=args.pid,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"aiperf phase gate failed: {exc}", file=sys.stderr)
        return 1
    print(start_ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
