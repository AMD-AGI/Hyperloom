#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Wait for an AIPerf credit phase through its local progress API."""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import math
import os
import re
import signal
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any


# Keep the standalone asset aligned with tools/_trace_rank.py's framework names.
_TRACE_RANK_PATTERNS = (
    re.compile(r"(?:^|[-_.])rank[-_]?(\d+)(?=[-_.]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_.])tp[-_](\d+)(?=[-_.]|$)", re.IGNORECASE),
    re.compile(r"^r(\d+)(?=[-.])", re.IGNORECASE),
)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant: {value}")


def is_auto_bounded(framework: str, body: str) -> bool:
    """Only SGLang's forwarded positive integer num_steps implies auto-stop."""
    try:
        payload = json.loads(body, parse_constant=_reject_json_constant)
    except ValueError:
        return False
    steps = payload.get("num_steps") if isinstance(payload, dict) else None
    return framework == "sglang" and type(steps) is int and steps > 0


def _trace_files(directories: list[str]) -> dict[str, list[int]]:
    files = {}
    for directory in directories:
        for path in Path(directory).resolve().rglob("*.trace.json*"):
            if path.name.startswith(("graph_capture_", "merged-")) or {"capture_traces", "trace_split"}.intersection(
                path.parts
            ):
                continue
            if path.is_file() and path.name.endswith((".trace.json", ".trace.json.gz")):
                stat = path.stat()
                files[str(path)] = [stat.st_mtime_ns, stat.st_size]
    return files


def snapshot_traces(directories: list[str]) -> dict[str, Any]:
    """Record the trace baseline immediately before start_profile is sent."""
    return {"started_ns": time.time_ns(), "files": _trace_files(directories)}


def current_traces(directories: list[str], snapshot: dict[str, Any]) -> dict[str, list[int]]:
    """Exclude unchanged paths and files older than the capture boundary."""
    return {
        path: stat
        for path, stat in _trace_files(directories).items()
        if stat[0] >= snapshot["started_ns"] and stat != snapshot["files"].get(path)
    }


_TRACE_READ_CHARS = 64 * 1024
_MAX_JSON_VALUE_CHARS = 1024 * 1024
_MAX_JSON_DEPTH = 128
_JSON_DECODER = json.JSONDecoder(parse_constant=_reject_json_constant)
_JSON_STRING_SPECIAL = re.compile(r'["\\\x00-\x1f]')


class _TraceValueTooLarge(ValueError):
    """Switch from whole-value decoding to field-wise streaming."""


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("trace check exceeded its remaining budget")


class _TraceJSONReader:
    """Read a complete trace document without retaining its event array."""

    def __init__(self, handle: Any, deadline: float | None) -> None:
        self.handle = handle
        self.deadline = deadline
        self.buffer = ""
        self.position = 0
        self.eof = False

    def refill(self) -> None:
        _check_deadline(self.deadline)
        self.buffer = self.buffer[self.position :]
        self.position = 0
        if len(self.buffer) >= _MAX_JSON_VALUE_CHARS:
            raise _TraceValueTooLarge("trace JSON value requires field-wise streaming")
        chunk = self.handle.read(min(_TRACE_READ_CHARS, _MAX_JSON_VALUE_CHARS - len(self.buffer)))
        _check_deadline(self.deadline)
        self.eof = not chunk
        self.buffer += chunk

    def peek(self) -> str:
        _check_deadline(self.deadline)
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position] in " \t\r\n":
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.refill()

    def expect(self, character: str) -> None:
        if self.peek() != character:
            raise ValueError(f"expected {character!r} in trace JSON")
        self.position += 1

    def value(self) -> Any:
        if not self.peek():
            raise ValueError("incomplete trace JSON value")
        while True:
            _check_deadline(self.deadline)
            try:
                value, end = _JSON_DECODER.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.refill()
                continue
            # A number can end at a chunk boundary before its exponent arrives.
            if end == len(self.buffer) and not self.eof:
                self.refill()
                continue
            if end < len(self.buffer) and self.buffer[end] not in " \t\r\n,]}":
                if type(value) in (int, float) and self.buffer[end] in ".eE" and not self.eof:
                    self.refill()
                    continue
                raise ValueError("invalid trace JSON value delimiter")
            self.position = end
            return value

    def string(self) -> str | None:
        """Validate strings of any length, retaining only short metadata values."""
        self.expect('"')
        parts = []
        length = 0
        while True:
            _check_deadline(self.deadline)
            match = _JSON_STRING_SPECIAL.search(self.buffer, self.position)
            end = match.start() if match else len(self.buffer)
            if length <= 64:
                length += end - self.position
                if length <= 64:
                    parts.append(self.buffer[self.position : end])
                else:
                    parts.clear()
            self.position = end
            if match is None:
                if self.eof:
                    raise ValueError("incomplete trace JSON string")
                self.refill()
                continue
            character = self.buffer[self.position]
            self.position += 1
            if character == '"':
                return "".join(parts) if length <= 64 else None
            if character != "\\":
                raise ValueError("unescaped control character in trace JSON string")
            while len(self.buffer) - self.position < 1 and not self.eof:
                self.refill()
            if self.position == len(self.buffer):
                raise ValueError("incomplete trace JSON escape")
            escape_length = 5 if self.buffer[self.position] == "u" else 1
            while len(self.buffer) - self.position < escape_length and not self.eof:
                self.refill()
            escaped = self.buffer[self.position : self.position + escape_length]
            decoded = _JSON_DECODER.decode('"\\' + escaped + '"')
            self.position += escape_length
            length += len(decoded)
            if length <= 64:
                parts.append(decoded)
            else:
                parts.clear()

    def event(self) -> bool:
        try:
            event = self.value()
        except _TraceValueTooLarge:
            if self.peek() != "{":
                self.skip()
                return False
            fields = {}
            for key in self.members():
                if key in {"cat", "ph"} and self.peek() == '"':
                    fields[key] = self.string()
                else:
                    if key in {"cat", "ph"}:
                        fields[key] = None
                    self.skip()
            return fields.get("cat") == "kernel" and fields.get("ph") == "X"
        return isinstance(event, dict) and event.get("cat") == "kernel" and event.get("ph") == "X"

    def members(self):
        self.expect("{")
        if self.peek() == "}":
            self.position += 1
            return
        while True:
            if self.peek() != '"':
                raise ValueError("trace JSON object key must be a string")
            key = self.string()
            self.expect(":")
            yield key
            if self.peek() == "}":
                self.position += 1
                return
            self.expect(",")

    def skip(self, depth: int = 0) -> None:
        if depth >= _MAX_JSON_DEPTH:
            raise ValueError("trace JSON nesting exceeds the parsing limit")
        character = self.peek()
        if character == "{":
            for _key in self.members():
                self.skip(depth + 1)
        elif character == "[":
            self.position += 1
            if self.peek() == "]":
                self.position += 1
                return
            while True:
                self.skip(depth + 1)
                if self.peek() == "]":
                    self.position += 1
                    return
                self.expect(",")
        elif character == '"':
            self.string()
        else:
            self.value()


def _read_trace_metadata(path: Path, *, deadline: float | None) -> dict[str, Any]:
    """Consume JSON and the gzip trailer, keeping only rank and GPU-event evidence."""
    rank = None
    for token, fullmatch in ((path.name, False), (path.parent.name, True)):
        for pattern in _TRACE_RANK_PATTERNS:
            match = pattern.fullmatch(token) if fullmatch else pattern.search(token)
            if match:
                rank = int(match.group(1))
                break
        if rank is not None:
            break
    opener = gzip.open if path.suffix == ".gz" else open
    has_kernel = False
    seen = set()
    with opener(path, "rt", encoding="utf-8") as handle:
        reader = _TraceJSONReader(handle, deadline)
        for key in reader.members():
            if key in {"traceEvents", "distributedInfo"}:
                if key in seen:
                    raise ValueError("duplicate trace metadata field")
                seen.add(key)
            if key == "traceEvents":
                reader.expect("[")
                if reader.peek() == "]":
                    reader.position += 1
                    continue
                while True:
                    if reader.event():
                        has_kernel = True
                    if reader.peek() == "]":
                        reader.position += 1
                        break
                    reader.expect(",")
            elif key == "distributedInfo" and reader.peek() == "{":
                rank_seen = False
                for field in reader.members():
                    if field == "rank":
                        header_rank = reader.value()
                        if rank_seen or type(header_rank) is not int or (rank is not None and rank != header_rank):
                            raise ValueError("conflicting trace rank")
                        rank = header_rank
                        rank_seen = True
                    else:
                        reader.skip()
            else:
                reader.skip()
        # Do not accept a valid prefix: consume EOF to verify gzip CRC/trailer too.
        if reader.peek():
            raise ValueError("trailing content after trace JSON document")
    return {"rank": rank, "has_kernel": has_kernel and "traceEvents" in seen}


def traces_complete(
    directories: list[str],
    snapshot: dict[str, Any],
    tp: int,
    *,
    cache: dict[str, Any] | None = None,
    deadline: float | None = None,
) -> bool:
    """Require fresh GPU traces; an unranked trace is unambiguous only at TP=1."""
    if tp <= 0:
        return False
    ranks = set()
    cache = cache if cache is not None else {}
    try:
        _check_deadline(deadline)
        files = current_traces(directories, snapshot)
        for obsolete in set(cache) - files.keys():
            del cache[obsolete]
        if len(files) != tp:
            return False
        for name, before in files.items():
            _check_deadline(deadline)
            path = Path(name)
            stat = path.stat()
            signature = [stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns, stat.st_ino, stat.st_dev]
            cached = cache.get(name)
            if isinstance(cached, dict) and cached.get("signature") == signature:
                metadata = cached.get("metadata")
            else:
                try:
                    metadata = _read_trace_metadata(path, deadline=deadline)
                except (gzip.BadGzipFile, EOFError, ValueError, RecursionError, zlib.error):
                    metadata = None
                _check_deadline(deadline)
                after = path.stat()
                if [after.st_mtime_ns, after.st_size, after.st_ctime_ns, after.st_ino, after.st_dev] != signature:
                    return False
                cache[name] = {"signature": signature, "metadata": metadata}
            _check_deadline(deadline)
            if [stat.st_mtime_ns, stat.st_size] != before or not isinstance(metadata, dict):
                return False
            rank = metadata.get("rank")
            if rank is None and tp == 1:
                rank = 0
            if (
                type(rank) is not int
                or rank not in range(tp)
                or rank in ranks
                or metadata.get("has_kernel") is not True
            ):
                return False
            ranks.add(rank)
        complete = ranks == set(range(tp)) and files == current_traces(directories, snapshot)
        _check_deadline(deadline)
        return complete
    except (OSError, EOFError, ValueError, RecursionError, zlib.error):
        return False


def _trace_timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError("trace check exceeded its remaining budget")


def _check_traces_command(args: argparse.Namespace, snapshot: dict[str, Any]) -> bool:
    timeout = args.timeout_seconds
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("trace timeout must be finite and positive")
    deadline = time.monotonic() + timeout if timeout is not None else None
    timed = timeout is not None and hasattr(signal, "setitimer")
    previous_handler = signal.getsignal(signal.SIGALRM) if timed else None
    temporary_path = None
    try:
        if timed:
            signal.signal(signal.SIGALRM, _trace_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout)
        cache: dict[str, Any] = {}
        if args.cache_file:
            try:
                saved = json.loads(Path(args.cache_file).read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                saved = None
            if isinstance(saved, dict) and saved.get("snapshot") == snapshot and isinstance(saved.get("files"), dict):
                cache = saved["files"]
        _check_deadline(deadline)
        complete = traces_complete(args.trace_dir, snapshot, args.tp, cache=cache, deadline=deadline)
        _check_deadline(deadline)
        if args.cache_file:
            output = Path(args.cache_file)
            output.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_path = tempfile.mkstemp(prefix=".trace-cache-", dir=output.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"snapshot": snapshot, "files": cache}, handle)
            _check_deadline(deadline)
            os.replace(temporary_path, output)
        _check_deadline(deadline)
        return complete
    finally:
        if timed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


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
    auto_parser = subparsers.add_parser("is-auto-bounded", help="check the forwarded native capture bound")
    auto_parser.add_argument("--framework", required=True)
    auto_parser.add_argument("--body", required=True)
    for command in ("snapshot-traces", "trace-stat", "traces-complete"):
        trace_parser = subparsers.add_parser(command)
        trace_parser.add_argument("--trace-dir", action="append", default=[])
        if command != "snapshot-traces":
            trace_parser.add_argument("--snapshot", required=True)
        if command == "traces-complete":
            trace_parser.add_argument("--tp", required=True, type=int)
            trace_parser.add_argument("--cache-file")
            trace_parser.add_argument("--timeout-seconds", type=float)

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
    if args.command == "is-auto-bounded":
        return 0 if is_auto_bounded(args.framework, args.body) else 1
    if args.command in {"snapshot-traces", "trace-stat", "traces-complete"}:
        try:
            if args.command == "snapshot-traces":
                print(json.dumps(snapshot_traces(args.trace_dir)))
                return 0
            snapshot = json.loads(args.snapshot)
            if args.command == "traces-complete":
                return 0 if _check_traces_command(args, snapshot) else 1
            files = current_traces(args.trace_dir, snapshot)
            print(len(files), sum(stat[1] for stat in files.values()))
            return 0
        except (OSError, ValueError) as exc:
            print(f"aiperf trace check failed: {exc}", file=sys.stderr)
            return 1
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
