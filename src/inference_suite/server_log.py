# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a serving framework's ``server.log`` says: ready, dead, and where a round's client phases begin.

:class:`LogCursor` follows the logs a round writes beside ``server.log`` incrementally, and :func:`death_excerpt` pulls
the line that names why an engine died.
"""

from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

#: Fatal server-init markers: once one appears in ``server.log`` the engine cannot recover within that process.
_DEAD_MARKERS: tuple[str, ...] = (
    # (1) runtime engine/worker bootstrap crashes
    "WorkerProc initialization failed",
    "EngineCore failed to start",
    "Engine core initialization failed",
    "Engine process failed to start",
    "AsyncEngineDeadError",
    "raise EngineDeadError",
    "Failed core proc(s)",
    # (2) config-validation-stage terminal failures (pre-engine).
    "does not recognize this architecture",
    "Transformers does not recognize",
    "ValidationError for ModelConfig",
    "are not supported for now",
)

#: Engine deaths that happen *after* the server reported ready and began
#: serving. These are read only when building a diagnostic excerpt.
_FATAL_ENGINE_MARKERS: tuple[str, ...] = (
    "EngineCore encountered a fatal error",
    "EngineCore encountered an issue",
    "EngineDeadError",
    "Engine core proc died",
    "died with exit code",
)

#: A framework line's exception, named anywhere in it: these logs prefix every
#: line with the worker pid, the level and the source location, so the exception
#: never starts the line.
_EXCEPTION_LINE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)\s*:")

#: Lines after a fatal marker that may carry its terminating exception.
_MARKER_WINDOW_LINES: int = 60

#: Lines of preceding context a legacy bootstrap marker keeps.
_MARKER_CONTEXT_LINES: int = 2


#: Lines that mean the server finished startup and accepts traffic: the uvicorn frontend (vLLM, sglang), vLLM's Rust
#: frontend and sglang's own banner.
READY_MARKERS: tuple[str, ...] = (
    "Application startup complete",
    "Uvicorn running on",
    # vLLM's Rust frontend (`vllm serve` with VLLM_USE_RUST_FRONTEND=1) runs no uvicorn; this is its ready line.
    "OpenAI server is ready to accept requests",
    "The server is fired up and ready to roll",
)

# The benchmark body's own stderr, beside ``server.log`` rather than in it or the parent's pipe.
_BENCHMARK_LOG_NAME: str = "benchmark_stderr.log"

# AgentX phase boundaries. The agentic client runs a long cold-cache warmup -- close to 18 minutes on the runs measured
# so far -- before the window under measurement opens, and mixing the two makes every KV statistic meaningless. aiperf
# already announces its phase transitions, so these match its own text; anchored on the parenthesised phase id rather
# than the whole line so a reword of the human-readable half does not silently stop the split.
_AGENTX_WARMUP_BEGIN_MARKERS: tuple[str, ...] = ("(warmup) started",)
_AGENTX_MEASURED_BEGIN_MARKERS: tuple[str, ...] = ("(profiling) started",)

# aiperf writes only this file: an AgentX benchmark directory has no ``benchmark_stdout.log`` / ``benchmark_stderr.log``
# the way a synthetic one does, so the phase lines are unreachable unless it is scanned explicitly.
_AGENTX_LOG_RELPATH: str = "aiperf_artifacts/logs/aiperf.log"


_SERVER_LOG_NAME: str = "server.log"


def death_excerpt(path: str, *, max_chars: int = 1200) -> str | None:
    """Return a short excerpt of ``path`` around its first terminal engine/worker-init marker.

    The whole file is streamed a line at a time rather than sampled from its
    head or tail. An engine that dies while serving logs one downstream error
    per rejected request afterwards, so the cause sits at the head of that
    cascade: a tail window never reaches it, and a bounded head read leaves the
    middle of a long log unsearched. Memory stays flat either way.

    Args:
        path: Log to read.
        max_chars: Cap on the returned excerpt.

    Returns:
        The excerpt, without surrounding whitespace, or ``None`` when the file
        cannot be read or names no terminal marker.
    """
    before: deque[str] = deque(maxlen=_MARKER_CONTEXT_LINES)
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                # Legacy first: every ``_FATAL_ENGINE_MARKERS`` entry is a
                # substring of some line these also match -- ``EngineDeadError``
                # of ``AsyncEngineDeadError`` -- and a legacy line must keep the
                # legacy extraction, whose cause sits above the marker.
                if any(marker in line for marker in _DEAD_MARKERS):
                    after = _read_lines(fh, _MARKER_CONTEXT_LINES)
                    excerpt = "\n".join([*before, line, *after])
                    return excerpt.strip()[-max_chars:].strip() or None
                if any(marker in line for marker in _FATAL_ENGINE_MARKERS):
                    return _cause_first_excerpt(
                        line.strip(), _read_lines(fh, _MARKER_WINDOW_LINES - 1), max_chars=max_chars
                    )
                before.append(line)
    except OSError:
        return None
    return None


def _read_lines(fh: Any, count: int) -> list[str]:
    """Return the next ``count`` lines of ``fh``, fewer at end of file."""
    out: list[str] = []
    for _ in range(max(0, count)):
        nxt = fh.readline()
        if not nxt:
            break
        out.append(nxt.rstrip("\n"))
    return out


def _cause_first_excerpt(head: str, window: list[str], *, max_chars: int) -> str | None:
    """Return the marker line followed by the exceptions it names, nearest first.

    A post-startup engine death names its cause *below* the marker, two dozen
    frames down. The frames carry no rule the classifier can use, and the window
    runs past this traceback into the errors it caused downstream -- so the last
    exception in it names a consequence (``EngineDeadError``) while the cause
    (``HIP out of memory``) sits above that. No leading context: the line above
    such a marker is routinely a scheduler-state dump tens of KB wide.
    """
    room = max_chars - len(head) - 1
    if room <= 0:
        return head[:max_chars].rstrip() or None
    causes: list[str] = []
    used = 0
    for line in window:
        if not _EXCEPTION_LINE.search(line):
            continue
        text = line.strip()
        if used + len(text) + 1 > room:
            break
        causes.append(text)
        used += len(text) + 1
    if causes:
        return head + "\n" + "\n".join(causes)
    rest = "\n".join(window).strip()[-room:].strip()
    return f"{head}\n{rest}" if rest else head[:max_chars].rstrip() or None


def _resolve_scan_logs(server_log_path: str) -> list[str]:
    """Return the log files to scan for markers: the server log, the benchmark body's stderr and the AgentX log."""
    primary = Path(server_log_path)
    names = (primary, primary.with_name(_BENCHMARK_LOG_NAME), primary.parent / _AGENTX_LOG_RELPATH)
    return [str(name) for name in names if name.exists()]


class LogScan(NamedTuple):
    """What one pass over the resolved logs found in the bytes appended since the last."""

    saw_ready: bool
    grew: bool
    saw_warmup_begin: bool = False
    saw_measured_begin: bool = False
    #: The last whole line a server log gained in this pass, ``None`` when none gained one.
    server_line: str | None = None

    def including(self, scan: "_IncrementScan", previous_offset: int) -> "LogScan":
        """What this pass has found once one log's increment is folded in."""
        advanced = scan.offset > previous_offset
        return LogScan(
            saw_ready=self.saw_ready or scan.saw_ready,
            grew=self.grew or (advanced and scan.active),
            saw_warmup_begin=self.saw_warmup_begin or scan.saw_warmup_begin,
            saw_measured_begin=self.saw_measured_begin or scan.saw_measured_begin,
            server_line=self.server_line,
        )


class _IncrementScan(NamedTuple):
    """One log's scan result: the advanced offset, the markers seen, and the partial line to carry forward."""

    offset: int
    saw_ready: bool
    saw_warmup_begin: bool = False
    saw_measured_begin: bool = False
    residual: str = ""
    active: bool = True
    #: The last whole line with text that the increment holds, other than a health probe's access line (a monitor's poll
    #: is answered while the engine builds), ``None`` when it holds none.
    last_line: str | None = None


#: Access lines for the endpoints monitors poll on their own tick. An HTTP front end whose engine died keeps answering
#: them, so a log that grew only by these says nothing about whether the server is still serving.
_HEALTH_PROBE_LINE = re.compile(r'"GET /(?:metrics|health|healthz|ping)(?:[/?][^" ]*)? HTTP/[\d.]+"')


# Cap on the partial line carried between scans. A log that appends a very long line without a newline -- or none at
# all -- must not let the buffer grow without bound; a marker is far shorter than this, so nothing real is lost.
_RESIDUAL_MAX_CHARS = 8192


@dataclass
class LogCursor:
    """Where a scan of the watched logs stopped: per path, the offset, the partial line and the file identity."""

    offsets: dict[str, int] = field(default_factory=dict)
    residuals: dict[str, str] = field(default_factory=dict)
    identities: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Pinned now, so a log an earlier round had already read is seen as replaced if it is.
        for path in self.offsets:
            try:
                stat = os.stat(path)
            except OSError:
                continue
            self.identities[path] = (stat.st_dev, stat.st_ino)

    def open_log(self, path: str) -> bool:
        """Begin reading ``path`` from its start, reporting whether it had not been read before."""
        if path in self.offsets:
            return False
        self.offsets[path] = 0
        return True

    def advance(self, server_log_path: str) -> LogScan:
        """Scan the bytes every watched log gained; replacing or truncating one is not activity."""
        found = LogScan(False, False)
        for path in _resolve_scan_logs(server_log_path):
            prev = self.offsets.get(path, 0)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            replaced = path in self.identities and self.identities[path] != identity
            self.identities[path] = identity
            if replaced or stat.st_size < prev:
                self.offsets[path] = stat.st_size
                self.residuals.pop(path, None)
                continue
            scan = _scan_increment(path, prev, self.residuals.get(path, ""))
            self.offsets[path] = scan.offset
            self.residuals[path] = scan.residual
            found = found.including(scan, prev)
            if scan.last_line is not None and (path == server_log_path or Path(path).name == _SERVER_LOG_NAME):
                found = found._replace(server_line=scan.last_line)
        return found


def _scan_increment(path: str, from_offset: int, residual: str = "") -> _IncrementScan:
    """Incrementally scan the bytes appended to ``server.log`` since ``from_offset`` for ready / AgentX phase markers."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return _IncrementScan(from_offset, False, residual=residual)
    start = from_offset
    carried = residual
    if size < start:  # truncated / rotated — rescan from the top.
        start = 0
        carried = ""  # the bytes it belonged to are gone
    if size <= start:  # nothing new appended
        return _IncrementScan(start, False, residual=carried)
    try:
        with open(path, "rb") as fh:
            fh.seek(start)
            chunk = fh.read().decode("utf-8", "ignore")
    except (OSError, ValueError):
        return _IncrementScan(from_offset, False, residual=carried)
    # A poll lands wherever the writer happens to be, so a marker is routinely split across two reads. Prepending the
    # previous tail and carrying the new one forward is what makes a marker findable exactly once, whole.
    chunk = carried + chunk
    tail_at = chunk.rfind("\n")
    next_residual = chunk if tail_at < 0 else chunk[tail_at + 1 :]
    next_residual = next_residual[-_RESIDUAL_MAX_CHARS:]
    saw_ready = any(marker in chunk for marker in READY_MARKERS)
    saw_warmup_begin = any(marker in chunk for marker in _AGENTX_WARMUP_BEGIN_MARKERS)
    saw_measured_begin = any(marker in chunk for marker in _AGENTX_MEASURED_BEGIN_MARKERS)
    complete = [line for line in chunk[: tail_at + 1].splitlines() if line.strip()] if tail_at >= 0 else []
    spoken = [line for line in complete if not _HEALTH_PROBE_LINE.search(line)]
    active = not complete or bool(spoken)
    return _IncrementScan(
        size,
        saw_ready,
        saw_warmup_begin,
        saw_measured_begin,
        residual=next_residual,
        active=active,
        last_line=spoken[-1] if spoken else None,
    )


__all__ = [
    "READY_MARKERS",
    "LogCursor",
    "LogScan",
    "death_excerpt",
]
