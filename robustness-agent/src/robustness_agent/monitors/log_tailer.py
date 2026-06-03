"""Log tailer — streams application log files and extracts error patterns."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Optional

from ..models import Alert, Severity

log = logging.getLogger(__name__)

# Common error patterns from marathon optimization error signature database
ERROR_PATTERNS: list[tuple[str, str, Severity]] = [
    (r"OutOfMemoryError|CUDA out of memory|HIP out of memory|oom-kill", "oom", Severity.CRITICAL),
    (r"NCCL\s+error|RCCL\s+error|collective.*timeout", "collective_error", Severity.CRITICAL),
    (r"Segmentation fault|SIGSEGV|core dumped", "segfault", Severity.CRITICAL),
    (r"RuntimeError.*CUDA|hipErrorNoBinaryForGpu|hipErrorInvalidDevice", "gpu_runtime", Severity.CRITICAL),
    (r"triton.*error|ptxas.*error|LLVM ERROR", "compiler_error", Severity.WARNING),
    (r"ConnectionRefused|ConnectionReset|BrokenPipeError", "connection_error", Severity.WARNING),
    (r"TimeoutError|asyncio\.TimeoutError|deadline exceeded", "timeout", Severity.WARNING),
    (r"ModuleNotFoundError|ImportError.*sgl_kernel|symbol.*not found", "import_error", Severity.WARNING),
    (r"AssertionError|assert.*failed", "assertion_error", Severity.WARNING),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), name, sev) for p, name, sev in ERROR_PATTERNS]


class LogTailer:
    """Tail a log file and emit alerts on error pattern matches."""

    def __init__(self, log_path: Optional[Path] = None, max_lines_per_check: int = 200):
        """Initialise the log tailer.

        Args:
            log_path (Optional[Path]): Path of the log file to tail; may
                be set later via :meth:`set_log_path`.
            max_lines_per_check (int): Maximum number of new lines to
                scan per :meth:`check` call.
        """
        self._log_path = log_path
        self._max_lines = max_lines_per_check
        self._file_pos: int = 0
        self._recent_matches: dict[str, float] = {}
        self._dedup_window_s: float = 60.0

    def set_log_path(self, path: Path) -> None:
        """Point the tailer at a new log file and reset the read offset.

        Args:
            path (Path): The log file to begin tailing.
        """
        self._log_path = path
        self._file_pos = 0

    async def check(self) -> list[Alert]:
        """Scan newly appended log lines for known error patterns.

        Returns:
            list[Alert]: Alerts for matched error patterns, deduplicated
            within the dedup window. Empty if no log file is set.
        """
        if self._log_path is None or not self._log_path.exists():
            return []

        alerts: list[Alert] = []
        try:
            new_lines = await self._read_new_lines()
            for line in new_lines:
                for regex, name, severity in _COMPILED_PATTERNS:
                    if regex.search(line):
                        if not self._is_dedup(name):
                            alerts.append(Alert(
                                check_name=f"log_error_{name}",
                                severity=severity,
                                summary=f"Error pattern '{name}' in {self._log_path.name}",
                                detail=line.strip()[:500],
                                evidence={"pattern": name, "file": str(self._log_path)},
                                timestamp=time.time(),
                            ))
                            self._recent_matches[name] = time.time()
                        break
        except Exception as exc:
            log.debug("Failed to tail %s: %s", self._log_path, exc)
        return alerts

    def _is_dedup(self, name: str) -> bool:
        """Report whether a pattern was matched within the dedup window.

        Args:
            name (str): The error-pattern name to check.

        Returns:
            bool: ``True`` if the pattern matched recently enough to be
            suppressed as a duplicate.
        """
        last = self._recent_matches.get(name, 0)
        return (time.time() - last) < self._dedup_window_s

    async def _read_new_lines(self) -> list[str]:
        """Read log lines appended since the last read, off the event loop.

        Returns:
            list[str]: Up to ``max_lines_per_check`` newly appended
            lines (empty if the file is missing).
        """
        def _read() -> list[str]:
            """Synchronously read new lines and advance the file offset.

            Returns:
                list[str]: New lines, trimmed to ``max_lines_per_check``.
            """
            if not self._log_path or not self._log_path.exists():
                return []
            with open(self._log_path, "r", errors="replace") as f:
                f.seek(self._file_pos)
                lines = f.readlines()
                self._file_pos = f.tell()
            return lines[-self._max_lines:] if len(lines) > self._max_lines else lines

        return await asyncio.get_event_loop().run_in_executor(None, _read)
