# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the runner leaves for the application to read in ``/out``: progress while it works, status when it is done."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["EXIT_CODES", "Progress", "RunnerFailure", "wait_for_file", "write_json_atomic", "write_status"]

#: Exit code by ``error_class``; anything not listed exits 1.
EXIT_CODES = {
    "no_spec": 0,
    "runner_start_failed": 2,
    "spec_rejected": 3,
    "preflight_mismatch": 4,
    "node_fault": 5,
}
_REFRESH_SEC = 20.0


class RunnerFailure(Exception):
    """The runner cannot go on; ``error_class`` is what ``status.json`` and the exit code carry."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.error_class, 1)


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """Write ``path`` so a reader sees the old file or the new one, never half of one or none.

    The content goes to a scratch file of its own in the same directory (two writers never share one), is flushed to
    disk, and then replaces ``path`` in one step.
    """
    fd, scratch_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as scratch:
            scratch.write(json.dumps(document, sort_keys=True))
            scratch.flush()
            os.fsync(scratch.fileno())
        os.replace(scratch_name, path)
    except BaseException:
        Path(scratch_name).unlink(missing_ok=True)
        raise


def write_status(out_dir: Path, *, state: str, error_class: str | None = None, message: str = "") -> None:
    """Write ``status.json``; it is always the last file, so its presence says the runner has ended."""
    write_json_atomic(out_dir / "status.json", {"state": state, "error_class": error_class, "message": message[-2000:]})


def wait_for_file(
    path: Path, timeout_sec: float, *, poll_sec: float = 0.5, sleep: Callable[[float], None] = time.sleep
) -> bool:
    """Wait until ``path`` exists; False when ``timeout_sec`` passes first."""
    deadline = time.monotonic() + timeout_sec
    while not path.exists():
        if time.monotonic() >= deadline:
            return False
        sleep(poll_sec)
    return True


class Progress:
    """``/out/progress.json``: the phase, and when the test last wrote output; refreshed in the background."""

    def __init__(self, out_dir: Path, *, refresh_sec: float = _REFRESH_SEC) -> None:
        self._path = out_dir / "progress.json"
        self._refresh_sec = refresh_sec
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"phase": "starting", "last_output_unix": None}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def phase(self, name: str) -> None:
        with self._lock:
            self._state["phase"] = name
        self.flush()

    def touch(self) -> None:
        """The test wrote output just now."""
        with self._lock:
            self._state["last_output_unix"] = time.time()

    def flush(self) -> None:
        with self._lock:
            document = {**self._state, "updated_unix": time.time()}
        write_json_atomic(self._path, document)

    def __enter__(self) -> Progress:
        def refresh() -> None:
            while not self._stop.wait(self._refresh_sec):
                self.flush()

        self.flush()
        self._thread = threading.Thread(target=refresh, name="hl-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._refresh_sec + 1)
