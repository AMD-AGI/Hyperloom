# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the runner leaves for the application to read in ``/out``: progress while it works, status when it is done."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from hyperloom.measure.failures import FOREIGN_PROCESS_ON_CARD
from hyperloom.measure.live_log import last_output

__all__ = [
    "EXIT_CODES",
    "Progress",
    "RunnerFailure",
    "join_notes",
    "wait_for_file",
    "write_json_atomic",
    "write_status",
]

#: Exit code by ``error_class``; anything not listed exits 1.
EXIT_CODES = {
    "no_spec": 0,
    "runner_start_failed": 2,
    "spec_rejected": 3,
    "preflight_mismatch": 4,
    "node_fault": 5,
    FOREIGN_PROCESS_ON_CARD: 6,
}
_REFRESH_SEC = 20.0
#: ``status.json`` keeps this many characters of its message.
_MESSAGE_LIMIT = 2000
_ROOM_FOR_COUNT = 40


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


def join_notes(notes: Sequence[str]) -> str:
    """The notes of a test that succeeded as one message: whole notes from the first while they fit, then how many did not.

    A message that is too long is cut at its start (the end of an error says the most), which for notes would drop the
    first and most important ones; so the notes are cut at their end instead, in the order the runner gave them.
    """
    message = "; ".join(notes)
    if len(message) <= _MESSAGE_LIMIT:
        return message
    kept: list[str] = []
    for note in notes:
        if len("; ".join([*kept, note])) + _ROOM_FOR_COUNT > _MESSAGE_LIMIT:
            if not kept:
                kept.append(note[: _MESSAGE_LIMIT - _ROOM_FOR_COUNT])
            break
        kept.append(note)
    return "; ".join([*kept, f"{len(notes) - len(kept)} more notes left out"])


def write_status(out_dir: Path, *, state: str, error_class: str | None = None, message: str = "") -> None:
    """Write ``status.json``; it is always the last file, so its presence says the runner has ended."""
    document = {"state": state, "error_class": error_class, "message": message[-_MESSAGE_LIMIT:]}
    write_json_atomic(out_dir / "status.json", document)


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
    """``/out/progress.json``: the phase, and when the test last wrote output; refreshed in the background.

    With ``files_dir``, the directory the test's logs are written below, it also says how long ago the newest live log was
    written (``last_log_age_sec``, as the runner's clock sees it) and that log's last line (``last_line``, cleaned and
    shortened), so the application can show how a running test is doing.
    """

    def __init__(self, out_dir: Path, *, refresh_sec: float = _REFRESH_SEC, files_dir: Path | None = None) -> None:
        self._path = out_dir / "progress.json"
        self._files_dir = files_dir
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
        write_json_atomic(self._path, {**document, **self._live_log()})

    def _live_log(self) -> dict[str, Any]:
        found = None if self._files_dir is None else last_output(self._files_dir, time.time())
        return {} if found is None else {"last_log_age_sec": found[0], "last_line": found[1]}

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
