# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A limit on how long a test's inference server may take to report ready.

The silence watchdog of the benchmark launcher starts only when the server's ready banner appears. A server that never
gets ready (a boot that tunes kernels for an hour) is therefore ended by the round's hard cap alone, and the same
candidate repeats it on the rerun. The guard here ends such a round at a share of that cap: it watches for the stamp the
launcher writes beside a round's server log the moment the banner appears and, when none has come within the limit,
cancels the scope the launcher's watchdog listens to, which reaps the whole process tree.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.actions.cancel_channel import CancelScope

__all__ = [
    "BOOT_CANCEL_REASON",
    "BOOT_SHARE",
    "BootGuard",
    "boot_limit_sec",
    "boot_seconds",
    "read_ready_stamp",
]

#: The ``reason`` the guard cancels the scope with.
BOOT_CANCEL_REASON = "server_boot_timeout"
#: The share of a round's time a boot may take. Boot, the warmup, the measured round and the accuracy evaluation all
#: run inside one cap, so a boot that has used half of it leaves the rest less than the boot took.
BOOT_SHARE = 0.5
#: The frameworks whose server prints a ready banner the launcher detects; another framework has no stamp to wait for.
_SERVER_FRAMEWORKS = ("sglang", "vllm", "atom")
#: Written by the launcher next to ``server.log`` (``_subprocess_kill``); its first field is the wall-clock second the
#: server reported ready, its second the seconds the boot took from the spawn (0 for a server the round found ready).
_READY_STAMP = "server_ready_at"
_MAX_POLL_SEC = 2.0


def boot_limit_sec(benchmark: Mapping[str, Any], timeout_sec: float) -> float | None:
    """How long the server of a round on ``benchmark`` may take to be ready; ``None`` when no stamp can say it is.

    The budget is the one the point carries (``server_lifecycle.server_ready_timeout_s``), never more than the round's
    ``timeout_sec``; a point that carries none gets the cap itself, which is what the launcher writes into that field.
    """
    if str(benchmark.get("framework") or "").lower() not in _SERVER_FRAMEWORKS:
        return None
    lifecycle = benchmark.get("server_lifecycle")
    carried = lifecycle.get("server_ready_timeout_s") if isinstance(lifecycle, Mapping) else None
    budget = float(carried) if isinstance(carried, (int, float)) and carried > 0 else timeout_sec
    return BOOT_SHARE * min(budget, timeout_sec)


class BootGuard:
    """Cancels ``scope`` when a watched round's server is not ready within its limit."""

    def __init__(self, output_dir: Path, scope: CancelScope) -> None:
        self._output_dir = output_dir
        self._scope = scope
        self._expired_after: float | None = None

    @property
    def expired_after(self) -> float | None:
        """The limit, in seconds, that ended a round; ``None`` while none has."""
        return self._expired_after

    @contextlib.contextmanager
    def watching(self, limit_sec: float) -> Iterator[None]:
        """Watch the round that runs inside the block."""
        started = time.time()
        stop = threading.Event()
        poll = min(_MAX_POLL_SEC, max(0.05, limit_sec / 20))

        def watch() -> None:
            while not stop.wait(poll):
                if self._ready_since(started):
                    return
                if time.time() - started >= limit_sec:
                    self._expired_after = limit_sec
                    self._scope.cancel(reason=BOOT_CANCEL_REASON)
                    return

        thread = threading.Thread(target=watch, name="hl-boot-watch", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=poll + 1)

    def _ready_since(self, started: float) -> bool:
        try:
            stamps = list(self._output_dir.rglob(_READY_STAMP))
        except OSError:
            return False
        return any(_stamped_at(stamp) >= started for stamp in stamps)


def _stamped_at(stamp: Path) -> float:
    """The wall-clock second a stamp says its server was ready; ``0.0`` for one that cannot be read."""
    try:
        return float(stamp.read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def boot_seconds(directory: Path) -> float | None:
    """The longest server boot the ready stamps below ``directory`` record, or ``None`` when no round booted a server.

    A test boots its server once, in its first round, and its later rounds reuse it and stamp no boot, so the longest
    one is the boot of the test. A stamp that cannot be read is no boot.
    """
    try:
        stamps = list(directory.rglob(_READY_STAMP))
    except OSError:
        return None
    booted = [seconds for seconds in map(_booted_in, stamps) if seconds > 0.0]
    return max(booted, default=None)


def read_ready_stamp(directory: Path) -> tuple[float, float] | None:
    """The ``(ready_unix, boot_seconds)`` the stamp beside a round's server log records, or ``None`` when the round left
    none or it cannot be read; ``boot_seconds`` is 0 for a round that found its server ready."""
    stamp = directory / _READY_STAMP
    ready = _stamped_at(stamp)
    return (ready, _booted_in(stamp)) if ready > 0.0 else None


def _booted_in(stamp: Path) -> float:
    """The seconds the boot a stamp records took; ``0.0`` for a stamp that records none or cannot be read."""
    try:
        seconds = float(stamp.read_text(encoding="utf-8").split()[1])
    except (OSError, ValueError, IndexError):
        return 0.0
    return seconds if math.isfinite(seconds) else 0.0
