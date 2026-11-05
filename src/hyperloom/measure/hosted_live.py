# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the runner of a running hosted test last said, as the backend read it from the runner's ``progress.json``.

The runner (:class:`~hyperloom.measure.runner.service.Progress`) writes the line and how long ago the log it comes from
was written, both by its own clock. The backend reads that file at every poll anyway; it keeps the newest reading and
adds the time since it read it, so no clock but one is ever compared with another.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from hyperloom.measure.live_log import clean_line

__all__ = ["LiveOutputs", "finite_number"]


def finite_number(value: object) -> float | None:
    """``value`` as a float when it is a finite number (a bool is not one); ``None`` for anything else, including an
    integer too large for a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class _Reading:
    stamp: tuple[object, float, str]
    age_sec: float
    line: str
    read_at: float


class LiveOutputs:
    """The newest live output of each running test; ``clock`` is the backend's own (wall) clock."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._readings: dict[str, _Reading] = {}

    def note(self, request_id: str, progress: Mapping[str, Any] | None) -> None:
        """Take what a ``progress.json`` says; a document without a live line (no log written yet, a runner that does
        not report one) changes nothing. The line is cleaned again here, as it is whatever the test printed."""
        document = progress or {}
        age, raw = finite_number(document.get("last_log_age_sec")), document.get("last_line")
        if age is None or age < 0 or not isinstance(raw, str):
            return
        line = clean_line(raw)
        stamp = (document.get("updated_unix"), age, line)
        with self._lock:
            earlier = self._readings.get(request_id)
            read_at = earlier.read_at if earlier is not None and earlier.stamp == stamp else self._clock()
            self._readings[request_id] = _Reading(stamp, age, line, read_at)

    def get(self, request_id: str) -> tuple[float, str] | None:
        """``(seconds since the log was written, its last line)``, or ``None`` before the runner has said anything."""
        with self._lock:
            reading = self._readings.get(request_id)
        if reading is None:
            return None
        return reading.age_sec + max(0.0, self._clock() - reading.read_at), reading.line

    def forget(self, request_id: str) -> None:
        """The test is over: nothing it said is reported any more."""
        with self._lock:
            self._readings.pop(request_id, None)
