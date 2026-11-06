# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The round's phase marks: the instants a round passes from one phase to the next, written as ``phase_marks.json``.

A mark carries the wall-clock instant readers align rounds by and the monotonic one the round's own timers use.
Listeners (the GPU sampler, the KV recorder) hear each phase as it is marked.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from inference_suite.contracts.layout import PHASE_MARKS, RunLayout


class Phase(str, Enum):
    SPAWN = "spawn"
    READY = "ready"
    WARMUP = "warmup"
    MEASURED = "measured"
    FINISHED = "finished"
    PROFILE_OPEN = "profile_open"
    PROFILE_CLOSE = "profile_close"
    EVAL = "eval"
    EVAL_END = "eval_end"
    TEARDOWN = "teardown"


@dataclass(frozen=True)
class PhaseMark:
    phase: Phase
    ts: float
    mono: float

    def to_document(self) -> dict[str, object]:
        return {"phase": self.phase.value, "ts": round(self.ts, 3), "mono": round(self.mono, 3)}


#: Hears a phase as it is marked, with the monotonic instant it began.
PhaseListener = Callable[[Phase, float], None]


class PhaseClock:
    """The marks of one round; each phase begins at most once, so a repeated mark keeps the first instant."""

    def __init__(self, listeners: Sequence[PhaseListener] = ()) -> None:
        self._listeners = tuple(listeners)
        self._marks: dict[Phase, PhaseMark] = {}

    def mark(self, phase: Phase, mono: float | None = None) -> PhaseMark:
        if phase in self._marks:
            return self._marks[phase]
        mark = PhaseMark(phase, time.time(), time.monotonic() if mono is None else mono)
        self._marks[phase] = mark
        for listener in self._listeners:
            listener(phase, mark.mono)
        return mark

    def first(self, phase: Phase) -> PhaseMark | None:
        return self._marks.get(phase)

    @property
    def marks(self) -> tuple[PhaseMark, ...]:
        return tuple(sorted(self._marks.values(), key=lambda mark: mark.mono))

    def write(self, layout: RunLayout) -> None:
        """Replace ``phase_marks.json`` whole, so a reader never sees half of it."""

        document = {"phase_marks": [mark.to_document() for mark in self.marks]}
        staging = layout.phase_marks.with_name(f".{PHASE_MARKS}.{os.getpid()}")
        staging.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, layout.phase_marks)


__all__ = ["Phase", "PhaseClock", "PhaseListener", "PhaseMark"]
