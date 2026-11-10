# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tune a shape list in chunks, each sized by the measured cost of the chunks before it.

A whole list in one tuner process is lost when the process is killed at the deadline, and a per-shape cost fixed in
the code goes stale with every aiter that changes its candidate count. A chunk's wall time, per wave of shapes (one
shape per card), is what the next chunk is sized by; the fixed cost only sizes the first.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

#: Waves in one chunk at most, unless one unit alone holds more: a chunk killed at the deadline loses all of its shapes,
#: and a later shape can cost more than the chunk that measured the rate.
MAX_WAVES_PER_CHUNK = 4

#: Where, in a tuner's directory, each chunk's own CSVs go: out of the way of the ``profile_*.csv`` and ``tuned_*.csv``
#: the merged run writes beside it, which callers glob for.
CHUNK_DIR = "chunks"
#: One untuned-CSV row per shape; a unit is the rows that must be tuned in the same chunk.
Unit = Sequence[str]


@dataclass(frozen=True)
class ChunkBudget:
    """The tuner's deadline (on ``clock``), the cards one chunk spreads its shapes over, the per-shape cost the first
    chunk is sized by, and the time kept after the last chunk for the work that follows it."""

    deadline: float
    cards: int
    first_shape_s: float
    reserve_s: float
    clock: Callable[[], float]


@dataclass
class ChunkTally:
    """What the chunks covered: the shapes handed to a tuner process, and the shapes no chunk reached."""

    chunks: int = 0
    shapes_run: int = 0
    shapes_left: int = 0
    wave_s: list[float] = field(default_factory=list)


class ChunkPacer:
    """How many shapes the next chunk may hold, and the timeout of its process."""

    def __init__(self, budget: ChunkBudget) -> None:
        self._budget = budget
        self._cards = max(1, budget.cards)
        self._wave_s: float | None = None

    def remaining_s(self) -> float:
        return self._budget.deadline - self._budget.clock()

    def room(self) -> int:
        """Shapes the time left before the reserve pays for; one wave for the first chunk, which nothing measured."""
        if self._wave_s is None:
            return self._cards
        return max(0, math.floor((self.remaining_s() - self._budget.reserve_s) / self._wave_s)) * self._cards

    def size(self) -> int:
        """Shapes the next chunk holds when its units are small enough: the room, at most the wave cap."""
        return min(self.room(), MAX_WAVES_PER_CHUNK * self._cards)

    def timeout_s(self) -> int:
        """The chunk process's kill timeout: half the reserve is left for the work after it."""
        return max(1, int(self.remaining_s() - self._budget.reserve_s / 2))

    def first_wave_fits(self) -> bool:
        return self.remaining_s() - self._budget.reserve_s >= self._budget.first_shape_s

    def record(self, shapes: int, elapsed_s: float) -> float:
        """Take a finished chunk's wall time as the cost of a wave; returns that cost."""
        waves = math.ceil(max(shapes, 1) / self._cards)
        self._wave_s = max(elapsed_s, 1e-3) / waves
        return self._wave_s


def take_units(pending: list[Unit], room: int, size: int, *, at_least_one: bool) -> list[Unit]:
    """Remove from ``pending`` and return the units, in order, that fit ``room`` and ``size``; a unit larger than
    ``size`` but within ``room`` goes alone. A unit that does not fit is passed over for a smaller one after it, and
    stays pending."""
    taken: list[Unit] = []
    used = 0
    for unit in list(pending):
        if used + len(unit) <= room and (not taken or used + len(unit) <= size):
            taken.append(unit)
            used += len(unit)
            pending.remove(unit)
    if not taken and at_least_one and pending:
        taken.append(pending.pop(0))
    return taken


#: Runs one chunk: its index, its untuned rows and its kill timeout; says whether the next chunk may start.
RunChunk = Callable[[int, list[str], int], bool]


def run_in_chunks(units: Sequence[Unit], budget: ChunkBudget, run_chunk: RunChunk) -> ChunkTally:
    """Run ``units`` chunk by chunk until they are done, the deadline leaves no room, or ``run_chunk`` says stop."""
    pacer, pending, tally = ChunkPacer(budget), list(units), ChunkTally()
    while pending:
        first = tally.chunks == 0
        units = take_units(pending, pacer.room(), pacer.size(), at_least_one=first)
        rows = [row for unit in units for row in unit]
        if not rows:
            break
        if first and not pacer.first_wave_fits():
            log.warning(
                "a first chunk of %d shape(s) may not finish in %.0fs; running it anyway",
                len(rows),
                pacer.remaining_s(),
            )
        started = budget.clock()
        go_on = run_chunk(tally.chunks, rows, pacer.timeout_s())
        tally.wave_s.append(pacer.record(len(rows), budget.clock() - started))
        tally.chunks += 1
        tally.shapes_run += len(rows)
        if not go_on:
            break
    tally.shapes_left = sum(len(unit) for unit in pending)
    if tally.shapes_left:
        log.warning(
            "%d shape(s) left untuned: %d chunk(s) ran %d shape(s), the last at %.0fs a wave of %d card(s)",
            tally.shapes_left,
            tally.chunks,
            tally.shapes_run,
            tally.wave_s[-1] if tally.wave_s else 0.0,
            max(1, budget.cards),
        )
    return tally


def decode_band_units(rows: Sequence[str], header: str, decode_m: Sequence[int]) -> list[Unit]:
    """One unit per row, except that a dispatch group's decode-band rows travel with the group's first row.

    A group tuned on part of its decode band serves the rest with a prefill tile, which is worse than leaving the
    whole group to the shipped table, so a chunk takes a group's band whole or not at all.
    """
    columns = {name.strip().upper(): i for i, name in enumerate(header.split(","))}
    if not {"M", "N", "K"} <= set(columns):
        return [[row] for row in rows]
    band = set(decode_m)

    def parsed(row: str) -> tuple[tuple[str, ...], int | None]:
        parts = [part.strip() for part in row.split(",")]
        group = tuple(parts[i] for i in sorted(i for name, i in columns.items() if name in ("N", "K", "Q_DTYPE_W")))
        try:
            return group, int(parts[columns["M"]])
        except (IndexError, ValueError):
            return group, None

    band_rows: dict[tuple[str, ...], list[str]] = {}
    for row in rows:
        group, m = parsed(row)
        if m in band:
            band_rows.setdefault(group, []).append(row)
    units: list[Unit] = []
    opened: set[tuple[str, ...]] = set()
    for row in rows:
        group, m = parsed(row)
        in_band = m in band
        if group not in opened:
            opened.add(group)
            units.append([*band_rows.get(group, []), *([] if in_band else [row])])
        elif not in_band:
            units.append([row])
    return units


def write_rows(path: Path, header: str, rows: Sequence[str]) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def csv_lines(path: Path) -> list[str]:
    """A CSV's non-blank lines, header first; none for a file that is not there."""
    try:
        return [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return []


class MergedCsv:
    """Rows of the chunks' CSVs under the first header seen, rewritten to ``path`` after every chunk so a kill of the
    whole run keeps what the finished chunks tuned."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.header: str | None = None
        self.rows: list[str] = []

    def absorb(self, lines: Sequence[str]) -> None:
        if not lines:
            return
        if self.header is None:
            self.header = lines[0]
        self.rows.extend(lines[1:])
        write_rows(self.path, self.header, self.rows)
