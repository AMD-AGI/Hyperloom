# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a round of a test left on disk about when its parts began, and the phases that follow from it.

``run_grid`` runs a round as one or two Magpie runs (a warmup run whose result is discarded, then the measured one) and
writes beside each run's server log what the launcher's watchdog saw: ``server_ready_at`` (the instant the server
reported ready and how long its boot took from the spawn) and ``kv_metrics.json`` (the instants the watchdog saw the
client start its evaluation, or, for an AgentX client, its warmup and measured phases begin and end; it looks every half
second, and the file's own time is when the run ended). The backend asks for no more than that of ``run_grid``; this
module reads those marks after the round and turns them into phase boundaries. A mark that is not there only makes a phase coarser: the boundaries tile the round's window, so the
seconds the marks cannot place stay in the phase before them and none are dropped.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from hyperloom.common.outside_facts import SGLANG_CUDA_GRAPH_CAPTURE_END
from hyperloom.measure.boot_watch import read_ready_stamp
from hyperloom.measure.phases import (
    ACCURACY_EVAL,
    BOOT_COMPILE,
    BOOT_GRAPHS,
    BOOT_OTHER,
    BOOT_WEIGHTS,
    CLIENT_START,
    MEASURED,
    ROUND,
    ROUND_COLLECT,
    ROUND_SETUP,
    SERVER_BOOT,
    TEARDOWN,
    WARMUP,
)

__all__ = [
    "ROLE_BOOT",
    "ROLE_MEASURE",
    "ROLE_WARMUP",
    "BootParts",
    "Boundary",
    "RoundShape",
    "RunMarks",
    "boot_parts",
    "read_runs",
    "round_boundaries",
]

#: The roles of a round's call: one that only boots the server, the warmup round of a graded test, and one that measures.
ROLE_BOOT = "boot"
ROLE_WARMUP = "warmup"
ROLE_MEASURE = "measure"
#: Where ``run_grid`` keeps the discarded warmup run of a round, below the round's own directory.
_WARMUP_DIR = "warmup_round"
_ROUND_GLOB = "variant_*"
_KV_FILE = "kv_metrics.json"
_SERVER_LOG = "server.log"
#: Magpie runs a server in a directory of its own below the run's: ``<run directory>/benchmark_sglang_<time>/server.log``.
_SERVER_LOG_DIRS = "benchmark_*"
_UTF8 = "utf-8"
#: The phases ``kv_metrics.json`` records the instants of: the client's evaluation, and an AgentX client's own phases.
_EVAL, _CLIENT_WARMUP, _CLIENT_MEASURED, _CLIENT_FINISHED = "eval", "warmup", "measured", "finished"
_MIB = 1 << 20
#: A boot is in the first bytes of a server's log, and a log of a long test is large.
_LOG_HEAD_BYTES = 8 * _MIB
_KV_MAX_BYTES = 64 * _MIB
_SECONDS = r"(\d+(?:\.\d+)?)"
_WEIGHTS_PATTERNS = (
    re.compile(rf"Loading weights took {_SECONDS} seconds"),
    re.compile(rf"Model loading took .*? and {_SECONDS} seconds"),
)
_COMPILE_PATTERNS = (re.compile(rf"torch\.compile takes {_SECONDS} s in total"),)
_GRAPH_PATTERNS = (
    re.compile(SGLANG_CUDA_GRAPH_CAPTURE_END),
    re.compile(rf"Graph capturing finished in {_SECONDS} secs?"),
)
#: SGLang states when it starts and ends loading the weights, not how long it took; each line begins with its time.
_STAMP = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\]"
_LOAD_BEGIN = re.compile(rf"{_STAMP} Load weight begin")
_LOAD_END = re.compile(rf"{_STAMP} Load weight end")
_STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class Boundary(NamedTuple):
    """A phase that begins at ``at`` (a unix time) and lasts until the next boundary."""

    name: str
    at: float


@dataclass(frozen=True)
class BootParts:
    """The seconds of a server's boot the engine's own log accounts for; ``None`` for a part the log does not say."""

    weights_load: float | None = None
    compile: float | None = None
    graph_capture: float | None = None

    def known(self) -> list[tuple[str, float]]:
        pairs = (
            (BOOT_WEIGHTS, self.weights_load),
            (BOOT_COMPILE, self.compile),
            (BOOT_GRAPHS, self.graph_capture),
        )
        return [(name, seconds) for name, seconds in pairs if seconds is not None]


@dataclass(frozen=True)
class RunMarks:
    """The instants one Magpie run of a round left: when its server was ready and how long it took to boot (0 for a run
    that found it ready), when the client began its evaluation or its own phases, and when the run ended.

    ``ended`` is the modification time of ``kv_metrics.json``, which the watchdog writes as the run ends; ``None`` when the
    run left none.
    """

    directory: Path
    ready: float
    boot_seconds: float = 0.0
    eval_at: float | None = None
    client_warmup_at: float | None = None
    client_measured_at: float | None = None
    client_finished_at: float | None = None
    ended: float | None = None


@dataclass(frozen=True)
class RoundShape:
    """What the caller knows of a round before it looks at the disk.

    ``role`` is one of ``ROLE_*``; ``agentx`` says the client is an AgentX replay; ``reuses_server`` that the round began on
    a server that was already ready; ``failed`` that the round's result failed; ``tail`` names the phase that follows the
    last run (the server's teardown, or only a collection when the server stays up).
    """

    role: str
    agentx: bool
    reuses_server: bool
    failed: bool
    tail: str = TEARDOWN


# -- reading the marks --------------------------------------------------------------------------------------------------


def read_runs(output_root: Path, window: tuple[float, float]) -> list[RunMarks]:
    """The runs that began inside ``window`` below ``output_root``, in the order they were ready.

    A run is found by its ready stamp, which holds the wall-clock instant the run's server was ready, so a stamp an
    earlier round left in a directory the same name serves again is not taken for this round's.
    """
    started, ended = window
    runs = []
    for directory in _run_directories(output_root):
        stamp = read_ready_stamp(directory)
        if stamp is not None and started <= stamp[0] <= ended + 1.0:
            runs.append(_marks_of(directory, stamp, started))
    return sorted(runs, key=lambda run: run.ready)


def _run_directories(output_root: Path) -> Iterator[Path]:
    try:
        rounds = sorted(Path(output_root).glob(_ROUND_GLOB))
    except OSError:
        return
    for directory in rounds:
        yield directory / _WARMUP_DIR
        yield directory


def _marks_of(directory: Path, stamp: tuple[float, float], since: float) -> RunMarks:
    ready, boot = stamp
    instants, ended = _kv_instants(directory / _KV_FILE, since)
    return RunMarks(
        directory,
        ready,
        boot,
        eval_at=instants.get(_EVAL),
        client_warmup_at=instants.get(_CLIENT_WARMUP),
        client_measured_at=instants.get(_CLIENT_MEASURED),
        client_finished_at=instants.get(_CLIENT_FINISHED),
        ended=ended,
    )


def _kv_instants(path: Path, since: float) -> tuple[dict[str, float], float | None]:
    """The first instant (at or after ``since``) the watchdog saw each phase begin, and when the file was written."""
    try:
        if path.stat().st_size > _KV_MAX_BYTES:
            return {}, None
        document = json.loads(path.read_text(encoding=_UTF8))
        written = path.stat().st_mtime
    except (OSError, ValueError):
        return {}, None
    marks = document.get("phase_marks") if isinstance(document, dict) else None
    instants: dict[str, float] = {}
    for mark in marks if isinstance(marks, list) else []:
        phase, at = (mark.get("phase"), _instant(mark.get("ts"))) if isinstance(mark, dict) else (None, None)
        if isinstance(phase, str) and at is not None and at >= since:
            instants.setdefault(phase, at)
    return instants, written if written >= since else None


def _instant(value: object) -> float | None:
    """``value`` as a unix time, or ``None`` when it is not a finite number a float can hold."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        instant = float(value)
    except OverflowError:
        return None
    return instant if math.isfinite(instant) else None


def server_log_of(run_directory: Path) -> Path:
    """The server log of the run in ``run_directory``: the one in it, else the newest launch's below it.

    Magpie keeps a launch's log in ``benchmark_<framework>_<time>`` below the run's directory; the time is in the name, so
    the last of them in name order is the launch that went on to serve (an earlier one is a launch that was started again).
    """
    direct = run_directory / _SERVER_LOG
    if direct.is_file():
        return direct
    try:
        launches = sorted(run_directory.glob(f"{_SERVER_LOG_DIRS}/{_SERVER_LOG}"))
    except OSError:
        return direct
    return launches[-1] if launches else direct


def boot_parts(server_log: Path) -> BootParts:
    """The parts of a boot the head of ``server_log`` accounts for (the longest mention of each: ranks boot in parallel)."""
    try:
        with server_log.open("rb") as handle:
            text = handle.read(_LOG_HEAD_BYTES).decode(_UTF8, errors="replace")
    except OSError:
        return BootParts()
    weights = _longest(text, _WEIGHTS_PATTERNS)
    if weights is None:
        weights = _sglang_weights_seconds(text)
    return BootParts(weights, _longest(text, _COMPILE_PATTERNS), _longest(text, _GRAPH_PATTERNS))


def _longest(text: str, patterns: Sequence[re.Pattern[str]]) -> float | None:
    found = [float(match.group(1)) for pattern in patterns for match in pattern.finditer(text)]
    return max(found, default=None)


def _sglang_weights_seconds(text: str) -> float | None:
    begins, ends = _LOAD_BEGIN.findall(text), _LOAD_END.findall(text)
    try:
        spans = [_seconds_between(begin, end) for begin, end in zip(begins, ends)]
    except ValueError:
        return None
    return max(spans, default=None)


def _seconds_between(begin: str, end: str) -> float:
    """Both instants are in the log's own zone, so only their difference means anything; UTC makes it exact."""
    return (_parsed(end) - _parsed(begin)).total_seconds()


def _parsed(text: str) -> datetime:
    return datetime.strptime(text, _STAMP_FORMAT).replace(tzinfo=timezone.utc)


# -- the boundaries -----------------------------------------------------------------------------------------------------


def round_boundaries(window: tuple[float, float], runs: Sequence[RunMarks], shape: RoundShape) -> list[Boundary]:
    """The phases of one round that began at ``window[0]``, as the boundaries between them, in order.

    The first begins at the start of the window and none lies outside it, so the boundaries tile it: an instant no mark
    places belongs to the phase before it. Where no run left a ready stamp, a round that failed while it was to boot its server is all boot, and
    any other is one phase the marks could not split.
    """
    started, ended = window
    if not runs:
        return [Boundary(SERVER_BOOT if shape.failed and not shape.reuses_server else ROUND, started)]
    boundaries = [Boundary(ROUND_SETUP, started)]
    for index, run in enumerate(runs):
        last = index == len(runs) - 1
        client = MEASURED if last and shape.role == ROLE_MEASURE else WARMUP
        boundaries.extend(_run_boundaries(run, shape, client))
        if run.ended is not None:
            boundaries.append(Boundary(shape.tail if last else ROUND_COLLECT, run.ended))
    return [Boundary(boundary.name, min(max(boundary.at, started), ended)) for boundary in boundaries]


def _run_boundaries(run: RunMarks, shape: RoundShape, client: str) -> list[Boundary]:
    """One run: the boot when it booted the server, then its client work, which counts as ``client`` and is split at the
    instants the client left.

    A discarded run, and the one run of a boot or a warmup call, is warmup. An AgentX measured run is split at its
    client's own phases when its measured phase began; any other measured run is measured up to its evaluation.
    """
    boundaries = _boot_boundaries(run) if run.boot_seconds > 0 else []
    if shape.agentx and client == MEASURED and run.client_measured_at is not None:
        steps = [
            Boundary(CLIENT_START, run.ready),
            Boundary(WARMUP, run.client_warmup_at),
            Boundary(MEASURED, run.client_measured_at),
            Boundary(ROUND_COLLECT, run.client_finished_at),
        ]
        return boundaries + [step for step in steps if step.at is not None]
    boundaries.append(Boundary(client, run.ready))
    if run.eval_at is not None:
        boundaries.append(Boundary(ACCURACY_EVAL, run.eval_at))
    return boundaries


def _boot_boundaries(run: RunMarks) -> list[Boundary]:
    """The boot of a run: one phase, or, when the engine's log says what part of it each took, its parts in the order an
    engine does them (what the log does not say first, then the weights, the compile, the graphs)."""
    spawned = run.ready - run.boot_seconds
    known = boot_parts(server_log_of(run.directory)).known()
    if not known:
        return [Boundary(SERVER_BOOT, spawned)]
    left, parts = run.boot_seconds, []
    for name, seconds in known:
        taken = min(seconds, left)
        left -= taken
        parts.append((name, taken))
    steps = ([(BOOT_OTHER, left)] if left > 0 else []) + parts
    boundaries, at = [], spawned
    for name, seconds in steps:
        boundaries.append(Boundary(name, at))
        at += seconds
    return boundaries
