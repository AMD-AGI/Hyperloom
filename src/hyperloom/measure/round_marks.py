# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a round of a test left on disk about when its parts began, and the phases that follow from it.

A round is one or two inference_suite runs (a warmup run whose result is discarded, then the measured one), and each
run's run_dir holds ``server_ready_at`` (the instant the server reported ready and how long its boot took from the
spawn) and ``kv_metrics.json`` (the instants the round's client began its evaluation, or, for an AgentX client, its
warmup and measured phases began and ended; the file's own time is when the run ended), both written by the
inference_suite round. This module reads those marks after the round and turns them into phase boundaries. A mark that
is not there only makes a phase coarser: the boundaries tile the round's window, so the seconds the marks cannot place
stay in the phase before them and none are dropped.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import yaml

from hyperloom.common.outside_facts import SGLANG_CUDA_GRAPH_CAPTURE_END
from hyperloom.measure import log_causes
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
from inference_suite.contracts import FailedStep, RunLayout, SuiteConfig

__all__ = [
    "ROLE_BOOT",
    "ROLE_EVAL",
    "ROLE_MEASURE",
    "ROLE_WARMUP",
    "BootParts",
    "Boundary",
    "RoundShape",
    "RunMarks",
    "boot_parts",
    "read_runs",
    "round_boundaries",
    "profile_stall",
    "stalled_in_profile_window",
]

#: The roles of a round's call: one that only boots the server, the warmup round of a test on a held server, one that
#: measures, and the round that only evaluates a graded test after its measured bench.
ROLE_BOOT = "boot"
ROLE_WARMUP = "warmup"
ROLE_MEASURE = "measure"
ROLE_EVAL = "eval"
#: Where ``run_grid`` keeps the discarded warmup run of a round, below the round's own directory.
_WARMUP_DIR = "warmup_round"
_ROUND_GLOB = "variant_*"
_UTF8 = "utf-8"
#: The phases ``kv_metrics.json`` records the instants of: the client's evaluation, and an AgentX client's own phases.
_EVAL, _CLIENT_WARMUP, _CLIENT_MEASURED, _CLIENT_FINISHED = "eval", "warmup", "measured", "finished"
#: The marks of a profile round's window in its ``phase_marks.json``, as inference_suite's ``Phase`` names them.
_PROFILE_OPEN, _PROFILE_CLOSE = "profile_open", "profile_close"
#: The control requests a profile window sends its server, as inference_suite's ``profile.inject`` names them.
_START_PROFILE, _STOP_PROFILE = "/start_profile", "/stop_profile"
_WINDOW_REQUESTS = (_START_PROFILE, _STOP_PROFILE)
#: How urllib words the error of a request the server answered with an error status.
_ANSWERED_WITH_STATUS = "HTTP Error "
#: How the suite's silence gate words the stall it ends a round for (``inference_suite.runtime.watch``): the seconds
#: from the server's last output, its log's or the client's, to the poll that ended the round.
_STALL_DETAIL = re.compile(r"no server log or client output for (\d+(?:\.\d+)?)s")
#: How long before an AgentX window's ``/start_profile`` went out the server may have last been heard and still count as
#: talking then. A server serving the replay writes its log, and the replay its own, every few seconds; the round sees
#: that output only at its polls, and the instant it ends the round is marked once the client has been reaped.
_TALKING_SLACK_SEC = 10.0
_MIB = 1 << 20
#: A boot is in the first bytes of a server's log, and a log of a long test is large.
_LOG_HEAD_BYTES = 8 * _MIB
#: How much of the end of a server log is read for its last line.
_LOG_END_BYTES = 64 * 1024
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
    """The instants one run of a round left: when its server was ready and how long it took to boot (0 for a run
    that found it ready), when the client began its evaluation or its own phases, and when the run ended.

    ``ended`` is the modification time of ``kv_metrics.json``, which the run writes as it ends; ``None`` when the run left
    none.
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
    instants, ended = _kv_instants(RunLayout(directory).kv_metrics, since)
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
    """The first instant (at or after ``since``) the run marked each phase begin, and when the file was written."""
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
        client = (
            ACCURACY_EVAL if shape.role == ROLE_EVAL else MEASURED if last and shape.role == ROLE_MEASURE else WARMUP
        )
        boundaries.extend(_run_boundaries(run, shape, client))
        if run.ended is not None:
            boundaries.append(Boundary(shape.tail if last else ROUND_COLLECT, run.ended))
    return [Boundary(boundary.name, min(max(boundary.at, started), ended)) for boundary in boundaries]


def _run_boundaries(run: RunMarks, shape: RoundShape, client: str) -> list[Boundary]:
    """One run: the boot when it booted the server, then its client work, which counts as ``client`` and is split at the
    instants the client left.

    A discarded run, and the one run of a boot or a warmup call, is warmup; the run of an evaluation call is the
    evaluation from the start. An AgentX measured run is split at its client's own phases when its measured phase began;
    any other measured run is measured up to its evaluation.
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
    if run.eval_at is not None and client != ACCURACY_EVAL:
        boundaries.append(Boundary(ACCURACY_EVAL, run.eval_at))
    return boundaries


def _boot_boundaries(run: RunMarks) -> list[Boundary]:
    """The boot of a run: one phase, or, when the engine's log says what part of it each took, its parts in the order an
    engine does them (what the log does not say first, then the weights, the compile, the graphs)."""
    spawned = run.ready - run.boot_seconds
    known = boot_parts(RunLayout(run.directory).server_log).known()
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


def stalled_in_profile_window(run_dir: Path) -> bool:
    """Whether the server of the round in ``run_dir``, which the round ended for its silence, went silent while the
    round's profile window was open.

    An AgentX window sends its ``/start_profile`` and ``/stop_profile`` from the round's poll and waits up to ten minutes
    for an answer, so its marks place no silence; ``profile/capture.json`` records a request the server never answered
    as ``<path>: <error>`` (one the server answered with an error status was answered). An unanswered ``/stop_profile``
    went out inside the opened window: the stall is in it. An unanswered ``/start_profile`` is the window's only when the
    server was still talking as it went out, which the round's own clock places: the request went out
    ``agentx_open_after_sec`` after the ``measured`` mark, and the server was last heard the silence the round's
    ``report.json`` states before the ``finished`` mark; a server last heard more than :data:`_TALKING_SLACK_SEC` before the request had gone
    silent before the window. A fixed load's window does not record its ``/stop_profile`` answer, so for it the marks
    decide. The round marks ``finished`` as the silence ends it, so the silence began the round's silence limit before
    that mark at the latest; it began in the window when ``profile_open`` is at or before that instant and
    ``profile_close`` is absent or after it (a window open when the round ended is closed then). A round whose capture,
    marks or config cannot be read is not known to have stalled there, nor is an unanswered ``/start_profile`` of a round
    whose report states no silence.
    """
    layout = RunLayout(run_dir)
    unanswered = _unanswered_window_request(layout)
    if unanswered == _STOP_PROFILE:
        return True
    try:
        config = SuiteConfig.from_document(yaml.safe_load(layout.config.read_text(encoding=_UTF8)))
        instants = _first_instants(json.loads(layout.phase_marks.read_text(encoding=_UTF8)))
    except (OSError, ValueError, yaml.YAMLError):
        return False
    if unanswered == _START_PROFILE:
        return _talking_when_window_started(instants, config.profile.agentx_open_after_sec, _failure_detail(layout))
    opened, finished = instants.get(_PROFILE_OPEN), instants.get(_CLIENT_FINISHED)
    if opened is None or finished is None:
        return False
    began = finished - config.deadlines.silence_sec
    closed = instants.get(_PROFILE_CLOSE)
    return opened <= began and (closed is None or closed > began)


def _first_instants(document: object) -> dict[str, float]:
    """The first monotonic instant of each phase a ``phase_marks.json`` document marks."""
    marks = document.get("phase_marks") if isinstance(document, dict) else None
    instants: dict[str, float] = {}
    for mark in marks if isinstance(marks, list) else []:
        phase, at = (mark.get("phase"), _instant(mark.get("mono"))) if isinstance(mark, dict) else (None, None)
        if isinstance(phase, str) and at is not None:
            instants.setdefault(phase, at)
    return instants


def _talking_when_window_started(instants: dict[str, float], open_after: float | None, detail: str) -> bool:
    """Whether the server was still heard when the AgentX window sent its ``/start_profile``, ``open_after`` seconds
    after the ``measured`` mark; ``detail`` states how long it had been silent when the ``finished`` mark ended it."""
    measured, finished = instants.get(_CLIENT_MEASURED), instants.get(_CLIENT_FINISHED)
    found = _STALL_DETAIL.search(detail)
    if measured is None or finished is None or open_after is None or found is None:
        return False
    last_heard = finished - float(found.group(1))
    return last_heard >= measured + open_after - _TALKING_SLACK_SEC


def _failure_detail(layout: RunLayout) -> str:
    """What the round's ``report.json`` says of why it failed; ``""`` when it says nothing or cannot be read."""
    try:
        document = json.loads(layout.report.read_text(encoding=_UTF8))
    except (OSError, ValueError):
        return ""
    detail = document.get("failure_detail") if isinstance(document, dict) else None
    return detail if isinstance(detail, str) else ""


def _unanswered_window_request(layout: RunLayout) -> str | None:
    """The window control request the round's ``profile/capture.json`` records its server never answered, if any."""
    try:
        document = json.loads(layout.profile_capture.read_text(encoding=_UTF8))
    except (OSError, ValueError):
        return None
    error = document.get("error") if isinstance(document, dict) else None
    if not isinstance(error, str):
        return None
    path, _, cause = error.partition(": ")
    return path if path in _WINDOW_REQUESTS and not cause.startswith(_ANSWERED_WITH_STATUS) else None


def profile_stall(run_dir: Path, step: FailedStep) -> str | None:
    """``None`` unless the round in ``run_dir`` failed at ``step`` ``server_stalled`` while its profile window was open
    (:func:`stalled_in_profile_window`); then the last line of its server log, ``""`` when it has none."""
    if step is not FailedStep.SERVER_STALLED or not stalled_in_profile_window(run_dir):
        return None
    try:
        with RunLayout(run_dir).server_log.open("rb") as handle:
            handle.seek(max(0, handle.seek(0, os.SEEK_END) - _LOG_END_BYTES))
            return log_causes.last_line(handle.read().decode(_UTF8, errors="replace"))
    except OSError:
        return ""
