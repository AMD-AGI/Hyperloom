# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-GPU telemetry over the measured phase of one benchmark round.

A benchmark-wide GPU reading does not describe the load being measured: a sampler that watches one card, or whose
window opens before the server launches and closes after the client exits, averages in model load, graph capture and
the idle tail, and on a TP4 round ignores three of the four serving cards.

:class:`GpuPowerRecorder` samples every card in scope with one read-only ``amd-smi metric --power --mem-usage --clock
--temperature --usage`` per interval and keeps only samples taken in the :attr:`GpuPhase.MEASURED` phase. It also reads
the cards' throttle accumulators (``amd-smi metric --violation``) as each measured window opens and closes, so a round
under a power cap reports how much of it the cards spent held back by that cap, and at what graphics clock they ran.
The serving cards are the ones holding a serving-sized share of VRAM during that phase, so a TP4 round on an unpinned
eight-card host is averaged over the four cards it runs on.

On a node with Node Power Management enabled, each sample also reads the node's power and the limit firmware enforces
on it. Firmware meets that limit by lowering every card's power, which the cards' own throttle accumulators do not
count, so a round held back by the node limit reports how much of the phase it spent there.

With ``energy`` on, the recorder also reads each card's energy counter (``amd-smi metric --energy``) as each measured
window opens and closes. Firmware integrates the card's power into that counter, so the energy it gives for the phase
holds however bursty the load is, which a mean of instantaneous readings two seconds apart does not. The summary's
``energy`` block reports it beside the sampled power, which it does not replace.

One batch of samples yields three views: the ``gpu_power.json`` v1 summary (:meth:`GpuPowerRecorder.summary`), one
``gpu_monitor`` block per serving card (:meth:`GpuPowerRecorder.gpu_monitor`), and every sample of every card as CSV.

Nothing here raises: telemetry that fails must cost the round nothing, and a metric with no reading is ``None`` rather
than zero.
"""

from __future__ import annotations

import csv
import glob
import io
import json
import logging
import subprocess  # nosec B404 - fixed read-only amd-smi invocation.
import threading
import time
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from hyperloom_common.amd_smi import (
    NPM_ENABLED,
    energy_joules,
    gfx_clock_limits,
    gpu_rows,
    metric_value,
    single_clock_limit,
)
from hyperloom_common.io import atomic_write_json, atomic_write_text

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_INTERVAL_SEC",
    "MIN_INTERVAL_SEC",
    "GpuMonitorBlock",
    "GpuPhase",
    "GpuPowerRecorder",
    "GpuSample",
    "MetricStats",
    "RecorderOptions",
    "parse_clock_sample",
    "parse_energy_sample",
    "parse_gpu_sample",
    "parse_power_sample",
    "parse_throttle_sample",
]

DEFAULT_INTERVAL_SEC = 2.0
MIN_INTERVAL_SEC = 0.5
_QUERY_TIMEOUT_SEC = 10.0

#: A card counts as serving when it held at least this share of its VRAM during the measured phase. Serving engines
#: reserve most of HBM up front (vLLM's default ``--gpu-memory-utilization`` is 0.9), while an idle card holds a few
#: hundred MB of driver state, so the split is wide and the threshold does not need to be exact.
_SERVING_VRAM_FRACTION = 0.10

#: A serving card held back by its power limit for at least this share of the measured phase is reported as power
#: throttled: its throughput was set by the power cap, not by the clock limit.
_POWER_THROTTLE_REPORT_FRACTION = 0.10
_FRACTION_DIGITS = 4

#: A node sample at or above this share of the NPM limit is one firmware is holding at that limit. A held node reads at
#: the limit, or above it once every card is down at its minimum power; a node the limit is not holding reads below.
_NODE_AT_LIMIT_FRACTION = 0.99

#: The card directories whose ``board`` may hold the node's NPM files. amdgpu creates them on a card whose XGMI physical
#: node id is 0 (``smu_v13_0_6_ppt.c``): one card of a hive, so one card on an MI350 node, and every card without a hive.
_BOARD_GLOB = "/sys/class/drm/card*/device/board"

#: The ``amd-smi metric`` fields each sample reads, and those read at each measured window's open and close.
_SAMPLE_METRICS = ("--power", "--mem-usage", "--clock", "--temperature", "--usage")
_THROTTLE_METRICS = ("--violation",)
#: Read at the same edges as the throttle accumulators, in a call of its own: an amd-smi that cannot read energy then
#: costs the round its energy reading and nothing else.
_ENERGY_METRICS = ("--energy",)

#: Temperature sensors in preference order: MI300-class cards report ``edge`` as N/A and ``hotspot`` as the die reading.
_TEMPERATURE_SENSORS = ("hotspot", "edge")


class GpuPhase(str, Enum):
    """The round phases a recorder is driven through; only :attr:`MEASURED` is sampled."""

    BOOT = "boot"
    WARMUP = "warmup"
    MEASURED = "measured"
    FINISHED = "finished"
    EVAL = "eval"


_PHASE_VALUES = frozenset(GpuPhase)


@dataclass(frozen=True)
class RecorderOptions:
    """What a recorder records beyond the ``gpu_power.json`` v1 summary; each is off unless asked for."""

    #: Where every measured sample of every card is written as CSV on :meth:`GpuPowerRecorder.close`.
    samples_path: str | None = None
    #: Also read each card's energy counter at the edges of each measured window, into the summary's ``energy`` block.
    energy: bool = False


@dataclass(frozen=True)
class GpuSample:
    """One card's reading at one sample; a metric amd-smi did not report is ``None``."""

    unix: float
    gpu: int
    power_w: float | None
    temperature_c: float | None
    gpu_clock_mhz: float | None
    mem_clock_mhz: float | None
    gpu_util_pct: float | None
    vram_pct: float | None


@dataclass(frozen=True)
class MetricStats:
    """``{min, max, avg}`` of one metric over the samples that read it."""

    min: float
    max: float
    avg: float

    @classmethod
    def over(cls, values: list[float | None]) -> MetricStats | None:
        """The statistics of the values read; ``None`` when none was."""
        read = [value for value in values if value is not None]
        if not read:
            return None
        return cls(min=min(read), max=max(read), avg=round(sum(read) / len(read), 2))


@dataclass(frozen=True)
class GpuMonitorBlock:
    """One card's measured-phase telemetry.

    The field names are the ``gpu_monitor`` keys a benchmark report already carries, so a reader of those reports
    reads this block without change.
    """

    gpu: int
    sample_count: int
    power_watts: MetricStats | None
    temperature_c: MetricStats | None
    gpu_clock_mhz: MetricStats | None
    mem_clock_mhz: MetricStats | None
    gpu_util_pct: MetricStats | None
    vram_pct: MetricStats | None

    @classmethod
    def over(cls, gpu: int, samples: list[GpuSample]) -> GpuMonitorBlock:
        """The block for ``gpu`` over its samples."""
        return cls(
            gpu=gpu,
            sample_count=len(samples),
            power_watts=MetricStats.over([s.power_w for s in samples]),
            temperature_c=MetricStats.over([s.temperature_c for s in samples]),
            gpu_clock_mhz=MetricStats.over([s.gpu_clock_mhz for s in samples]),
            mem_clock_mhz=MetricStats.over([s.mem_clock_mhz for s in samples]),
            gpu_util_pct=MetricStats.over([s.gpu_util_pct for s in samples]),
            vram_pct=MetricStats.over([s.vram_pct for s in samples]),
        )

    def as_dict(self) -> dict[str, Any]:
        """The block as JSON-ready data."""
        return asdict(self)


def parse_power_sample(payload: Any) -> dict[int, tuple[float | None, float | None]]:
    """``{gpu: (socket_power_w, vram_used_fraction)}`` from ``amd-smi metric --power --mem-usage --json``."""
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    out: dict[int, tuple[float | None, float | None]] = {}
    for row in rows or []:
        if not isinstance(row, dict) or not isinstance(row.get("gpu"), int):
            continue
        power = row.get("power") if isinstance(row.get("power"), dict) else {}
        watts = metric_value(power.get("socket_power"))
        mem = row.get("mem_usage") if isinstance(row.get("mem_usage"), dict) else {}
        used, total = metric_value(mem.get("used_vram")), metric_value(mem.get("total_vram"))
        fraction = used / total if used is not None and total else None
        # No card draws 0 W: a non-positive socket power is a sensor that failed to read, the same as N/A.
        out[row["gpu"]] = (watts if watts is not None and watts > 0 else None, fraction)
    return out


def parse_clock_sample(payload: Any) -> dict[int, tuple[float | None, frozenset[float]]]:
    """``{gpu: (mean current sclk over the busy XCDs, sclk max limits)}`` from ``amd-smi metric --clock --json``.

    An XCD in deep sleep reports a clock below its own ``min_clk`` (tens of MHz on MI355X); that is idle time, not the
    clock the card runs at under load, so it is left out, and a card with every XCD asleep has no current clock.
    The limits are :func:`gfx_clock_limits`; collected over a round, more than one means the limit changed during it.
    """
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    out: dict[int, tuple[float | None, frozenset[float]]] = {}
    for row in rows or []:
        if not isinstance(row, dict) or not isinstance(row.get("gpu"), int) or not isinstance(row.get("clock"), dict):
            continue
        blocks = [
            block for name, block in row["clock"].items() if str(name).startswith("gfx_") and isinstance(block, dict)
        ]
        current = [
            clk
            for block in blocks
            if (clk := metric_value(block.get("clk"))) is not None
            and clk >= (metric_value(block.get("min_clk")) or 0.0)
        ]
        out[row["gpu"]] = (sum(current) / len(current) if current else None, gfx_clock_limits(row["clock"]))
    return out


def parse_throttle_sample(payload: Any) -> dict[int, tuple[float, float]]:
    """``{gpu: (accumulation_counter, ppt_accumulated)}`` from ``amd-smi metric --violation --json``.

    Both count the same firmware ticks, so the share of ticks in a window spent power (PPT) limited is the ratio of
    their deltas. A card without both counters is left out.
    """
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    out: dict[int, tuple[float, float]] = {}
    for row in rows or []:
        if not isinstance(row, dict) or not isinstance(row.get("gpu"), int):
            continue
        throttle = row.get("throttle")
        block = throttle if isinstance(throttle, dict) else {}
        ticks, ppt = metric_value(block.get("accumulation_counter")), metric_value(block.get("ppt_accumulated"))
        if ticks is not None and ppt is not None:
            out[row["gpu"]] = (ticks, ppt)
    return out


def parse_energy_sample(payload: Any) -> dict[int, float]:
    """``{gpu: joules}`` from ``amd-smi metric --energy --json``: each card's energy counter, which only grows while the
    driver stays loaded.

    amd-smi prints the counter as ``ENERGY: TOTAL_ENERGY_CONSUMPTION: <n> J``, read by :func:`energy_joules`. A card
    that reports it N/A is left out.

    Raises:
        ValueError: cards were reported and not one carried a counter. The message quotes the first card's energy
            block, so an amd-smi whose output changed shape says how, instead of reading as cards with nothing to count.
    """
    rows = gpu_rows(payload)
    out: dict[int, float] = {}
    for row in rows:
        block = row.get("energy")
        joules = energy_joules(block.get("total_energy_consumption")) if isinstance(block, dict) else None
        if joules is not None:
            out[row["gpu"]] = joules
    if rows and not out:
        raise ValueError(
            f"amd-smi metric --energy read no counter: card {rows[0]['gpu']} energy={rows[0].get('energy')!r}"
        )
    return out


def _read_node_power() -> tuple[float, float] | None:
    """``(node power W, enforced node limit W)`` while NPM is enabled; ``None`` on a node without it, or with it off.

    Read from sysfs rather than ``amd-smi``, which reports the limit but not the node's power.

    Raises:
        OSError, ValueError: the node's files exist but cannot be read, or report no limit.
    """
    for board in sorted(glob.glob(_BOARD_GLOB)):
        try:
            status = _read_board(board, "npm_status").strip()
        except FileNotFoundError:
            continue
        if status != NPM_ENABLED:
            # Without an XGMI hive every card publishes its own copy; one reading disabled says nothing of the others.
            continue
        limit = float(_read_board(board, "cur_node_power_limit"))
        if limit <= 0:
            raise ValueError(f"{board}: NPM is enabled with a node power limit of {limit:g} W")
        return float(_read_board(board, "node_power")), limit
    return None


def _read_board(board: str, name: str) -> str:
    return Path(board, name).read_bytes().decode()


def _node_fields(samples: list[tuple[float, float]]) -> dict[str, Any] | None:
    """The node's measured-phase power under NPM and the share of samples held at its limit; ``None`` with none."""
    if not samples:
        return None
    powers = [power for power, _ in samples]
    limits = {limit for _, limit in samples}
    held = sum(1 for power, limit in samples if power >= limit * _NODE_AT_LIMIT_FRACTION)
    return {
        "limit_w": limits.pop() if len(limits) == 1 else None,
        "avg_power_w": round(sum(powers) / len(powers), 2),
        "max_power_w": round(max(powers), 2),
        "samples": len(samples),
        "limited_fraction": round(held / len(samples), _FRACTION_DIGITS),
    }


def _query_amd_smi(run: Callable[..., subprocess.CompletedProcess], *metrics: str) -> Any:
    done = run(
        ["amd-smi", "metric", *metrics, "--json"],  # nosec B603 B607
        capture_output=True,
        text=True,
        timeout=_QUERY_TIMEOUT_SEC,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"amd-smi exited {done.returncode}: {(done.stderr or '').strip()[:200]}")
    return json.loads(done.stdout)


def _clock_fields(
    clocks: list[float] | None, limits: set[float] | None, delta: tuple[float, float] | None
) -> dict[str, Any]:
    """A card's measured-phase sclk and power-throttle share; ``None`` for whatever was not read."""
    return {
        "sclk_mean_mhz": round(sum(clocks) / len(clocks), 1) if clocks else None,
        "sclk_max_mhz": single_clock_limit(limits or ()),
        "power_throttle_fraction": round(delta[1] / delta[0], _FRACTION_DIGITS) if delta and delta[0] > 0 else None,
    }


def _power_fields(watts: list[float], peak_vram: float) -> dict[str, Any]:
    """A card's measured-phase power."""
    return {
        "avg_power_w": round(sum(watts) / len(watts), 2),
        "max_power_w": round(max(watts), 2),
        "samples": len(watts),
        "peak_vram_fraction": round(peak_vram, _FRACTION_DIGITS),
    }


def _energy_fields(joules: float, seconds: float, sampled_w: float | None) -> dict[str, Any]:
    """A card's counted energy over the measured windows, its mean power, and that mean over the sampled mean."""
    watts = joules / seconds
    return {
        "energy_j": round(joules, 1),
        "avg_power_w": round(watts, 2),
        "vs_sampled": round(watts / sampled_w, _FRACTION_DIGITS) if sampled_w else None,
    }


def _mhz(value: float | None) -> str:
    return "unread" if value is None else f"{value} MHz"


def _power_throttled(per_gpu: dict[str, dict[str, Any]], serving: list[int]) -> list[int]:
    """The serving cards held back by their power limit for at least the reporting share of the measured phase."""
    return sorted(
        gpu
        for gpu in serving
        if (per_gpu[str(gpu)].get("power_throttle_fraction") or 0.0) >= _POWER_THROTTLE_REPORT_FRACTION
    )


def _first_reading(block: Any, names: tuple[str, ...]) -> float | None:
    readings = block if isinstance(block, dict) else {}
    return next((value for name in names if (value := metric_value(readings.get(name))) is not None), None)


def parse_gpu_sample(payload: Any, unix: float) -> dict[int, GpuSample]:
    """``{gpu: GpuSample}`` from ``amd-smi metric --power --mem-usage --clock --temperature --usage --json``."""
    power = parse_power_sample(payload)
    clocks = parse_clock_sample(payload)
    out: dict[int, GpuSample] = {}
    for row in gpu_rows(payload):
        gpu = row["gpu"]
        watts, vram = power.get(gpu, (None, None))
        clock = row.get("clock") if isinstance(row.get("clock"), dict) else {}
        out[gpu] = GpuSample(
            unix=unix,
            gpu=gpu,
            power_w=watts,
            temperature_c=_first_reading(row.get("temperature"), _TEMPERATURE_SENSORS),
            gpu_clock_mhz=clocks.get(gpu, (None, frozenset()))[0],
            mem_clock_mhz=_first_reading(clock.get("mem_0"), ("clk",)),
            gpu_util_pct=_first_reading(row.get("usage"), ("gfx_activity",)),
            vram_pct=None if vram is None else round(vram * 100, 2),
        )
    return out


def _samples_csv(samples: list[GpuSample]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    columns = [field.name for field in fields(GpuSample)]
    writer.writerow(columns)
    for sample in samples:
        writer.writerow(["" if (value := getattr(sample, name)) is None else value for name in columns])
    return out.getvalue()


class GpuPowerRecorder:
    """Samples GPU telemetry on a background thread while the round is in its ``measured`` phase.

    Exposes the same ``note_phase`` / ``tick`` / ``close`` surface as the KV-metrics recorder so the watchdog loop
    drives both identically. Sampling runs on its own thread because an ``amd-smi`` call takes a few hundred ms and the
    watchdog loop must not stall on it; only :meth:`close` waits, for the sampler and the closing counter reads (about
    a second, a little more with energy on).
    """

    def __init__(
        self,
        *,
        output_path: str | None,
        gpus: set[int] | None = None,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        query: Callable[..., Any] | None = None,
        options: RecorderOptions | None = None,
    ) -> None:
        """Prepare a recorder without running anything.

        ``output_path`` receives the ``gpu_power.json`` v1 summary on :meth:`close`, or nothing when ``None``;
        ``options`` adds the per-sample CSV and the energy counters (:class:`RecorderOptions`).
        """
        options = options or RecorderOptions()
        self._output_path = output_path
        self._samples_path = options.samples_path
        self._rows: list[GpuSample] = []
        self._gpus = None if gpus is None else set(gpus)
        self._interval = max(MIN_INTERVAL_SEC, float(interval_sec))
        self._query = query or (lambda *metrics: _query_amd_smi(subprocess.run, *metrics))
        self._node: list[tuple[float, float]] = []
        self._node_errors = 0
        self._clocks: dict[int, list[float]] = {}
        self._clock_limits: dict[int, set[float]] = {}
        # Throttle counters at the open of the current measured window; per card ``[ticks, ppt, windows read]`` over
        # the windows it was read across, and how many windows closed. A card is published only when read across every
        # window: summing only the windows that were read would publish the share of a different phase.
        self._window_open: dict[int, tuple[float, float]] | None = None
        self._throttle_totals: dict[int, list[float]] = {}
        self._windows_closed = 0
        # The energy counters the same way: their reading at the open of the window, stamped with the monotonic middle
        # of the read, and per card ``[joules, seconds, windows read]``.
        self._energy = options.energy
        self._energy_open: tuple[float, dict[int, float]] | None = None
        self._energy_totals: dict[int, list[float]] = {}
        self._energy_errors = 0
        self._energy_last_error: str | None = None
        self._phase = GpuPhase.BOOT
        self._samples: list[dict[int, tuple[float | None, float | None]]] = []
        self._errors = 0
        self._last_error: str | None = None
        self._throttle_errors = 0
        self._throttle_last_error: str | None = None
        self._started_unix = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # Set when the round leaves the measured phase, so the sampler closes the window then and not up to an interval
        # later, when the counters would also have counted the start of what follows.
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def phase(self) -> GpuPhase:
        """Current collection phase."""
        return self._phase

    def note_phase(self, phase: str, mono: float) -> None:
        """Record a phase transition; sampling starts the first time the round enters ``measured``.

        ``phase`` is a :class:`GpuPhase` value; any other phase name the driving loop knows is ignored.
        """
        if phase not in _PHASE_VALUES or self._closed:
            return
        self._phase = GpuPhase(phase)
        if self._phase == GpuPhase.MEASURED and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="gpu-power-sampler", daemon=True)
            self._thread.start()
        elif self._phase != GpuPhase.MEASURED and self._thread is not None:
            self._wake.set()

    def tick(self, mono: float) -> None:
        """No-op: the sampler keeps its own cadence."""

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._phase == GpuPhase.MEASURED:
                # Sampled first: the counter reads take about a second and must not delay the first power sample.
                self._sample_once()
                self._sample_node()
                if self._window_open is None:
                    self._open_window()
            elif self._window_open is not None:
                self._close_window()
            self._wake.wait(self._interval)
            self._wake.clear()

    def _open_window(self) -> None:
        """Read the counters a measured window is measured from: energy first, the faster read, so its stamp sits
        nearer the window's edge."""
        self._energy_open = self._read_energy()
        self._window_open = self._read_throttle()

    def _read_throttle(self) -> dict[int, tuple[float, float]]:
        try:
            counters = parse_throttle_sample(self._query(*_THROTTLE_METRICS))
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            with self._lock:
                self._throttle_errors += 1
                self._throttle_last_error = str(exc)[:200]
            return {}
        return {gpu: row for gpu, row in counters.items() if self._gpus is None or gpu in self._gpus}

    def _read_energy(self) -> tuple[float, dict[int, float]] | None:
        """``(monotonic middle of the read, {gpu: joules})`` for the cards in scope, empty when the read failed;
        ``None`` with energy off."""
        if not self._energy:
            return None
        start = time.monotonic()
        try:
            counters = parse_energy_sample(self._query(*_ENERGY_METRICS))
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, TypeError) as exc:
            log.debug("gpu_power: amd-smi energy read failed", exc_info=True)
            with self._lock:
                self._energy_errors += 1
                self._energy_last_error = str(exc)[:200]
            counters = {}
        middle = (start + time.monotonic()) / 2
        return middle, {gpu: joules for gpu, joules in counters.items() if self._gpus is None or gpu in self._gpus}

    def _close_window(self) -> None:
        """Add the counters' movement over the measured window that just ended."""
        energy_opened, self._energy_open = self._energy_open, None
        energy_closed = self._read_energy() if energy_opened and energy_opened[1] else None
        opened, self._window_open = self._window_open or {}, None
        closed = self._read_throttle()
        with self._lock:
            self._windows_closed += 1
            for gpu in opened.keys() & closed.keys():
                (ticks_0, ppt_0), (ticks_1, ppt_1) = opened[gpu], closed[gpu]
                if ticks_1 > ticks_0 and ppt_1 >= ppt_0:
                    total = self._throttle_totals.setdefault(gpu, [0.0, 0.0, 0])
                    total[0] += ticks_1 - ticks_0
                    total[1] += ppt_1 - ppt_0
                    total[2] += 1
            if energy_opened and energy_closed:
                self._add_energy(energy_opened, energy_closed)

    def _add_energy(self, opened: tuple[float, dict[int, float]], closed: tuple[float, dict[int, float]]) -> None:
        """Add each card's counter movement over one window. A counter that went back was reset, by a driver reload or
        a GPU reset, so its card has no reading for that window. Called with the lock held."""
        seconds = closed[0] - opened[0]
        if seconds <= 0:
            return
        for gpu in opened[1].keys() & closed[1].keys():
            joules = closed[1][gpu] - opened[1][gpu]
            if joules >= 0:
                total = self._energy_totals.setdefault(gpu, [0.0, 0.0, 0])
                total[0] += joules
                total[1] += seconds
                total[2] += 1

    def _sample_once(self) -> None:
        try:
            payload = self._query(*_SAMPLE_METRICS)
            reading = parse_power_sample(payload)
            rows = parse_gpu_sample(payload, time.time())
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, TypeError) as exc:
            log.debug("gpu_power: amd-smi query failed", exc_info=True)
            with self._lock:
                self._errors += 1
                self._last_error = str(exc)[:200]
            return
        if self._gpus is not None:
            reading = {gpu: row for gpu, row in reading.items() if gpu in self._gpus}
        if reading and self._phase == GpuPhase.MEASURED:
            clocks = parse_clock_sample(payload)
            with self._lock:
                self._samples.append(reading)
                self._rows.extend(rows[gpu] for gpu in sorted(reading))
                for gpu, (current, limits) in clocks.items():
                    if gpu not in reading:
                        continue
                    if current is not None:
                        self._clocks.setdefault(gpu, []).append(current)
                    self._clock_limits.setdefault(gpu, set()).update(limits)

    def _sample_node(self) -> None:
        try:
            sample = _read_node_power()
        except (OSError, ValueError):
            log.debug("gpu_power: node power read failed", exc_info=True)
            with self._lock:
                self._node_errors += 1
            return
        if sample is not None and self._phase == GpuPhase.MEASURED:
            with self._lock:
                self._node.append(sample)

    def _node_summary(self) -> dict[str, Any]:
        with self._lock:
            return {"node": _node_fields(list(self._node)), "node_query_errors": self._node_errors}

    def _complete_throttle_totals(self) -> dict[int, tuple[float, float]]:
        """Each card's ``(ticks, power-limited ticks)`` summed over every measured window, for the cards read across
        all of them. Called with the lock held."""
        return {
            gpu: (ticks, ppt)
            for gpu, (ticks, ppt, windows) in self._throttle_totals.items()
            if windows == self._windows_closed
        }

    def _complete_energy_totals(self) -> dict[int, tuple[float, float]]:
        """Each card's ``(joules, seconds)`` summed over every measured window, for the cards read across all of them.
        Called with the lock held."""
        return {
            gpu: (joules, seconds)
            for gpu, (joules, seconds, windows) in self._energy_totals.items()
            if windows == self._windows_closed
        }

    def _energy_summary(
        self, per_gpu: dict[str, dict[str, Any]], serving: list[int], unread: list[int]
    ) -> dict[str, Any]:
        """The ``energy`` block: each card's counted energy over the measured windows, and the serving cards' total.

        The total fails closed like the sampled mean: a serving card without a reading in every window, or a card that
        may be serving and could not be read, leaves it ``None`` rather than a sum over the cards that happened to read.
        """
        with self._lock:
            totals = self._complete_energy_totals()
            windows, errors, last_error = self._windows_closed, self._energy_errors, self._energy_last_error
        cards = {
            str(gpu): _energy_fields(joules, seconds, (per_gpu.get(str(gpu)) or {}).get("avg_power_w"))
            for gpu, (joules, seconds) in sorted(totals.items())
        }
        counted = bool(serving) and not unread and all(gpu in totals for gpu in serving)
        joules = round(sum(totals[gpu][0] for gpu in serving), 1) if counted else None
        # One amd-smi call reads every card, so every card read across every window shares the same seconds.
        seconds = totals[serving[0]][1] if counted else None
        return {
            "enabled": self._energy,
            "source": "amd-smi metric --energy",
            "boundary": "gpu",
            "windows": windows,
            "seconds": None if seconds is None else round(seconds, 3),
            "total_energy_j": joules,
            "total_power_w": round(joules / seconds, 2) if joules is not None and seconds else None,
            "per_gpu": cards,
            "query_errors": errors,
            "last_error": last_error,
        }

    def summary(self, *, aborted: bool = False) -> dict[str, Any]:
        """The round's measured-phase power, averaged over the cards that served it."""
        with self._lock:
            samples = list(self._samples)
            errors, last_error = self._errors, self._last_error
            throttle_errors, throttle_last_error = self._throttle_errors, self._throttle_last_error
            clocks = {gpu: list(values) for gpu, values in self._clocks.items()}
            limits = {gpu: set(values) for gpu, values in self._clock_limits.items()}
            deltas = self._complete_throttle_totals()
        peak_vram: dict[int, float] = {}
        watts: dict[int, list[float]] = {}
        for reading in samples:
            for gpu, (power, fraction) in reading.items():
                if fraction is not None:
                    peak_vram[gpu] = max(peak_vram.get(gpu, 0.0), fraction)
                if power is not None:
                    watts.setdefault(gpu, []).append(power)
        loaded = {gpu for gpu, frac in peak_vram.items() if frac >= _SERVING_VRAM_FRACTION}
        serving = sorted(loaded.intersection(watts))
        # A card that may be serving but cannot be read would drop out of the total and escape its own limit, so it
        # leaves the whole round unmeasured: a card holding a model with no power reading, or a card in scope whose
        # VRAM was never read (absent from amd-smi, or reported N/A), so it cannot be ruled out as serving.
        in_scope = self._gpus if self._gpus is not None else {gpu for reading in samples for gpu in reading}
        unread = sorted(in_scope.difference(peak_vram) | loaded.difference(watts)) if samples else []
        per_gpu = {
            str(gpu): {
                **_power_fields(watts[gpu], peak_vram.get(gpu, 0.0)),
                **_clock_fields(clocks.get(gpu), limits.get(gpu), deltas.get(gpu)),
            }
            for gpu in sorted(watts)
        }
        avg = (
            round(sum(per_gpu[str(g)]["avg_power_w"] for g in serving) / len(serving), 2)
            if serving and not unread
            else None
        )
        peak = max((per_gpu[str(g)]["max_power_w"] for g in serving), default=None)
        return {
            "schema_version": 1,
            "source": "amd-smi",
            "phase": "measured",
            "started_unix": round(self._started_unix, 3),
            "interval_sec": self._interval,
            "samples": len(samples),
            "serving_gpus": serving,
            "unread_gpus": unread,
            "avg_power_w": avg,
            "max_power_w": peak,
            "per_gpu": per_gpu,
            "power_throttled_gpus": _power_throttled(per_gpu, serving),
            "query_errors": errors,
            "last_error": last_error,
            "throttle_query_errors": throttle_errors,
            "throttle_last_error": throttle_last_error,
            **self._node_summary(),
            "aborted": bool(aborted),
            # Added after the v1 keys, which keep their order: a reader of v1 reads this summary unchanged.
            "energy": self._energy_summary(per_gpu, serving, unread),
        }

    def close(self, *, aborted: bool = False) -> dict[str, Any]:
        """Stop sampling and write the artifact. Idempotent."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=_QUERY_TIMEOUT_SEC + self._interval)
            if self._thread.is_alive():
                # Still blocked in amd-smi, possibly opening or closing a window: that window was not read across.
                with self._lock:
                    self._windows_closed += 1
        if self._window_open is not None and not self._closed:
            self._close_window()
        payload = self.summary(aborted=aborted)
        if payload["power_throttled_gpus"] and not self._closed:
            log.warning(
                "gpu_power: GPU(s) %s were power throttled for %s of the measured phase, at a mean clock of %s",
                ", ".join(map(str, payload["power_throttled_gpus"])),
                ", ".join(
                    f"{payload['per_gpu'][str(gpu)]['power_throttle_fraction']:.0%}"
                    for gpu in payload["power_throttled_gpus"]
                ),
                ", ".join(
                    _mhz(payload["per_gpu"][str(gpu)]["sclk_mean_mhz"]) for gpu in payload["power_throttled_gpus"]
                ),
            )
        node = payload["node"]
        if node and node["limited_fraction"] >= _POWER_THROTTLE_REPORT_FRACTION and not self._closed:
            log.warning(
                "gpu_power: the node was held at its NPM power limit (%s W) for %.0f%% of the measured phase",
                node["limit_w"] if node["limit_w"] is not None else "changing",
                node["limited_fraction"] * 100,
            )
        if not self._closed:
            self._closed = True
            self._write_artifacts(payload)
        return payload

    def gpu_monitor(self) -> tuple[GpuMonitorBlock, ...]:
        """One block per serving card over the measured phase; empty when no card was found serving."""
        serving = self.summary()["serving_gpus"]
        with self._lock:
            rows = list(self._rows)
        return tuple(GpuMonitorBlock.over(gpu, [row for row in rows if row.gpu == gpu]) for gpu in serving)

    def _write_artifacts(self, payload: dict[str, Any]) -> None:
        if self._output_path:
            try:
                atomic_write_json(Path(self._output_path), payload)
            except (OSError, TypeError, ValueError):
                log.warning("gpu_power: could not write %s", self._output_path, exc_info=True)
        if self._samples_path:
            with self._lock:
                rows = list(self._rows)
            try:
                atomic_write_text(Path(self._samples_path), _samples_csv(rows))
            except OSError:
                log.warning("gpu_power: could not write %s", self._samples_path, exc_info=True)
