# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-GPU power over the measured phase of one benchmark round, sampled by Hyperloom itself.

Anything power-aware needs a number that describes the load being measured. The benchmark report's ``gpu_monitor``
block is not one: Magpie samples a single card (the first visible one), and its window opens before the server launches and
closes after the client exits, so model load, graph capture and idle tail all average in. On a TP4 round that
reading ignores three of the four serving cards, and a candidate that only boots slower reads as drawing less power.

This recorder hangs off the same watchdog loop that drives the KV-metrics phase machine and samples every card the
round could use with read-only ``amd-smi metric --power --mem-usage --clock``, keeping only samples taken in the
``measured`` phase. It also reads the cards' throttle accumulators (``amd-smi metric --violation``) as each measured
window opens and closes, so a round under a power cap reports how much of it the cards spent held back by that cap,
and at what graphics clock they actually ran. The serving cards are the ones holding a serving-sized share of VRAM
during that phase, so a TP4 round on an unpinned eight-card host is averaged over the four cards it runs on.

Nothing here raises: telemetry that fails must cost the round nothing, and a round with no reading reports ``None``
rather than zero.
"""

from __future__ import annotations

import json
import logging
import subprocess  # nosec B404 - fixed read-only amd-smi invocation.
import threading
import time
from pathlib import Path
from typing import Any, Callable

from hyperloom.common.env import EnvValueError, env_float
from hyperloom.common.gpu_power_settings import gpu_power_sampling_unavailable

log = logging.getLogger(__name__)

__all__ = [
    "GPU_POWER_ARTIFACT_NAME",
    "GpuPowerRecorder",
    "ServerLogPhaseDriver",
    "build_gpu_power_recorder",
    "parse_clock_sample",
    "parse_power_sample",
    "parse_throttle_sample",
    "read_measured_gpu_power",
    "read_measured_gpu_power_by_gpu",
]

#: Artifact written beside the round's ``server.log``.
GPU_POWER_ARTIFACT_NAME = "gpu_power.json"

_INTERVAL_ENV = "HYPERLOOM_GPU_POWER_INTERVAL_S"
_DEFAULT_INTERVAL_SEC = 2.0
_MIN_INTERVAL_SEC = 0.5
_QUERY_TIMEOUT_SEC = 10.0

#: A card counts as serving when it held at least this share of its VRAM during the measured phase. Serving engines
#: reserve most of HBM up front (vLLM's default ``--gpu-memory-utilization`` is 0.9), while an idle card holds a few
#: hundred MB of driver state, so the split is wide and the threshold does not need to be exact.
_SERVING_VRAM_FRACTION = 0.10

#: A serving card held back by its power limit for at least this share of the measured phase is reported as power
#: throttled: its throughput was set by the power cap, not by the clock limit.
_POWER_THROTTLE_REPORT_FRACTION = 0.10

#: The ``amd-smi metric`` fields each power sample reads, and those read at each measured window's open and close.
_SAMPLE_METRICS = ("--power", "--mem-usage", "--clock")
_THROTTLE_METRICS = ("--violation",)

#: How far a recorder's start may precede the subprocess it is paired with and still be that round's.
_START_SLACK_SEC = 5.0

_BOOT = "boot"
_PHASES = (_BOOT, "warmup", "measured", "finished", "eval")


def _value(block: Any) -> float | None:
    if isinstance(block, dict):
        block = block.get("value")
    if isinstance(block, bool) or not isinstance(block, (int, float)):
        return None
    return float(block)


def parse_power_sample(payload: Any) -> dict[int, tuple[float | None, float | None]]:
    """``{gpu: (socket_power_w, vram_used_fraction)}`` from ``amd-smi metric --power --mem-usage --json``."""
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    out: dict[int, tuple[float | None, float | None]] = {}
    for row in rows or []:
        if not isinstance(row, dict) or not isinstance(row.get("gpu"), int):
            continue
        power = row.get("power") if isinstance(row.get("power"), dict) else {}
        watts = _value(power.get("socket_power"))
        mem = row.get("mem_usage") if isinstance(row.get("mem_usage"), dict) else {}
        used, total = _value(mem.get("used_vram")), _value(mem.get("total_vram"))
        fraction = used / total if used is not None and total else None
        # No card draws 0 W: a non-positive socket power is a sensor that failed to read, the same as N/A.
        out[row["gpu"]] = (watts if watts is not None and watts > 0 else None, fraction)
    return out


def parse_clock_sample(payload: Any) -> dict[int, tuple[float | None, frozenset[float]]]:
    """``{gpu: (mean current sclk over the busy XCDs, sclk max limits)}`` from ``amd-smi metric --clock --json``.

    An XCD in deep sleep reports a clock below its own ``min_clk`` (tens of MHz on MI355X); that is idle time, not the
    clock the card runs at under load, so it is left out, and a card with every XCD asleep has no current clock.
    amd-smi prints all eight ``gfx_<n>`` slots and fills only those with an XCD behind them (a partitioned card has
    fewer), copying the card's one limit into each, so the limits are those the filled slots report: empty when none is
    filled. Collected over a round, more than one means the limit changed during it.
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
            if (clk := _value(block.get("clk"))) is not None and clk >= (_value(block.get("min_clk")) or 0.0)
        ]
        limits = frozenset(
            limit
            for block in blocks
            if _value(block.get("clk")) is not None and (limit := _value(block.get("max_clk"))) is not None
        )
        out[row["gpu"]] = (sum(current) / len(current) if current else None, limits)
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
        ticks, ppt = _value(block.get("accumulation_counter")), _value(block.get("ppt_accumulated"))
        if ticks is not None and ppt is not None:
            out[row["gpu"]] = (ticks, ppt)
    return out


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
    limit = next(iter(limits)) if limits and len(limits) == 1 else None
    return {
        "sclk_mean_mhz": round(sum(clocks) / len(clocks), 1) if clocks else None,
        "sclk_max_mhz": limit,
        "power_throttle_fraction": round(delta[1] / delta[0], 4) if delta and delta[0] > 0 else None,
    }


def _power_fields(watts: list[float], peak_vram: float) -> dict[str, Any]:
    """A card's measured-phase power."""
    return {
        "avg_power_w": round(sum(watts) / len(watts), 2),
        "max_power_w": round(max(watts), 2),
        "samples": len(watts),
        "peak_vram_fraction": round(peak_vram, 4),
    }


def _mhz(value: float | None) -> str:
    return "unread" if value is None else f"{value} MHz"


def _serving_gpus(peak_vram: dict[int, float], watts: dict[int, list[float]]) -> list[int]:
    """The cards whose peak VRAM share says they served the round and whose power was read."""
    return sorted(gpu for gpu, frac in peak_vram.items() if frac >= _SERVING_VRAM_FRACTION and gpu in watts)


def _unread_gpus(in_scope: set[int], peak_vram: dict[int, float], watts: dict[int, list[float]]) -> list[int]:
    """The cards in scope that cannot be ruled out as serving because a reading is missing.

    Such a card would drop out of the total and escape its own limit, so it leaves the whole round
    unmeasured rather than under-counted.
    """
    return sorted(
        gpu
        for gpu in in_scope
        if gpu not in peak_vram or (peak_vram[gpu] >= _SERVING_VRAM_FRACTION and gpu not in watts)
    )


def _power_throttled(per_gpu: dict[str, dict[str, Any]], serving: list[int]) -> list[int]:
    """The serving cards held back by their power limit for at least the reporting share of the measured phase."""
    return sorted(
        gpu
        for gpu in serving
        if (per_gpu[str(gpu)].get("power_throttle_fraction") or 0.0) >= _POWER_THROTTLE_REPORT_FRACTION
    )


class GpuPowerRecorder:
    """Samples GPU power on a background thread while the round is in its ``measured`` phase.

    Exposes the same ``note_phase`` / ``tick`` / ``close`` surface as the KV-metrics recorder so the watchdog loop
    drives both identically. Sampling runs on its own thread because an ``amd-smi`` call takes a few hundred ms and the
    watchdog loop must not stall on it; only :meth:`close` waits, for the sampler and the closing throttle-counter read
    (about a second).
    """

    def __init__(
        self,
        *,
        output_path: str | None,
        gpus: set[int] | None = None,
        interval_sec: float = _DEFAULT_INTERVAL_SEC,
        query: Callable[..., Any] | None = None,
    ) -> None:
        """Prepare a recorder without running anything."""
        self._output_path = output_path
        self._gpus = None if gpus is None else set(gpus)
        self._interval = max(_MIN_INTERVAL_SEC, float(interval_sec))
        self._query = query or (lambda *metrics: _query_amd_smi(subprocess.run, *metrics))
        self._clocks: dict[int, list[float]] = {}
        self._clock_limits: dict[int, set[float]] = {}
        # Throttle counters at the open of the current measured window; per card ``[ticks, ppt, windows read]`` over
        # the windows it was read across, and how many windows closed. A card is published only when read across every
        # window: summing only the windows that were read would publish the share of a different phase.
        self._window_open: dict[int, tuple[float, float]] | None = None
        self._throttle_totals: dict[int, list[float]] = {}
        self._throttle_windows = 0
        self._phase = _BOOT
        self._samples: list[dict[int, tuple[float | None, float | None]]] = []
        self._errors = 0
        self._last_error: str | None = None
        self._throttle_errors = 0
        self._throttle_last_error: str | None = None
        self._started_unix = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def phase(self) -> str:
        """Current collection phase."""
        return self._phase

    def note_phase(self, phase: str, mono: float) -> None:
        """Record a phase transition; sampling starts the first time the round enters ``measured``."""
        if phase not in _PHASES or self._closed:
            return
        self._phase = phase
        if phase == "measured" and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="gpu-power-sampler", daemon=True)
            self._thread.start()

    def tick(self, mono: float) -> None:
        """No-op: the sampler keeps its own cadence."""

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._phase == "measured":
                # Sampled first: the counter read takes about a second and must not delay the first power sample.
                self._sample_once()
                if self._window_open is None:
                    self._window_open = self._read_throttle()
            elif self._window_open is not None:
                self._close_window()
            self._stop.wait(self._interval)

    def _read_throttle(self) -> dict[int, tuple[float, float]]:
        try:
            counters = parse_throttle_sample(self._query(*_THROTTLE_METRICS))
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            with self._lock:
                self._throttle_errors += 1
                self._throttle_last_error = str(exc)[:200]
            return {}
        return {gpu: row for gpu, row in counters.items() if self._gpus is None or gpu in self._gpus}

    def _close_window(self) -> None:
        """Add the throttle counters' movement over the measured window that just ended."""
        opened, self._window_open = self._window_open or {}, None
        closed = self._read_throttle()
        with self._lock:
            self._throttle_windows += 1
            for gpu in opened.keys() & closed.keys():
                (ticks_0, ppt_0), (ticks_1, ppt_1) = opened[gpu], closed[gpu]
                if ticks_1 > ticks_0 and ppt_1 >= ppt_0:
                    total = self._throttle_totals.setdefault(gpu, [0.0, 0.0, 0])
                    total[0] += ticks_1 - ticks_0
                    total[1] += ppt_1 - ppt_0
                    total[2] += 1

    def _sample_once(self) -> None:
        try:
            payload = self._query(*_SAMPLE_METRICS)
            reading = parse_power_sample(payload)
        except Exception as exc:
            log.debug("gpu_power: amd-smi query failed", exc_info=True)
            with self._lock:
                self._errors += 1
                self._last_error = str(exc)[:200]
            return
        if self._gpus is not None:
            reading = {gpu: row for gpu, row in reading.items() if gpu in self._gpus}
        if reading and self._phase == "measured":
            clocks = parse_clock_sample(payload)
            with self._lock:
                self._samples.append(reading)
                for gpu, (current, limits) in clocks.items():
                    if gpu not in reading:
                        continue
                    if current is not None:
                        self._clocks.setdefault(gpu, []).append(current)
                    self._clock_limits.setdefault(gpu, set()).update(limits)

    def _complete_throttle_totals(self) -> dict[int, tuple[float, float]]:
        """Each card's ``(ticks, power-limited ticks)`` summed over every measured window, for the cards read across
        all of them. Called with the lock held."""
        return {
            gpu: (ticks, ppt)
            for gpu, (ticks, ppt, windows) in self._throttle_totals.items()
            if windows == self._throttle_windows
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
        serving = _serving_gpus(peak_vram, watts)
        in_scope = self._gpus if self._gpus is not None else {gpu for reading in samples for gpu in reading}
        unread = _unread_gpus(in_scope, peak_vram, watts) if samples else []
        per_gpu = {
            str(gpu): {
                **_power_fields(watts[gpu], peak_vram.get(gpu, 0.0)),
                **_clock_fields(clocks.get(gpu), limits.get(gpu), deltas.get(gpu)),
            }
            for gpu in sorted(watts)
        }
        throttled = _power_throttled(per_gpu, serving)
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
            "power_throttled_gpus": throttled,
            "query_errors": errors,
            "last_error": last_error,
            "throttle_query_errors": throttle_errors,
            "throttle_last_error": throttle_last_error,
            "aborted": bool(aborted),
        }

    def close(self, *, aborted: bool = False) -> dict[str, Any]:
        """Stop sampling and write the artifact. Idempotent."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_QUERY_TIMEOUT_SEC + self._interval)
            if self._thread.is_alive():
                # Still blocked in amd-smi, possibly opening or closing a window: that window was not read across.
                with self._lock:
                    self._throttle_windows += 1
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
        if self._closed or not self._output_path:
            self._closed = True
            return payload
        self._closed = True
        try:
            from hyperloom.common.io import atomic_write_json

            atomic_write_json(Path(self._output_path), payload)
        except (OSError, TypeError, ValueError):
            log.warning("gpu_power: could not write %s", self._output_path, exc_info=True)
        return payload


class ServerLogPhaseDriver:
    """Drives a recorder's phase from server logs that appear while one subprocess runs.

    For a child the watchdog loop does not own, such as a GEAK replay that boots its own server once per replica. Each
    newly seen log is a fresh boot, so the recorder returns to ``boot`` until that server reports ready; the markers are
    the watchdog's own, so a replay is cut into phases exactly like a native round.
    """

    def __init__(
        self,
        recorder: Any,
        find_logs: Callable[[], list[str]],
        *,
        cursor: Any,
        interval_sec: float = _DEFAULT_INTERVAL_SEC,
    ) -> None:
        """Prepare a driver; nothing is read until :meth:`start`.

        ``cursor`` is the watchdog's own log cursor, passed in rather than imported: the watchdog module builds this
        module's recorder, so importing it back from here would close a cycle.
        """
        self._recorder = recorder
        self._find_logs = find_logs
        self._cursor = cursor
        self._interval = max(_MIN_INTERVAL_SEC, float(interval_sec))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._read_failed = False

    def poll(self) -> None:
        """Read what every log appended since the last poll and move the recorder's phase on."""
        for path in self._find_logs():
            if self._cursor.open_log(path):
                self._recorder.note_phase(_BOOT, time.monotonic())
            scan = self._cursor.advance(path)
            now = time.monotonic()
            if scan.saw_ready:
                self._recorder.note_phase("measured", now)
            if scan.saw_warmup_begin:
                self._recorder.note_phase("warmup", now)
            if scan.saw_measured_begin:
                self._recorder.note_phase("measured", now)
            if scan.saw_eval_start:
                self._recorder.note_phase("eval", now)

    def _poll_logs(self) -> None:
        """:meth:`poll`, with a replay log that cannot be listed or read reported once at warning, then at debug."""
        try:
            self.poll()
        except OSError as exc:
            level = logging.DEBUG if self._read_failed else logging.WARNING
            self._read_failed = True
            log.log(level, "gpu_power: cannot read the replay's server logs, so its phase may not advance: %s", exc)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_logs()
            self._stop.wait(self._interval)

    def start(self) -> None:
        """Begin polling on a background thread."""
        self._thread = threading.Thread(target=self._loop, name="gpu-power-phase", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop polling, after one last read so a phase that began just before exit is not lost."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + _QUERY_TIMEOUT_SEC)
        self._poll_logs()


def build_gpu_power_recorder(server_log_path: str | None, env: dict[str, str] | None) -> GpuPowerRecorder | None:
    """The round's power recorder, or ``None`` when sampling is off or there is no ``amd-smi`` to sample with."""
    if not server_log_path or gpu_power_sampling_unavailable():
        return None
    try:
        from hyperloom.common.gpu_power_settings import visible_gpu_indices

        return GpuPowerRecorder(
            output_path=str(Path(server_log_path).parent / GPU_POWER_ARTIFACT_NAME),
            gpus=visible_gpu_indices(env or {}),
            interval_sec=env_float(_INTERVAL_ENV, _DEFAULT_INTERVAL_SEC),
        )
    except EnvValueError:
        log.warning("gpu_power: recorder could not be built; this round samples nothing", exc_info=True)
        return None


def _read_round_artifact(workspace: Path | None, subprocess_started_unix: float | None) -> dict[str, Any] | None:
    """This round's ``gpu_power.json``, beside the report or one level up; ``None`` when no recorder ran for it."""
    if workspace is None:
        return None
    for directory in (Path(workspace), Path(workspace).parent):
        path = directory / GPU_POWER_ARTIFACT_NAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        started = _value(payload.get("started_unix"))
        if subprocess_started_unix is not None and (
            started is None or started < float(subprocess_started_unix) - _START_SLACK_SEC
        ):
            continue
        return payload
    return None


def read_measured_gpu_power(
    workspace: Path | None, *, subprocess_started_unix: float | None = None
) -> tuple[bool, float | None]:
    """``(found, avg_power_w)`` from the round's ``gpu_power.json``.

    ``found`` is ``True`` whenever this round's artifact exists, even when it holds no reading, so a caller can tell
    "the recorder ran and measured nothing" (fail closed) from "no recorder ran" (fall back to the report).
    """
    payload = _read_round_artifact(workspace, subprocess_started_unix)
    return (False, None) if payload is None else (True, _value(payload.get("avg_power_w")))


def read_measured_gpu_power_by_gpu(
    workspace: Path | None, *, subprocess_started_unix: float | None = None
) -> tuple[bool, dict[str, float] | None]:
    """``(found, {gpu_id: avg_power_w})`` for the round's serving cards; ``None`` when no serving card had a reading,
    or when a card that may be serving could not be read (``unread_gpus``), since the budget cannot cover it.

    Keyed by the physical ``amd-smi`` index as a string, the same ids ``--max-per-gpu-power-w`` names.
    """
    payload = _read_round_artifact(workspace, subprocess_started_unix)
    if payload is None:
        return False, None
    if payload.get("unread_gpus"):
        return True, None
    per_gpu = payload.get("per_gpu") if isinstance(payload.get("per_gpu"), dict) else {}
    readings = {
        str(gpu): watts
        for gpu in payload.get("serving_gpus") or []
        if (watts := _value((per_gpu.get(str(gpu)) or {}).get("avg_power_w"))) is not None
    }
    return True, (readings or None)
