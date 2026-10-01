# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-GPU power over the measured phase of one benchmark round, sampled by Hyperloom itself.

``--max-power-w`` needs a number that describes the load being graded. The benchmark report's ``gpu_monitor`` block is
not one: Magpie samples a single card (the first visible one), and its window opens before the server launches and
closes after the client exits, so model load, graph capture and idle tail all average in. On a TP4 round that
reading ignores three of the four serving cards, and a candidate that only boots slower reads as drawing less power.

This recorder hangs off the same watchdog loop that drives the KV-metrics phase machine and samples every card the
round could use with read-only ``amd-smi metric --power --mem-usage``, keeping only samples taken in the ``measured``
phase. The serving cards are the ones holding a serving-sized share of VRAM during that phase, so a TP4 round on an
unpinned eight-card host is averaged over the four cards it runs on.

Nothing here raises: telemetry that fails must cost the round nothing, and a round with no reading reports ``None``
rather than zero.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess  # nosec B404 - fixed read-only amd-smi invocation.
import threading
import time
from pathlib import Path
from typing import Any, Callable

from hyperloom.common.env import env_flag, env_float

log = logging.getLogger(__name__)

__all__ = [
    "GPU_POWER_ARTIFACT_NAME",
    "GPU_POWER_ENV",
    "GpuPowerRecorder",
    "ServerLogPhaseDriver",
    "build_gpu_power_recorder",
    "parse_power_sample",
    "read_measured_gpu_power",
]

#: Artifact written beside the round's ``server.log``.
GPU_POWER_ARTIFACT_NAME = "gpu_power.json"

#: Escape hatch for a host where running ``amd-smi`` every couple of seconds is not acceptable.
GPU_POWER_ENV = "HYPERLOOM_GPU_POWER_SAMPLING"
_INTERVAL_ENV = "HYPERLOOM_GPU_POWER_INTERVAL_S"
_DEFAULT_INTERVAL_SEC = 2.0
_MIN_INTERVAL_SEC = 0.5
_QUERY_TIMEOUT_SEC = 10.0

#: A card counts as serving when it held at least this share of its VRAM during the measured phase. Serving engines
#: reserve most of HBM up front (vLLM's default ``--gpu-memory-utilization`` is 0.9), while an idle card holds a few
#: hundred MB of driver state, so the split is wide and the threshold does not need to be exact.
_SERVING_VRAM_FRACTION = 0.10

#: How far a recorder's start may precede the subprocess it is paired with and still be that round's.
_START_SLACK_SEC = 5.0

_PHASES = ("boot", "warmup", "measured", "eval")


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
        out[row["gpu"]] = (watts, fraction)
    return out


def _query_amd_smi(run: Callable[..., subprocess.CompletedProcess]) -> Any:
    done = run(
        ["amd-smi", "metric", "--power", "--mem-usage", "--json"],  # nosec B603 B607
        capture_output=True,
        text=True,
        timeout=_QUERY_TIMEOUT_SEC,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"amd-smi exited {done.returncode}: {(done.stderr or '').strip()[:200]}")
    return json.loads(done.stdout)


class GpuPowerRecorder:
    """Samples GPU power on a background thread while the round is in its ``measured`` phase.

    Exposes the same ``note_phase`` / ``tick`` / ``close`` surface as the KV-metrics recorder so the watchdog loop
    drives both identically. Sampling runs on its own thread because an ``amd-smi`` call takes a few hundred ms and the
    watchdog loop must not stall on it.
    """

    def __init__(
        self,
        *,
        output_path: str | None,
        gpus: set[int] | None = None,
        interval_sec: float = _DEFAULT_INTERVAL_SEC,
        query: Callable[[], Any] | None = None,
    ) -> None:
        """Prepare a recorder without running anything."""
        self._output_path = output_path
        self._gpus = None if gpus is None else set(gpus)
        self._interval = max(_MIN_INTERVAL_SEC, float(interval_sec))
        self._query = query or (lambda: _query_amd_smi(subprocess.run))
        self._phase = "boot"
        self._samples: list[dict[int, tuple[float | None, float | None]]] = []
        self._errors = 0
        self._last_error: str | None = None
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
                self._sample_once()
            self._stop.wait(self._interval)

    def _sample_once(self) -> None:
        try:
            reading = parse_power_sample(self._query())
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail the round
            with self._lock:
                self._errors += 1
                self._last_error = str(exc)[:200]
            return
        if self._gpus is not None:
            reading = {gpu: row for gpu, row in reading.items() if gpu in self._gpus}
        if reading and self._phase == "measured":
            with self._lock:
                self._samples.append(reading)

    def summary(self, *, aborted: bool = False) -> dict[str, Any]:
        """The round's measured-phase power, averaged over the cards that served it."""
        with self._lock:
            samples = list(self._samples)
            errors, last_error = self._errors, self._last_error
        peak_vram: dict[int, float] = {}
        watts: dict[int, list[float]] = {}
        for reading in samples:
            for gpu, (power, fraction) in reading.items():
                if fraction is not None:
                    peak_vram[gpu] = max(peak_vram.get(gpu, 0.0), fraction)
                if power is not None:
                    watts.setdefault(gpu, []).append(power)
        serving = sorted(gpu for gpu, frac in peak_vram.items() if frac >= _SERVING_VRAM_FRACTION and gpu in watts)
        per_gpu = {
            str(gpu): {
                "avg_power_w": round(sum(watts[gpu]) / len(watts[gpu]), 2),
                "max_power_w": round(max(watts[gpu]), 2),
                "samples": len(watts[gpu]),
                "peak_vram_fraction": round(peak_vram.get(gpu, 0.0), 4),
            }
            for gpu in sorted(watts)
        }
        avg = round(sum(per_gpu[str(g)]["avg_power_w"] for g in serving) / len(serving), 2) if serving else None
        peak = max((per_gpu[str(g)]["max_power_w"] for g in serving), default=None)
        return {
            "schema_version": 1,
            "source": "amd-smi",
            "phase": "measured",
            "started_unix": round(self._started_unix, 3),
            "interval_sec": self._interval,
            "samples": len(samples),
            "serving_gpus": serving,
            "avg_power_w": avg,
            "max_power_w": peak,
            "per_gpu": per_gpu,
            "query_errors": errors,
            "last_error": last_error,
            "aborted": bool(aborted),
        }

    def close(self, *, aborted: bool = False) -> dict[str, Any]:
        """Stop sampling and write the artifact. Idempotent."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_QUERY_TIMEOUT_SEC + self._interval)
        payload = self.summary(aborted=aborted)
        if self._closed or not self._output_path:
            self._closed = True
            return payload
        self._closed = True
        try:
            from hyperloom.common.io import atomic_write_json

            atomic_write_json(Path(self._output_path), payload)
        except Exception:
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
        scan: Callable[..., Any],
        interval_sec: float = _DEFAULT_INTERVAL_SEC,
    ) -> None:
        """Prepare a driver; nothing is read until :meth:`start`.

        ``scan`` is the watchdog's own log scanner, passed in rather than imported: the watchdog module builds this
        module's recorder, so importing it back from here would close a cycle.
        """
        self._recorder = recorder
        self._find_logs = find_logs
        self._scan = scan
        self._interval = max(_MIN_INTERVAL_SEC, float(interval_sec))
        self._offsets: dict[str, int] = {}
        self._residuals: dict[str, str] = {}
        self._identities: dict[str, tuple[int, int]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll(self) -> None:
        """Read what every log appended since the last poll and move the recorder's phase on."""
        for path in self._find_logs():
            if path not in self._offsets:
                self._offsets[path] = 0
                self._recorder.note_phase("boot", time.monotonic())
            scan = self._scan(path, self._offsets, self._residuals, self._identities)
            now = time.monotonic()
            if scan.saw_ready:
                self._recorder.note_phase("measured", now)
            if scan.saw_warmup_begin:
                self._recorder.note_phase("warmup", now)
            if scan.saw_measured_begin:
                self._recorder.note_phase("measured", now)
            if scan.saw_eval_start:
                self._recorder.note_phase("eval", now)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:
                log.debug("gpu_power: replay log poll failed", exc_info=True)
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
        try:
            self.poll()
        except Exception:
            log.debug("gpu_power: final replay log poll failed", exc_info=True)


def build_gpu_power_recorder(server_log_path: str | None, env: dict[str, str] | None) -> GpuPowerRecorder | None:
    """The round's power recorder, or ``None`` when sampling is off or there is no ``amd-smi`` to sample with."""
    if not server_log_path or not env_flag(GPU_POWER_ENV, default=not os.environ.get("PYTEST_CURRENT_TEST")):
        return None
    if shutil.which("amd-smi") is None:
        return None
    try:
        from hyperloom.common.gpu_power_settings import visible_gpu_indices

        return GpuPowerRecorder(
            output_path=str(Path(server_log_path).parent / GPU_POWER_ARTIFACT_NAME),
            gpus=visible_gpu_indices(env or {}),
            interval_sec=env_float(_INTERVAL_ENV, _DEFAULT_INTERVAL_SEC),
        )
    except Exception:
        log.warning("gpu_power: recorder could not be built; this round samples nothing", exc_info=True)
        return None


def read_measured_gpu_power(
    workspace: Path | None, *, subprocess_started_unix: float | None = None
) -> tuple[bool, float | None]:
    """``(found, avg_power_w)`` from the round's ``gpu_power.json``, beside the report or one level up.

    ``found`` is ``True`` whenever this round's artifact exists, even when it holds no reading, so a caller can tell
    "the recorder ran and measured nothing" (fail closed) from "no recorder ran" (fall back to the report).
    """
    if workspace is None:
        return False, None
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
        return True, _value(payload.get("avg_power_w"))
    return False, None
