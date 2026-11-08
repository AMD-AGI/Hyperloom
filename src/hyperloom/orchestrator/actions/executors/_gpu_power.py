# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hyperloom's wiring of the measured-phase GPU recorder, and the readers of what it wrote.

A benchmark report's ``gpu_monitor`` block samples one card over the whole process, so it describes
neither the serving cards nor the load being measured; :class:`hyperloom_common.gpu_telemetry.GpuPowerRecorder` does.
This module builds that recorder for a round (:func:`build_gpu_power_recorder`), names the same sampling for an
inference_suite round, which runs the recorder itself (:func:`round_gpu_telemetry`), drives it from the server logs of a
child the watchdog does not own (:class:`ServerLogPhaseDriver`), and reads the round's ``gpu_power.json`` back for the
power-budget checks.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from hyperloom_common.amd_smi import metric_value
from hyperloom_common.env import EnvValueError, env_float
from hyperloom_common.gpu_telemetry import DEFAULT_INTERVAL_SEC, MIN_INTERVAL_SEC, GpuPhase, GpuPowerRecorder
from hyperloom.common.gpu_power_settings import gpu_power_sampling_unavailable
from inference_suite.contracts import GpuTelemetry

log = logging.getLogger(__name__)

__all__ = [
    "GPU_POWER_ARTIFACT_NAME",
    "ServerLogPhaseDriver",
    "build_gpu_power_recorder",
    "read_measured_gpu_power",
    "read_measured_gpu_power_by_gpu",
    "round_gpu_telemetry",
]

#: Artifact written beside the round's ``server.log``.
GPU_POWER_ARTIFACT_NAME = "gpu_power.json"

_INTERVAL_ENV = "HYPERLOOM_GPU_POWER_INTERVAL_S"

#: How long :meth:`ServerLogPhaseDriver.stop` waits for a poll in flight, beyond one interval.
_DRIVER_JOIN_SLACK_SEC = 10.0

#: How far a recorder's start may precede the subprocess it is paired with and still be that round's.
_START_SLACK_SEC = 5.0


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
        interval_sec: float = DEFAULT_INTERVAL_SEC,
    ) -> None:
        """Prepare a driver; nothing is read until :meth:`start`.

        ``cursor`` is the watchdog's own log cursor, passed in rather than imported: the watchdog module builds this
        module's recorder, so importing it back from here would close a cycle.
        """
        self._recorder = recorder
        self._find_logs = find_logs
        self._cursor = cursor
        self._interval = max(MIN_INTERVAL_SEC, float(interval_sec))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._read_failed = False

    def poll(self) -> None:
        """Read what every log appended since the last poll and move the recorder's phase on."""
        for path in self._find_logs():
            if self._cursor.open_log(path):
                self._recorder.note_phase(GpuPhase.BOOT, time.monotonic())
            scan = self._cursor.advance(path)
            now = time.monotonic()
            if scan.saw_ready:
                self._recorder.note_phase(GpuPhase.MEASURED, now)
            if scan.saw_warmup_begin:
                self._recorder.note_phase(GpuPhase.WARMUP, now)
            if scan.saw_measured_begin:
                self._recorder.note_phase(GpuPhase.MEASURED, now)
            if scan.saw_eval_start:
                self._recorder.note_phase(GpuPhase.EVAL, now)

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
            self._thread.join(timeout=self._interval + _DRIVER_JOIN_SLACK_SEC)
        self._poll_logs()


def build_gpu_power_recorder(server_log_path: str | None, env: dict[str, str] | None) -> GpuPowerRecorder | None:
    """The round's power recorder, or ``None`` when sampling is off or there is no ``amd-smi`` to sample with."""
    if not server_log_path or gpu_power_sampling_unavailable():
        return None
    try:
        from hyperloom_common.amd_smi import AmdSmiError

        from hyperloom.common.gpu_power_settings import visible_gpu_indices

        try:
            gpus = visible_gpu_indices(env or {})
        except AmdSmiError as exc:
            # Serving cards are still found by the VRAM they hold, as on a host with no mask.
            log.warning("gpu_power: %s; sampling every card", exc)
            gpus = None
        return GpuPowerRecorder(
            output_path=str(Path(server_log_path).parent / GPU_POWER_ARTIFACT_NAME),
            gpus=gpus,
            interval_sec=env_float(_INTERVAL_ENV, DEFAULT_INTERVAL_SEC),
        )
    except EnvValueError:
        log.warning("gpu_power: recorder could not be built; this round samples nothing", exc_info=True)
        return None


def round_gpu_telemetry() -> GpuTelemetry:
    """The GPU sampling an inference_suite round runs: on where ``amd-smi`` can sample, at the operator's interval."""
    try:
        interval_sec = env_float(_INTERVAL_ENV, DEFAULT_INTERVAL_SEC)
    except EnvValueError:
        log.warning("gpu_power: the sampling interval is unreadable; this round samples nothing", exc_info=True)
        return GpuTelemetry(enabled=False, interval_sec=DEFAULT_INTERVAL_SEC)
    return GpuTelemetry(enabled=not gpu_power_sampling_unavailable(), interval_sec=max(MIN_INTERVAL_SEC, interval_sec))


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
        started = metric_value(payload.get("started_unix"))
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
    return (False, None) if payload is None else (True, metric_value(payload.get("avg_power_w")))


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
        if (watts := metric_value((per_gpu.get(str(gpu)) or {}).get("avg_power_w"))) is not None
    }
    return True, (readings or None)
