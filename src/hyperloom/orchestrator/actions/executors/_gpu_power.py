# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hyperloom's wiring of the power recorder, and the reader of what it wrote.

A benchmark report's ``gpu_monitor`` block samples one card over the whole process, so it describes
neither the serving cards nor the load being measured; :class:`hyperloom_common.gpu_telemetry.PowerRecorder` does.
This module builds that recorder for a GEAK replay (:func:`build_gpu_power_recorder`), names the same sampling for an
inference_suite round, which runs the recorder itself (:func:`round_gpu_telemetry`), drives it from the server logs of a
child the watchdog does not own (:class:`ServerLogPhaseDriver`), and reads the round's energy back as measurement
fields (:func:`round_power_fields`) for the power-budget checks and the search.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from hyperloom_common.amd_smi import metric_value
from hyperloom_common.env import EnvValueError, env_float
from hyperloom_common.gpu_telemetry import (
    DEFAULT_INTERVAL_SEC,
    MIN_INTERVAL_SEC,
    GpuPhase,
    PowerRecorder,
    RecorderScope,
    RecorderTiming,
    fast_interval,
)
from hyperloom.common.gpu_power_settings import gpu_power_sampling_unavailable
from inference_suite.contracts import REPORT_SCHEMA, GpuTelemetry, PowerSummary
from inference_suite.contracts.layout import POWER

log = logging.getLogger(__name__)

__all__ = [
    "POWER_FIELDS",
    "ServerLogPhaseDriver",
    "build_gpu_power_recorder",
    "round_gpu_telemetry",
    "round_power_fields",
    "round_power_summary",
]

#: The window a GEAK replay's recorder measures comes from polling its server logs, not from phase marks.
_SERVER_LOG_WINDOW = "server_log_poll"

#: The measurement fields a round's power fills, each from the :class:`PowerSummary` figure it names (``None`` for
#: ``power_valid``, which is the summary's own). Scalars only: ``hosted_executor`` publishes a round's finite numbers
#: and drops dicts and ``None``, so an invalid round carries ``power_valid`` 0 and no energy.
POWER_FIELDS = {
    "gpu_power_avg_w": "avg_power_w",
    "gpu_power_p75_w": "p75_power_w",
    "gpu_power_p90_w": "p90_power_w",
    "gpu_energy_j": "total_gpu_energy_j",
    "joules_per_input_token": "joules_per_input_token",
    "joules_per_output_token": "joules_per_output_token",
    "joules_per_total_token": "joules_per_total_token",
    "joules_per_successful_query": "joules_per_successful_query",
}

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


def build_gpu_power_recorder(server_log_path: str | None, env: dict[str, str] | None) -> PowerRecorder | None:
    """A GEAK replay's power recorder, writing ``power.json`` beside ``server_log_path``; ``None`` when sampling is off
    or there is no ``amd-smi`` to sample with. The caller starts it."""
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
        scope = RecorderScope(gpus=None if gpus is None else tuple(sorted(gpus)), window_source=_SERVER_LOG_WINDOW)
        return PowerRecorder(
            output_path=str(Path(server_log_path).parent / POWER),
            scope=scope,
            timing=RecorderTiming(fast_sec=_interval_sec()),
        )
    except EnvValueError:
        log.warning("gpu_power: recorder could not be built; this round samples nothing", exc_info=True)
        return None


def _interval_sec() -> float:
    """The operator's sampling interval, as :func:`fast_interval` takes it.

    Raises:
        EnvValueError: the variable is not a number, or :func:`fast_interval` refuses it.
    """
    requested = env_float(_INTERVAL_ENV, DEFAULT_INTERVAL_SEC)
    try:
        return fast_interval(requested)
    except ValueError as exc:
        raise EnvValueError(f"{_INTERVAL_ENV}: {exc}") from exc


def round_gpu_telemetry() -> GpuTelemetry:
    """The GPU sampling an inference_suite round runs: on where ``amd-smi`` can sample, at the operator's interval."""
    try:
        interval_sec = _interval_sec()
    except EnvValueError:
        log.warning("gpu_power: the sampling interval is unusable; this round samples nothing", exc_info=True)
        return GpuTelemetry(enabled=False, interval_sec=DEFAULT_INTERVAL_SEC)
    return GpuTelemetry(enabled=not gpu_power_sampling_unavailable(), interval_sec=interval_sec)


def _read_power_json(workspace: Path, subprocess_started_unix: float | None) -> dict[str, Any] | None:
    """This round's ``power.json``, beside the report or one level up; ``None`` when no recorder ran for it."""
    for directory in (workspace, workspace.parent):
        try:
            payload = json.loads((directory / POWER).read_text(encoding="utf-8"))
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


def round_power_summary(
    report: Mapping[str, Any] | None, workspace: Path | None, *, subprocess_started_unix: float | None = None
) -> PowerSummary | None:
    """The round's power: an inference_suite report's ``power`` section, else the ``power.json`` a GEAK replay left
    in ``workspace``; ``None`` when no recorder ran for the round."""
    if report and report.get("schema") == REPORT_SCHEMA:
        section = report.get("power")
        return None if section is None else PowerSummary.from_document(section)
    if workspace is None:
        return None
    document = _read_power_json(Path(workspace), subprocess_started_unix)
    return None if document is None else PowerSummary.of(POWER, document)


def round_power_fields(summary: PowerSummary) -> dict[str, Any]:
    """The measurement fields of a round's power.

    ``gpu_power_by_gpu_w`` is ``None`` when any card's energy is, so a per-card budget never passes a card it cannot
    see; every figure is ``None`` unless ``power_valid`` is 1.
    """
    return {
        **{field: summary.metrics[name] for field, name in POWER_FIELDS.items()},
        "gpu_power_by_gpu_w": None if summary.per_gpu_avg_power_w is None else dict(summary.per_gpu_avg_power_w),
        "power_valid": summary.power_valid,
    }
