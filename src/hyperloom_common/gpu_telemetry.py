# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The power recorder: every power-related reading a node gives, for one round, into one ``power.json``.

:class:`PowerRecorder` samples every card from :meth:`PowerRecorder.start` to :meth:`PowerRecorder.close`, on two
threads. The fast thread reads sysfs each second: every card's power, temperatures, clocks, board voltage, busy
percentages and memory use, the node's NPM power and the CPU's energy counters where the node has them. The slow
thread reads the cards' energy accumulators (``amd-smi metric -E``) at each phase mark the round passes, and every
``amd-smi metric`` field every 10 s and at each mark, so throttle accumulators bracket every phase exactly. ``amd-smi
static`` and the sysfs caps and DPM tables are read at both ends.

Phase marks carry the monotonic instant the phase began, which bounds the window; when the recorder hears of a mark
does not move it. The recording (:class:`~hyperloom_common.power_recording.PowerRecording`) is written every 60 s while
the round runs, so a killed round leaves its samples, and once more by the caller, with the round's token counts,
after :meth:`PowerRecorder.close`.

Once a recorder is built nothing here raises: telemetry that fails must cost the round nothing, so each failed read is
counted under ``errors`` in the file and a metric with no reading is ``None`` rather than zero. Building one refuses a
sampling interval :func:`fast_interval` refuses.
"""

from __future__ import annotations

import logging
import socket
import subprocess  # nosec B404 - the exceptions amd-smi reads raise.
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from hyperloom_common.amd_smi import NPM_ENABLED, gpu_rows
from hyperloom_common.io import JsonFormat, atomic_write_json
from hyperloom_common.power_energy import MAX_SAMPLE_GAP_S, TimedRun, TokenCounts
from hyperloom_common.power_recording import MEASURED, MetricStats, PowerRecording, RecorderScope, throttle_fraction
from hyperloom_common.power_sources import (
    CpuDomain,
    GpuFiles,
    NodeFiles,
    SysfsNode,
    amd_smi_query,
    bdf_by_gpu,
    energy_by_gpu,
    flatten_metric,
)

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_INTERVAL_SEC",
    "MAX_INTERVAL_SEC",
    "MIN_INTERVAL_SEC",
    "GpuMonitorBlock",
    "GpuPhase",
    "MetricStats",
    "PowerRecorder",
    "PowerRecording",
    "RecorderScope",
    "RecorderSources",
    "RecorderTiming",
    "TokenCounts",
    "fast_interval",
    "gpu_monitor",
    "publish_power",
    "recording_host",
]

DEFAULT_INTERVAL_SEC = 1.0
MIN_INTERVAL_SEC = 0.5
#: A fast read that overruns its slot is followed one whole interval after it ends, so two samples can sit two
#: intervals apart; at this ceiling that stays within the gap that invalidates a window's energy.
MAX_INTERVAL_SEC = MAX_SAMPLE_GAP_S / 2
_JOIN_SLACK_SEC = 35.0
_ERROR_DIGITS = 200

#: A card held back by its power limit for at least this share of the measured phase is logged as power throttled.
_POWER_THROTTLE_REPORT_FRACTION = 0.10
_PPT = "ppt"
_COMPACT = JsonFormat(indent=None, sort_keys=False, trailing_newline=True)
_READ_ERRORS = (OSError, subprocess.SubprocessError, RuntimeError, ValueError, TypeError)


class GpuPhase(str, Enum):
    """The phases a recorder hears; the measured one is the energy window."""

    BOOT = "boot"
    READY = "ready"
    WARMUP = "warmup"
    MEASURED = MEASURED
    FINISHED = "finished"
    EVAL = "eval"
    TEARDOWN = "teardown"


_PHASE_VALUES = frozenset(phase.value for phase in GpuPhase)


def recording_host() -> str:
    """The host a recorder runs on, as ``power.json`` names it; a round's report names the same one."""
    return socket.gethostname()


def fast_interval(seconds: float) -> float:
    """The interval a recorder reads sysfs at for a requested ``seconds``: raised to :data:`MIN_INTERVAL_SEC`.

    Raises:
        ValueError: ``seconds`` is above :data:`MAX_INTERVAL_SEC`; every window sampled at it could be invalid.
    """
    if seconds > MAX_INTERVAL_SEC:
        raise ValueError(
            f"a {seconds} s GPU power sampling interval is above {MAX_INTERVAL_SEC} s, so samples could sit more "
            f"than the {MAX_SAMPLE_GAP_S} s apart that invalidates a window's energy"
        )
    return max(MIN_INTERVAL_SEC, seconds)


@dataclass(frozen=True)
class RecorderTiming:
    """How often each source is read: sysfs each ``fast_sec``, every ``amd-smi metric`` field each ``full_sec``, and
    the recording written each ``write_sec``.

    On an MI355X node a 60 s run with six full reads cost about 4.7% of one core in ``amd-smi`` subprocesses; the
    throttle accumulators it carries are cumulative, so the windows' deltas come from the reads at the marks and do
    not depend on it.
    """

    fast_sec: float = DEFAULT_INTERVAL_SEC
    full_sec: float = 10.0
    write_sec: float = 60.0


@dataclass(frozen=True)
class RecorderSources:
    """Where readings come from: ``amd-smi`` and the sysfs tree."""

    query: Callable[..., Any] = amd_smi_query
    sysfs: SysfsNode = field(default_factory=SysfsNode)


@dataclass(frozen=True)
class GpuMonitorBlock:
    """One card's measured-phase telemetry, under the ``gpu_monitor`` keys a benchmark report already carries."""

    gpu: int
    sample_count: int
    power_watts: MetricStats | None
    temperature_c: MetricStats | None
    gpu_clock_mhz: MetricStats | None
    mem_clock_mhz: MetricStats | None
    gpu_util_pct: MetricStats | None
    vram_pct: MetricStats | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def gpu_monitor(recording: PowerRecording) -> tuple[GpuMonitorBlock, ...]:
    """One block per card whose energy is the round's, over the measured phase."""
    sampled = set(recording.sampled_gpus())
    return tuple(
        GpuMonitorBlock(gpu=gpu, **recording.monitor_stats(gpu)) for gpu in recording.energy_scope() if gpu in sampled
    )


class PowerRecorder:
    """Samples the node from :meth:`start` to :meth:`close`; see the module docstring for what and when."""

    def __init__(
        self,
        *,
        output_path: str | None,
        scope: RecorderScope,
        timing: RecorderTiming = RecorderTiming(),
        sources: RecorderSources | None = None,
    ) -> None:
        """Prepare a recorder without reading anything; ``output_path`` receives the periodic writes."""
        self._output_path = output_path
        self._scope = scope
        self._timing = replace(timing, fast_sec=fast_interval(timing.fast_sec))
        self._sources = sources or RecorderSources()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []
        self._files: dict[int, GpuFiles] = {}
        self._node_files: NodeFiles | None = None
        self._cpu: tuple[str | None, tuple[CpuDomain, ...]] = (None, ())
        self._recording = PowerRecording(
            host=recording_host(),
            started_unix=time.time(),
            unix_offset=time.time() - time.monotonic(),
            ended_mono=time.monotonic(),
            scope=scope,
            recorder={},
        )
        self._closed: PowerRecording | None = None

    def start(self) -> None:
        """Read the static facts and start both sampling threads."""
        self._recording.static["start"] = self._read_static(first=True)
        self._recording.recorder = self._recorder_facts()
        self._threads = [
            threading.Thread(target=self._fast_loop, name="power-fast", daemon=True),
            threading.Thread(target=self._slow_loop, name="power-slow", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def note_phase(self, phase: str, mono: float) -> None:
        """Mark ``phase`` as begun at ``mono``; a phase name the recorder does not know is ignored."""
        if phase not in _PHASE_VALUES or self._closed is not None:
            return
        name = GpuPhase(phase).value
        with self._lock:
            marks = self._recording.marks
            marks.append((name, mono + self._recording.unix_offset, mono))
            marks.sort(key=lambda mark: mark[2])
        self._wake.set()

    def note_timed_run(self, run: TimedRun) -> None:
        """Bound the measured phase by ``run``, for a load whose own stamps are its measured window."""
        offset = self._recording.unix_offset
        with self._lock:
            self._recording.reported_duration_s = run.duration_s
        self.note_phase(GpuPhase.MEASURED.value, run.start_unix - offset)
        self.note_phase(GpuPhase.FINISHED.value, run.end_unix - offset)

    def close(self, *, aborted: bool = False) -> PowerRecording:
        """Stop sampling, take the closing reads and return everything recorded. Idempotent."""
        if self._closed is not None:
            return self._closed
        ended = time.monotonic()
        self._stop.set()
        self._wake.set()
        for thread in self._threads:
            thread.join(timeout=_JOIN_SLACK_SEC)
        if self._threads:
            # Read after the end, so a phase still open at close is bracketed by samples on both sides.
            self._read_fast()
            self._read_energy()
            self._read_full()
            self._recording.static["end"] = self._read_static(first=False)
        self._closed = self.recording(final=True, aborted=aborted, ended_mono=ended)
        return self._closed

    def finish(self, counts: TokenCounts | None, *, aborted: bool = False) -> tuple[PowerRecording, dict[str, Any]]:
        """Close, and write ``power.json`` with ``counts``, what the measured window served; the recording and the
        document written."""
        recording = self.close(aborted=aborted)
        document = recording.document(counts)
        if self._output_path:
            publish_power(self._output_path, document)
        return recording, document

    def recording(
        self, *, final: bool = False, aborted: bool = False, ended_mono: float | None = None
    ) -> PowerRecording:
        """A copy of everything recorded so far."""
        with self._lock:
            current = self._recording
            return replace(
                current,
                final=final,
                aborted=aborted,
                ended_mono=time.monotonic() if ended_mono is None else ended_mono,
                marks=list(current.marks),
                fast=list(current.fast),
                energy=list(current.energy),
                node=list(current.node),
                cpu=list(current.cpu),
                full=list(current.full),
                units=dict(current.units),
                errors={name: dict(error) for name, error in current.errors.items()},
                static=dict(current.static),
            )

    def _error(self, source: str, exc: object) -> None:
        with self._lock:
            entry = self._recording.errors.setdefault(source, {"count": 0, "last": None})
            entry["count"] += 1
            entry["last"] = str(exc)[:_ERROR_DIGITS]

    def _query(self, source: str, *args: str) -> Any:
        try:
            return self._sources.query(*args)
        except _READ_ERRORS as exc:
            log.debug("power: amd-smi %s failed", " ".join(args), exc_info=True)
            self._error(source, exc)
            return None

    def _recorder_facts(self) -> dict[str, Any]:
        version = self._query("amd_smi_version", "version")
        tool = version[0] if isinstance(version, list) and version and isinstance(version[0], dict) else {}
        return {
            "fast_interval_s": self._timing.fast_sec,
            "full_interval_s": self._timing.full_sec,
            "write_interval_s": self._timing.write_sec,
            "amd_smi_version": tool.get("version"),
            "sources": {
                "gpu_power": "sysfs_hwmon",
                "energy": "amd-smi metric -E",
                "full": "amd-smi metric",
                "node": _node_source(self._node_files),
                "cpu": {
                    "source": self._cpu[0],
                    "reason": None if self._cpu[0] else "no RAPL powercap zone or amd_energy hwmon",
                },
            },
        }

    def _read_static(self, *, first: bool) -> dict[str, Any]:
        payload = self._query("amd_smi_static", "static")
        rows = {row["gpu"]: row for row in gpu_rows(payload)}
        if first:
            sysfs = self._sources.sysfs
            self._files = sysfs.gpu_files(bdf_by_gpu(payload))
            self._node_files = sysfs.node_files()
            self._cpu = sysfs.cpu_domains()
            self._recording.vram_total_b = {
                gpu: total for gpu, files in self._files.items() if (total := files.static()["vram_total_b"])
            }
            self._recording.cpu_counted = {domain.name: domain.wrap_uj for domain in self._cpu[1] if domain.counted}
        return {
            "unix": round(time.time(), 3),
            "gpus": {
                str(gpu): {**files.static(), "amd_smi_static": rows.get(gpu)} for gpu, files in self._files.items()
            },
            "unmapped_amd_smi_gpus": sorted(set(rows) - set(self._files)),
            "npm": None if self._node_files is None else self._node_files.static(),
            "cpu": {
                "source": self._cpu[0],
                "domains": [{"name": d.name, "counted": d.counted, "wrap_uj": d.wrap_uj} for d in self._cpu[1]],
            },
        }

    def _fast_loop(self) -> None:
        due = time.monotonic()
        while not self._stop.is_set():
            self._read_fast()
            due += self._timing.fast_sec
            now = time.monotonic()
            if due < now:
                due = now + self._timing.fast_sec
            self._stop.wait(due - now)

    def _read_fast(self) -> None:
        mono, unix = time.monotonic(), time.time()
        stamp = (mono, unix)
        rows = []
        for gpu, files in self._files.items():
            values = files.read()
            if values[0] is None and "power_w" in files.columns:
                self._error("sysfs_gpu_power", f"gpu {gpu}: {files.columns['power_w'][0]} did not read")
            rows.append((*stamp, gpu, *values))
        node = None if self._node_files is None else (*stamp, *self._node_files.read())
        cpu = [(*stamp, domain.name, domain.read()) for domain in self._cpu[1]]
        with self._lock:
            self._recording.fast.extend(rows)
            if node is not None:
                self._recording.node.append(node)
            self._recording.cpu.extend(cpu)

    def _slow_loop(self) -> None:
        written = time.monotonic()
        self._read_energy()
        self._read_full()
        while not self._stop.is_set():
            if self._wake.wait(self._timing.full_sec):
                self._wake.clear()
                if self._stop.is_set():
                    return
                self._read_energy()
            self._read_full()
            if self._output_path and time.monotonic() - written >= self._timing.write_sec:
                written = time.monotonic()
                publish_power(self._output_path, self.recording().document(None), warn=False)

    def _read_energy(self) -> None:
        before = time.monotonic()
        payload = self._query("amd_smi_energy", "metric", "-E")
        after = time.monotonic()
        if payload is None:
            return
        mid = (before + after) / 2
        stamp = (mid, mid + self._recording.unix_offset)
        rows = [
            (*stamp, gpu, joules, round(after - before, 3)) for gpu, joules in sorted(energy_by_gpu(payload).items())
        ]
        with self._lock:
            self._recording.energy.extend(rows)

    def _read_full(self) -> None:
        before = time.monotonic()
        payload = self._query("amd_smi_metric", "metric")
        after = time.monotonic()
        if payload is None:
            return
        rows, units = flatten_metric(payload)
        mid = (before + after) / 2
        with self._lock:
            self._recording.units.update(units)
            self._recording.full.extend(
                (mid, mid + self._recording.unix_offset, gpu, values) for gpu, values in rows.items()
            )


def _node_source(files: NodeFiles | None) -> dict[str, Any]:
    """Where node power comes from, and why there is none when NPM does not report it."""
    if files is None:
        return {"source": None, "status": None, "reason": "no card publishes board/npm_status"}
    status = files.status()
    reason = None if status == NPM_ENABLED else f"npm_status is {status}: firmware reports no node power"
    return {"source": "sysfs_npm", "status": status, "reason": reason}


def publish_power(path: str | Path, document: Mapping[str, Any], *, warn: bool = True) -> None:
    """Write ``power.json``; with ``warn``, log the cards held back by their power limit and a node held at NPM's."""
    try:
        atomic_write_json(Path(path), document, fmt=_COMPACT)
    except (OSError, TypeError, ValueError):
        log.warning("power: could not write %s", path, exc_info=True)
    if warn:
        _warn_throttling(document)


def _warn_throttling(document: Mapping[str, Any]) -> None:
    scope = document.get("scope") or {}
    throttled = {
        gpu: fraction
        for gpu in scope.get("energy_gpus") or []
        if (fraction := throttle_fraction(document, gpu, _PPT)) is not None
        and fraction >= _POWER_THROTTLE_REPORT_FRACTION
    }
    if throttled:
        log.warning(
            "power: GPU(s) %s were power throttled for %s of the measured phase",
            ", ".join(map(str, throttled)),
            ", ".join(f"{fraction:.0%}" for fraction in throttled.values()),
        )
    node = (((document.get("phases") or {}).get(MEASURED) or {}).get("node")) or {}
    if (node.get("limited_fraction") or 0.0) >= _POWER_THROTTLE_REPORT_FRACTION:
        log.warning(
            "power: the node was held at its NPM power limit for %.0f%% of the measured phase",
            node["limited_fraction"] * 100,
        )
