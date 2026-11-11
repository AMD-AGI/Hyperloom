# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one power recorder saw, and ``power.json`` (``schema_version`` 2) built from it.

:class:`PowerRecording` holds every reading the recorder took and the phase marks it heard; :meth:`PowerRecording.document`
turns them into the one file that records a round's power: the node's static facts at both ends, a summary of every
phase, the measured window's energy (:mod:`hyperloom_common.power_energy`), and every sample, column-major.

A phase lasts from its mark to the next mark the recorder heard, or to the end of the recording. The cards whose energy
is the round's are the configured ones; with none configured, those found holding a serving-sized share of VRAM. Every
card on the node is sampled, but with a configured scope only its cards are asked whether they serve: on a node shared
with another server, the cards that one holds are not this round's.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hyperloom_common.power_energy import EnergyInputs, TokenCounts, Window, counter_delta, energy_block, integrate
from hyperloom_common.power_sources import FAST_COLUMNS

__all__ = [
    "MEASURED",
    "POWER_SCHEMA_VERSION",
    "MetricStats",
    "PowerRecording",
    "RecorderScope",
    "throttle_fraction",
]

POWER_SCHEMA_VERSION = 2
MEASURED = "measured"

#: A card counts as serving when it held at least this share of its VRAM during the measured phase. Serving engines
#: reserve most of HBM up front (vLLM's default ``--gpu-memory-utilization`` is 0.9), while an idle card holds a few
#: hundred MB of driver state, so the split is wide and the threshold does not need to be exact.
SERVING_VRAM_FRACTION = 0.10
#: A node sample at or above this share of the NPM limit is one firmware is holding at that limit.
_NODE_AT_LIMIT_FRACTION = 0.99
_THROTTLE_PREFIX = "throttle."
_THROTTLE_SUFFIX = "_accumulated"
_TICKS = "throttle.accumulation_counter"
_FRACTION_DIGITS = 4
_PERCENTILES = {"p50": 50, "p75": 75, "p90": 90}
_STAMP = ("mono", "unix")
_GPU_STAMP = (*_STAMP, "gpu")
#: Where each fast column sits in a fast row, after its stamp and card.
_FAST = {name: index + len(_GPU_STAMP) for index, name in enumerate(FAST_COLUMNS)}
_POWER = "power_w"
_DIGITS = 3


@dataclass(frozen=True)
class RecorderScope:
    """Which cards the recording is about, and on which server.

    ``gpus`` are the amd-smi indices of the cards the round configured, which every sample is keyed by; ``None`` finds
    them by the VRAM they hold. A round against an
    external server records this host but measures no energy: the load ran on cards it does not see.
    """

    gpus: tuple[int, ...] | None
    external_server: bool = False
    nnodes: int = 1
    node_rank: int = 0
    window_source: str = "phase_marks"


@dataclass(frozen=True)
class MetricStats:
    """``{min, max, avg}`` of one metric over the samples that read it."""

    min: float
    max: float
    avg: float

    @classmethod
    def over(cls, values: Sequence[float | None]) -> MetricStats | None:
        read = [value for value in values if value is not None]
        if not read:
            return None
        return cls(min=min(read), max=max(read), avg=round(sum(read) / len(read), 2))


@dataclass
class PowerRecording:
    """Every reading one recorder took. Rows are tuples in the column order of ``power.json``'s ``samples``."""

    host: str
    started_unix: float
    unix_offset: float
    ended_mono: float
    scope: RecorderScope
    recorder: dict[str, Any]
    final: bool = True
    aborted: bool = False
    marks: list[tuple[str, float, float]] = field(default_factory=list)
    static: dict[str, Any] = field(default_factory=dict)
    fast: list[tuple[Any, ...]] = field(default_factory=list)
    energy: list[tuple[Any, ...]] = field(default_factory=list)
    node: list[tuple[Any, ...]] = field(default_factory=list)
    cpu: list[tuple[Any, ...]] = field(default_factory=list)
    full: list[tuple[float, float, int, dict[str, Any]]] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    vram_total_b: dict[int, float] = field(default_factory=dict)
    cpu_counted: dict[str, float | None] = field(default_factory=dict)
    reported_duration_s: float | None = None

    def intervals(self, phase: str) -> tuple[tuple[float, float], ...]:
        """Every ``(start, end)`` the recording spent in ``phase``, on the monotonic clock."""
        ends = [mono for _, _, mono in self.marks[1:]] + [self.ended_mono]
        return tuple((mono, end) for (name, _, mono), end in zip(self.marks, ends) if name == phase and end > mono)

    def window(self) -> Window | None:
        """The measured window, or ``None`` when the recording never entered the measured phase."""
        intervals = self.intervals(MEASURED)
        if not intervals:
            return None
        start, end = intervals[0][0] + self.unix_offset, intervals[-1][1] + self.unix_offset
        return Window(MEASURED, intervals, start, end, self.scope.window_source, self.reported_duration_s)

    def _column(self, gpu: int, column: str, intervals: Sequence[tuple[float, float]]) -> list[tuple[float, Any]]:
        index = _FAST[column]
        return [
            (row[0], row[index])
            for row in self.fast
            if row[2] == gpu and any(start <= row[0] <= end for start, end in intervals)
        ]

    def power_series(self, gpu: int) -> list[tuple[float, float | None]]:
        return [(row[0], row[_FAST[_POWER]]) for row in self.fast if row[2] == gpu]

    def sampled_gpus(self) -> list[int]:
        return sorted({row[2] for row in self.fast})

    def serving_gpus(self) -> tuple[int, ...]:
        """The cards holding a serving-sized share of their VRAM during the measured phase, among the configured ones
        when a scope is configured, else among every card sampled."""
        intervals = self.intervals(MEASURED)
        candidates = self.sampled_gpus() if self.scope.gpus is None else self.scope.gpus
        return tuple(gpu for gpu in candidates if (self._vram_peak(gpu, intervals) or 0.0) >= SERVING_VRAM_FRACTION)

    def _vram_peak(self, gpu: int, intervals: Sequence[tuple[float, float]]) -> float | None:
        total = self.vram_total_b.get(gpu)
        used = [value for _, value in self._column(gpu, "vram_used_b", intervals) if value is not None]
        return round(max(used) / total, _FRACTION_DIGITS) if used and total else None

    def energy_scope(self) -> tuple[int, ...]:
        return self.scope.gpus if self.scope.gpus is not None else self.serving_gpus()

    def unread_gpus(self) -> list[int]:
        """Scope cards with no power reading in the measured window: the round's energy cannot cover them."""
        intervals = self.intervals(MEASURED)
        return [
            gpu
            for gpu in self.energy_scope()
            if not any(value is not None for _, value in self._column(gpu, _POWER, intervals))
        ]

    def energy_inputs(self) -> EnergyInputs:
        accumulator: dict[int, list[tuple[float, float]]] = {}
        for mono, _, gpu, joules, _ in self.energy:
            accumulator.setdefault(gpu, []).append((mono, joules))
        return EnergyInputs(
            window=self.window(),
            power={gpu: self.power_series(gpu) for gpu in self.sampled_gpus()},
            accumulator=accumulator,
            scope=self.energy_scope(),
            serving=self.serving_gpus(),
            expected_scope=self.scope.gpus is not None,
            external_server=self.scope.external_server,
            aborted=self.aborted,
        )

    def document(self, counts: TokenCounts | None) -> dict[str, Any]:
        """``power.json``; ``counts`` are what the measured window served, ``None`` when the load counts no tokens."""
        energy = energy_block(self.energy_inputs(), counts)
        window = self.window()
        intervals = () if window is None or self.scope.external_server else window.intervals
        energy["node_energy_j"] = _round(self._node_energy(intervals), _DIGITS)
        energy["cpu_energy_j"] = _round(self._cpu_energy(intervals), _DIGITS)
        return {
            "schema_version": POWER_SCHEMA_VERSION,
            "host": self.host,
            "started_unix": round(self.started_unix, _DIGITS),
            "final": self.final,
            "aborted": self.aborted,
            "recorder": self.recorder,
            "scope": self._scope_document(),
            "static": self.static,
            "phase_marks": [
                {"phase": name, "unix": round(unix, _DIGITS), "mono": round(mono, _DIGITS)}
                for name, unix, mono in self.marks
            ],
            "phases": {name: self._phase(name) for name in dict.fromkeys(name for name, _, _ in self.marks)},
            "energy": energy,
            "samples": self._samples(),
            "errors": self.errors,
        }

    def _scope_document(self) -> dict[str, Any]:
        return {
            "gpus_configured": None if self.scope.gpus is None else list(self.scope.gpus),
            "gpus_sampled": self.sampled_gpus(),
            "serving_gpus": list(self.serving_gpus()),
            "energy_gpus": list(self.energy_scope()),
            "unread_gpus": self.unread_gpus(),
            "external_server": self.scope.external_server,
            "nnodes": self.scope.nnodes,
            "node_rank": self.scope.node_rank,
        }

    def _phase(self, phase: str) -> dict[str, Any]:
        intervals = self.intervals(phase)
        return {
            "duration_s": round(sum(end - start for start, end in intervals), _DIGITS),
            "per_gpu": {str(gpu): self._gpu_phase(gpu, intervals) for gpu in self.sampled_gpus()},
            "node": self._node_phase(intervals),
            "cpu": self._cpu_phase(intervals),
        }

    def _gpu_phase(self, gpu: int, intervals: Sequence[tuple[float, float]]) -> dict[str, Any]:
        def stats(column: str) -> dict[str, float] | None:
            values = [value for _, value in self._column(gpu, column, intervals) if value is not None]
            return None if not values else {"avg": round(statistics.fmean(values), 2), "max": max(values)}

        power = [value for _, value in self._column(gpu, _POWER, intervals) if value is not None]
        series = [(mono, value) for mono, value in self.power_series(gpu) if value is not None]
        energies = [integrate(series, start, end) for start, end in intervals]
        return {
            "power_w": _distribution(power),
            "energy_j": None if not energies or None in energies else round(sum(e for e in energies if e), _DIGITS),
            "junction_c": stats("junction_c"),
            "mem_c": stats("mem_c"),
            "sclk_mhz": stats("sclk_mhz"),
            "mclk_mhz": stats("mclk_mhz"),
            "gfx_busy_pct": stats("gfx_busy_pct"),
            "umc_busy_pct": stats("umc_busy_pct"),
            "vram_peak_frac": self._vram_peak(gpu, intervals),
            "throttle": self._throttle(gpu, intervals),
        }

    def _throttle(self, gpu: int, intervals: Sequence[tuple[float, float]]) -> dict[str, Any] | None:
        """Each throttle accumulator's share of the phase's firmware ticks, from the full snapshots at its bounds."""
        rows = [(mono, values) for mono, _, row_gpu, values in self.full if row_gpu == gpu]
        totals: dict[str, float] = {}
        for start, end in intervals:
            opened = next((values for mono, values in rows if mono >= start), None)
            closed = next((values for mono, values in rows if mono >= end), None)
            if opened is None or closed is None or opened is closed:
                return None
            for path, value in closed.items():
                before = opened.get(path)
                if _is_number(value) and _is_number(before) and (path == _TICKS or _is_accumulator(path)):
                    totals[path] = totals.get(path, 0.0) + value - before
        ticks = totals.pop(_TICKS, 0.0)
        if ticks <= 0:
            return None
        return {_accumulator_name(path): round(delta / ticks, _FRACTION_DIGITS) for path, delta in totals.items()}

    def _node_phase(self, intervals: Sequence[tuple[float, float]]) -> dict[str, Any] | None:
        rows = [row for row in self.node if any(start <= row[0] <= end for start, end in intervals)]
        powers = [row[2] for row in rows if row[2] is not None]
        if not powers:
            return None
        held = [row for row in rows if row[2] is not None and row[3] and row[2] >= row[3] * _NODE_AT_LIMIT_FRACTION]
        return {
            "avg_w": round(statistics.fmean(powers), 2),
            "max_w": max(powers),
            "limited_fraction": round(len(held) / len(powers), _FRACTION_DIGITS),
            "energy_j": _round(self._node_energy(intervals), _DIGITS),
        }

    def _node_energy(self, intervals: Sequence[tuple[float, float]]) -> float | None:
        series = [(row[0], row[2]) for row in self.node if row[2] is not None]
        energies = [integrate(series, start, end) for start, end in intervals]
        return None if not energies or None in energies else sum(e for e in energies if e is not None)

    def _cpu_energy(self, intervals: Sequence[tuple[float, float]]) -> float | None:
        if not self.cpu_counted or not intervals:
            return None
        total = 0.0
        for domain, wrap in self.cpu_counted.items():
            readings = [(row[0], row[3]) for row in self.cpu if row[2] == domain and row[3] is not None]
            for start, end in intervals:
                delta = counter_delta(readings, start, end, wrap)
                if delta is None:
                    return None
                total += delta / 1e6
        return total

    def _cpu_phase(self, intervals: Sequence[tuple[float, float]]) -> dict[str, Any] | None:
        energy = self._cpu_energy(intervals)
        seconds = sum(end - start for start, end in intervals)
        if energy is None or seconds <= 0:
            return None
        return {"energy_j": round(energy, _DIGITS), "avg_w": round(energy / seconds, 2)}

    def _samples(self) -> dict[str, Any]:
        paths = list(dict.fromkeys(path for *_, values in self.full for path in values))
        return {
            "fast": {"columns": [*_GPU_STAMP, *FAST_COLUMNS], "rows": _stamped(self.fast)},
            "energy": {
                "columns": [*_GPU_STAMP, "energy_j", "read_s"],
                "rows": _stamped(self.energy),
            },
            "node": {"columns": [*_STAMP, "node_power_w", "limit_w"], "rows": _stamped(self.node)},
            "cpu": {"columns": [*_STAMP, "domain", "energy_uj"], "rows": _stamped(self.cpu)},
            "full": {
                "columns": [*_GPU_STAMP, *paths],
                "units": {path: self.units[path] for path in paths if path in self.units},
                "rows": _stamped(
                    [(mono, unix, gpu, *(values.get(path) for path in paths)) for mono, unix, gpu, values in self.full]
                ),
            },
        }

    def monitor_stats(self, gpu: int) -> dict[str, Any]:
        """One card's ``gpu_monitor`` metrics over the measured phase: the fields a benchmark report's block holds."""
        intervals = self.intervals(MEASURED)
        total = self.vram_total_b.get(gpu)

        def over(column: str, scale: float = 1.0) -> MetricStats | None:
            return MetricStats.over([None if v is None else v * scale for _, v in self._column(gpu, column, intervals)])

        return {
            "sample_count": len(self._column(gpu, _POWER, intervals)),
            "power_watts": over(_POWER),
            "temperature_c": over("junction_c"),
            "gpu_clock_mhz": over("sclk_mhz"),
            "mem_clock_mhz": over("mclk_mhz"),
            "gpu_util_pct": over("gfx_busy_pct"),
            "vram_pct": over("vram_used_b", 100.0 / total) if total else None,
        }


def _stamped(rows: Sequence[tuple[Any, ...]]) -> list[list[Any]]:
    """Rows with their ``mono`` and ``unix`` stamps at millisecond precision."""
    return [[round(row[0], _DIGITS), round(row[1], _DIGITS), *row[2:]] for row in rows]


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_accumulator(path: str) -> bool:
    head = path.split(".")[1] if path.startswith(_THROTTLE_PREFIX) and path.count(".") >= 1 else ""
    return head.endswith(_THROTTLE_SUFFIX)


def _accumulator_name(path: str) -> str:
    return path[len(_THROTTLE_PREFIX) :].replace(_THROTTLE_SUFFIX, "", 1)


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    out = {"avg": round(statistics.fmean(ordered), 2), "min": ordered[0]}
    for name, percent in _PERCENTILES.items():
        out[name] = round(_percentile(ordered, percent), 2)
    out["max"] = ordered[-1]
    return out


def _percentile(ordered: Sequence[float], percent: float) -> float:
    rank = (len(ordered) - 1) * percent / 100.0
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def throttle_fraction(document: Mapping[str, Any], gpu: int, name: str) -> float | None:
    """One card's measured-phase share of firmware ticks counted by throttle accumulator ``name``."""
    phase = (document.get("phases") or {}).get(MEASURED) or {}
    throttle = ((phase.get("per_gpu") or {}).get(str(gpu)) or {}).get("throttle") or {}
    value = throttle.get(name)
    return float(value) if _is_number(value) else None
