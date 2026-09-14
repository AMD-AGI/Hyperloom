# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-round GPU telemetry, normalised out of the round's own benchmark report.

The numbers are already collected -- Magpie's ``GPUMonitor`` on a single node, the pod-side ``rocm-smi`` sampler on
several -- and land in ``benchmark_report.json`` under ``gpu_monitor``. What was missing is a reading of them that a
consumer can trust, per round, without knowing which producer wrote the block.

Two things make that non-trivial, and both have burned this data before:

* **The producers disagree on names and on shape.** Magpie writes ``power_watts`` as a pre-aggregated
  ``{min, max, avg}`` block; the multi-node harvester writes ``power_w`` as a flat scalar per sample. Reading only the
  short spelling is what left every single-node session's GPU numbers reading 0.0 -- not absent, *zero*, which is a
  plausible-looking lie.
* **Absent and zero are different findings.** A card that drew no power and a card nobody sampled must not produce the
  same number. Every metric here is ``float | None`` and is never coerced.

Written per round rather than aggregated per session on purpose: a session mixes baseline, explore and roofline rounds
whose power and thermal behaviour have nothing to do with each other, and averaging them describes no round that
actually ran.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


__all__ = [
    "GPU_ARTIFACT_NAME",
    "gpu_metrics_from_report",
    "write_gpu_metrics",
]

#: Artifact written into the round's workspace, beside ``benchmark_report.json``.
GPU_ARTIFACT_NAME = "gpu_metrics.json"

# ``gpu_monitor`` metric aliases, current producer name first. Magpie emits ``power_watts`` / ``temperature_c`` /
# ``gpu_clock_mhz`` / ``mem_clock_mhz``; the shorter spellings are older shapes, kept so archived reports still parse.
_POWER_KEYS = ("power_watts", "power_w", "power")
_TEMP_KEYS = ("temperature_c", "temp_c", "temperature")
_CLOCK_KEYS = ("gpu_clock_mhz", "clock_mhz", "sclk_mhz")
_MEM_CLOCK_KEYS = ("mem_clock_mhz", "memory_clock_mhz", "mclk_mhz")

# Occupancy, in percent. ``gpu_util_pct`` / ``vram_pct`` are what the multi-node harvester writes (see
# ``benchmark_result._row_to_gpu_sample``); the rest are spellings of the same percentage a producer might use.
#
# Percent-named aliases only, deliberately. An absolute reading -- ``vram_used_mb``, ``memory_used_bytes`` -- is a
# different quantity, and folding one into a field called ``_pct`` would put 81920 where a percentage belongs.
_UTIL_KEYS = ("gpu_util_pct", "gpu_utilization_pct", "gpu_use_pct", "utilization_pct")
_VRAM_KEYS = ("vram_pct", "vram_usage_pct", "vram_used_pct", "memory_used_pct")

#: Every metric this reader knows. A block counts as contributing when it yields any one of them.
_ALL_KEYS = (_POWER_KEYS, _TEMP_KEYS, _CLOCK_KEYS, _MEM_CLOCK_KEYS, _UTIL_KEYS, _VRAM_KEYS)


def _to_float(raw: Any) -> float | None:
    """Coerce a reading to a float, or ``None`` when it is not one."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def _source(block: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Pick the one alias in this block that carries a usable reading.

    Resolved once per block rather than once per statistic. Choosing per statistic lets the mean come from
    ``power_watts`` and the peak from a stale ``power_w`` in the same block, which can report a maximum below the
    average -- a self-contradictory number of exactly the kind this exists to stop emitting.
    """
    for key in keys:
        if key not in block:
            continue
        raw = block[key]
        if isinstance(raw, dict):
            if any(_to_float(raw.get(stat)) is not None for stat in ("avg", "max", "min")):
                return raw
        elif _to_float(raw) is not None:
            return raw
    return None


def _metric(block: dict[str, Any], keys: tuple[str, ...], field: str) -> float | None:
    """Read one metric out of a single ``gpu_monitor`` block.

    Both producer shapes are read here: a flat scalar per sample, where the scalar is that sample's mean and its peak
    alike, and a pre-aggregated ``{min, max, avg}`` block, where ``field`` picks the statistic. ``None`` rather than
    ``0.0`` when the winning alias omits this particular statistic -- a metric that was never sampled has to stay
    distinguishable from one that measured zero.
    """
    raw = _source(block, keys)
    if raw is None:
        return None
    return _to_float(raw.get(field)) if isinstance(raw, dict) else _to_float(raw)


def _blocks(report: Any) -> list[dict[str, Any]]:
    """Every ``gpu_monitor`` entry in a report, whichever shape it was written in."""
    if not isinstance(report, dict):
        return []
    monitor = report.get("gpu_monitor")
    if isinstance(monitor, list):
        return [b for b in monitor if isinstance(b, dict)]
    return [monitor] if isinstance(monitor, dict) else []


def gpu_metrics_from_report(report: Any) -> dict[str, Any]:
    """Normalise one round's GPU telemetry. ``{}`` when the report carried none.

    Means are weighted by each block's ``sample_count``: a Magpie block summarises many samples while a flat per-sample
    block is one, and unweighted a 10-sample block would pull the round's mean as hard as a 10,000-sample one.
    """
    blocks = _blocks(report)
    if not blocks:
        return {}

    def _weight(block: dict[str, Any]) -> float:
        """Underlying samples behind one block; 1.0 when it does not say.

        ``or 1.0`` would promote a monitor that started and sampled nothing into one sample -- the same conflation of
        "no reading" with "a reading of zero" this module exists to avoid.
        """
        declared = _to_float(block.get("sample_count"))
        return 1.0 if declared is None else max(0.0, declared)

    weights = [_weight(b) for b in blocks]
    # Only blocks that yielded a metric count toward ``samples``. A block carrying ``sample_count: 27000`` and no
    # recognised key measured nothing this reader can report, and crediting it would put a large sample count beside a
    # row of nulls.
    contributing = [w for b, w in zip(blocks, weights) if any(_source(b, keys) is not None for keys in _ALL_KEYS)]

    def _avg(keys: tuple[str, ...]) -> float | None:
        """Sample-count-weighted mean of one metric, or ``None`` if unread."""
        total = 0.0
        weight_sum = 0.0
        for block, weight in zip(blocks, weights):
            value = _metric(block, keys, "avg")
            if value is None:
                continue
            total += value * weight
            weight_sum += weight
        return round(total / weight_sum, 2) if weight_sum else None

    def _max(keys: tuple[str, ...]) -> float | None:
        """Peak of one metric across all blocks, or ``None`` if unread."""
        values = [v for b in blocks if (v := _metric(b, keys, "max")) is not None]
        return round(max(values), 2) if values else None

    return {
        "schema_version": 1,
        "samples": round(sum(contributing)),
        "blocks": len(blocks),
        "avg_power_w": _avg(_POWER_KEYS),
        "max_power_w": _max(_POWER_KEYS),
        "avg_temp_c": _avg(_TEMP_KEYS),
        "max_temp_c": _max(_TEMP_KEYS),
        "avg_clock_mhz": _avg(_CLOCK_KEYS),
        "max_clock_mhz": _max(_CLOCK_KEYS),
        "avg_mem_clock_mhz": _avg(_MEM_CLOCK_KEYS),
        # Compute occupancy and memory pressure. Only the multi-node harvester reports these today -- Magpie's monitor
        # samples neither -- so a single-node round reports null for both. Null, not 0.0: "the GPU sat idle" and
        # "nobody asked" lead to opposite conclusions about a round that looks slow.
        "avg_gpu_util_pct": _avg(_UTIL_KEYS),
        "max_gpu_util_pct": _max(_UTIL_KEYS),
        "avg_vram_pct": _avg(_VRAM_KEYS),
        "max_vram_pct": _max(_VRAM_KEYS),
    }


def write_gpu_metrics(workspace: Path | str) -> str | None:
    """Normalise the round's GPU telemetry into ``gpu_metrics.json``. Returns the path, or ``None``.

    Best-effort throughout: this runs while a round is finishing, and a round that produced a good benchmark number and
    no GPU artifact is a far better outcome than one that failed here.
    """
    try:
        root = Path(workspace)
        report_path = root / "benchmark_report.json"
        if not report_path.is_file():
            return None
        payload = gpu_metrics_from_report(json.loads(report_path.read_text(encoding="utf-8")))
        if not payload:
            return None
        payload["source"] = report_path.name
        out = root / GPU_ARTIFACT_NAME

        from hyperloom.common.io import atomic_write_json

        atomic_write_json(out, payload)
        return str(out)
    except Exception:  # noqa: BLE001 - telemetry must never fail a round
        log.debug("gpu_metrics: could not write telemetry for %s", workspace, exc_info=True)
        return None
