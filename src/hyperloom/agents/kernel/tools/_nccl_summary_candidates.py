# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Create source-resolved collective candidates from TraceLens NCCL metrics."""

from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

_DEVICE_SOURCE_SUFFIXES = (".cuh", ".cu", ".hip", ".h", ".hpp", ".cpp")
_MAX_SCANNED_FILES = 4000

_METRICS_RELPATH = "category_data/multi_kernel_metrics.json"


def _itanium_components(mangled: str) -> list[str]:
    """Split a nested Itanium-mangled name into symbol components."""
    if not mangled.startswith("_ZN"):
        return []
    out: list[str] = []
    i = 3
    while i < len(mangled) and mangled[i].isdigit():
        j = i
        while j < len(mangled) and mangled[j].isdigit():
            j += 1
        length = int(mangled[i:j])
        if length <= 0 or j + length > len(mangled):
            break
        out.append(mangled[j : j + length])
        i = j + length
    return out


def collective_symbol(kernel_name: str) -> str:
    """Extract the bare device symbol from a trace name."""
    name = (kernel_name or "").strip()
    if not name:
        return ""
    components = _itanium_components(name)
    if components:
        return components[-1]
    head = re.split(r"[<(]", name, maxsplit=1)[0]
    return head.rsplit("::", 1)[-1].strip()


def _iter_device_sources(roots: Iterable[str]) -> Iterable[Path]:
    """Yield at most the configured number of device-source files."""
    seen = 0
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if seen >= _MAX_SCANNED_FILES:
                return
            if path.suffix.lower() in _DEVICE_SOURCE_SUFFIXES and path.is_file():
                seen += 1
                yield path


def collect_device_sources(roots: Sequence[str]) -> tuple[list[Path], bool]:
    """Return the device sources to scan, and whether the cap truncated them.

    Every symbol scans the same tree, so walking it once turns the lookup from
    ``O(symbols x files)`` into one pass. The truncation flag matters because a
    capped scan and a genuinely absent symbol both end in ``None``.
    """
    sources = list(_iter_device_sources(roots))
    return sources, len(sources) >= _MAX_SCANNED_FILES


def locate_device_symbol(
    symbol: str,
    roots: Sequence[str],
    *,
    sources: Sequence[Path] | None = None,
) -> tuple[str, int, str] | None:
    """Locate the ``__global__`` definition of a device symbol."""
    if not symbol:
        return None
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\s*\(")
    scanned = sources if sources is not None else _iter_device_sources(roots)
    for path in scanned:
        try:
            lines = Path(path).read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines):
            if not pattern.search(line):
                continue
            window = " ".join(lines[max(0, idx - 2) : idx + 1])
            if "__global__" in window:
                return (str(path), idx + 1, symbol)
    return None


def _prorated_totals(
    top_ops: Sequence[dict[str, Any]],
    total_time_ms: float,
    total_count: int,
) -> "OrderedDict[str, tuple[float, int, int]]":
    """Prorate summary totals by sampled kernel duration.

    Returns:
        Kernel name -> ``(duration_us, call_count, stream)``, ordered by
        descending sampled duration.
    """
    if (
        isinstance(total_time_ms, bool)
        or not isinstance(total_time_ms, (int, float))
        or not math.isfinite(float(total_time_ms))
        or total_time_ms <= 0
    ):
        raise ValueError(
            "nccl_summary.total_time_ms must be finite and positive"
        )
    if (
        isinstance(total_count, bool)
        or not isinstance(total_count, int)
        or total_count <= 0
    ):
        raise ValueError("nccl_summary.total_count must be a positive integer")
    sampled: "OrderedDict[str, float]" = OrderedDict()
    streams: dict[str, int] = {}
    for index, op in enumerate(top_ops):
        if not isinstance(op, dict):
            raise ValueError(f"nccl_summary.top_ops[{index}] must be an object")
        name_raw = op.get("name")
        if not isinstance(name_raw, str) or not name_raw.strip():
            raise ValueError(f"nccl_summary.top_ops[{index}] has no name")
        name = name_raw.strip()
        duration_raw = op.get("duration_us")
        stream_raw = op.get("stream", 0)
        if (
            isinstance(duration_raw, bool)
            or not isinstance(duration_raw, (int, float))
            or not math.isfinite(float(duration_raw))
            or duration_raw < 0
            or isinstance(stream_raw, bool)
            or not isinstance(stream_raw, int)
        ):
            raise ValueError(f"invalid nccl_summary.top_ops[{index}]")
        dur = float(duration_raw)
        stream = stream_raw
        sampled[name] = sampled.get(name, 0.0) + dur
        streams.setdefault(name, stream)
    weight_sum = sum(sampled.values())
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("nccl_summary.top_ops must have positive sampled duration")
    total_us = float(total_time_ms) * 1000.0
    if not math.isfinite(total_us):
        raise ValueError("nccl_summary.total_time_ms is too large")
    out: "OrderedDict[str, tuple[float, int, int]]" = OrderedDict()
    for name, weight in sorted(sampled.items(), key=lambda kv: -kv[1]):
        share = weight / weight_sum
        out[name] = (
            round(total_us * share, 3),
            max(1, int(round(total_count * share))),
            streams.get(name, 0),
        )
    return out


def extract_collective_candidates(
    tracelens_dir: Path | str,
    source_roots: Sequence[str],
    *,
    log_fn: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Build source-resolved candidates from TraceLens NCCL metrics."""
    metrics_path = Path(tracelens_dir) / _METRICS_RELPATH
    if not metrics_path.is_file():
        return []
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TraceLens metrics file: {metrics_path}") from exc
    if not isinstance(metrics, dict):
        raise ValueError(f"TraceLens metrics must be an object: {metrics_path}")
    summary = metrics.get("nccl_summary")
    if summary is None:
        return []
    if not isinstance(summary, dict):
        raise ValueError("nccl_summary must be an object")
    top_ops = summary.get("top_ops")
    if top_ops in (None, []):
        return []
    if not isinstance(top_ops, list):
        raise ValueError("nccl_summary.top_ops must be a list")
    total_time_raw = summary.get("total_time_ms")
    total_count = summary.get("total_count")
    prorated = _prorated_totals(
        top_ops,
        total_time_raw,
        total_count,
    )
    total_time_ms = float(total_time_raw)

    roots = [r for r in source_roots if r]
    sources, truncated = collect_device_sources(roots)
    if truncated and log_fn is not None:
        log_fn(
            f"nccl_summary: device-source scan stopped at {_MAX_SCANNED_FILES} "
            f"files under {', '.join(roots)}; a symbol reported missing below "
            "may simply lie past the cap"
        )
    candidates: list[dict[str, Any]] = []
    for name, (duration_us, call_count, stream) in prorated.items():
        symbol = collective_symbol(name)
        located = locate_device_symbol(symbol, roots, sources=sources)
        if located is None:
            if log_fn is not None:
                log_fn(
                    f"nccl_summary: no device source for symbol {symbol!r} "
                    f"({name[:60]}); dropping collective candidate"
                )
            continue
        source_file, source_line, source_function = located
        candidates.append(
            {
                "name": name,
                "duration_us": duration_us,
                "call_count": call_count,
                "efficiency_percent": 0.0,
                "impact_score": 0.0,
                "bound_type": "communication",
                "tracelens_category": "collective",
                "tracelens_pitem_rank": 0,
                # Both feed the invocation spec that forge-loop's task preparer
                # reads to author run_candidate; the device source is the only
                # launcher a summary row can attribute.
                "kernel_path": source_file,
                "tracelens_launcher_path": source_file,
                "source_file": source_file,
                "source_line": source_line,
                "source_function": source_function,
                "source_resolution_method": "nccl_summary_symbol_lookup",
                "shapes": [],
                "input_shapes": [],
                "library": "",
                "is_multigpu": True,
                "candidate_source": "nccl_summary",
                "collective_stream": stream,
                "nccl_summary_total_ms": round(total_time_ms, 3),
                # Prorated by each row's sampled duration share of summary time.
                "duration_provenance": "nccl_summary_prorated_from_top_ops_sample",
            }
        )
    candidates.sort(key=lambda c: c.get("duration_us", 0.0), reverse=True)
    return candidates
