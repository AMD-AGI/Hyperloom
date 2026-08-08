"""Build hot-kernel candidates from TraceLens' ``nccl_summary`` block.

TraceLens reports collective kernels in ``category_data/multi_kernel_metrics.json``
under ``nccl_summary``. That block is separate from the per-category
``operations`` lists that feed the deterministic hot-kernel extractor, and it
carries only ``{name, duration_us, stream}`` per sampled op -- no launcher path,
no category, no shapes. Collectives therefore never reach the candidate pool,
and the collective optimisation lane can only ever report
``no_collective_candidate`` even when exposed communication dominates the trace.

This module bridges that gap: it turns the summary into candidates shaped like
the deterministic extractor's output, resolving each mangled device symbol to
its ``__global__`` definition so the patchability gate and the forge collective
lane have a real source file to work with.

A candidate is emitted only when its source is located. An unresolved symbol
(vendor RCCL kernels in particular, which ship no source) is logged and dropped
rather than injected with an empty ``source_file``, which would send the
downstream grep fallback hunting for a definition that does not exist.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# Device-source extensions worth scanning for a __global__ definition.
_DEVICE_SOURCE_SUFFIXES = (".cuh", ".cu", ".hip", ".h", ".hpp", ".cpp")

# Cap the directory walk; collective symbols live in a handful of headers and an
# unbounded walk over a site-packages tree would dominate analysis time.
_MAX_SCANNED_FILES = 4000

_METRICS_RELPATH = "category_data/multi_kernel_metrics.json"


def _itanium_components(mangled: str) -> list[str]:
    """Split a nested Itanium-mangled name into its length-prefixed components.

    ``_ZN5aiter26cross_device_reduce_2stageI...`` yields
    ``["aiter", "cross_device_reduce_2stage"]``. Parsing stops at the first
    non-digit, which is where template arguments and parameter types begin.

    Args:
        mangled: A possibly mangled kernel symbol.

    Returns:
        The decoded namespace components, empty when the name is not a nested
        Itanium symbol.
    """
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
    """Extract the bare device-function symbol from a kernel name.

    Falls back to the name itself (minus any template or argument tail) when the
    name is not Itanium-mangled, so demangled trace names work too.

    Args:
        kernel_name: Kernel name as reported by TraceLens.

    Returns:
        The device-function symbol, or ``""`` when nothing usable is found.
    """
    name = (kernel_name or "").strip()
    if not name:
        return ""
    components = _itanium_components(name)
    if components:
        return components[-1]
    # Demangled form: drop template args, parameter list and namespace prefix.
    head = re.split(r"[<(]", name, maxsplit=1)[0]
    return head.rsplit("::", 1)[-1].strip()


def _iter_device_sources(roots: Iterable[str]) -> Iterable[Path]:
    """Yield device-source files under the given roots, bounded by a file cap.

    Args:
        roots: Directory roots to scan.

    Yields:
        Paths to candidate device-source files.
    """
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


def locate_device_symbol(symbol: str, roots: Sequence[str]) -> tuple[str, int, str] | None:
    """Locate the ``__global__`` definition of a device symbol.

    Matches ``symbol`` only when directly followed by an argument list, so
    ``cross_device_reduce_2stage`` does not bind to
    ``cross_device_reduce_2stage_naive``. A ``__global__``-qualified definition
    wins over a plain one; the qualifier may sit on a preceding line, so a small
    look-back window is used.

    Args:
        symbol: Bare device-function symbol.
        roots: Source roots to search.

    Returns:
        ``(source_file, line, function)`` for the best match, or ``None``.
    """
    if not symbol:
        return None
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\s*\(")
    fallback: tuple[str, int, str] | None = None
    for path in _iter_device_sources(roots):
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines):
            if not pattern.search(line):
                continue
            window = " ".join(lines[max(0, idx - 2) : idx + 1])
            if "__global__" in window:
                return (str(path), idx + 1, symbol)
            if fallback is None:
                fallback = (str(path), idx + 1, symbol)
    return fallback


def _prorated_totals(
    top_ops: Sequence[dict[str, Any]],
    total_time_ms: float,
    total_count: int,
) -> "OrderedDict[str, tuple[float, int, int]]":
    """Spread the summary totals across the sampled op names.

    ``top_ops`` is a slowest-N sample, not the full population, so its durations
    only give the relative weight of each distinct kernel. Those weights are
    applied to ``total_time_ms``/``total_count`` to approximate each kernel's
    real share. With a single distinct name the whole total lands on it.

    Args:
        top_ops: Sampled collective ops from ``nccl_summary``.
        total_time_ms: Total collective time reported by the summary.
        total_count: Total collective invocation count.

    Returns:
        Mapping of kernel name to ``(duration_us, call_count, stream)``,
        ordered by descending sampled weight.
    """
    sampled: "OrderedDict[str, float]" = OrderedDict()
    streams: dict[str, int] = {}
    for op in top_ops:
        if not isinstance(op, dict):
            continue
        name = str(op.get("name") or "").strip()
        if not name:
            continue
        try:
            dur = float(op.get("duration_us") or 0.0)
        except (TypeError, ValueError):
            dur = 0.0
        sampled[name] = sampled.get(name, 0.0) + dur
        if name not in streams:
            try:
                streams[name] = int(op.get("stream") or 0)
            except (TypeError, ValueError):
                streams[name] = 0
    weight_sum = sum(sampled.values())
    if weight_sum <= 0:
        return OrderedDict()
    total_us = max(0.0, float(total_time_ms)) * 1000.0
    out: "OrderedDict[str, tuple[float, int, int]]" = OrderedDict()
    for name, weight in sorted(sampled.items(), key=lambda kv: -kv[1]):
        share = weight / weight_sum
        out[name] = (
            round(total_us * share, 3),
            max(1, int(round(max(0, total_count) * share))),
            streams.get(name, 0),
        )
    return out


def extract_collective_candidates(
    tracelens_dir: Path | str,
    source_roots: Sequence[str],
    *,
    log_fn: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Build collective candidates from a TraceLens run's ``nccl_summary``.

    Args:
        tracelens_dir: TraceLens output directory.
        source_roots: Device-source roots used to resolve kernel symbols.
        log_fn: Optional sink for messages about dropped symbols.

    Returns:
        Candidate dicts shaped like the deterministic extractor's output, sorted
        by descending duration. Empty when the summary is absent or unusable.
    """
    metrics_path = Path(tracelens_dir) / _METRICS_RELPATH
    if not metrics_path.is_file():
        return []
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    summary = metrics.get("nccl_summary")
    if not isinstance(summary, dict):
        return []
    top_ops = summary.get("top_ops")
    if not isinstance(top_ops, list) or not top_ops:
        return []
    try:
        total_time_ms = float(summary.get("total_time_ms") or 0.0)
    except (TypeError, ValueError):
        total_time_ms = 0.0
    try:
        total_count = int(summary.get("total_count") or 0)
    except (TypeError, ValueError):
        total_count = 0
    if total_time_ms <= 0:
        return []

    roots = [r for r in source_roots if r]
    candidates: list[dict[str, Any]] = []
    for name, (duration_us, call_count, stream) in _prorated_totals(
        top_ops, total_time_ms, total_count
    ).items():
        symbol = collective_symbol(name)
        located = locate_device_symbol(symbol, roots)
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
                "kernel_path": "",
                "tracelens_launcher_path": "",
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
                # top_ops is a slowest-N sample, so a kernel that dominates the
                # sample absorbs the share of any collective that never made the
                # cut. Treat the resulting duration as an upper bound.
                "duration_provenance": "nccl_summary_prorated_from_top_ops_sample",
            }
        )
    candidates.sort(key=lambda c: c.get("duration_us", 0.0), reverse=True)
    return candidates
