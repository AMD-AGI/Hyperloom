# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which trace files a run reads: discovery, capture-fragment classification, and the order they are tried in."""

from __future__ import annotations

import gzip
import json
import re
import zlib
from pathlib import Path
from typing import Any, NamedTuple

# Capture-vs-workload trace classification, shared across routes so a sidecar is
# recognised identically whichever backend reads the profile.
from ._capture_shapes import is_capture_fragment
from ._kernel_classification import is_kernel_event


def open_json(path: Path) -> dict[str, Any]:
    """Load a JSON file, transparently handling ``.gz`` compression.

    Args:
        path (Path): JSON or gzipped-JSON file to read.

    Returns:
        dict[str, Any]: The parsed JSON payload.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


#: What :func:`open_json` raises on a missing, truncated, or corrupt (possibly gzipped) trace.
UNREADABLE_TRACE_ERRORS: tuple[type[Exception], ...] = (OSError, EOFError, ValueError, zlib.error)


def count_gpu_kernel_events(trace_file: Path, max_events: int = 1_000_000) -> int:
    """Count GPU kernel events in a torch_profiler trace.

    Used as a pre-flight check for CPU-only traces. Counts only real GPU
    kernels via :func:`is_kernel_event`, not host-side wrappers.

    Args:
        trace_file: Path to the torch_profiler trace (JSON or ``.gz``).
        max_events: Early-exit cap on the number of events counted.

    Returns:
        The GPU kernel event count, or 0 when the trace is unreadable.
    """
    try:
        payload = open_json(trace_file)
    except UNREADABLE_TRACE_ERRORS:
        return 0
    events = payload.get("traceEvents") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return 0
    count = 0
    for ev in events:
        if isinstance(ev, dict) and is_kernel_event(ev):
            count += 1
            if count >= max_events:
                break
    return count


#: Categories that live on the device timeline. Everything else in a torch
#: trace -- ``cpu_op``, ``python_function``, ``user_annotation``,
#: ``cuda_runtime`` -- is host-side.
_SPLIT_DIR_NAME = "trace_split"

#: Per-phase fragment names the splitter emits. Matched as well as the directory
#: because a flat layout would otherwise leave them in the default bucket, and a
#: capture with eight ranks would then spend the whole probe budget on fragments
#: before reaching a rank file.
_PHASE_FRAGMENT_RE = re.compile(
    r"^(?:decode_only|mixed|prefill_only|prefilldecode)\w*_steady_state\w*"
    r"_rank_\d+\.trace\.json(?:\.gz)?$"
)


def _capture_classification_root(trace_input: Path) -> Path:
    """Root to classify candidates discovered from ``trace_input`` against.

    A directory input is its own root, so a ``capture_traces/`` component below
    it counts. A file input resolves to its parent, which leaves the filename as
    the only component and keeps an unrelated ancestor directory out of the
    decision.

    Args:
        trace_input: The ``--trace-input`` path.

    Returns:
        The directory candidate paths are made relative to.
    """
    return trace_input if trace_input.is_dir() else trace_input.parent


def _is_derived_trace(path: Path, root: Path | None = None) -> bool:
    """Whether a trace path is splitter output rather than a raw capture.

    Three shapes, all derived: anything under the splitter's own output
    directory, the per-iteration annotation sidecars it writes beside a capture,
    and the per-phase fragment names themselves. They are a few hundred bytes
    each and cover one phase of one iteration, so an analysis pointed at them
    describes a sliver of the run.

    The name test is not redundant with the directory test. Production nests
    these under ``trace_split/`` today -- 276 of 276 capture directories with
    fragments -- but the demotion should not depend on a layout that the
    splitter is free to change.

    ``root`` bounds the directory test to the capture being analysed. Paths
    arrive absolute, so testing every component would demote *every* candidate
    whenever an ancestor happened to be named ``trace_split`` -- pointing
    ``--trace-input`` inside a previous split, say. With all candidates in the
    same bucket the ordering collapses back to alphabetical and the original bug
    returns, which is a lot of damage for a coincidence of naming.
    """
    relative = path
    if root is not None:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
    if _SPLIT_DIR_NAME in relative.parts:
        return True
    if _PHASE_FRAGMENT_RE.match(path.name):
        return True
    return bool(re.search(r"trace_annotation_iteration_\d+", path.name))


def _trace_file_size(path: Path) -> int:
    """Size in bytes, or 0 when it cannot be read."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _count_kernels_if_readable(path: Path) -> tuple[bool, int]:
    """``(readable, kernel_count)`` for a candidate trace.

    :func:`count_gpu_kernel_events` answers 0 both for a trace with no GPU work
    and for one it could not parse. Those need different readers: the first is a
    profiler problem, the second is a truncated or corrupt file. Collapsing them
    is the same misdirection this module already sent people on once.

    Still counts through :func:`count_gpu_kernel_events`, so it remains the one
    place kernel events are recognised. The disambiguating parse runs only on a
    zero answer, which is the only answer that is ambiguous -- a trace with
    kernels in it demonstrably parsed.
    """
    count = count_gpu_kernel_events(path)
    if count:
        return True, count
    try:
        payload = open_json(path)
    except UNREADABLE_TRACE_ERRORS:
        return False, 0
    if not isinstance(payload, dict) or not isinstance(payload.get("traceEvents"), list):
        return False, 0
    return True, 0


def _trace_input_sort_key(path: Path, root: Path | None = None) -> tuple[int, int, str]:
    """Compute the discovery sort key for a trace input path.

    Prefers the merged annotated trace over rank/phase shards (the TraceLens
    splitter needs the large trace).

    Sorts by (priority bucket, -size, name). Size is bucket-independent: a real capture is
    orders of magnitude larger than any phase shard or sidecar, so descending size puts the
    right file first even for unrecognised fragment names.

    Args:
        path: The trace file path to rank.
        root: The capture directory being analysed, so the ``trace_split``
            component is looked for below it rather than anywhere in an
            absolute path.

    Returns:
        A ``(priority, -size, name)`` sort key (lower sorts first).
    """
    name = path.name
    size = _trace_file_size(path)
    if _is_derived_trace(path, root):
        return (4, -size, name)
    if name.startswith("merged-"):
        return (0, -size, name)
    if re.search(r"TP-\d+-DECODE\.trace\.json(?:\.gz)?$", name):
        return (2, -size, name)
    if is_capture_fragment(path, root):
        return (3, -size, name)
    return (1, -size, name)


class TraceInputs(NamedTuple):
    """The trace files a trace input resolves to."""

    input_type: str
    files: list[Path]


def discover_trace_inputs(trace_input: Path) -> TraceInputs:
    """Resolve a trace input path into a list of trace files.

    Accepts either a single trace file or a capture directory; directories
    are searched recursively for known trace extensions, deduplicated, and
    ordered via :func:`_trace_input_sort_key`.

    Args:
        trace_input (Path): A trace file or a capture directory.

    Returns:
        TraceInputs: ``("file", [path])`` for a single file or
            ``("capture_dir", paths)`` for a directory.

    Raises:
        FileNotFoundError: When the path does not exist or no trace files are
            found under a supplied directory.
    """
    if trace_input.is_file():
        return TraceInputs("file", [trace_input])
    if not trace_input.is_dir():
        raise FileNotFoundError(f"trace_input does not exist: {trace_input}")

    traces: list[Path] = []
    for pattern in ("*.json", "*.json.gz", "*.trace", "*.trace.json", "*.trace.json.gz"):
        traces.extend(sorted(trace_input.rglob(pattern)))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique = []
    for trace in traces:
        if trace not in seen:
            seen.add(trace)
            unique.append(trace)
    unique.sort(key=lambda p: _trace_input_sort_key(p, trace_input))
    if not unique:
        raise FileNotFoundError(f"no trace files found under capture directory: {trace_input}")
    return TraceInputs("capture_dir", unique)
