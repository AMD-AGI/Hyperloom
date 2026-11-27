###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Shared classifier for CUDA-graph capture sidecars vs workload traces."""

from __future__ import annotations

import re
from pathlib import Path

from inference_suite.contracts.layout import CAPTURE_TRACE_DIRS

#: Directory shapes a profile writes capture sidecars into, matched per path
#: component: a run_dir's capture directory exactly, or a name *starting* with
#: ``graph_capture`` (a bare ``graph_capture`` directory). Anchored,
#: unlike the filename rule below: a directory is named for what it holds, so an
#: unanchored token would also condemn e.g. ``torch_profiler_with_graph_capture/``
#: -- and because the capture-only preflight is an ``all(...)``, one such false
#: positive rejects the entire input.
CAPTURE_DIR_RE = re.compile(
    "|".join([*(rf"\A{re.escape(name)}\Z" for name in CAPTURE_TRACE_DIRS), r"\Agraph_capture"]), re.IGNORECASE
)

#: Sidecar filename shapes: ``bs_<batch>[_rank<n>]`` anchored to the start, and
#: ``graph_capture`` anywhere in the name (an unpatched SGLang prefixes it with
#: ``cuda_``). The batch number is required rather than a bare ``bs_`` prefix
#: because this classifier can *reject* an input rather than only sort it, and a
#: real trace that merely starts with those three characters must not be thrown
#: out. ``graph_capture`` needs no such guard: a workload trace is not named
#: after graph capture.
CAPTURE_FRAGMENT_RE = re.compile(r"\Abs_\d+|graph_capture", re.IGNORECASE)


def is_capture_dir_name(name: str) -> bool:
    """Whether one path component names a graph-capture output directory."""
    return CAPTURE_DIR_RE.search(name) is not None


def is_capture_fragment(path: str | Path, root: str | Path | None = None) -> bool:
    """Whether a trace path is a CUDA-graph capture sidecar."""
    resolved = Path(path)
    if CAPTURE_FRAGMENT_RE.search(resolved.name) is not None:
        return True
    parts: tuple[str, ...] = resolved.parts
    if root is not None:
        try:
            parts = resolved.relative_to(root).parts
        except ValueError:
            parts = resolved.parts
    return any(is_capture_dir_name(part) for part in parts)


def _holds_capture_traces(folder: Path) -> bool:
    # TraceLens reads every .json / .json.gz in the capture folder as a capture trace, and when there is none it
    # exits 0 without writing any report.
    try:
        return any(child.is_file() and child.name.endswith((".json", ".json.gz")) for child in folder.iterdir())
    except OSError:
        return False


def discover_capture_folder(trace_input: Path, trace_files: list[Path]) -> Path | None:
    """Find a graph-capture folder near a torch_trace input.

    Scans the trace input directory and the first trace file's neighbourhood for
    a subdirectory whose name matches the shared capture-directory shape, so a
    layout that ranking already demotes is also a layout discovery can find.
    Matching by shape rather than by two hard-coded names is what lets an
    unpatched SGLang's ``graph_capture_profile/`` through, so the capture folder
    is passed on whenever the workload trace was picked. A folder with no capture
    trace in it is passed over: the capture path fired but wrote nothing, and
    the workload trace is still analysable without one.

    Args:
        trace_input (Path): The trace input path (file or directory).
        trace_files (list[Path]): Discovered trace files; only the first is used.

    Returns:
        Path | None: The capture folder if one holding a capture trace exists nearby, else ``None``.
    """

    capture_dir_priority = ("capture_traces", "graph_capture_profile", "graph_capture")
    search_roots: list[Path] = []
    if trace_input.is_dir():
        search_roots.append(trace_input)
    for trace_file in trace_files[:1]:
        search_roots.extend([trace_file.parent, trace_file.parent.parent])
    seen: set[Path] = set()
    for root in search_roots:
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        try:
            children = [child for child in root.iterdir() if child.is_dir() and _holds_capture_traces(child)]
        except OSError:
            continue
        children_by_name = {child.name.lower(): child for child in children}
        for name in capture_dir_priority:
            if child := children_by_name.get(name):
                return child
        for child in sorted(children):
            if is_capture_dir_name(child.name):
                return child
    return None
