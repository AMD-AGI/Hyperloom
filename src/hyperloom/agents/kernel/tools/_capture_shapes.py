###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Shared classifier for CUDA-graph capture sidecars vs workload traces.

Single source of truth for the trace-analysis pipeline, so a capture sidecar is
recognised identically no matter where in the pipeline the profile is read.

A capture sidecar is recorded while the CUDA/HIP graph is being *built*, so its
launches go into the graph instead of onto the device. What lands is a host-side
call tree with a handful of stray kernels and no iteration loop, which is
useless to a steady-state splitter and must never be mistaken for the workload
trace beside it.

Matching is by *shape*, not by an exact name, because the profile's layout
varies with framework and patch level: an SGLang carrying Hyperloom's profiler
patch writes ``capture_traces/bs_<batch>_rank<n>``, an unpatched one writes
``graph_capture_profile/cuda_graph_capture-<runner>-TP-<n>``, and vLLM writes
``graph_capture_*``. An exact-name whitelist silently misses the shapes it has
not been taught, and a missed sidecar is not merely ranked late: it shares the
default discovery bucket with the workload trace, where the tie-break is
descending file size, and a capture is the larger file.

Kept dependency-free (stdlib only) so it can be imported anywhere in the
pipeline without pulling in TraceLens.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Directory shapes a profile writes capture sidecars into, matched per path
#: component: ``capture_traces`` exactly, or a name *starting* with
#: ``graph_capture`` (``graph_capture``, ``graph_capture_profile``), which
#: covers every layout observed. Anchored, unlike the filename rule below: a
#: directory is named for what it holds, so an unanchored token would also
#: condemn e.g. ``torch_profiler_with_graph_capture/`` -- and because the
#: capture-only preflight is an ``all(...)``, one such false positive rejects
#: the entire input.
CAPTURE_DIR_RE = re.compile(r"\Acapture_traces\Z|\Agraph_capture", re.IGNORECASE)

#: Sidecar filename shapes: ``bs_<batch>[_rank<n>]`` anchored to the start, and
#: ``graph_capture`` anywhere in the name (an unpatched SGLang prefixes it with
#: ``cuda_``). The batch number is required rather than a bare ``bs_`` prefix
#: because this classifier can *reject* an input rather than only sort it, and a
#: real trace that merely starts with those three characters must not be thrown
#: out. ``graph_capture`` needs no such guard: a workload trace is not named
#: after graph capture.
CAPTURE_FRAGMENT_RE = re.compile(r"\Abs_\d+|graph_capture", re.IGNORECASE)


def is_capture_dir_name(name: str) -> bool:
    """Whether one path component names a graph-capture output directory.

    Exposed separately from :func:`is_capture_fragment` because discovery has to
    *find* the capture folder to hand it to TraceLens as ``--capture_folder``,
    not merely recognise files already under it. Both answers come from one
    pattern so a layout that is demoted during ranking is also the layout that
    gets located here.

    Args:
        name: A single path component.

    Returns:
        True when the component names a capture directory.
    """
    return CAPTURE_DIR_RE.search(name) is not None


def is_capture_fragment(path: str | Path, root: str | Path | None = None) -> bool:
    """Whether a trace path is a CUDA-graph capture sidecar.

    Two signals, because either alone has a blind spot: the directory name
    catches a sidecar whose filename nobody has seen before, and the filename
    catches a flat layout that never made the directory.

    ``root`` bounds the directory test to the input being analysed. Paths arrive
    absolute, so testing every component would condemn every candidate whenever
    some ancestor happened to be named after graph capture. Callers pass the
    input directory, or a file's parent so a single-file input is judged on its
    name alone.

    Args:
        path: The trace file path to classify.
        root: Directory the classification is relative to, when known.

    Returns:
        True when ``path`` is a graph-capture sidecar rather than a workload
        trace.
    """
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
