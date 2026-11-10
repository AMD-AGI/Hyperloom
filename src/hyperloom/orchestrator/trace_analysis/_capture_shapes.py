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
