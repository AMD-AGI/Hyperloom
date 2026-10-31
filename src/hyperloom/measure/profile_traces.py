# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a ``profile`` test leaves its torch-profiler traces, so the signed bundle carries them and an analysis finds them.

The profiled round writes its traces below the round's workspace (``torch_trace``, ``capture_traces``). The backend moves
them into ``<output_dir>/profile`` the way the old flow reads them, keeping the directory names so the capture-sidecar
detection of the trace analysis still works. The result's ``profile_location`` is that directory, relative to the
session, and the bundle's manifest lists its files, so the location is verifiable.

A profile whose traces are all CUDA-graph capture sidecars (``graph_capture_profile/``, ``capture_traces/``) recorded the
graphs the server captured at boot and no serving step: the profiler of the round never started. Measured on a NEXTN
speculative-decoding head, 11 of 11 profiles were of that shape (the run ended before the step the capture window starts
at), and an analysis of such a trace can only fail; :func:`has_step_traces` tells the two apart.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

from hyperloom.orchestrator.trace_analysis import is_capture_fragment

__all__ = ["PROFILE_DIR", "collect_traces", "has_step_traces", "has_traces", "is_trace_file", "round_has_traces"]

#: The directory, below a test's output directory, that holds the traces of a ``profile`` test.
PROFILE_DIR = "profile"
#: Directories a Magpie profile round writes its traces into, relative to the round's workspace.
_TRACE_DIRS = ("torch_trace", "capture_traces")
_TRACE_SUFFIXES = (".json", ".json.gz")


def _candidates(workspace: Path) -> Iterator[Path]:
    for name in _TRACE_DIRS:
        yield workspace / name
    yield workspace.parent / "capture_traces"


def is_trace_file(name: str) -> bool:
    """Whether a file name is a torch-profiler trace."""
    return name.endswith(_TRACE_SUFFIXES)


def _trace_files(root: Path) -> Iterator[Path]:
    if root.is_dir():
        yield from (p for p in root.rglob("*") if p.is_file() and is_trace_file(p.name))


def has_traces(directory: Path) -> bool:
    """Whether ``directory`` holds at least one trace file."""
    return any(_trace_files(Path(directory)))


def has_step_traces(directory: Path) -> bool:
    """Whether ``directory`` holds a trace of serving steps: a trace file that is not a CUDA-graph capture sidecar."""
    root = Path(directory)
    return any(not is_capture_fragment(path, root) for path in _trace_files(root))


def round_has_traces(workspace: Path) -> bool:
    """Whether the round at ``workspace`` left a trace where :func:`collect_traces` looks, without moving it."""
    return any(has_traces(source) for source in _candidates(Path(workspace)))


def collect_traces(workspace: Path, destination: Path) -> int:
    """Move the trace directories of the round at ``workspace`` below ``destination``; return how many it moved.

    A directory that holds no trace file is left where it is. A stale directory of the same name in ``destination``
    (an earlier attempt of the same test) is replaced, never merged.
    """
    moved = 0
    for source in _candidates(Path(workspace)):
        if not has_traces(source):
            continue
        target = Path(destination) / source.name
        if target.exists():
            shutil.rmtree(target)
        Path(destination).mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved += 1
    return moved
