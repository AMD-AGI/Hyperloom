# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a ``profile`` test leaves its torch-profiler traces, so the signed bundle carries them and an analysis finds them.

An inference_suite run_dir holds its traces in ``profile/`` (:meth:`RunLayout.profile_trace_dirs`); a Magpie round
writes them below its workspace (``torch_trace``, ``capture_traces``). The backend moves them into
``<output_dir>/profile``, keeping each directory's name rather than flattening them, so the capture-sidecar detection of
the trace analysis still tells a boot capture from a serving step. The result's ``profile_location`` is that directory,
relative to the session, and the bundle's manifest lists its files, so the location is verifiable.

A profile whose traces are all CUDA-graph capture sidecars (``graph_capture_profile/``, ``capture_traces/``) recorded the
graphs the server captured at boot and no serving step: the profiler of the round never started, because the run
ended before the step the capture window starts at (a speculative-decoding head, which takes several tokens per step,
gets there sooner). An analysis of such a trace can only fail; :func:`has_step_traces` tells the two apart.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

from hyperloom.orchestrator.trace_analysis import is_capture_fragment
from inference_suite.contracts import RunLayout
from inference_suite.contracts.layout import CAPTURE_TRACES_DIR, PROFILE_DIR, TORCH_TRACE_DIR

__all__ = ["PROFILE_DIR", "collect_traces", "has_step_traces", "has_traces", "is_trace_file", "round_has_traces"]

#: Directories a Magpie profile round writes its traces into, relative to its workspace; they go with Magpie (5.03).
_MAGPIE_TRACE_DIRS = (TORCH_TRACE_DIR, CAPTURE_TRACES_DIR)
_TRACE_SUFFIXES = (".json", ".json.gz")


def _candidates(workspace: Path) -> Iterator[Path]:
    yield from RunLayout(workspace).profile_trace_dirs()
    for name in _MAGPIE_TRACE_DIRS:
        yield workspace / name
    yield workspace.parent / CAPTURE_TRACES_DIR


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
    (an earlier attempt of the same test) is replaced, never merged. A run_dir's three trace directories land under
    their own names. A Magpie round's two ``capture_traces`` directories, in the workspace and beside it, both land in
    ``destination/capture_traces``: the second's files join the first's, each under its own name unless the first holds
    that name, when it takes a numbered one (a directory both hold is merged the same way); no trace is overwritten.
    """
    destination = Path(destination)
    moved = 0
    placed: set[str] = set()
    for source in _candidates(Path(workspace)):
        if not has_traces(source):
            continue
        target = destination / source.name
        if source.name in placed:
            _merge(source, target)
        else:
            if target.exists():
                shutil.rmtree(target)
            destination.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            placed.add(source.name)
        moved += 1
    return moved


def _merge(source: Path, target: Path) -> None:
    """Move what ``source`` holds to the same place below ``target``, then remove ``source``.

    A directory both hold is merged the same way; any other entry whose name ``target`` already holds (a file, or a file
    on one side and a directory on the other) is moved whole under a free name, so both are kept.
    """
    pending = [(source, target)]
    while pending:
        here, there = pending.pop()
        for entry in sorted(here.iterdir()):
            landing = there / entry.name
            if _real_dir(entry) and _real_dir(landing):
                pending.append((entry, landing))
            else:
                shutil.move(str(entry), str(_free(landing)))
    shutil.rmtree(source)


def _real_dir(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _free(path: Path) -> Path:
    """``path``, or when it is taken the first ``<stem>.<n><suffixes>`` that is not: the name keeps its leading part
    (what a capture sidecar is recognised by) and its suffixes (what a trace is)."""
    stem, dot, suffixes = path.name.partition(".")
    candidate, number = path, 0
    while candidate.exists() or candidate.is_symlink():
        number += 1
        candidate = path.with_name(f"{stem}.{number}{dot}{suffixes}")
    return candidate
