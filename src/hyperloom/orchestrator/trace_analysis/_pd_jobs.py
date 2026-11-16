# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One TraceLens job per PD server node, for a multi-trace analysis aggregated with ``combine_by_node``.

Each server contributes the first GPU's trace of each node. Decode traces go through the TraceLens splitter and the
decode-only steady-state chunk (else the generic one), stream-overlap clamped, is the job's trace; prefill traces are
analysed whole. Decode jobs come first, so the first decode node is the focus job. Every job uses its own server's
graph-capture folder when that folder holds files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ._io_utils import append_log
from ._stream_overlap_repair import clamp_stream_overlaps
from ._trace_split import DECODE_ONLY_OR_GENERIC_MODE, _collect_split_chunks, _validate_selected_chunk
from ._tracelens_checkout import run_command

SPLIT_NUM_STEPS = 32
_CAPTURE_DIR_NAMES = ("capture_traces", "graph_capture_profile", "graph_capture")
_PHASE_ORDER = ("decode", "prefill")
_SPLIT_TIMEOUT_S = 1800


def role_capture_folder(trace_root: Path, role: str) -> str:
    """The graph-capture folder ``role``'s server wrote under ``trace_root``; '' when none holds files."""
    for name in _CAPTURE_DIR_NAMES:
        folder = trace_root / role / name
        if folder.is_dir() and any(path.is_file() for path in folder.iterdir()):
            return str(folder)
    return ""


def pd_node_name(phase: str, node: int) -> str:
    """The TraceLens node id of one PD job, e.g. ``decode_0``."""
    return f"{phase}_{node}"


def _decode_chunk(
    trace_path: Path,
    split_dir: Path,
    *,
    tracelens_root: Path,
    log_path: Path,
    mode: str = DECODE_ONLY_OR_GENERIC_MODE,
    health_warnings: list[dict[str, Any]] | None = None,
) -> Path:
    """Split one decode node's trace and return its stream-overlap-clamped chunk for ``mode``.

    With ``health_warnings`` the chunk also passes the low-quality gate, and every refusal is recorded there.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "TraceLens.TraceUtils.split_trace.main",
        str(trace_path),
        "-o",
        str(split_dir),
        "--find-steady-state",
        "--num-steps",
        str(SPLIT_NUM_STEPS),
    ]
    rc = run_command(cmd, cwd=tracelens_root, log_path=log_path, timeout_s=_SPLIT_TIMEOUT_S)
    if rc != 0:
        raise RuntimeError(f"trace_split_failed: TraceLens splitter exited with code {rc} for {trace_path}")
    warnings = health_warnings if health_warnings is not None else []
    chunk, _meta, mode_to_chunks = _collect_split_chunks(split_dir, mode, trace_path, warnings)
    if health_warnings is not None:
        _validate_selected_chunk(split_dir, chunk, mode, mode_to_chunks, health_warnings)
    try:
        summary = clamp_stream_overlaps(chunk)
    except (OSError, ValueError) as exc:
        append_log(log_path, f"stream overlap clamp skipped for {chunk}: {exc}")
    else:
        append_log(log_path, f"stream overlap clamp: {summary['clamped_events']} event(s) in {chunk}")
    return chunk


def pd_report_jobs(
    node_traces: dict[str, list[str]],
    *,
    trace_root: Path,
    output_dir: Path,
    platform: str,
    tracelens_root: Path,
    log_path: Path,
    focus_mode: str = DECODE_ONLY_OR_GENERIC_MODE,
    health_warnings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One job per server node, decode first: decode nodes as split chunks, prefill nodes as captured.

    The first decode node's chunk is selected with ``focus_mode`` and gated for quality, its refusals recorded in
    ``health_warnings``; every other decode node takes the decode-only or generic chunk.
    """
    jobs: list[dict[str, Any]] = []
    for phase in _PHASE_ORDER:
        capture_folder = role_capture_folder(trace_root, phase)
        for node, trace in enumerate(node_traces.get(phase) or []):
            trace_path = Path(trace)
            if phase == "decode":
                focus = node == 0
                trace_path = _decode_chunk(
                    trace_path,
                    output_dir / "trace_split" / pd_node_name(phase, node),
                    tracelens_root=tracelens_root,
                    log_path=log_path,
                    mode=focus_mode if focus else DECODE_ONLY_OR_GENERIC_MODE,
                    health_warnings=health_warnings if focus else None,
                )
            job: dict[str, Any] = {"phase": phase, "node": node, "trace_path": str(trace_path), "platform": platform}
            if capture_folder:
                job["capture_folder"] = capture_folder
            jobs.append(job)
    return jobs
