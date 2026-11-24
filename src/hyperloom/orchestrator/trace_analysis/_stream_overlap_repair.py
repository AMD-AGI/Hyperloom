###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Truncate device events whose recorded duration runs past the next event on the same stream."""

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._analysis_run import AnalysisRun
from ._io_utils import append_log
from ._run_status import note_step
from ._trace_reader import _GPU_CATS, _STREAM_OVERLAP_EPS_US


def clamp_stream_overlaps(trace_path: Path) -> dict[str, Any]:
    """Clamp each GPU event to end at the next event's start on its (pid, tid) stream; rewrite the trace if changed."""
    opener: Callable[..., Any] = gzip.open if trace_path.name.endswith(".gz") else open
    with opener(trace_path, "rt", encoding="utf-8") as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace

    streams: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        if (
            isinstance(ev, dict)
            and ev.get("ph") == "X"
            and ev.get("cat") in _GPU_CATS
            and isinstance(ev.get("ts"), (int, float))
            and isinstance(ev.get("dur"), (int, float))
        ):
            streams[(ev.get("pid"), ev.get("tid"))].append(ev)

    clamped = 0
    removed_us = 0.0
    worst_name = ""
    worst_us = 0.0
    for stream in streams.values():
        stream.sort(key=lambda ev: ev["ts"])
        for cur, nxt in zip(stream, stream[1:]):
            gap = nxt["ts"] - cur["ts"]
            over = cur["dur"] - gap
            if gap <= 0 or over <= _STREAM_OVERLAP_EPS_US:
                continue
            cur["dur"] = gap
            clamped += 1
            removed_us += over
            if over > worst_us:
                worst_name, worst_us = str(cur.get("name", "")), over

    if clamped:
        tmp_path = trace_path.with_name(trace_path.name + ".tmp")
        with opener(tmp_path, "wt", encoding="utf-8") as fh:
            json.dump(trace, fh)
        os.replace(tmp_path, trace_path)

    return {
        "clamped_events": clamped,
        "removed_ms": round(removed_us / 1000.0, 4),
        "worst_event": worst_name,
        "worst_removed_ms": round(worst_us / 1000.0, 4),
    }


def clamp_chunk_stream_overlaps(run: AnalysisRun, chunk_path: Path) -> None:
    """Truncate same-stream overrunning GPU events in the steady-state chunk when ``--clamp-stream-overlaps`` is set."""
    if not run.args.clamp_stream_overlaps:
        return
    try:
        summary = clamp_stream_overlaps(chunk_path)
    except (OSError, ValueError) as exc:
        append_log(run.log_path, f"stream overlap clamp skipped for {chunk_path.name}: {exc}")
        note_step(run.run_meta, "clamp_stream_overlaps", category="split", status="skipped", skip_reason=str(exc))
        return
    append_log(
        run.log_path,
        f"stream overlap clamp: {summary['clamped_events']} event(s), {summary['removed_ms']} ms removed "
        f"(worst {summary['worst_event'] or '-'} {summary['worst_removed_ms']} ms) in {chunk_path.name}",
    )
    note_step(run.run_meta, "clamp_stream_overlaps", category="split", status="ok", **summary)
