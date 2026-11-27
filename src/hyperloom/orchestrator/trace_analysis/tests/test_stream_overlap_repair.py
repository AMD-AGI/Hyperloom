###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis import _stream_overlap_repair, tracelens_analysis as tla
from hyperloom.orchestrator.trace_analysis._analysis_run import start_run
from hyperloom.orchestrator.trace_analysis._stream_overlap_repair import clamp_stream_overlaps
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute


def _kernel(name: str, ts: float, dur: float, tid: int = 6, cat: str = "kernel") -> dict:
    return {"ph": "X", "cat": cat, "name": name, "pid": 2, "tid": tid, "ts": ts, "dur": dur, "args": {"stream": tid}}


def _write(path: Path, events: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"traceEvents": events}, fh)


def _durations(path: Path) -> dict[str, float]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {ev["name"]: ev["dur"] for ev in json.load(fh)["traceEvents"]}


def test_kernel_overrunning_later_kernels_on_its_stream_is_truncated(tmp_path: Path) -> None:
    trace = tmp_path / "chunk.trace.json.gz"
    _write(
        trace,
        [
            _kernel("index_elementwise_kernel", 1000.0, 533_100.0),
            _kernel("gemm", 1004.5, 10.0),
            _kernel("copy", 1020.0, 3.0, cat="gpu_memcpy"),
            _kernel("other_stream", 1001.0, 50.0, tid=7),
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::_index_put_impl_",
                "pid": 1,
                "tid": 1,
                "ts": 990.0,
                "dur": 20.0,
            },
        ],
    )

    summary = clamp_stream_overlaps(trace)

    assert _durations(trace) == {
        "index_elementwise_kernel": 4.5,
        "gemm": 10.0,
        "copy": 3.0,
        "other_stream": 50.0,
        "aten::_index_put_impl_": 20.0,
    }
    assert summary["clamped_events"] == 1
    assert summary["worst_event"] == "index_elementwise_kernel"


def test_sub_microsecond_rounding_overlap_leaves_the_trace_untouched(tmp_path: Path) -> None:
    trace = tmp_path / "chunk.trace.json.gz"
    _write(trace, [_kernel("a", 0.0, 10.5), _kernel("b", 10.0, 5.0)])
    before = trace.read_bytes()

    summary = clamp_stream_overlaps(trace)

    assert summary["clamped_events"] == 0
    assert trace.read_bytes() == before


@pytest.mark.parametrize(("argv", "clamped"), [([], False), (["--clamp-stream-overlaps"], True)])
def test_steady_state_chunk_is_clamped_only_when_asked(tmp_path: Path, monkeypatch, argv, clamped) -> None:
    seen: list[Path] = []
    summary = {"clamped_events": 0, "removed_ms": 0.0, "worst_event": "", "worst_removed_ms": 0.0}
    monkeypatch.setattr(_stream_overlap_repair, "clamp_stream_overlaps", lambda p: seen.append(p) or summary)
    args = tla._build_parser().parse_args(
        ["--trace-input", str(tmp_path / "t.json.gz"), "--workspace-path", str(tmp_path / "ws"), *argv]
    )
    chunk = tmp_path / "chunk.json.gz"

    _stream_overlap_repair.clamp_chunk_stream_overlaps(start_run(args, AnalysisRoute.AGENT), chunk)

    assert seen == ([chunk] if clamped else [])
