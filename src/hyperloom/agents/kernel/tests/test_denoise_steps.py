###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the shared diffusion per-denoise-step divisor helpers."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _denoise_steps as ds  # noqa: E402


def test_divisor_prefers_inferred_over_requested():
    # The bug: per-step divided by the requested full schedule (20) instead of
    # the steps actually in the analyzed window (8). Inferred must win.
    assert ds.resolve_perstep_divisor(8, 20) == 8
    assert ds.resolve_perstep_divisor(16, 0) == 16


def test_divisor_falls_back_to_requested_when_no_inferred():
    assert ds.resolve_perstep_divisor(0, 20) == 20
    assert ds.resolve_perstep_divisor(None, 20) == 20


def test_divisor_none_when_nothing_known():
    assert ds.resolve_perstep_divisor(0, 0) is None
    assert ds.resolve_perstep_divisor(None, None) is None


def _write_trace(path: Path, names: list[str], gz: bool):
    ev = {"traceEvents": [{"name": n, "ph": "X", "ts": i, "dur": 1} for i, n in enumerate(names)]}
    raw = json.dumps(ev).encode("utf-8")
    if gz:
        with gzip.open(path, "wb") as fh:
            fh.write(raw)
    else:
        path.write_bytes(raw)


def test_count_profiler_steps_gz_and_plain(tmp_path):
    names = ["ProfilerStep#1", "aten::mm", "ProfilerStep#2", "ProfilerStep#2", "ProfilerStep#3"]
    gzp = tmp_path / "t.pt.trace.json.gz"
    _write_trace(gzp, names, gz=True)
    assert ds.count_profiler_steps(str(gzp)) == 3  # distinct #1/#2/#3
    plain = tmp_path / "t.json"
    _write_trace(plain, names, gz=False)
    assert ds.count_profiler_steps(str(plain)) == 3


def test_count_profiler_steps_across_chunk_boundary(tmp_path, monkeypatch):
    # Tiny chunks force ProfilerStep#N markers to straddle chunk boundaries; the
    # overlap carry must still match them and the set must not double-count.
    monkeypatch.setattr(ds, "_CHUNK_BYTES", 8)
    names = ["ProfilerStep#1", "aten::mm", "ProfilerStep#2", "ProfilerStep#3", "ProfilerStep#2"]
    gzp = tmp_path / "t.pt.trace.json.gz"
    _write_trace(gzp, names, gz=True)
    assert ds.count_profiler_steps(str(gzp)) == 3


def test_count_profiler_steps_none_and_dir(tmp_path):
    no_steps = tmp_path / "n.json"
    _write_trace(no_steps, ["aten::mm", "aten::add"], gz=False)
    assert ds.count_profiler_steps(str(no_steps)) == 0
    # directory input picks the first trace file
    d = tmp_path / "run"
    d.mkdir()
    _write_trace(d / "x.pt.trace.json.gz", ["ProfilerStep#5", "ProfilerStep#6"], gz=True)
    assert ds.count_profiler_steps(str(d)) == 2
    assert ds.count_profiler_steps(str(tmp_path / "missing.json")) == 0
