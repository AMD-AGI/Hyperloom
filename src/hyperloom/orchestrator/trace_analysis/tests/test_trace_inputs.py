# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a trace input resolves to, and what an unreadable candidate costs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis._trace_inputs import (
    _count_kernels_if_readable,
    _is_derived_trace,
    _trace_file_size,
    count_gpu_kernel_events,
    discover_trace_inputs,
)


def _kernel_event(name: str = "gemm") -> dict:
    return {"name": name, "cat": "kernel", "dur": 1.0}


def _write_trace(path: Path, events: list[dict]) -> Path:
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return path


def test_an_unreadable_trace_counts_no_kernels(tmp_path: Path):
    """A truncated or absent trace is zero kernels, not an exception the caller has to catch."""
    assert count_gpu_kernel_events(tmp_path / "gone.json") == 0
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"traceEvents": [', encoding="utf-8")
    assert count_gpu_kernel_events(truncated) == 0


def test_a_trace_without_an_event_list_counts_no_kernels(tmp_path: Path):
    """``traceEvents`` has to be a list: a dict or a missing key is a trace with nothing on the device."""
    shaped_wrong = tmp_path / "wrong.json"
    shaped_wrong.write_text(json.dumps({"traceEvents": {"not": "a list"}}), encoding="utf-8")
    assert count_gpu_kernel_events(shaped_wrong) == 0


def test_the_kernel_count_stops_at_its_cap(tmp_path: Path):
    """The count is a pre-flight probe, so it stops at ``max_events`` rather than reading a whole capture."""
    trace = _write_trace(tmp_path / "many.json", [_kernel_event() for _ in range(10)])
    assert count_gpu_kernel_events(trace, max_events=3) == 3


def test_a_gzipped_trace_is_read_like_a_plain_one(tmp_path: Path):
    """Captures arrive gzipped as often as not, and the probe reads both."""
    trace = tmp_path / "capture.json.gz"
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": [_kernel_event(), _kernel_event()]}, handle)
    assert count_gpu_kernel_events(trace) == 2


def test_a_size_that_cannot_be_read_is_zero(tmp_path: Path):
    """Size orders the candidates, so a path that cannot be stat'd sorts last instead of failing the run."""
    assert _trace_file_size(tmp_path / "gone.json") == 0


def test_an_unreadable_candidate_is_not_readable(tmp_path: Path):
    """``_count_kernels_if_readable`` answers the readability question for the caller that orders candidates."""
    assert _count_kernels_if_readable(tmp_path / "gone.json") == (False, 0)
    not_a_trace = tmp_path / "notes.json"
    not_a_trace.write_text(json.dumps({"something": "else"}), encoding="utf-8")
    assert _count_kernels_if_readable(not_a_trace) == (False, 0)


def test_a_candidate_outside_the_capture_root_is_judged_by_name_alone(tmp_path: Path):
    """A path the root does not contain cannot be demoted by its directory, only by its own name."""
    outside = tmp_path / "elsewhere" / "trace.json"
    outside.parent.mkdir()
    assert _is_derived_trace(outside, root=tmp_path / "capture") is False


def test_a_missing_trace_input_names_itself(tmp_path: Path):
    """Neither a file nor a directory: the operator gets the path they passed, not a later shape error."""
    missing = tmp_path / "gone"
    with pytest.raises(FileNotFoundError, match="trace_input does not exist"):
        discover_trace_inputs(missing)


def test_an_empty_capture_directory_is_a_failure_not_an_empty_list(tmp_path: Path):
    """An empty directory is a capture that did not happen; every reader downstream expects at least one file."""
    empty = tmp_path / "capture"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no trace files found"):
        discover_trace_inputs(empty)


def test_a_single_file_resolves_to_itself(tmp_path: Path):
    """The file route is the one an operator uses by hand, and it reports the type the reports record."""
    trace = _write_trace(tmp_path / "one.json", [_kernel_event()])
    resolved = discover_trace_inputs(trace)
    assert resolved.input_type == "file"
    assert resolved.files == [trace]
