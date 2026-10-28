# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which step classes replay a captured graph and which still launch every kernel."""

from __future__ import annotations

from kernelforge import graph_coverage as gc
from kernelforge.tests.trace_fixtures import eager_step, graphed_step, kernel, launch, step, write_trace

_DECODE = "step[VERIFY bs=3]"
_PREFILL = "step[EXTEND bs=1 toks=900]"


def _run(tmp_path, events):
    return gc.graph_coverage(write_trace(tmp_path / "t.json", events))


def test_a_graphed_class_and_an_eager_one_are_told_apart(tmp_path):
    events = [
        *graphed_step(_DECODE, start=0.0, count=10),
        *eager_step(_PREFILL, start=200.0, count=10),
    ]

    out = _run(tmp_path, events)

    assert out["EXTEND bs=1"]["eager_time_pct"] == 100.0
    assert out["EXTEND bs=1"]["ungraphed"] is True
    assert out["VERIFY bs=3"]["eager_time_pct"] == 0.0
    assert out["VERIFY bs=3"]["ungraphed"] is False


def test_the_eager_class_is_listed_first(tmp_path):
    events = [*graphed_step(_DECODE, start=0.0, count=10), *eager_step(_PREFILL, start=200.0, count=10)]

    assert list(_run(tmp_path, events)) == ["EXTEND bs=1", "VERIFY bs=3"]


def test_launches_per_step_are_averaged_over_the_class(tmp_path):
    events = [
        *eager_step(_PREFILL, start=0.0, count=6, first_correlation=0),
        *eager_step("step[EXTEND bs=1 toks=901]", start=200.0, count=4, first_correlation=100),
    ]

    row = _run(tmp_path, events)["EXTEND bs=1"]

    assert row["eager_kernels_per_step"] == 5.0


def test_a_class_graphed_except_for_a_few_kernels_is_not_flagged(tmp_path):
    events = [*graphed_step(_DECODE, start=0.0, count=20), kernel("stray", 150.0, 5.0, correlation=1), launch(1)]

    row = _run(tmp_path, events)["VERIFY bs=3"]

    assert 0 < row["eager_time_pct"] < 100.0
    assert row["ungraphed"] is False


def test_a_capture_with_no_launch_calls_cannot_be_read(tmp_path):
    """Without launch records every kernel would read as graphed, which is a wrong answer, not an empty one."""
    assert _run(tmp_path, [step(_DECODE, 0.0, 100.0), kernel("k", 1.0, 5.0, correlation=1)]) == {}


def test_a_capture_without_steps_or_a_missing_one_reports_nothing(tmp_path):
    assert _run(tmp_path, [kernel("k", 1.0, 5.0, correlation=1), launch(1)]) == {}
    assert gc.graph_coverage("") == {}
    assert gc.graph_coverage(None) == {}
    assert gc.graph_coverage(tmp_path / "absent.json") == {}
