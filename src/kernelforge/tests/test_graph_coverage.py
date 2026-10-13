# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which step classes replay a captured graph and which still launch every kernel."""

from __future__ import annotations

import json

from kernelforge import graph_coverage as gc


def _step(name: str, ts: float, dur: float) -> dict:
    return {"cat": "gpu_user_annotation", "name": name, "ts": ts, "dur": dur, "pid": 2}


def _kernel(name: str, ts: float, dur: float, correlation: int) -> dict:
    return {"cat": "kernel", "name": name, "ts": ts, "dur": dur, "pid": 2, "args": {"correlation": correlation}}


def _launch(correlation: int) -> dict:
    return {
        "cat": "cuda_runtime",
        "name": "hipLaunchKernel",
        "ts": 0.0,
        "dur": 1.0,
        "args": {"correlation": correlation},
    }


def _run(tmp_path, events):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return gc.graph_coverage(path)


def test_a_graphed_class_and_an_eager_one_are_told_apart(tmp_path):
    events = [
        _step("step[VERIFY bs=3]", 0.0, 100.0),
        # Replayed: no launch call carries these correlations.
        *[_kernel(f"graphed_{i}", 1.0 + i, 5.0, 900 + i) for i in range(10)],
        _step("step[EXTEND bs=1 toks=900]", 200.0, 100.0),
        *[_kernel(f"eager_{i}", 201.0 + i, 5.0, i) for i in range(10)],
        *[_launch(i) for i in range(10)],
    ]

    out = _run(tmp_path, events)

    assert out["EXTEND bs=1"]["eager_time_pct"] == 100.0
    assert out["EXTEND bs=1"]["ungraphed"] is True
    assert out["VERIFY bs=3"]["eager_time_pct"] == 0.0
    assert out["VERIFY bs=3"]["ungraphed"] is False


def test_the_eager_class_is_listed_first(tmp_path):
    events = [
        _step("step[VERIFY bs=3]", 0.0, 100.0),
        *[_kernel(f"g{i}", 1.0 + i, 5.0, 900 + i) for i in range(10)],
        _step("step[EXTEND bs=1 toks=900]", 200.0, 100.0),
        *[_kernel(f"e{i}", 201.0 + i, 5.0, i) for i in range(10)],
        *[_launch(i) for i in range(10)],
    ]

    assert list(_run(tmp_path, events)) == ["EXTEND bs=1", "VERIFY bs=3"]


def test_launches_per_step_are_averaged_over_the_class(tmp_path):
    events = [
        _step("step[EXTEND bs=1 toks=900]", 0.0, 100.0),
        *[_kernel(f"a{i}", 1.0 + i, 5.0, i) for i in range(6)],
        _step("step[EXTEND bs=1 toks=901]", 200.0, 100.0),
        *[_kernel(f"b{i}", 201.0 + i, 5.0, 100 + i) for i in range(4)],
        *[_launch(i) for i in range(6)],
        *[_launch(100 + i) for i in range(4)],
    ]

    row = _run(tmp_path, events)["EXTEND bs=1"]

    assert row["kernels"] == 10
    assert row["eager_kernels_per_step"] == 5.0


def test_a_class_graphed_except_for_a_few_kernels_is_not_flagged(tmp_path):
    events = [
        _step("step[VERIFY bs=3]", 0.0, 200.0),
        *[_kernel(f"g{i}", 1.0 + i, 5.0, 900 + i) for i in range(20)],
        _kernel("stray", 150.0, 5.0, 1),
        _launch(1),
    ]

    row = _run(tmp_path, events)["VERIFY bs=3"]

    assert 0 < row["eager_time_pct"] < 100.0
    assert row["ungraphed"] is False


def test_a_capture_with_no_launch_calls_cannot_be_read(tmp_path):
    """Without launch records every kernel would read as graphed, which is a wrong answer, not an empty one."""
    events = [_step("step[VERIFY bs=3]", 0.0, 100.0), _kernel("k", 1.0, 5.0, 1)]

    assert _run(tmp_path, events) == {}


def test_a_capture_without_steps_or_a_missing_one_reports_nothing(tmp_path):
    assert _run(tmp_path, [_kernel("k", 1.0, 5.0, 1), _launch(1)]) == {}
    assert gc.graph_coverage("") == {}
    assert gc.graph_coverage(None) == {}
    assert gc.graph_coverage(tmp_path / "absent.json") == {}


def test_the_rendering_marks_the_ungraphed_class_and_says_what_to_check(tmp_path):
    events = [
        _step("step[EXTEND bs=1 toks=900]", 0.0, 100.0),
        *[_kernel(f"e{i}", 1.0 + i, 5.0, i) for i in range(10)],
        *[_launch(i) for i in range(10)],
    ]

    body = "\n".join(gc.render_graph_coverage(_run(tmp_path, events)))

    assert "`EXTEND bs=1` | prefill" in body
    assert "100.0% ⚠" in body
    assert "no `layers` for the capture to walk" in body


def test_a_fully_graphed_run_renders_without_advice(tmp_path):
    events = [
        _step("step[VERIFY bs=3]", 0.0, 100.0),
        *[_kernel(f"g{i}", 1.0 + i, 5.0, 900 + i) for i in range(10)],
        _kernel("stray", 50.0, 1.0, 1),
        _launch(1),
    ]

    body = "\n".join(gc.render_graph_coverage(_run(tmp_path, events)))

    assert "⚠" not in body
    assert "pays a launch for every kernel" not in body


def test_nothing_renders_without_coverage():
    assert gc.render_graph_coverage({}) == []
    assert gc.render_graph_coverage(None) == []
