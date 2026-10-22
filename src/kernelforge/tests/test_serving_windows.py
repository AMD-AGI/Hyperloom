# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernels ranked inside the step class they ran in, because the classes are not comparable."""

from __future__ import annotations


from kernelforge.tests.trace_fixtures import kernel, step, write_trace
from kernelforge import serving_windows as sw


def test_the_step_kind_and_batch_size_come_from_the_annotation():
    windows = sw.step_windows(
        [
            step("step[VERIFY bs=3]", 100.0, 10.0),
            step("step[EXTEND bs=1 toks=17555 c_sq=17555 c_sqsq=308178025]", 200.0, 600.0),
        ]
    )

    assert [(w.kind, w.batch_size, w.window_class, w.is_decode) for w in windows] == [
        ("VERIFY", 3, "VERIFY bs=3", True),
        ("EXTEND", 1, "EXTEND bs=1", False),
    ]


def test_the_host_copy_of_an_annotation_is_not_a_window():
    """The host records the same step; its duration is submission, and its clock is not the kernels'."""
    windows = sw.step_windows(
        [
            step("step[VERIFY bs=3]", 100.0, 500.0, cat="user_annotation"),
            step("step[VERIFY bs=3]", 100.0, 14300.0),
        ]
    )

    assert [w.end_us - w.start_us for w in windows] == [14300.0]


def test_anything_that_is_not_a_step_is_ignored():
    assert sw.step_windows([step("scheduler.run_batch", 1.0, 2.0)]) == []
    assert sw.step_windows([step("step[VERIFY bs=3]", 1.0, 0.0)]) == []
    assert sw.step_windows([{"cat": "gpu_user_annotation", "name": "step[VERIFY bs=3]"}]) == []


def test_prefill_and_decode_are_ranked_apart(tmp_path):
    # One prefill step 40x a decode step, as a real AgentX capture holds: 660 ms against 14 ms.
    events = [
        step("step[EXTEND bs=1 toks=900]", 0.0, 600.0),
        kernel("attention_prefill", 10.0, 400.0),
        kernel("moe_gemm", 420.0, 100.0),
        step("step[VERIFY bs=3]", 1000.0, 14.0),
        kernel("moe_gemm", 1001.0, 9.0),
        kernel("allreduce", 1010.0, 3.0),
    ]

    out = sw.hot_kernels_by_class(write_trace(tmp_path / "t.json.gz", events))

    assert out["EXTEND bs=1"]["is_decode"] is False
    assert out["VERIFY bs=3"]["is_decode"] is True
    assert [k["name"] for k in out["EXTEND bs=1"]["kernels"]] == ["attention_prefill", "moe_gemm"]
    assert [k["name"] for k in out["VERIFY bs=3"]["kernels"]] == ["moe_gemm", "allreduce"]
    # The decode ranking is not swamped by the prefill step that is 40x its size.
    assert out["VERIFY bs=3"]["kernels"][0]["pct"] == 75.0


def test_steps_of_one_class_are_pooled(tmp_path):
    events = [
        step("step[VERIFY bs=3]", 0.0, 10.0),
        kernel("k", 1.0, 4.0),
        step("step[VERIFY bs=3]", 100.0, 10.0),
        kernel("k", 101.0, 6.0),
    ]

    out = sw.hot_kernels_by_class(write_trace(tmp_path / "t.json.gz", events))

    assert out["VERIFY bs=3"]["steps"] == 2
    assert out["VERIFY bs=3"]["kernels"][0]["ms"] == 0.01
    assert out["VERIFY bs=3"]["step_ms_median"] == 0.01


def test_one_batch_size_is_not_pooled_with_another(tmp_path):
    events = [
        step("step[VERIFY bs=1]", 0.0, 10.0),
        kernel("k", 1.0, 4.0),
        step("step[VERIFY bs=3]", 100.0, 10.0),
        kernel("k", 101.0, 6.0),
    ]

    out = sw.hot_kernels_by_class(write_trace(tmp_path / "t.json.gz", events))

    assert sorted(out) == ["VERIFY bs=1", "VERIFY bs=3"]


def test_a_kernel_outside_every_step_is_not_counted(tmp_path):
    events = [
        step("step[VERIFY bs=3]", 100.0, 10.0),
        kernel("inside", 101.0, 5.0),
        kernel("between_steps", 50.0, 5.0),
    ]

    out = sw.hot_kernels_by_class(write_trace(tmp_path / "t.json.gz", events))

    assert [k["name"] for k in out["VERIFY bs=3"]["kernels"]] == ["inside"]


def test_a_capture_without_step_annotations_ranks_nothing(tmp_path):
    """Told nothing beats handing back a ranking of everything."""
    events = [kernel("k", 1.0, 5.0), step("scheduler.run_batch", 0.0, 10.0)]

    assert sw.hot_kernels_by_class(write_trace(tmp_path / "t.json.gz", events)) == {}
    assert sw.hot_kernels_by_class("") == {}
    assert sw.hot_kernels_by_class(None) == {}
    assert sw.hot_kernels_by_class(tmp_path / "absent.json") == {}


def test_a_plain_json_capture_reads_the_same(tmp_path):
    events = [step("step[VERIFY bs=3]", 0.0, 10.0), kernel("k", 1.0, 4.0)]

    assert sw.hot_kernels_by_class(write_trace(tmp_path / "t.json", events))["VERIFY bs=3"]["steps"] == 1


def test_the_rendering_names_the_phase_and_says_why_classes_are_apart(tmp_path):
    events = [
        step("step[EXTEND bs=1 toks=900]", 0.0, 600.0),
        kernel("attention_prefill", 10.0, 400.0),
        step("step[VERIFY bs=3]", 1000.0, 14.0),
        kernel("moe_gemm", 1001.0, 9.0),
    ]

    body = "\n".join(
        sw.render_hot_kernels_by_class(sw.hot_kernels_by_class(write_trace(tmp_path / "t.json.gz", events)))
    )

    assert "### EXTEND bs=1 (prefill)" in body
    assert "### VERIFY bs=3 (decode)" in body
    assert "a ranking of prefill" in body


def test_a_templated_kernel_name_is_cut_for_the_reader(tmp_path):
    long_name = "void ck::kernel_gemm_xdl<" + "ck::Sequence<1, 0, 2>, " * 40 + ">"
    events = [step("step[VERIFY bs=3]", 0.0, 10.0), kernel(long_name, 1.0, 4.0)]

    body = "\n".join(
        sw.render_hot_kernels_by_class(sw.hot_kernels_by_class(write_trace(tmp_path / "t.json.gz", events)))
    )

    assert "…" in body
    assert "void ck::kernel_gemm_xdl<" in body
    kernel_rows = [line for line in body.splitlines() if line.startswith("| ") and "`" in line]
    assert len(max(kernel_rows, key=len)) < len(long_name)


def test_nothing_renders_without_a_ranking():
    assert sw.render_hot_kernels_by_class({}) == []
    assert sw.render_hot_kernels_by_class(None) == []
