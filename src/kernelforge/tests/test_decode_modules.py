# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A decode step's device time by model module, read off the raw capture rather than a TraceLens report."""

from __future__ import annotations

from kernelforge import decode_modules as dm
from kernelforge.tests.trace_fixtures import kernel, step, write_trace

# Kernel names as GLM-5.3's VERIFY graph replay records them on ROCm 10.
_MOE = "_ZN5aiter48fmoe_bf16_blockscaleFp8_g1u1_vs_silu_mfma_moe1_afp4"
_GEMM = "ck_tile::kentry<gemm_a8w8_blockscale_b_preshuffle>"
_AR = "aiter::allreduce_fusion_kernel_1stage"
_MLA = "_sparse_mla_split_k"
_QUANT = "aiter::dynamic_per_group_scaled_quant_kernel"


def _verify_step(bs: int, start: float) -> list[dict]:
    return [
        step(f"step[VERIFY bs={bs}]", start, 100.0),
        kernel(_MOE, start + 1.0, 30.0 * bs),
        kernel(_GEMM, start + 40.0, 20.0),
        kernel(_AR, start + 61.0, 10.0),
        kernel(_MLA, start + 72.0, 8.0),
        kernel(_QUANT, start + 81.0, 2.0),
    ]


def test_each_decode_class_splits_its_step_time_by_module(tmp_path):
    events = [*_verify_step(1, 0.0), *_verify_step(1, 200.0), *_verify_step(2, 400.0)]
    events.append(step("step[EXTEND bs=1 toks=4096]", 600.0, 900.0))
    events.append(kernel(_GEMM, 601.0, 800.0))
    trace = write_trace(tmp_path / "trace.json.gz", events)

    table = dm.decode_modules(trace)

    assert list(table) == ["VERIFY bs=1", "VERIFY bs=2"]
    assert table["VERIFY bs=1"]["steps"] == 2
    assert [(m["module"], m["ms_per_step"], m["pct"]) for m in table["VERIFY bs=1"]["modules"]] == [
        ("moe", 0.03, 42.9),
        ("gemm", 0.02, 28.6),
        ("allreduce/comm", 0.01, 14.3),
        ("attention/mla", 0.008, 11.4),
        ("quant", 0.002, 2.9),
    ]
    assert table["VERIFY bs=2"]["modules"][0] == {"module": "moe", "ms_per_step": 0.06, "pct": 60.0, "top_kernel": _MOE}


def test_a_capture_without_decode_steps_has_no_table(tmp_path):
    trace = write_trace(tmp_path / "t.json", [step("step[EXTEND bs=1 toks=9]", 0.0, 10.0), kernel(_GEMM, 1.0, 5.0)])

    assert dm.decode_modules(trace) == {}


def test_the_rendered_table_has_one_column_per_decode_class_and_one_row_per_module(tmp_path):
    trace = write_trace(tmp_path / "t.json", [*_verify_step(1, 0.0), *_verify_step(4, 200.0)])

    lines = dm.render_decode_modules(dm.decode_modules(trace))

    assert lines[:4] == [
        "| module | VERIFY bs=1 | VERIFY bs=4 | top kernel |",
        "|---|---:|---:|---|",
        f"| moe | 0.030 ms (42.9%) | 0.120 ms (75.0%) | `{_MOE}` |",
        f"| gemm | 0.020 ms (28.6%) | 0.020 ms (12.5%) | `{_GEMM}` |",
    ]


def test_one_modules_lines_keep_its_row_in_every_decode_class(tmp_path):
    trace = write_trace(tmp_path / "t.trace.json", [*_verify_step(1, 0.0), *_verify_step(4, 200.0)])

    assert dm.decode_module_lines(trace, module="allreduce/comm") == [
        "| module | VERIFY bs=1 | VERIFY bs=4 | top kernel |",
        "|---|---:|---:|---|",
        f"| allreduce/comm | 0.010 ms (14.3%) | 0.010 ms (6.2%) | `{_AR}` |",
    ]
    assert dm.decode_module_lines(trace, module="indexer/topk") == []
