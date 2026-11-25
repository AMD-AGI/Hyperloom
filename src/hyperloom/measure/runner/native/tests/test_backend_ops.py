# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""test-backend-ops as the sweep drives it: the case file numbered by the tree's enums, and its output read back."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.measure.runner.native.backend_ops import (
    BackendOpsError,
    MulMatCase,
    case_file,
    check_cases,
    ggml_enum,
    time_cases,
)

_HEADER = """
    enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1,
        GGML_TYPE_ALIAS   = GGML_TYPE_F16,
        GGML_TYPE_COUNT,
    };
    enum ggml_op {
        GGML_OP_NONE = 0,
        GGML_OP_DUP, /* copy */
        GGML_OP_ADD,
        GGML_OP_MUL_MAT,
    };
"""
_PERF = (
    "Testing 1 devices\n"
    "  MUL_MAT(name=a_1_64x128,type=f32,ne=[64,1,1,1],op_params=[],sources=q8_0[128,64,1,1],f32[128,1,1,1]):"
    "      \x1b[1;32m 8192 runs -     1.25 us/run -       10 kB/run -  \x1b[1;34m  7.00 GB/s\x1b[0m\n"
    "  MUL_MAT(name=a_2_64x128,type=f32,ne=[64,2,1,1],op_params=[],sources=q8_0[128,64,1,1],f32[128,2,1,1]):"
    "       4096 runs -     2.50 us/run -       11 kB/run -    8.00 GB/s\n"
    "  Backend ROCm0: OK\n"
)
_CHECK = (
    "  MUL_MAT(name=a_1_64x128,type=f32,ne=[64,1,1,1],op_params=[],sources=q8_0[128,64,1,1]): \x1b[1;32mOK\x1b[0m\n"
    "  MUL_MAT(name=a_2_64x128,type=f32,ne=[64,2,1,1],op_params=[],sources=q8_0[128,64,1,1]): "
    "[MUL_MAT] NMSE = 0.5 > 0.0005 \x1b[1;31mFAIL\x1b[0m\n"
)


def _tool(tmp_path: Path, output: str, code: int = 0) -> Path:
    (tmp_path / "out.txt").write_text(output, encoding="utf-8")
    tool = tmp_path / "test-backend-ops"
    tool.write_text(f'#!/bin/sh\necho "$@" > "{tmp_path}/argv"\ncat "{tmp_path}/out.txt"\nexit {code}\n')
    tool.chmod(0o755)
    return tool


def test_the_enums_are_numbered_as_c_numbers_them() -> None:
    types = ggml_enum(_HEADER, "ggml_type")

    assert types == {
        "GGML_TYPE_F32": 0,
        "GGML_TYPE_F16": 1,
        "GGML_TYPE_Q8_0": 8,
        "GGML_TYPE_Q8_1": 9,
        "GGML_TYPE_ALIAS": 1,
        "GGML_TYPE_COUNT": 2,
    }
    assert ggml_enum(_HEADER, "ggml_op")["GGML_OP_MUL_MAT"] == 3
    with pytest.raises(BackendOpsError, match="no enum ggml_prec"):
        ggml_enum(_HEADER, "ggml_prec")


def test_a_case_line_names_the_op_and_types_by_the_trees_enums() -> None:
    text = case_file([MulMatCase("a_3_17408x5120", "Q8_0", 17408, 5120, 3)], _HEADER)

    assert text == "3 0 17408 3 1 1 0 2 8 5120 17408 1 1 0 0 0 0 0 5120 3 1 1 0 0 0 0 a_3_17408x5120\n"
    copied = case_file([MulMatCase("c", "Q8_0", 17408, 5120, 3, copies=4)], _HEADER)
    assert copied == "3 0 17408 3 4 1 0 2 8 5120 17408 4 1 0 0 0 0 0 5120 3 4 1 0 0 0 0 c\n"
    with pytest.raises(BackendOpsError, match="no GGML_TYPE_IQ4_XS"):
        case_file([MulMatCase("b", "IQ4_XS", 1, 32, 1)], _HEADER)


def test_the_perf_mode_is_read_back_by_case_name(tmp_path: Path) -> None:
    tool = _tool(tmp_path, _PERF)
    env = {"PATH": os.environ["PATH"]}

    times = time_cases(tool, tmp_path / "cases.txt", ["a_2_64x128", "a_1_64x128"], env, 30)

    assert times == {"a_2_64x128": 2.5, "a_1_64x128": 1.25}
    assert (tmp_path / "argv").read_text().split() == [
        "perf",
        "-b",
        "ROCm0",
        "--test-file",
        str(tmp_path / "cases.txt"),
    ]


def test_a_case_the_output_lacks_or_a_failed_run_is_an_error(tmp_path: Path) -> None:
    env = {"PATH": os.environ["PATH"]}

    with pytest.raises(BackendOpsError, match="no result of"):
        time_cases(_tool(tmp_path, _PERF), tmp_path / "c", ["a_1_64x128", "a_3_64x128"], env, 30)
    with pytest.raises(BackendOpsError, match="exited 1"):
        time_cases(_tool(tmp_path, _PERF, code=1), tmp_path / "c", ["a_1_64x128"], env, 30)


def test_the_check_reads_a_mismatch_as_a_result(tmp_path: Path) -> None:
    tool = _tool(tmp_path, _CHECK, code=1)

    verdicts = check_cases(tool, tmp_path / "c", ["a_1_64x128", "a_2_64x128"], {"PATH": os.environ["PATH"]}, 30)

    assert verdicts["a_1_64x128"] == "OK"
    assert verdicts["a_2_64x128"].endswith("FAIL")
    assert (tmp_path / "argv").read_text().split()[0] == "test"
