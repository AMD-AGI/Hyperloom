# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.common.tracelens_report: the name every reader gives an analysis.json compute member."""

from __future__ import annotations

import pytest

from hyperloom.common.tracelens_report import member_name


@pytest.mark.parametrize(
    ("operation", "kernel_name", "expected"),
    [
        (" aiter::gemm_a16w16 ", ["Cijk_gemm_a"], "aiter::gemm_a16w16"),
        (None, ["", "Cijk_gemm_a", "Cijk_gemm_a_tail"], "Cijk_gemm_a"),
        (None, [], ""),
    ],
)
def test_a_member_is_named_by_its_task_operation_else_by_its_first_device_symbol(operation, kernel_name, expected):
    assert member_name({"operation": operation}, {"kernel_name": kernel_name}) == expected
