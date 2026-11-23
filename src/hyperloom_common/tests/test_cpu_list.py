# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A CPU list as ``taskset -c`` takes it."""

from __future__ import annotations

import pytest

from hyperloom_common.cpu_list import cpu_list


@pytest.mark.parametrize(
    ("text", "cpus"),
    [("3", (3,)), ("0-3", (0, 1, 2, 3)), ("8-9,2,9-10", (8, 9, 2, 10))],
    ids=["one", "a-range", "ranges-in-order-each-once"],
)
def test_a_cpu_list_names_its_cpus_in_order_each_once(text: str, cpus: tuple[int, ...]) -> None:
    assert cpu_list(text) == cpus


@pytest.mark.parametrize(
    ("text", "why"),
    [("", "not a CPU list"), ("0-", "not a CPU list"), ("1 2", "not a CPU list"), ("5-3", "runs backwards")],
)
def test_what_names_no_cpu_is_refused(text: str, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        cpu_list(text)
