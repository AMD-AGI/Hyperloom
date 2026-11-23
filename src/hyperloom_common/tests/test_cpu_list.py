# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A CPU list as ``taskset -c`` takes it."""

from __future__ import annotations

import os

import pytest

from hyperloom_common.cpu_list import cpu_list, runnable_cpus

#: Above every CPU number a Linux kernel can be built for.
_NO_HOST_HAS = 100_000


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


def test_the_cpus_a_process_may_run_on_here_are_the_ones_the_kernel_grants() -> None:
    own = sorted(os.sched_getaffinity(0))

    assert runnable_cpus((own[0], _NO_HOST_HAS)) == {own[0]}
    assert runnable_cpus((_NO_HOST_HAS, 2**40)) == frozenset()
    assert sorted(os.sched_getaffinity(0)) == own
