# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.kernel.request_handlers import (
    _GEMM_PER_TUNER_TIMEOUT_SHARE,
    _gemm_per_tuner_timeout_sec,
)


def test_a_gemm_tuner_is_capped_strictly_below_the_session() -> None:
    per_tuner = _gemm_per_tuner_timeout_sec(10_000)
    assert 0 < per_tuner < 10_000


@pytest.mark.parametrize("total", [2, 3, 10, 100, 10_000])
def test_the_cap_stays_below_any_usable_session(total: int) -> None:
    assert 0 < _gemm_per_tuner_timeout_sec(total) < total


def test_a_one_second_session_is_degenerate_and_yields_one_second() -> None:
    assert _gemm_per_tuner_timeout_sec(1) == 1


@pytest.mark.parametrize("value", [0, -1, None, "x", True])
def test_an_unbounded_gemm_session_passes_through(value: Any) -> None:
    assert _gemm_per_tuner_timeout_sec(value) == 0


def test_the_gemm_cap_follows_its_share() -> None:
    assert _gemm_per_tuner_timeout_sec(1000) == int(1000 * _GEMM_PER_TUNER_TIMEOUT_SHARE)
