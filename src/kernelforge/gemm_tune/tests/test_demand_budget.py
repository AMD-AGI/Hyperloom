# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for sizing the demand shape list against the mode's real cost.

A `--libtype all` shape does not cost what a hipblaslt-only shape costs. A
per-backend breakdown over four shapes on an 8-GPU MI355X box measured 169s for
hipblaslt+asm+triton+skinny+opus+torch together and 1458s for flydsl on its own
-- and flydsl is not droppable, it won two of the four shapes (by 37% at M=16,
N=1536, K=7168). Thorough mode is genuinely ~5.5x more expensive per shape.

Sizing it with the fast figure is not a mild over-estimate: the batch claims
5.5x the shapes it can finish, `--shape_grouped` spends the entire allowance on
the first few, and the remainder are written as nothing. The report then cannot
distinguish that from a tuner that ran fine and found no improvement, which is
the reading that hid the original breakage.
"""

from __future__ import annotations

import json
from pathlib import Path

from kernelforge.gemm_tune.tuners import _aiter_dense_common as adc


class _Ctx:
    """Minimal stand-in: _demand_budget reads only these two attributes."""

    def __init__(self, timeout_s: int, thorough: bool = False):
        self.timeout_s = timeout_s
        self.thorough = thorough
        self.output_dir = Path(".")


def test_thorough_claims_fewer_shapes_than_fast():
    fast = adc._demand_budget(_Ctx(3_600))
    thorough = adc._demand_budget(_Ctx(3_600, thorough=True))
    assert fast == (3_600 - adc._DEMAND_RESERVE_S) // adc._DEMAND_PER_SHAPE_COST_S
    assert thorough == (3_600 - adc._DEMAND_RESERVE_S) // adc._DEMAND_PER_SHAPE_COST_THOROUGH_S
    assert thorough < fast


def test_claimed_shapes_fit_the_budget_in_both_modes():
    for timeout_s in (900, 1_800, 3_600, 7_200):
        for thorough, cost in (
            (False, adc._DEMAND_PER_SHAPE_COST_S),
            (True, adc._DEMAND_PER_SHAPE_COST_THOROUGH_S),
        ):
            n = adc._demand_budget(_Ctx(timeout_s, thorough=thorough))
            assert n * cost <= timeout_s, (timeout_s, thorough, n)


def test_override_wins_in_either_mode(monkeypatch):
    monkeypatch.setenv(adc._DEMAND_MAX_SHAPES_ENV, "5")
    assert adc._demand_budget(_Ctx(3_600)) == 5
    assert adc._demand_budget(_Ctx(3_600, thorough=True)) == 5


def test_garbage_override_falls_back_to_the_measured_cost(monkeypatch):
    monkeypatch.setenv(adc._DEMAND_MAX_SHAPES_ENV, "not-a-number")
    assert adc._demand_budget(_Ctx(3_600, thorough=True)) == (
        (3_600 - adc._DEMAND_RESERVE_S) // adc._DEMAND_PER_SHAPE_COST_THOROUGH_S
    )


def test_never_claims_zero_shapes():
    # A budget smaller than one shape still has to tune something, or the run
    # reports "no shapes" for what is really "no time".
    assert adc._demand_budget(_Ctx(1, thorough=True)) == 1
    assert adc._demand_budget(_Ctx(0)) == 1


def test_a_context_without_the_flag_is_treated_as_fast():
    class _Old:
        timeout_s = 3_600

    assert adc._demand_budget(_Old()) == ((3_600 - adc._DEMAND_RESERVE_S) // adc._DEMAND_PER_SHAPE_COST_S)


def test_quantized_demand_uses_the_runtime_lookup_buckets(tmp_path):
    """a8w8/a4w4 use the same padded-M retry sequence as a16w16."""
    demand = tmp_path / "demand.json"
    demand.write_text(
        json.dumps(
            {
                "demands": [
                    {
                        "tuner": "a8w8_blockscale",
                        "distinct_keys": 2,
                        "keys": [
                            {"M": 300, "N": 4096, "K": 4096, "requests": 7},
                            {"M": 400, "N": 4096, "K": 4096, "requests": 3},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ctx = _Ctx(3_600)
    ctx.demand_json = demand

    out = adc._demand_input_csv(ctx, tmp_path, "a8w8_blockscale")

    assert out is not None
    assert out.read_text(encoding="utf-8").splitlines() == [
        "M,N,K",
        "512,4096,4096",
    ]
