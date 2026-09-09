# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for sizing the demand shape list against the mode's real cost."""

from __future__ import annotations

import json
from pathlib import Path

from kernelforge.gemm_tune.dense_shapes import compute_decode_m_values
from kernelforge.gemm_tune.tuners import _aiter_dense_common as adc


class _Ctx:
    """Minimal stand-in for the TuneContext fields the demand path reads."""

    def __init__(self, timeout_s: int, thorough: bool = False, conc: int = 64):
        self.timeout_s = timeout_s
        self.thorough = thorough
        self.conc = conc
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
    # A budget smaller than one shape still has to tune something, or the run reports "no shapes" for what is really
    # "no time".
    assert adc._demand_budget(_Ctx(1, thorough=True)) == 1
    assert adc._demand_budget(_Ctx(0)) == 1


def test_a_context_without_the_flag_is_treated_as_fast():
    class _Old:
        timeout_s = 3_600

    assert adc._demand_budget(_Old()) == ((3_600 - adc._DEMAND_RESERVE_S) // adc._DEMAND_PER_SHAPE_COST_S)


def _write_demand(path: Path, keys: list[dict], tuner: str = "a8w8_blockscale") -> Path:
    path.write_text(
        json.dumps({"demands": [{"tuner": tuner, "distinct_keys": len(keys), "keys": keys}]}),
        encoding="utf-8",
    )
    return path


def _rows(csv_path: Path) -> list[str]:
    return csv_path.read_text(encoding="utf-8").splitlines()


def test_quantized_demand_uses_the_runtime_lookup_buckets(tmp_path):
    """a8w8/a4w4 use the same padded-M retry sequence as a16w16."""
    demand = _write_demand(
        tmp_path / "demand.json",
        [
            {"M": 300, "N": 4096, "K": 4096, "requests": 7},
            {"M": 400, "N": 4096, "K": 4096, "requests": 3},
        ],
    )
    ctx = _Ctx(3_600)
    ctx.demand_json = demand

    out = adc._demand_input_csv(ctx, tmp_path, "a8w8_blockscale")

    assert out is not None
    # 512 is the bucket both observed M pad into; the rest is the decode-band
    # guarantee, which the demand ranking cannot supply (see below).
    assert "512,4096,4096" in _rows(out)


class TestDemandKeepsTheDecodeBand:
    """The demand ranking cannot decide whether the decode band gets tuned.

    ``requests`` counts memoized log lines, so it scores a bucket by how many
    distinct M happen to land in it. ``padded_m`` buckets double in width, so
    the prefill tail always outranks the narrow decode buckets -- on a real
    Qwen3-14B-FP8 sglang arm every decode bucket ranked 33rd-56th of 56 and a
    budget of 14 cut the band entirely, shipping a prefill-only table that lost
    its e2e gate at +1.30%.
    """

    #: One wide prefill bucket plus enough distinct M to outrank everything.
    PREFILL_KEYS = [{"M": 1024 + i, "N": 4096, "K": 4096, "requests": 3} for i in range(40)]

    def test_decode_band_survives_a_budget_that_only_funds_prefill(self, tmp_path, monkeypatch):
        monkeypatch.setenv(adc._DEMAND_MAX_SHAPES_ENV, "1")
        demand = _write_demand(tmp_path / "demand.json", self.PREFILL_KEYS)
        ctx = _Ctx(3_600, conc=64)
        ctx.demand_json = demand

        out = adc._demand_input_csv(ctx, tmp_path, "a8w8_blockscale")

        assert out is not None
        tuned_m = {int(line.split(",")[0]) for line in _rows(out)[1:]}
        # Every M the scheduler can run at decode must resolve to a tuned row.
        for m in compute_decode_m_values(64):
            assert tuned_m & adc._dispatch_lookup_ms(m, 4096), (m, sorted(tuned_m))

    def test_prefill_selection_is_not_crowded_out(self, tmp_path):
        demand = _write_demand(tmp_path / "demand.json", self.PREFILL_KEYS)
        ctx = _Ctx(3_600, conc=64)
        ctx.demand_json = demand

        out = adc._demand_input_csv(ctx, tmp_path, "a8w8_blockscale")

        assert out is not None
        tuned_m = {int(line.split(",")[0]) for line in _rows(out)[1:]}
        assert any(m > 64 for m in tuned_m), sorted(tuned_m)

    def test_decode_rows_are_per_dispatch_group(self, tmp_path, monkeypatch):
        # A row at (M, N1, K1) is never consulted for (N2, K2), so the guarantee
        # has to hold per (N,K) rather than once for the table.
        monkeypatch.setenv(adc._DEMAND_MAX_SHAPES_ENV, "2")
        keys = [
            {"M": 2048, "N": 4096, "K": 4096, "requests": 9},
            {"M": 2048, "N": 5120, "K": 17408, "requests": 9},
        ]
        demand = _write_demand(tmp_path / "demand.json", keys)
        ctx = _Ctx(3_600, conc=64)
        ctx.demand_json = demand

        out = adc._demand_input_csv(ctx, tmp_path, "a8w8_blockscale")

        assert out is not None
        by_group: dict[tuple[int, int], set[int]] = {}
        for line in _rows(out)[1:]:
            m, n, k = (int(v) for v in line.split(",")[:3])
            by_group.setdefault((n, k), set()).add(m)
        assert set(by_group) == {(4096, 4096), (5120, 17408)}
        for (n, _k), ms in by_group.items():
            for m in compute_decode_m_values(64):
                assert ms & adc._dispatch_lookup_ms(m, n), (n, m, sorted(ms))
