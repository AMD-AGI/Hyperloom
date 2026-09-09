# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for sizing the demand shape list against the mode's real cost."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
        # Four shapes is one band plus one prefill row; the ranking would have
        # spent all four on prefill.
        monkeypatch.setenv(adc._DEMAND_MAX_SHAPES_ENV, "4")
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
        # has to hold per (N,K) rather than once for the table. Two prefill
        # shapes plus two bands of three is what that costs.
        monkeypatch.setenv(adc._DEMAND_MAX_SHAPES_ENV, "8")
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


class TestTheBandIsPaidForOutOfTheBudget:
    """The band is reserved out of the budget, never added on top of it.

    ``ctx.timeout_s`` is the tuner subprocess deadline, and a dense tuner killed
    there returns no candidate at all -- so a shape list the lane cannot finish
    does not merely lose its tail, it loses the prefill rows that used to
    complete. The band's cost scales with the dispatch group count, which the
    budget never saw.
    """

    #: Enough distinct prefill buckets per group that the budget, not the demand
    #: report, is what bounds the selection.
    PREFILL_M = (65, 129, 257, 513, 1_025, 2_049)

    @staticmethod
    def _demand(path: Path, groups: int, prefill_m: tuple[int, ...] = PREFILL_M) -> Path:
        keys = [
            {"M": m, "N": 4_096 + 1_024 * g, "K": 4_096, "requests": len(prefill_m) - i}
            for g in range(groups)
            for i, m in enumerate(prefill_m)
        ]
        return _write_demand(path, keys)

    @pytest.mark.parametrize("groups", [1, 2, 4, 6, 8])
    @pytest.mark.parametrize(("timeout_s", "thorough"), [(1_216, False), (3_600, False), (3_600, True)])
    def test_prefill_plus_band_fits(self, tmp_path, groups, timeout_s, thorough):
        ctx = _Ctx(timeout_s, thorough=thorough, conc=64)
        ctx.demand_json = self._demand(tmp_path / "demand.json", groups)

        out = adc._demand_input_csv(ctx, tmp_path, "a8w8_blockscale")

        assert out is not None
        assert len(_rows(out)) - 1 <= adc._demand_budget(ctx)

    @pytest.mark.parametrize("groups", [1, 2, 4, 6, 8])
    def test_every_group_it_tunes_gets_a_whole_band(self, tmp_path, groups):
        # Trimming groups is the concession, never the band inside a group: a
        # group holding part of its band serves the rest with a prefill tile.
        ctx = _Ctx(1_216, conc=64)
        ctx.demand_json = self._demand(tmp_path / "demand.json", groups)

        out = adc._demand_input_csv(ctx, tmp_path, "a8w8_blockscale")

        assert out is not None
        by_group: dict[int, set[int]] = {}
        for line in _rows(out)[1:]:
            m, n = (int(v) for v in line.split(",")[:2])
            by_group.setdefault(n, set()).add(m)
        for n, ms in by_group.items():
            for m in compute_decode_m_values(64):
                assert ms & adc._dispatch_lookup_ms(m, n), (n, m, sorted(ms))

    def test_the_prefill_tail_pays_for_the_extra_groups(self, tmp_path):
        # Same budget, more groups: the band costs more, so the discretionary
        # tail is what gives way.
        def prefill_rows(groups: int, sub: Path) -> int:
            sub.mkdir()
            ctx = _Ctx(1_216, conc=64)
            ctx.demand_json = self._demand(sub / "demand.json", groups)
            out = adc._demand_input_csv(ctx, sub, "a8w8_blockscale")
            assert out is not None
            return sum(int(line.split(",")[0]) > 64 for line in _rows(out)[1:])

        assert prefill_rows(1, tmp_path / "one") > prefill_rows(4, tmp_path / "four")

    def test_an_unaffordable_group_is_skipped_not_a_stop(self, tmp_path):
        # The ranking interleaves groups, so the shape that would open a fourth
        # band is unaffordable while later shapes in the three groups already
        # paid for still are. Stopping at the first miss would strand budget
        # the lane can spend.
        ctx = _Ctx(1_216, conc=64)
        ctx.demand_json = self._demand(tmp_path / "demand.json", 4, prefill_m=(1_025, 2_049, 4_097))

        out = adc._demand_input_csv(ctx, tmp_path, "a8w8_blockscale")

        assert out is not None
        rows = _rows(out)[1:]
        groups = {int(line.split(",")[1]) for line in rows}
        prefill = [line for line in rows if int(line.split(",")[0]) > 64]
        assert len(groups) == 3
        assert len(rows) == adc._demand_budget(ctx)
        assert len(prefill) > len(groups)

    def test_an_unaffordable_band_is_still_kept_whole(self, tmp_path, monkeypatch, caplog):
        # Not even one group fits, so there is no group left to trim. Shipping
        # a partial band would reintroduce the regression; overrun loudly.
        monkeypatch.setenv(adc._DEMAND_MAX_SHAPES_ENV, "1")
        ctx = _Ctx(1_216, conc=64)
        ctx.demand_json = self._demand(tmp_path / "demand.json", 1)

        with caplog.at_level("WARNING"):
            out = adc._demand_input_csv(ctx, tmp_path, "a8w8_blockscale")

        assert out is not None
        tuned_m = {int(line.split(",")[0]) for line in _rows(out)[1:]}
        for m in compute_decode_m_values(64):
            assert tuned_m & adc._dispatch_lookup_ms(m, 4_096), (m, sorted(tuned_m))
        assert any("exceeds the 1-shape budget" in r.message for r in caplog.records)
