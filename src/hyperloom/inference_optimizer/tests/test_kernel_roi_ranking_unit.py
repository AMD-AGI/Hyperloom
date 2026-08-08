# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for headroom-weighted ranking of kernel-opt candidates.

Ranking on ``gpu_pct`` alone spends the attempt budget on whatever owns the
most trace time, including kernels already at their roofline with nothing left
to give. These tests pin the ROI ordering (share x headroom), the floor that
must stay on the raw share, and the degrade-to-``gpu_pct`` path for traces that
carry no roofline at all.

Headroom is read from ``roofline_attainment_pct`` (the binding side), never
from the compute-side ``efficiency_percent`` alone: a memory-bound kernel
pinned at its bandwidth roof reports ~0 there, and scoring it on that axis
would rank the one kernel with nothing to recover at the very top.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.kernel import _kernel_decisions as kd


def _state(hot):
    return SimpleNamespace(
        last_trace_analyze={"hot_kernels_top15": hot, "task_groups": []},
        optimization_stack=[],
        rejected_kernel_ids=[],
        kernel_opt_attempts={},
        kernel_opt_task_attempts=None,
    )


def _hot(kid, *, name, gpu_pct, attain=None, eff=None, bound=None, roi=None, src="model.py"):
    row = {
        "kernel_id": kid,
        "name": name,
        "source_file": src,
        "gpu_pct": gpu_pct,
        "reusable_native_kernel": True,
    }
    if attain is not None:
        row["roofline_attainment_pct"] = attain
    if eff is not None:
        row["efficiency_percent"] = eff
    if bound is not None:
        row["bound_type"] = bound
    if roi is not None:
        row["optimization_priority"] = roi
    return row


class TestOptimizationRoi:
    """The ROI helper itself."""

    def test_attainment_scales_the_share(self):
        row = {"roofline_attainment_pct": 25.0}
        assert kd._kernel_optimization_roi(row, 40.0) == 30.0

    def test_precomputed_priority_wins_over_recomputation(self):
        # The bypass report already published an ROI; recomputing it here would
        # fork the definition.
        row = {"roofline_attainment_pct": 25.0, "optimization_priority": 3.5}
        assert kd._kernel_optimization_roi(row, 40.0) == 3.5

    def test_missing_roofline_degrades_to_raw_share(self):
        # Rows without a roofline must rank exactly as before.
        assert kd._kernel_optimization_roi({}, 12.5) == 12.5

    def test_non_numeric_attainment_degrades_to_raw_share(self):
        assert kd._kernel_optimization_roi({"roofline_attainment_pct": "n/a"}, 12.5) == 12.5

    def test_attainment_is_clamped(self):
        # A kernel reported at or above roofline has no headroom, not negative.
        assert kd._kernel_optimization_roi({"roofline_attainment_pct": 140.0}, 10.0) == 0.0
        assert kd._kernel_optimization_roi({"roofline_attainment_pct": -20.0}, 10.0) == 10.0

    def test_bool_is_not_treated_as_a_number(self):
        assert kd._kernel_optimization_roi({"roofline_attainment_pct": True}, 9.0) == 9.0
        assert kd._kernel_optimization_roi({"optimization_priority": True}, 9.0) == 9.0


class TestHeadroomIsReadOnTheBindingSide:
    """``efficiency_percent`` is compute-side and must not be trusted alone."""

    def test_saturated_memory_bound_kernel_is_not_given_headroom(self):
        # Regression: aiter::add_rmsnorm sits at 100% of its bandwidth roof but
        # reports efficiency_percent=0.279 because that number is compute-side.
        # Scoring on it would hand a fully saturated kernel ~all its share.
        row = {
            "efficiency_percent": 0.279,
            "roofline_attainment_pct": 100.0,
            "bound_type": "memory_bound",
        }
        assert kd._kernel_optimization_roi(row, 1.7465) == 0.0

    def test_memory_bound_without_attainment_is_unknown_not_full_headroom(self):
        # The TraceLens route carries no attainment. A memory-bound row there
        # cannot be scored on the compute axis, so it degrades to the raw share
        # rather than being scored on the wrong one.
        row = {"efficiency_percent": 0.3, "bound_type": "memory_bound"}
        assert kd._kernel_optimization_roi(row, 4.0) == 4.0

    def test_compute_bound_efficiency_is_the_binding_side(self):
        # For a compute-bound kernel the compute-side number IS the attainment,
        # so the TraceLens route can still rank it.
        row = {"efficiency_percent": 75.0, "bound_type": "compute_bound"}
        assert kd._kernel_optimization_roi(row, 8.0) == 2.0

    def test_attainment_wins_when_both_are_present(self):
        row = {
            "efficiency_percent": 10.0,
            "roofline_attainment_pct": 90.0,
            "bound_type": "memory_bound",
        }
        assert kd._kernel_optimization_roi(row, 10.0) == pytest.approx(1.0)

    def test_efficiency_alone_without_bound_type_is_not_scored(self):
        # Without knowing which side binds, the compute-side number could be
        # either meaningful or ~0 by construction; refuse to guess.
        assert kd._kernel_optimization_roi({"efficiency_percent": 90.0}, 5.0) == 5.0


class TestCappedEstimatesAreRefused:
    """A clamped roofline means the model missed, not that the kernel is full."""

    def test_capped_attainment_falls_back_to_raw_share(self):
        # Regression: aiter::moe_cktile2stages_gemm2_ck is a grouped GEMM billed
        # for all 128 experts by the elementwise form when only topk=4 run. The
        # estimate implied ~68 TB/s, overshot the roof, and was clamped to 100%
        # -- which would zero the ROI of a kernel worth 14.6% of GPU time.
        row = {
            "roofline_attainment_pct": 100.0,
            "bound_type": "memory_bound",
            "roofline_estimate_capped": True,
        }
        assert kd._kernel_optimization_roi(row, 14.6232) == 14.6232

    def test_capped_compute_side_estimate_is_also_refused(self):
        row = {
            "efficiency_percent": 100.0,
            "bound_type": "compute_bound",
            "roofline_estimate_capped": True,
        }
        assert kd._kernel_optimization_roi(row, 9.0) == 9.0

    def test_uncapped_attainment_is_still_trusted(self):
        row = {
            "roofline_attainment_pct": 100.0,
            "bound_type": "memory_bound",
            "roofline_estimate_capped": False,
        }
        assert kd._kernel_optimization_roi(row, 14.0) == 0.0


class TestRankingPrefersHeadroom:
    """Selection order under the top_n cap."""

    def test_smaller_kernel_with_headroom_outranks_a_saturated_bigger_one(self):
        # 12% at 90% attainment has ROI 1.2; 11% at 30% has ROI 7.7. Ranking on
        # gpu_pct alone would burn the single attempt on the kernel that cannot
        # move.
        hot = [
            _hot("k_big", name="gemm_saturated", gpu_pct=12.0, attain=90.0),
            _hot("k_headroom", name="gemm_slack", gpu_pct=11.0, attain=30.0),
        ]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=1.0, top_n=1)
        assert untried == ["k_headroom"]

    def test_saturated_memory_bound_kernel_loses_to_a_smaller_one(self):
        # The bug in trace form: the saturated norm owns more GPU time, but all
        # of it is already at the bandwidth roof.
        hot = [
            _hot("k_norm", name="add_rmsnorm", gpu_pct=1.75, attain=100.0, eff=0.279, bound="memory_bound"),
            _hot("k_rope", name="rotary", gpu_pct=0.87, attain=22.8, eff=0.035, bound="memory_bound"),
        ]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=0.5, top_n=1)
        assert untried == ["k_rope"]

    def test_without_roofline_order_is_unchanged(self):
        # No roofline anywhere => pure gpu_pct ordering, as before the change.
        hot = [
            _hot("k_small", name="a", gpu_pct=11.0),
            _hot("k_big", name="b", gpu_pct=12.0),
        ]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=1.0, top_n=1)
        assert untried == ["k_big"]

    def test_precomputed_priority_drives_selection(self):
        hot = [
            _hot("k_a", name="a", gpu_pct=30.0, roi=0.5),
            _hot("k_b", name="b", gpu_pct=5.0, roi=4.0),
        ]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=1.0, top_n=1)
        assert untried == ["k_b"]

    def test_mixed_rows_rank_together(self):
        # A row with no roofline ranks on its raw share alongside ROI rows.
        hot = [
            _hot("k_eff", name="a", gpu_pct=20.0, attain=95.0),  # ROI 1.0
            _hot("k_raw", name="b", gpu_pct=6.0),  # ROI 6.0
            _hot("k_mid", name="c", gpu_pct=10.0, attain=70.0),  # ROI 3.0
        ]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=1.0, top_n=3)
        assert untried == ["k_raw", "k_mid", "k_eff"]


class TestFloorStaysOnRawShare:
    """``min_gpu_pct`` asks whether a kernel is big enough to bother with."""

    def test_large_saturated_kernel_still_clears_the_floor(self):
        # ROI is 0.5, well under the 1.0 floor, but the floor is not an ROI
        # test: the kernel owns half the trace and stays a candidate.
        hot = [_hot("k_big", name="a", gpu_pct=50.0, attain=99.0)]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=1.0, top_n=5)
        assert untried == ["k_big"]

    def test_tiny_kernel_with_full_headroom_is_still_excluded(self):
        # Headroom must not be able to promote a kernel that is too small to
        # matter past the floor.
        hot = [_hot("k_tiny", name="a", gpu_pct=0.4, attain=0.0)]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=1.0, top_n=5)
        assert untried == []
