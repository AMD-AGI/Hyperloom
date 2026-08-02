# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for headroom-weighted ranking of kernel-opt candidates.

Ranking on ``gpu_pct`` alone spends the attempt budget on whatever owns the
most trace time, including kernels already at their roofline with nothing left
to give. These tests pin the ROI ordering (share x headroom), the floor that
must stay on the raw share, and the degrade-to-``gpu_pct`` path for traces that
carry no efficiency at all.
"""

from __future__ import annotations

from types import SimpleNamespace

from hyperloom.orchestrator.kernel import _kernel_decisions as kd


def _state(hot):
    return SimpleNamespace(
        last_trace_analyze={"hot_kernels_top15": hot, "task_groups": []},
        optimization_stack=[],
        rejected_kernel_ids=[],
        kernel_opt_attempts={},
        kernel_opt_task_attempts=None,
    )


def _hot(kid, *, name, gpu_pct, eff=None, roi=None, src="model.py"):
    row = {
        "kernel_id": kid,
        "name": name,
        "source_file": src,
        "gpu_pct": gpu_pct,
        "reusable_native_kernel": True,
    }
    if eff is not None:
        row["efficiency_percent"] = eff
    if roi is not None:
        row["optimization_priority"] = roi
    return row


class TestOptimizationRoi:
    """The ROI helper itself."""

    def test_efficiency_scales_the_share(self):
        row = {"efficiency_percent": 25.0}
        assert kd._kernel_optimization_roi(row, 40.0) == 30.0

    def test_precomputed_priority_wins_over_recomputation(self):
        # The bypass report already published an ROI; recomputing it here would
        # fork the definition.
        row = {"efficiency_percent": 25.0, "optimization_priority": 3.5}
        assert kd._kernel_optimization_roi(row, 40.0) == 3.5

    def test_missing_efficiency_degrades_to_raw_share(self):
        # TraceLens rows without an efficiency must rank exactly as before.
        assert kd._kernel_optimization_roi({}, 12.5) == 12.5

    def test_non_numeric_efficiency_degrades_to_raw_share(self):
        assert kd._kernel_optimization_roi({"efficiency_percent": "n/a"}, 12.5) == 12.5

    def test_efficiency_is_clamped(self):
        # A kernel reported at or above roofline has no headroom, not negative.
        assert kd._kernel_optimization_roi({"efficiency_percent": 140.0}, 10.0) == 0.0
        assert kd._kernel_optimization_roi({"efficiency_percent": -20.0}, 10.0) == 10.0

    def test_bool_is_not_treated_as_a_number(self):
        assert kd._kernel_optimization_roi({"efficiency_percent": True}, 9.0) == 9.0
        assert kd._kernel_optimization_roi({"optimization_priority": True}, 9.0) == 9.0


class TestRankingPrefersHeadroom:
    """Selection order under the top_n cap."""

    def test_smaller_kernel_with_headroom_outranks_a_saturated_bigger_one(self):
        # 12% at 90% efficiency has ROI 1.2; 11% at 30% has ROI 7.7. Ranking on
        # gpu_pct alone would burn the single attempt on the kernel that cannot
        # move.
        hot = [
            _hot("k_big", name="gemm_saturated", gpu_pct=12.0, eff=90.0),
            _hot("k_headroom", name="gemm_slack", gpu_pct=11.0, eff=30.0),
        ]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=1.0, top_n=1)
        assert untried == ["k_headroom"]

    def test_without_efficiency_order_is_unchanged(self):
        # No efficiency anywhere => pure gpu_pct ordering, as before the change.
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
        # A row with no efficiency ranks on its raw share alongside ROI rows.
        hot = [
            _hot("k_eff", name="a", gpu_pct=20.0, eff=95.0),  # ROI 1.0
            _hot("k_raw", name="b", gpu_pct=6.0),  # ROI 6.0
            _hot("k_mid", name="c", gpu_pct=10.0, eff=70.0),  # ROI 3.0
        ]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=1.0, top_n=3)
        assert untried == ["k_raw", "k_mid", "k_eff"]


class TestFloorStaysOnRawShare:
    """``min_gpu_pct`` asks whether a kernel is big enough to bother with."""

    def test_large_saturated_kernel_still_clears_the_floor(self):
        # ROI is 0.5, well under the 1.0 floor, but the floor is not an ROI
        # test: the kernel owns half the trace and stays a candidate.
        hot = [_hot("k_big", name="a", gpu_pct=50.0, eff=99.0)]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=1.0, top_n=5)
        assert untried == ["k_big"]

    def test_tiny_kernel_with_full_headroom_is_still_excluded(self):
        # Headroom must not be able to promote a kernel that is too small to
        # matter past the floor.
        hot = [_hot("k_tiny", name="a", gpu_pct=0.4, eff=0.0)]
        untried = kd.untried_hot_reusable_kernels(_state(hot), min_gpu_pct=1.0, top_n=5)
        assert untried == []
