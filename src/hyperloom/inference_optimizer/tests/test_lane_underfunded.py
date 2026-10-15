# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A forge lane whose share pays for no target declines instead of starting work the phase cannot finish."""

from __future__ import annotations

from hyperloom.orchestrator.kernel import lane_budget as lb
from hyperloom.orchestrator.kernel.request_handlers import _unfunded_lane_skip


def test_an_underfunded_lane_reports_a_skip_not_a_verdict():
    """Nothing was attempted, so the ledgers must not learn the work is impossible."""
    out = _unfunded_lane_skip(lb.LANE_FUSION)

    assert out["status"] == "skipped"
    assert "kept" not in out and "decision" not in out


def test_the_decline_names_the_lane_and_the_budget_it_refused_on():
    out = _unfunded_lane_skip(lb.LANE_FUSION)

    assert lb.LANE_FUSION in out["skip_reason"]
    assert "phase time budget" in out["skip_reason"]


def test_both_forge_lanes_decline_through_the_same_shape():
    fusion = _unfunded_lane_skip(lb.LANE_FUSION)
    gemm = _unfunded_lane_skip(lb.LANE_GEMM)

    assert fusion["backend"] == gemm["backend"] == "forge"
    assert fusion["error_class"] == gemm["error_class"]


def test_a_fusion_share_too_small_for_one_recipe_funds_nothing():
    """A fusion share was counted fundable whenever it was non-zero, so the skip could not fire for this lane.

    An 18h session entered KERNEL with minutes left, the fusion share landed at
    223 s, and forge-fuse was started on a 31 MB trace anyway --- writing
    nothing but the input it was handed. Pricing a recipe is what lets the
    share report that it pays for no target.
    """
    starved = lb.allocate(6.0)[lb.LANE_FUSION]

    assert starved.budget_sec > 0
    assert starved.max_targets == 0
    assert starved.is_fundable is False


def test_a_share_that_pays_for_recipes_still_funds_them():
    generous = lb.allocate(600.0)[lb.LANE_FUSION]

    assert generous.is_fundable is True
    assert generous.max_targets >= 1
