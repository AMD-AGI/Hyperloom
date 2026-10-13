# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A forge lane whose share pays for no target declines instead of starting work the phase cannot finish."""

from __future__ import annotations

from hyperloom.orchestrator.kernel import lane_budget as lb
from hyperloom.orchestrator.kernel.request_handlers import _lane_underfunded


def test_an_underfunded_lane_reports_a_skip_not_a_failure():
    out = _lane_underfunded("forge_fusion", lb.LANE_FUSION, 223)

    # Skipped, not failed: nothing was attempted, so the ledgers must not learn the work is impossible.
    assert out["status"] == "skipped"
    assert out["error_class"] == "lane_underfunded"
    assert out["kept"] is False
    assert out["decision"] == "REVERT"


def test_the_decline_names_the_share_it_refused_on():
    out = _lane_underfunded("forge_fusion", lb.LANE_FUSION, 223)

    assert out["lane_budget_sec"] == 223
    assert "223s" in out["error"]
    assert lb.LANE_FUSION in out["error"]
    assert "override" in out["error"]


def test_both_forge_lanes_use_the_same_decline():
    fusion = _lane_underfunded("forge_fusion", lb.LANE_FUSION, 10)
    gemm = _lane_underfunded("forge_gemm_tune", lb.LANE_GEMM, 10)

    assert fusion["engine"] == "forge_fusion" and gemm["engine"] == "forge_gemm_tune"
    assert fusion["backend"] == gemm["backend"] == "forge"
    assert fusion["error_class"] == gemm["error_class"] == "lane_underfunded"


def test_the_phase_share_an_observed_session_dispatched_on_is_now_unfundable():
    # 18h session, KERNEL phase entered with minutes left: the fusion share landed at 223 s and forge-fuse was
    # started on a 31 MB trace anyway, writing nothing but the input it was handed.
    allocation = lb.allocate(6.0)[lb.LANE_FUSION]

    assert allocation.budget_sec < lb.FUSION_DEFAULT_TARGET_SEC
    assert allocation.is_fundable is False


def test_a_phase_with_room_still_funds_the_lane():
    allocation = lb.allocate(240.0)[lb.LANE_FUSION]

    assert allocation.is_fundable is True
    assert allocation.max_targets >= 1
