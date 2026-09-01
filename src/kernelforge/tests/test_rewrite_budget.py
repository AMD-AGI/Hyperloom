"""Tests for the centralized FlyDSL rewrite wall-clock policy."""

from __future__ import annotations

from kernelforge.rewrite_by_flydsl import budget, report


def test_search_reserves_applyback_finalization_time():
    policy = budget.DEFAULT_REWRITE_BUDGET
    deadline = 10_000.0

    assert policy.search_stop_unix(deadline) == (deadline - policy.applyback_reserve_sec)
    assert policy.applyback_reserve_sec == 20 * 60


def test_host_validation_timeout_tracks_the_remaining_wall_clock(monkeypatch):
    policy = budget.DEFAULT_REWRITE_BUDGET
    now = 1_000.0
    monkeypatch.setattr(budget.time, "time", lambda: now)

    deadline = now + policy.applyback_post_agent_reserve_sec + 300
    assert policy.host_validation_timeout_sec(deadline) == 300
    # A deadline already spent still leaves a timeout the caller can pass on.
    assert policy.host_validation_timeout_sec(now) == 1


def test_applyback_start_threshold_is_derived_from_named_reserves(monkeypatch):
    policy = budget.DEFAULT_REWRITE_BUDGET
    now = 1_000.0
    monkeypatch.setattr(budget.time, "time", lambda: now)

    assert policy.can_start_applyback(now + policy.applyback_start_min_remaining_sec + 1)
    assert not policy.can_start_applyback(now + policy.applyback_start_min_remaining_sec)


def test_retry_agent_budget_is_split_without_consuming_host_reserve(monkeypatch):
    policy = budget.DEFAULT_REWRITE_BUDGET
    now = 1_000.0
    monkeypatch.setattr(budget.time, "time", lambda: now)
    deadline = now + policy.applyback_host_validation_reserve_sec + 1_200

    assert (
        policy.agent_timeout_sec(
            deadline_unix=deadline,
            configured_timeout_sec=1_800,
            attempts_left=2,
        )
        == 600
    )
    assert (
        policy.agent_timeout_sec(
            deadline_unix=deadline,
            configured_timeout_sec=1_800,
            attempts_left=1,
        )
        == 1_200
    )


def test_result_reports_the_effective_budget_policy():
    result = report.build_result(
        op_name="softmax",
        port_ok=False,
        port_attempts=0,
        source_ms=None,
        optimize_result={},
    )

    assert result.budget_policy == budget.DEFAULT_REWRITE_BUDGET.to_dict()
    assert result.budget_policy["applyback_reserve_sec"] == 1_200
