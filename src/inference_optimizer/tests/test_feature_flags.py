"""Tests for orchestrator/feature_flags.py — DESIGN §3.4.4 matrix."""

from __future__ import annotations

from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.feature_flags import build_feature_flags


def test_quick_flag_set():
    flags = build_feature_flags(ExecutionMode.QUICK_PARAM_SWEEP)
    assert flags.enable_critic_reactor is False
    assert flags.enable_watchdog_reactor is False
    assert flags.enable_sage_reactor is False
    assert flags.enable_sage_query_service is True
    assert flags.enable_subagent_delegate is False
    assert flags.enable_kb_read is True
    assert flags.enable_kb_write is True
    assert flags.enable_strategic_review is False
    assert flags.enable_persona_distill is False


def test_guided_flag_set():
    flags = build_feature_flags(ExecutionMode.GUIDED_KERNEL_OPT)
    assert flags.enable_critic_reactor is True
    assert flags.enable_watchdog_reactor is False
    assert flags.enable_subagent_delegate is True
    assert flags.enable_sage_reactor is False
    assert flags.enable_sage_query_service is True
    assert flags.enable_persona_distill is False
    assert flags.enable_strategic_review is False
    assert flags.enable_parliament is False


def test_marathon_flag_set():
    flags = build_feature_flags(ExecutionMode.MARATHON_MULTI_AGENT)
    # All optional features on for marathon
    assert all(
        getattr(flags, attr) for attr in (
            "enable_critic_reactor",
            "enable_watchdog_reactor",
            "enable_sage_reactor",
            "enable_sage_query_service",
            "enable_subagent_delegate",
            "enable_persona_distill",
            "enable_kb_read",
            "enable_kb_write",
            "enable_strategic_review",
            "enable_cross_run_synthesis",
            "enable_parliament",
            "enable_event_driven_alert",
        )
    )


def test_kb_is_always_enabled():
    """ADR-28: L4 KB read/write are on in every mode."""
    for mode in ExecutionMode:
        flags = build_feature_flags(mode)
        assert flags.enable_kb_read is True, mode
        assert flags.enable_kb_write is True, mode


def test_sage_query_service_is_always_enabled():
    """ADR-29: Sage is reachable as a query helper even in quick/guided."""
    for mode in ExecutionMode:
        flags = build_feature_flags(mode)
        assert flags.enable_sage_query_service is True, mode
