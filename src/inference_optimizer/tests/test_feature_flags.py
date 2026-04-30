"""Tests for orchestrator/feature_flags.py — v0.4 MVP flag matrix
(standalone_agent_design §13).
"""

from __future__ import annotations

from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.feature_flags import build_feature_flags


def test_quick_flag_set():
    flags = build_feature_flags(ExecutionMode.QUICK_PARAM_SWEEP)
    assert flags.enable_critic_reactor is False
    assert flags.enable_triage_reactor is True   # always-on per §13.2
    assert flags.enable_kernel_reactor is False
    # Post-Phase 8a: action_executor bridge requires delegate to be on
    # in EVERY mode.
    assert flags.enable_subagent_delegate is True
    assert flags.enable_kb_read is False
    assert flags.enable_kb_write is False
    assert flags.enable_strategic_review is False
    assert flags.enable_persona_distill is False


def test_guided_flag_set():
    flags = build_feature_flags(ExecutionMode.GUIDED_KERNEL_OPT)
    assert flags.enable_critic_reactor is True
    assert flags.enable_triage_reactor is True
    assert flags.enable_kernel_reactor is True
    assert flags.enable_subagent_delegate is True
    assert flags.enable_persona_distill is False
    assert flags.enable_strategic_review is False


def test_marathon_flag_set():
    flags = build_feature_flags(ExecutionMode.MARATHON_MULTI_AGENT)
    # v0.4 — guided/marathon roster is identical; marathon adds strategic
    # review + event_driven_alert.
    assert flags.enable_critic_reactor is True
    assert flags.enable_triage_reactor is True
    assert flags.enable_kernel_reactor is True
    assert flags.enable_subagent_delegate is True
    assert flags.enable_strategic_review is True
    assert flags.enable_event_driven_alert is True
    # v0.4 disabled persona distill in MVP (KB system not implemented).
    assert flags.enable_persona_distill is False


def test_triage_reactor_always_on_v04():
    """Triage is always-on across every mode."""
    for mode in ExecutionMode:
        flags = build_feature_flags(mode)
        assert flags.enable_triage_reactor is True, (
            f"triage must be enabled in mode={mode!r}"
        )


def test_no_legacy_flags_remain_v04():
    """v0.4 — these flags were removed; the dataclass should not carry them."""
    flags = build_feature_flags(ExecutionMode.MARATHON_MULTI_AGENT)
    for legacy in (
        "enable_parliament",
        "enable_sage_reactor",
        "enable_sage_query_service",
        "enable_watchdog_reactor",
        "enable_cross_run_synthesis",
    ):
        assert not hasattr(flags, legacy), (
            f"v0.4 removed flag {legacy!r}; FeatureFlags should no longer carry it"
        )
