"""FeatureFlags — DESIGN §3.4.4.

The Conductor uses these flags to decide which reactors to spawn, which
periodic jobs to schedule, and whether to enable strategic review etc.
The flag table is the *single* source of truth for "what does this mode
actually run".
"""

from __future__ import annotations

from dataclasses import dataclass

from .execution_mode import ExecutionMode


@dataclass(frozen=True)
class FeatureFlags:
    enable_critic_reactor: bool        # F09
    enable_watchdog_reactor: bool      # F10
    enable_sage_reactor: bool          # F11 resident
    enable_sage_query_service: bool    # F11 KB query helper
    enable_subagent_delegate: bool     # F12
    enable_persona_distill: bool       # F23
    enable_kb_read: bool               # F25
    enable_kb_write: bool               # F24
    enable_strategic_review: bool      # F27
    enable_cross_run_synthesis: bool   # F28
    enable_parliament: bool            # F20
    enable_event_driven_alert: bool    # F21


def build_feature_flags(mode: ExecutionMode) -> FeatureFlags:
    if mode is ExecutionMode.QUICK_PARAM_SWEEP:
        return FeatureFlags(
            enable_critic_reactor=False,
            enable_watchdog_reactor=False,
            enable_sage_reactor=False,
            enable_sage_query_service=True,
            # Post-Phase 8a (action_executor bridge): quick mode now NEEDS
            # ``delegate`` enabled. The bridge dispatches queued delegate
            # tasks to bundled ``scripts/*.sh`` shell wrappers — this is
            # the ONLY way a quick run can actually produce a baseline
            # number. Keeping it False forced the executor to shell out
            # via raw Bash, which is slower and fragile. Critic/sage/
            # watchdog stay off (still single-role).
            enable_subagent_delegate=True,
            enable_persona_distill=False,
            enable_kb_read=True,
            enable_kb_write=True,
            enable_strategic_review=False,
            enable_cross_run_synthesis=False,
            enable_parliament=False,
            enable_event_driven_alert=False,
        )
    if mode is ExecutionMode.GUIDED_KERNEL_OPT:
        return FeatureFlags(
            enable_critic_reactor=True,
            enable_watchdog_reactor=False,
            enable_sage_reactor=False,
            enable_sage_query_service=True,
            enable_subagent_delegate=True,
            enable_persona_distill=False,
            enable_kb_read=True,
            enable_kb_write=True,
            enable_strategic_review=False,
            enable_cross_run_synthesis=False,
            enable_parliament=False,
            enable_event_driven_alert=False,
        )
    # MARATHON_MULTI_AGENT
    return FeatureFlags(
        enable_critic_reactor=True,
        enable_watchdog_reactor=True,
        enable_sage_reactor=True,
        enable_sage_query_service=True,
        enable_subagent_delegate=True,
        enable_persona_distill=True,
        enable_kb_read=True,
        enable_kb_write=True,
        enable_strategic_review=True,
        enable_cross_run_synthesis=True,
        enable_parliament=True,
        enable_event_driven_alert=True,
    )
