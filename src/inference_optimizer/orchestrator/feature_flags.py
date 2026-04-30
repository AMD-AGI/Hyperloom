"""FeatureFlags — DESIGN §3.4.4 / standalone_agent_design §13 (v0.4 MVP).

The Conductor uses these flags to decide which reactors to spawn, which
periodic jobs to schedule, and whether to enable strategic review etc.
The flag table is the *single* source of truth for "what does this mode
actually run".

v0.4 MVP changes (vs v0.3 Plan A):
    * removed `enable_sage_reactor` / `enable_sage_query_service` (sage
      role deleted entirely)
    * removed `enable_watchdog_reactor` (watchdog renamed to triage)
    * removed `enable_parliament` (parliament mode gone — OBJECTION/VOTE
      intents and 4 related topics deleted)
    * added `enable_triage_reactor` — true in every mode (always-on per
      §13.2)
"""

from __future__ import annotations

from dataclasses import dataclass

from .execution_mode import ExecutionMode


@dataclass(frozen=True)
class FeatureFlags:
    enable_critic_reactor: bool        # F09
    enable_triage_reactor: bool        # F10 (v0.4 — replaces watchdog reactor)
    enable_kernel_reactor: bool        # F10b — Plan A kernel agent
    enable_subagent_delegate: bool     # F12
    enable_persona_distill: bool       # F23
    enable_kb_read: bool               # F25 (v0.4: KB system not implemented;
                                       # flag retained for v0.5 wire-up)
    enable_kb_write: bool              # F24 (same — placeholder)
    enable_strategic_review: bool      # F27
    enable_event_driven_alert: bool    # F21


def build_feature_flags(mode: ExecutionMode) -> FeatureFlags:
    if mode is ExecutionMode.QUICK_PARAM_SWEEP:
        return FeatureFlags(
            enable_critic_reactor=False,
            enable_triage_reactor=True,   # always-on per §13.2
            enable_kernel_reactor=False,
            # Post-Phase 8a (action_executor bridge): quick mode now NEEDS
            # ``delegate`` enabled. The bridge dispatches queued delegate
            # tasks to bundled ``scripts/*.sh`` shell wrappers — this is
            # the ONLY way a quick run can actually produce a baseline
            # number. Critic/kernel stay off (single-mode-eligible).
            enable_subagent_delegate=True,
            enable_persona_distill=False,
            enable_kb_read=False,
            enable_kb_write=False,
            enable_strategic_review=False,
            enable_event_driven_alert=False,
        )
    if mode is ExecutionMode.GUIDED_KERNEL_OPT:
        return FeatureFlags(
            enable_critic_reactor=True,
            enable_triage_reactor=True,
            enable_kernel_reactor=True,
            enable_subagent_delegate=True,
            enable_persona_distill=False,
            enable_kb_read=False,
            enable_kb_write=False,
            enable_strategic_review=False,
            enable_event_driven_alert=False,
        )
    # MARATHON_MULTI_AGENT — guided + checkpointing; same roster as guided
    # in v0.4 (per §13.2 the difference is prompt length + checkpoint
    # cadence rather than a different reactor set).
    return FeatureFlags(
        enable_critic_reactor=True,
        enable_triage_reactor=True,
        enable_kernel_reactor=True,
        enable_subagent_delegate=True,
        enable_persona_distill=False,    # v0.4: distill disabled even in marathon
        enable_kb_read=False,
        enable_kb_write=False,
        enable_strategic_review=True,
        enable_event_driven_alert=True,
    )
