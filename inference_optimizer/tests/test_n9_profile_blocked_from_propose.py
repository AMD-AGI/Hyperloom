"""F3-1 N9: direct `profile` propose_action is hard-blocked when the
roofline composite is on.

Pins the v0.8 form of the N9 rule. Main's N9
(``INFERENCE_OPTIMIZER_ALLOW_DIRECT_PROFILE`` + ``execution_order``
denial inside ``_sequence_denial_for_action``) was rewritten in F3-1 as
``PolicyGate._validate_roofline_composite_supersedes_profile`` —
fires at ``propose_action`` validation time, gated by the
SharedState two-toggle pair (``use_roofline_composite`` +
``deny_direct_profile``), with rule
``n9_deny_direct_profile_when_composite_on``. The atomic-snapshot
guarantee is the same: when the composite is on, every ``profile``
must come from the ``roofline`` executor's internal sub-step rather
than a direct LLM propose.

Tests pin:

* propose_action(profile) denied when both toggles are on, with the
  F3-1 rule + an actionable hint pointing at ``roofline``.
* Either toggle off restores the legacy direct-profile path (lets
  operators A/B the rule before committing to it).
* Defers to ``phase_incompatible`` when ``profile`` is not in the
  current phase's allow-set — single source of truth for the more
  fundamental denial.
* orchestration.md surfaces the rule so the LLM sees the constraint
  before hitting PolicyGate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import PolicyDenied, PolicyGate
from inference_optimizer.orchestrator.shared_state import SharedState


def _policy_with_state(**state_overrides: Any) -> PolicyGate:
    ss = SharedState()
    for key, value in state_overrides.items():
        setattr(ss, key, value)
    gate = PolicyGate(role_registry=default_role_registry())
    gate.shared_state = ss
    return gate


def _propose_profile() -> Intent:
    return Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "profile", "rationale": "test"},
    )


# ---------------------------------------------------------------------------
# Default behaviour — propose(profile) is hard-blocked when both toggles on
# ---------------------------------------------------------------------------
def test_profile_propose_denied_when_both_toggles_on():
    gate = _policy_with_state(
        use_roofline_composite=True,
        deny_direct_profile=True,
        phase="KERNEL",
    )
    with pytest.raises(PolicyDenied) as exc_info:
        gate.validate_intent("orchestration", _propose_profile())
    assert exc_info.value.rule == "n9_deny_direct_profile_when_composite_on"


def test_profile_denial_hint_mentions_roofline():
    gate = _policy_with_state(
        use_roofline_composite=True,
        deny_direct_profile=True,
        phase="KERNEL",
    )
    with pytest.raises(PolicyDenied) as exc_info:
        gate.validate_intent("orchestration", _propose_profile())
    hint = exc_info.value.hint or ""
    assert "roofline" in hint
    # The hint must explain WHY profile is blocked (composite runs the
    # atomic profile + trace_analyze snapshot) so the LLM self-corrects
    # rather than blind-retrying.
    assert "atomic" in hint.lower() or "composite" in hint.lower()


# ---------------------------------------------------------------------------
# Either toggle off → legacy path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_composite,deny_direct", [
    (False, True),     # composite off, deny on -> no-op
    (True, False),     # composite on, deny off -> escape hatch
    (False, False),    # both off -> legacy default
])
def test_either_toggle_off_allows_profile_propose(use_composite, deny_direct):
    gate = _policy_with_state(
        use_roofline_composite=use_composite,
        deny_direct_profile=deny_direct,
        phase="KERNEL",
    )
    # No raise — N9 should defer when either toggle is off.
    gate.validate_intent("orchestration", _propose_profile())


# ---------------------------------------------------------------------------
# Defers to phase_incompatible when phase disallows profile
# ---------------------------------------------------------------------------
def test_n9_defers_to_phase_incompatible_in_prelude(monkeypatch):
    """In PRELUDE, the phase allow-set already denies profile; the N9
    rule defers so the operator sees the more fundamental
    `phase_incompatible` rule rather than a competing denial.

    Strict-phase enforcement is normally opt-in
    (``INFERENCE_OPTIMIZER_STRICT_PHASE=1``); turn it on so the
    phase_incompatible denial is observable in the test."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_STRICT_PHASE", "1")
    gate = _policy_with_state(
        use_roofline_composite=True,
        deny_direct_profile=True,
        phase="PRELUDE",
    )
    with pytest.raises(PolicyDenied) as exc_info:
        gate.validate_intent("orchestration", _propose_profile())
    # Whichever rule fires, it must NOT be N9 — phase_incompatible
    # owns the denial when the phase allow-set rejects the action.
    assert exc_info.value.rule != "n9_deny_direct_profile_when_composite_on"


# ---------------------------------------------------------------------------
# N9 does NOT affect other actions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", [
    "roofline", "baseline", "target_analysis", "explore", "sweep", "report",
])
def test_other_actions_unaffected_by_n9_gate(action):
    gate = _policy_with_state(
        use_roofline_composite=True,
        deny_direct_profile=True,
        phase="KERNEL",
    )
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": action, "rationale": "test"},
    )
    try:
        gate.validate_intent("orchestration", intent)
    except PolicyDenied as denied:
        # Any denial is fine, but it must not come from N9.
        assert denied.rule != "n9_deny_direct_profile_when_composite_on"


# ---------------------------------------------------------------------------
# orchestration.md guidance section presence
# ---------------------------------------------------------------------------
def test_orchestration_md_includes_n9_hard_rule():
    """The orchestration system prompt must include the explicit
    'NEVER propose profile directly' rule so the LLM sees the
    constraint even before hitting PolicyGate."""
    from inference_optimizer.paths import asset_system_prompts_dir
    text = (asset_system_prompts_dir() / "orchestration.md").read_text(
        encoding="utf-8",
    )
    assert "NEVER propose `profile` directly" in text
    # Reference the v0.8 rule name (F3-1).
    assert "n9_deny_direct_profile_when_composite_on" in text
    # Must explain the alternative.
    assert "roofline" in text
