"""F3-1 / F3-2 / F3-5 — PolicyGate gates for the Roofline-v2 toggles.

Three independent ``propose_action`` rules:

* **F3-1 N9** — direct ``profile`` denied while
  ``--use-roofline-composite`` AND ``--deny-direct-profile`` are both
  on. Forces atomic ``roofline`` so the snapshot_id counter +
  analysis.md cache stay aligned with the trace.
* **F3-2 N19c** — ``kernel_opt`` denied when
  ``--gain-driven-kernel-opt`` is on AND the moving average of the
  last three accepted ``gain_per_stack_entry`` ``delta_pct`` values is
  ``>= 0.5%``: cheap rounds are still earning, kernel_opt is
  premature.
* **F3-5** (always-on) — ``kernel_opt`` denied when
  ``gain_per_stack_entry`` is empty (replaces v0.6's
  ``backends_attempts < 1`` / ``params_attempts < 1`` sequence
  denials, which had no v0.8 writers).

All three are wired into both ``_validate_propose_action`` and
``_validate_delegate`` so kernel_opt cannot bypass the gates by
skipping propose_action.

Reference: ``plan_roofline_framework/F3_policygate_advisory.MD``.
"""
from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.policy import (
    PolicyDenied,
    PolicyGate,
)
from inference_optimizer.orchestrator.shared_state import SharedState


def _gate(state: SharedState | None) -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        action_registry=None,
        session_dir=None,
        strict_paths=False,
        shared_state=state,
    )


def _orch(gate: PolicyGate):
    return gate.role_registry["orchestration"]


def _propose(action: str, **extra) -> dict:
    payload = {"action_name": action}
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# F3-1 — N9 deny direct profile when composite is on
# ---------------------------------------------------------------------------


def _state_n9_on() -> SharedState:
    s = SharedState()
    s.use_roofline_composite = True
    s.deny_direct_profile = True
    # F3-5 baseline correctness rule needs at least one accepted
    # variant on record so kernel_opt isn't ALSO blocked by the
    # explore-minimum gate; F3-1 tests don't touch kernel_opt anyway,
    # but seeding here keeps the test fixtures share-friendly.
    s.gain_per_stack_entry = [{"delta_pct": 0.0}]
    return s


def test_n9_denies_direct_profile_when_both_toggles_on():
    gate = _gate(_state_n9_on())
    with pytest.raises(PolicyDenied) as exc_info:
        gate._validate_propose_action(_orch(gate), _propose("profile"))
    assert exc_info.value.rule == "n9_deny_direct_profile_when_composite_on"
    assert "roofline" in (exc_info.value.hint or "")


def test_n9_allows_roofline_when_both_toggles_on():
    gate = _gate(_state_n9_on())
    gate._validate_propose_action(_orch(gate), _propose("roofline"))


def test_n9_allows_profile_when_composite_off():
    s = _state_n9_on()
    s.use_roofline_composite = False
    gate = _gate(s)
    gate._validate_propose_action(_orch(gate), _propose("profile"))


def test_n9_allows_profile_when_only_composite_on():
    """The deny-toggle is the operator escape hatch — until it flips,
    direct ``profile`` stays legal so operators can A/B test."""
    s = _state_n9_on()
    s.deny_direct_profile = False
    gate = _gate(s)
    gate._validate_propose_action(_orch(gate), _propose("profile"))


def test_n9_also_fires_on_delegate_channel():
    """Defense in depth: delegate creates the real task, so the gate
    must apply there too. Otherwise the LLM could skip propose_action
    and still create a profile task."""
    gate = _gate(_state_n9_on())
    payload = {"action_name": "profile", "params": {}}
    with pytest.raises(PolicyDenied) as exc_info:
        gate._validate_delegate(_orch(gate), payload)
    assert exc_info.value.rule == "n9_deny_direct_profile_when_composite_on"


# ---------------------------------------------------------------------------
# F3-2 — N19c gain-driven kernel_opt lock
# ---------------------------------------------------------------------------


def _state_n19c_on(deltas: list[float]) -> SharedState:
    s = SharedState()
    s.phase = "KERNEL"
    s.gain_driven_kernel_opt = True
    s.gain_per_stack_entry = [{"delta_pct": float(d)} for d in deltas]
    return s


def test_n19c_off_by_default_allows_kernel_opt():
    s = SharedState()
    s.phase = "KERNEL"
    s.gain_driven_kernel_opt = False
    s.gain_per_stack_entry = [{"delta_pct": 5.0}]   # would deny if N19c on
    gate = _gate(s)
    gate._validate_propose_action(_orch(gate), _propose("kernel_opt"))


def test_n19c_skipped_outside_kernel_phase():
    """N19c is a within-KERNEL gate; outside the phase, defer to the
    phase allowlist (phase_incompatible owns that denial)."""
    s = SharedState()
    s.phase = "EXPLORE"
    s.gain_driven_kernel_opt = True
    s.gain_per_stack_entry = [{"delta_pct": 5.0}, {"delta_pct": 5.0},
                              {"delta_pct": 5.0}]
    # No phase allowlist wired here -> propose_action passes.
    gate = _gate(s)
    gate._validate_propose_action(_orch(gate), _propose("kernel_opt"))


def test_n19c_denies_when_window_short():
    """Need 3 entries for the moving-average window."""
    s = _state_n19c_on([3.0, 1.0])
    gate = _gate(s)
    with pytest.raises(PolicyDenied, match="cheap-round"):
        gate._validate_propose_action(_orch(gate), _propose("kernel_opt"))


def test_n19c_denies_when_average_above_epsilon():
    s = _state_n19c_on([2.0, 1.5, 1.2])  # avg ~ 1.57 > 0.5
    gate = _gate(s)
    with pytest.raises(PolicyDenied, match="still earning"):
        gate._validate_propose_action(_orch(gate), _propose("kernel_opt"))


def test_n19c_allows_when_average_below_epsilon():
    s = _state_n19c_on([0.4, 0.3, 0.2])  # avg 0.3 < 0.5
    gate = _gate(s)
    gate._validate_propose_action(_orch(gate), _propose("kernel_opt"))


def test_n19c_skips_resumed_entries_with_none_delta():
    """Resumed/seeded entries carry delta_pct=None; the rule must skip
    them and reach back further for real numbers."""
    s = SharedState()
    s.phase = "KERNEL"
    s.gain_driven_kernel_opt = True
    s.gain_per_stack_entry = [
        {"delta_pct": 0.1},
        {"delta_pct": 0.2},
        {"delta_pct": 0.3},
        {"delta_pct": None},
        {"delta_pct": None},
    ]
    gate = _gate(s)
    gate._validate_propose_action(_orch(gate), _propose("kernel_opt"))


def test_n19c_does_not_affect_non_kernel_opt_actions():
    s = _state_n19c_on([5.0, 5.0, 5.0])
    gate = _gate(s)
    for action in ("explore", "specialist", "roofline", "report"):
        gate._validate_propose_action(_orch(gate), _propose(action))


# ---------------------------------------------------------------------------
# F3-5 — explore_attempts_minimum_before_kernel_opt (always on)
# ---------------------------------------------------------------------------


def test_f3_5_denies_kernel_opt_when_no_explore_succeeded():
    s = SharedState()
    s.phase = "KERNEL"
    s.gain_per_stack_entry = []
    gate = _gate(s)
    with pytest.raises(PolicyDenied) as exc_info:
        gate._validate_propose_action(_orch(gate), _propose("kernel_opt"))
    assert exc_info.value.rule == \
        "explore_attempts_minimum_before_kernel_opt"


def test_f3_5_allows_kernel_opt_when_at_least_one_entry():
    s = SharedState()
    s.phase = "KERNEL"
    s.gain_per_stack_entry = [{"delta_pct": 1.5}]
    gate = _gate(s)
    gate._validate_propose_action(_orch(gate), _propose("kernel_opt"))


def test_f3_5_skipped_outside_kernel_phase():
    """Phase allowlist owns the 'kernel_opt outside KERNEL' denial."""
    s = SharedState()
    s.phase = "EXPLORE"
    s.gain_per_stack_entry = []
    gate = _gate(s)
    gate._validate_propose_action(_orch(gate), _propose("kernel_opt"))


def test_f3_5_does_not_affect_non_kernel_opt_actions():
    s = SharedState()
    s.phase = "KERNEL"
    s.gain_per_stack_entry = []  # would block kernel_opt
    gate = _gate(s)
    for action in ("explore", "specialist", "roofline", "integrate_patch"):
        gate._validate_propose_action(_orch(gate), _propose(action))


def test_f3_5_no_legacy_backends_or_params_attempts_rules():
    """Defensive: ensure the v0.6 ``backends_attempts < 1`` /
    ``params_attempts < 1`` sequence_denials are not present on this
    branch — they had no v0.8 writer and would have permanently
    locked kernel_opt.

    A rule is identified by either a ``rule="..."`` argument to
    PolicyDenied or a ``raise PolicyDenied(...)`` line whose nearest
    ``rule=`` literal carries the name. We grep for active rule_id
    strings rather than free-form mentions so the F3-5 helper's
    docstring (which intentionally references the retired names)
    does not trip the guard.
    """
    import re
    from inference_optimizer.orchestrator import policy as policy_mod

    src = open(policy_mod.__file__).read()
    legacy_rule_ids = re.findall(
        r'rule\s*=\s*"((?:backends|params)_attempts[^"]*)"', src,
    )
    assert legacy_rule_ids == [], (
        "Legacy v0.6 rule_id(s) still active in policy.py: "
        f"{legacy_rule_ids!r}"
    )
