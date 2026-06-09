# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""atom-PolicyGate anti-regression guards.

The ``framework_atom_action_unsupported`` rule scaffold (constant, helper, and
two dispatch sites) was deleted from ``policy``; atom needs no framework-specific
denial here. This file makes a future reintroduction intentional.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator import policy as policy_module
from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.protocol.intent import (
    Intent, IntentType,
)
from inference_optimizer.orchestrator.policy import (
    PolicyDenied,
    PolicyGate,
)


# G2 — source-level guards: the scaffold is fully removed
_POLICY_PATH = Path(policy_module.__file__)


def test_no_atom_unsupported_actions_constant_in_source():
    """``_ATOM_UNSUPPORTED_ACTIONS`` must not appear as a constant definition in policy.py."""
    src = _POLICY_PATH.read_text(encoding="utf-8")
    assert "_ATOM_UNSUPPORTED_ACTIONS:" not in src, (
        "_ATOM_UNSUPPORTED_ACTIONS constant was removed; "
        "re-introducing it must be intentional (and should re-add a "
        "dispatch site too)."
    )


def test_no_validate_framework_atom_action_unsupported_helper_in_source():
    """The validator helper must not be defined in policy.py."""
    src = _POLICY_PATH.read_text(encoding="utf-8")
    assert "def _validate_framework_atom_action_unsupported" not in src, (
        "_validate_framework_atom_action_unsupported helper was removed."
    )


def test_no_framework_atom_action_unsupported_rule_name_in_source():
    """The rule name must not appear as a ``rule=`` parameter in policy.py."""
    src = _POLICY_PATH.read_text(encoding="utf-8")
    assert 'rule="framework_atom_action_unsupported"' not in src, (
        "framework_atom_action_unsupported rule is no longer emitted; "
        "if you need an atom-specific denial, pick a different rule name "
        "to avoid confusing operators who searched logs for the old one."
    )


def test_atom_unsupported_actions_attribute_removed_from_policy_gate():
    """Runtime-level guard: PolicyGate must not expose the constant."""
    assert not hasattr(PolicyGate, "_ATOM_UNSUPPORTED_ACTIONS"), (
        "PolicyGate._ATOM_UNSUPPORTED_ACTIONS was removed; "
        "reintroducing requires updating this guard intentionally."
    )


def test_validate_helper_removed_from_policy_gate():
    """Runtime-level guard: PolicyGate must not expose the validator."""
    assert not hasattr(PolicyGate, "_validate_framework_atom_action_unsupported"), (
        "PolicyGate._validate_framework_atom_action_unsupported was "
        "removed."
    )


def test_historical_rule_name_mention_is_documented_provenance_only():
    """The rule name still appears in policy.py, but only inside a NOTE provenance comment."""
    src = _POLICY_PATH.read_text(encoding="utf-8")
    assert "framework_atom_action_unsupported" in src, (
        "NOTE provenance comment was removed; if intentional, drop "
        "this test together with it. The other three guards above "
        "will continue to enforce that no rule of this name is emitted."
    )
    idx = src.find("framework_atom_action_unsupported")
    window = src[max(0, idx - 600): idx + 600]
    assert "NOTE" in window, (
        "framework_atom_action_unsupported mention is no longer "
        "documented as historical provenance; if you re-add the rule, "
        "drop this guard AND update the other three guards above."
    )
    assert "no" in window.lower() and "exists" in window.lower(), (
        "framework_atom_action_unsupported mention is not framed as "
        "a 'no such rule exists' note; see the comment block at the "
        "top of the rule section in policy.py."
    )


# Behavioural guards: the cross-framework LLM-proposability rule still covers
# framework_pr under atom, keeping the LLM → framework_pr hole closed.
class _BareSharedState:
    """Minimal SharedState surface for PolicyGate."""

    def __init__(self, framework: str = "atom", phase: str = "EXPLORE"):
        self.framework = framework
        self.phase = phase
        self.tick = 0

    def record_policy_denial(self, **_kwargs):
        return 1


def _gate(state) -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=state,
    )


def _delegate(action_name: str, **extra) -> Intent:
    payload = {"action_name": action_name}
    payload.update(extra)
    return Intent(type=IntentType.DELEGATE, payload=payload)


def _propose(action_name: str, **extra) -> Intent:
    payload = {"action_name": action_name}
    payload.update(extra)
    return Intent(type=IntentType.PROPOSE_ACTION, payload=payload)


@pytest.mark.parametrize("channel", ["delegate", "propose_action"])
def test_framework_pr_still_denied_via_phase_incompatible_under_atom(
    channel: str,
):
    """LLM-proposed ``framework_pr`` under atom is still denied by R1 ``phase_incompatible``, not the removed atom rule."""
    gate = _gate(_BareSharedState(framework="atom"))
    intent = _delegate("framework_pr") if channel == "delegate" else _propose("framework_pr")
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "phase_incompatible", (
        f"expected R1 phase_incompatible to deny framework_pr under "
        f"atom; got rule={exc.value.rule!r}"
    )


@pytest.mark.parametrize("action_name", ["kernel_opt", "integrate_patch"])
def test_kernel_actions_not_denied_by_removed_atom_rule(action_name, monkeypatch):
    """``kernel_opt`` / ``integrate_patch`` may be denied for other reasons, but never by the removed atom rule."""
    monkeypatch.setenv("FRAMEWORK", "atom")
    gate = _gate(_BareSharedState(framework="atom"))
    try:
        gate.validate_intent("orchestration", _delegate(action_name))
    except PolicyDenied as exc:
        assert exc.rule != "framework_atom_action_unsupported", (
            f"{action_name} must never trigger the removed atom rule; "
            f"got rule={exc.rule!r}"
        )
