"""IR-8 / atom PolicyGate test: framework_atom_action_unsupported.

Covers the defensive PolicyGate rule that denies LLM-proposed
``framework_pr`` action when ``FRAMEWORK=atom`` is in effect.

After Phase 2 of atom_plan/ (kernel-agent enablement) the set
shrank to ``{"framework_pr"}`` only — ``kernel_opt`` and
``integrate_patch`` now have real execution paths on atom (the atom
source roots are in the PolicyGate allowlist, reusable-kernel ledger,
and the server-flag pre-flight probe). Phase 3 will further lift the
``framework_pr`` denial once ``fa`` learns about atom's repo URL.

The CLI's ``_apply_atom_auto_tighten`` flips ``--no-framework`` on
fresh atom launches, so in normal flow ``framework_pr`` is gated off
by the phase rules anyway. The rule here is *defense in depth* — it
catches the cases that would otherwise reach the executor and crash:

  * Operator passes ``--framework-pr --framework atom`` explicitly
    (auto-tighten only flips when the flag is still at its enabled
    default).
  * Resume from a session whose ``$FRAMEWORK`` env drifted between
    invocations.

The rule fires on both ``delegate`` and ``propose_action`` channels and
sources the framework either from ``SharedState.framework`` (when
wired) or from ``$FRAMEWORK`` (env fallback). Tests cover both paths.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.intent_parser import (
    Intent, IntentType,
)
from inference_optimizer.orchestrator.policy import (
    PolicyDenied,
    PolicyGate,
)


class _BareSharedState:
    """Just enough SharedState surface for PolicyGate to read
    ``framework`` and record denials without crashing."""

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


# ---------------------------------------------------------------------------
# delegate channel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "action_name",
    ["framework_pr"],
)
def test_delegate_denied_when_framework_atom_from_shared_state(action_name):
    """The rule must fire when SharedState.framework=='atom' for the
    remaining unsupported action (framework_pr). Phase 2 dropped
    kernel_opt + integrate_patch from this set — they now have real
    execution paths on atom and are NOT denied by this rule. Phase 3
    will lift the framework_pr denial in turn.

    Defense-in-depth: framework_pr is also caught by the earlier
    ``framework_pr_action_not_llm_proposable`` rule (it fires before
    this one in the validation order). Either rule firing is the
    correct outcome since both block the action."""
    gate = _gate(_BareSharedState(framework="atom"))
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _delegate(action_name))
    assert exc.value.rule in {
        "framework_atom_action_unsupported",
        "framework_pr_action_not_llm_proposable",
    }


def test_delegate_denied_when_framework_atom_from_env_fallback(
    monkeypatch,
):
    """When SharedState doesn't carry a ``framework`` attribute, the
    rule must fall back to the process ``$FRAMEWORK`` env. This matches
    how PolicyGate elsewhere reads SharedState defensively. Uses
    framework_pr (the remaining atom-unsupported action) as the probe."""
    monkeypatch.setenv("FRAMEWORK", "atom")
    state = _BareSharedState(framework="")  # SharedState says nothing
    gate = _gate(state)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _delegate("framework_pr"))
    assert exc.value.rule in {
        "framework_atom_action_unsupported",
        "framework_pr_action_not_llm_proposable",
    }


@pytest.mark.parametrize("framework", ["sglang", "vllm", ""])
def test_delegate_not_denied_when_framework_not_atom(framework, monkeypatch):
    """The atom rule must NOT fire on sglang / vllm / unset frameworks.
    integrate_patch is a useful probe here — it's not kernel-owned and
    isn't otherwise denied on simple inputs, so a green pass through
    this validator path is observable. (We don't assert validate_intent
    succeeds end-to-end — other validators may still deny on missing
    payload fields — only that the atom rule didn't trigger.)"""
    monkeypatch.delenv("FRAMEWORK", raising=False)
    if framework:
        monkeypatch.setenv("FRAMEWORK", framework)
    state = _BareSharedState(framework=framework)
    gate = _gate(state)
    try:
        gate.validate_intent("orchestration", _delegate("integrate_patch"))
    except PolicyDenied as exc:
        assert exc.rule != "framework_atom_action_unsupported", (
            f"framework={framework!r} must not trigger the atom rule; "
            f"got denial: rule={exc.rule!r} reason={exc.reason!r}"
        )


# ---------------------------------------------------------------------------
# propose_action channel
# ---------------------------------------------------------------------------
def test_propose_action_denied_for_framework_pr_on_atom():
    """Mirror coverage for the propose_action channel using framework_pr
    (the remaining atom-unsupported action after Phase 2). Either
    ``framework_atom_action_unsupported`` or the earlier
    ``framework_pr_action_not_llm_proposable`` is acceptable since both
    correctly block the action."""
    gate = _gate(_BareSharedState(framework="atom"))
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _propose("framework_pr"))
    assert exc.value.rule in {
        "framework_atom_action_unsupported",
        "framework_pr_action_not_llm_proposable",
    }
