"""atom-PolicyGate anti-regression guards.

The ``framework_atom_action_unsupported`` rule scaffold was deleted
from ``inference_optimizer.orchestrator.policy``:

* ``_ATOM_UNSUPPORTED_ACTIONS`` constant — gone.
* ``_validate_framework_atom_action_unsupported`` helper — gone.
* The two dispatch sites in ``_validate_delegate`` /
  ``_validate_propose_action`` — gone.

atom no longer has any action that needs framework-specific denial at
this layer:

* ``kernel_opt`` / ``integrate_patch`` — atom source roots are in
  PolicyGate's allowlist, ``_REUSABLE_SOURCE_ROOTS``, and the
  server-flag pre-flight probe.
* ``framework_pr`` — Coordinator-driven, not LLM-proposable
  regardless of framework. R1 ``phase_incompatible`` still fires for
  any LLM that tries to propose / delegate it directly (it never
  sits in any phase's LLM-proposable set).
* ``--nodes >= 2`` — guarded at the CLI level
  (``_apply_atom_auto_tighten``).

This file's only job is to make a future reintroduction of the rule
intentional.
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


# ---------------------------------------------------------------------------
# G2 — source-level guards: the scaffold is fully removed
# ---------------------------------------------------------------------------
_POLICY_PATH = Path(policy_module.__file__)


def test_no_atom_unsupported_actions_constant_in_source():
    """``_ATOM_UNSUPPORTED_ACTIONS`` must not appear as a constant
    definition in policy.py. A bare mention in a comment / docstring
    (e.g. release-notes context) is fine and explicitly tolerated."""
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
    """The rule name must not appear as a ``rule=`` parameter in
    policy.py — no validator emits it any more."""
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
    """The rule name ``framework_atom_action_unsupported`` DOES still
    appear in ``policy.py`` — but only inside an explanatory NOTE
    block-comment that documents *why* the rule does not exist. This
    test pins that the bare mention is documentation (no surrounding
    ``def`` / ``rule=`` / constant assignment) so a future reader
    doesn't accidentally tighten the guard above and trip on the
    provenance comment.

    If you ever physically delete the explanatory comment, you can
    also delete this test — the three guards above will continue to
    hold the contract.
    """
    src = _POLICY_PATH.read_text(encoding="utf-8")
    # The rule name appears.
    assert "framework_atom_action_unsupported" in src, (
        "NOTE provenance comment was removed; if intentional, drop "
        "this test together with it. The other three guards above "
        "will continue to enforce that no rule of this name is emitted."
    )
    # ...but only inside a NOTE comment block (the word ``NOTE``
    # appears in the surrounding ~30 lines).
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


# ---------------------------------------------------------------------------
# Behavioural guards: the cross-framework LLM-proposability rule still
# covers framework_pr under atom, so the hole the atom-specific rule
# guarded against (LLM → framework_pr) stays closed.
# ---------------------------------------------------------------------------
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
    """LLM-proposed ``framework_pr`` under FRAMEWORK=atom must still
    be denied — now by R1 ``phase_incompatible`` (it is
    Coordinator-managed and never LLM-proposable), not by the now-
    removed atom-specific rule. Confirms removing the atom-specific
    gate didn't open up the LLM-proposability hole."""
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
    """``kernel_opt`` and ``integrate_patch`` have real execution paths
    on atom. They may still be denied for
    unrelated reasons (e.g. ``kernel_owned_by_kernel_agent``), but the
    removed atom rule must not be the source of denial."""
    monkeypatch.setenv("FRAMEWORK", "atom")
    gate = _gate(_BareSharedState(framework="atom"))
    try:
        gate.validate_intent("orchestration", _delegate(action_name))
    except PolicyDenied as exc:
        assert exc.rule != "framework_atom_action_unsupported", (
            f"{action_name} must never trigger the removed atom rule; "
            f"got rule={exc.rule!r}"
        )
