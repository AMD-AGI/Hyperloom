# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Orchestration permission widenings for scheduling-police intents.

Orchestration may now emit PRUNE_BRANCH (Roofline-v2 C3) and
ESCALATE_STRATEGY_CHANGE in addition to the robustness path;
FORCE_DISPATCH stays robustness-only, and Kernel/Critic cannot emit any.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.policy import PolicyDenied, PolicyGate


@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry())


# Orchestration (new permission) — happy path + payload-shape guard
def test_orchestration_can_emit_prune_branch_with_family(gate):
    """C3 enabler: Orchestration forwards roofline advice as PRUNE_BRANCH."""
    gate.validate_intent("orchestration", Intent(
        type=IntentType.PRUNE_BRANCH,
        payload={"family": "kernel_opt", "reason": "compute saturated 92%"},
    ))


def test_orchestration_prune_branch_missing_family_rejected(gate):
    """Payload-shape check must still fire for the new source: an empty ``family`` is denied."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"family": "", "reason": "missing target"},
        ))
    assert exc.value.rule == "payload"


def test_orchestration_prune_branch_missing_family_key_rejected(gate):
    """``family`` key entirely absent → payload denial (not KeyError)."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"reason": "missing target"},
        ))
    assert exc.value.rule == "payload"


# Robustness (pre-existing path) — still works
def test_robustness_can_still_emit_prune_branch(gate):
    """Pre-existing path unchanged — both sources are in the PRUNE_BRANCH allowlist."""
    gate.validate_intent("robustness", Intent(
        type=IntentType.PRUNE_BRANCH,
        payload={"family": "kernel_opt", "reason": "five sequential denials"},
    ))


# Per-intent override widens PRUNE_BRANCH and ESCALATE_STRATEGY_CHANGE;
# FORCE_DISPATCH stays robustness-only.
def test_orchestration_cannot_emit_force_dispatch(gate):
    """FORCE_DISPATCH stays robustness-only — Orchestration's role gate fires first."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.FORCE_DISPATCH,
            payload={"task_id": "t-1", "reason": "x"},
        ))
    assert exc.value.rule == "role"


def test_orchestration_can_emit_escalate_strategy_change(gate):
    """Orchestration may emit phase-advance hints directly."""
    gate.validate_intent("orchestration", Intent(
        type=IntentType.ESCALATE_STRATEGY_CHANGE,
        payload={"next_action_hint": "skip_to_kernel"},
    ))


def test_robustness_can_still_emit_escalate_strategy_change(gate):
    """Robustness retains the original authority for the same intent."""
    gate.validate_intent("robustness", Intent(
        type=IntentType.ESCALATE_STRATEGY_CHANGE,
        payload={"next_action_hint": "skip_to_close"},
    ))


def test_kernel_cannot_emit_escalate_strategy_change(gate):
    """Kernel responder-only — no scheduling-police intents."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("kernel", Intent(
            type=IntentType.ESCALATE_STRATEGY_CHANGE,
            payload={"next_action_hint": "skip_to_close"},
        ))
    assert exc.value.rule == "role"


# Kernel / Critic — not in the PRUNE_BRANCH source allowlist either
def test_kernel_cannot_emit_prune_branch(gate):
    """Kernel responder-only — no scheduling-police authority."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("kernel", Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"family": "params", "reason": "x"},
        ))
    assert exc.value.rule == "role"


def test_critic_cannot_emit_prune_branch(gate):
    """Critic limited to review_verdict / send_message / advice."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("critic", Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"family": "params", "reason": "x"},
        ))
    assert exc.value.rule == "role"
