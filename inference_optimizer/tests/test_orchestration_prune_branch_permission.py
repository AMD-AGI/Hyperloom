"""Orchestration permission widenings for scheduling-police intents.

Two intents are reachable from Orchestration in addition to the
robustness path:

* PRUNE_BRANCH (Roofline-v2 C3) — Orchestration forwards the
  structured ``suggested_prunes`` advice from the ``roofline`` action.
* ESCALATE_STRATEGY_CHANGE (loosen P3_18 18A) — Orchestration emits
  phase-advance hints (``skip_to_kernel`` / ``skip_to_sweep`` /
  ``skip_to_close``) directly instead of bouncing through robustness.

FORCE_DISPATCH stays robustness-only: it is a recovery-shaped intent
that bypasses normal task accounting. Kernel / Critic cannot emit any
of these (the role.allowed_intents check runs before the per-intent
source allowlist).
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.policy import PolicyDenied, PolicyGate


@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate(role_registry=default_role_registry())


# ---------------------------------------------------------------------------
# Orchestration (new permission) — happy path + payload-shape guard
# ---------------------------------------------------------------------------
def test_orchestration_can_emit_prune_branch_with_family(gate):
    """Roofline-driven advice → Orchestration forwards as PRUNE_BRANCH.

    This is the C3 enabler — without it the ``roofline`` action's
    ``suggested_prunes`` would have no path to ``pruned_families`` and
    the entire C4/C5 chain would degrade to soft hint only.
    """
    gate.validate_intent("orchestration", Intent(
        type=IntentType.PRUNE_BRANCH,
        payload={"family": "kernel_opt", "reason": "compute saturated 92%"},
    ))


def test_orchestration_prune_branch_missing_family_rejected(gate):
    """Payload-shape check must still fire for the new source.

    The roofline-analyzer LLM output is not contractually trusted; even
    after C2 normalization a downstream wiring bug could surface an
    empty ``family``. PolicyGate is the last line of defense before
    ``add_pruned_family`` would commit garbage.
    """
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


# ---------------------------------------------------------------------------
# Robustness (pre-existing path) — still works
# ---------------------------------------------------------------------------
def test_robustness_can_still_emit_prune_branch(gate):
    """Pre-existing path unchanged — both sources are now in the
    PRUNE_BRANCH-specific allowlist."""
    gate.validate_intent("robustness", Intent(
        type=IntentType.PRUNE_BRANCH,
        payload={"family": "kernel_opt", "reason": "five sequential denials"},
    ))


# ---------------------------------------------------------------------------
# Per-intent override widens PRUNE_BRANCH and ESCALATE_STRATEGY_CHANGE.
# FORCE_DISPATCH stays robustness-only.
# ---------------------------------------------------------------------------
def test_orchestration_cannot_emit_force_dispatch(gate):
    """FORCE_DISPATCH stays robustness-only — Orchestration's role
    intent set doesn't list it, so the role gate fires before the
    per-intent override is consulted."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.FORCE_DISPATCH,
            payload={"task_id": "t-1", "reason": "x"},
        ))
    assert exc.value.rule == "role"


def test_orchestration_can_emit_escalate_strategy_change(gate):
    """Loosen P3_18 18A — Orchestration may emit phase-advance hints
    directly (skip_to_kernel / skip_to_sweep / skip_to_close). The
    Coordinator's ``_handle_escalate_strategy_change`` is source-
    agnostic; the per-intent allowlist widening here unlocks the path
    end-to-end. FORCE_DISPATCH remains robustness-only."""
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
    """Kernel responder-only — its role intent set excludes the
    scheduling-police intents entirely."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("kernel", Intent(
            type=IntentType.ESCALATE_STRATEGY_CHANGE,
            payload={"next_action_hint": "skip_to_close"},
        ))
    assert exc.value.rule == "role"


# ---------------------------------------------------------------------------
# Kernel / Critic — not in the PRUNE_BRANCH source allowlist either
# ---------------------------------------------------------------------------
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
