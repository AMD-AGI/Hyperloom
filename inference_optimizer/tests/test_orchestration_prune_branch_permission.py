"""Roofline-v2 C3: Orchestration is allowed to emit PRUNE_BRANCH.

These tests pin the permission boundary the ``roofline`` action (C4)
and the renderer (C5) depend on:

* Orchestration can emit ``PRUNE_BRANCH`` with a non-empty ``family``
  (the structured advice in ``last_roofline_analysis.suggested_prunes``
  is consumed by the main LLM and forwarded as a normal PRUNE_BRANCH
  intent — Coordinator's ``_handle_prune_branch`` is source-agnostic
  so this widening alone unlocks the full path).
* Orchestration is still rejected when ``family`` is missing — the
  payload-shape check (``_validate_robustness_only``) still runs for
  the new source, so a malformed analyzer-LLM forwarded intent cannot
  silently write garbage into ``pruned_families``.
* Robustness can still emit PRUNE_BRANCH (the original allowlist
  member); ESCALATE_STRATEGY_CHANGE and FORCE_DISPATCH remain
  robustness-only — the per-intent override widens **only**
  PRUNE_BRANCH, not the whole robustness-only set.
* Kernel / Critic cannot emit PRUNE_BRANCH (both fall under
  ``role.allowed_intents`` check before reaching the source allowlist).
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
# Per-intent override is scoped to PRUNE_BRANCH — sibling scheduling-
# police intents stay robustness-only
# ---------------------------------------------------------------------------
def test_orchestration_cannot_emit_force_dispatch(gate):
    """The PRUNE_BRANCH widening must NOT leak to FORCE_DISPATCH —
    Orchestration's role intent set doesn't even list it, so this is
    caught at the role gate before reaching the per-intent override."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.FORCE_DISPATCH,
            payload={"task_id": "t-1", "reason": "x"},
        ))
    assert exc.value.rule == "role"


def test_orchestration_cannot_emit_escalate_strategy_change(gate):
    """Same as FORCE_DISPATCH — Orchestration's role intent set
    intentionally excludes ESCALATE_STRATEGY_CHANGE."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", Intent(
            type=IntentType.ESCALATE_STRATEGY_CHANGE,
            payload={"reason": "x"},
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
