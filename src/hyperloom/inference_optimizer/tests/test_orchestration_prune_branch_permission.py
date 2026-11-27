# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Orchestration permission widenings for scheduling-police intents."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy.denial import PolicyDenied
from hyperloom.orchestrator.policy.gate import PolicyGate


@pytest.fixture
def gate() -> PolicyGate:
    return PolicyGate()


# Orchestration (new permission) — happy path + payload-shape guard
def test_orchestration_can_emit_prune_branch_with_family(gate):
    """Orchestration forwards roofline advice as PRUNE_BRANCH."""
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"family": "kernel_opt", "reason": "compute saturated 92%"},
        ),
    )


@pytest.mark.parametrize(
    "payload",
    [{"family": "", "reason": "missing target"}, {"reason": "missing target"}],
    ids=["empty_family", "family_key_absent"],
)
def test_orchestration_prune_branch_missing_family_rejected(gate, payload):
    """Payload-shape check must still fire for the new source: an empty or absent ``family`` is denied (not KeyError)."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(type=IntentType.PRUNE_BRANCH, payload=payload),
        )
    assert exc.value.rule == "payload"


def test_orchestration_prune_branch_accepts_queued_scope(gate):
    """scope='queued' drains a backlog without retiring the family."""
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={
                "family": "baseline",
                "reason": "anchor already established",
                "scope": "queued",
            },
        ),
    )


def test_orchestration_prune_branch_unknown_scope_rejected(gate):
    """An unrecognised scope is denied rather than silently treated as a full prune."""
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.PRUNE_BRANCH,
                payload={"family": "baseline", "reason": "x", "scope": "running"},
            ),
        )
    assert exc.value.rule == "prune_scope"


# Per-intent override widens PRUNE_BRANCH and ESCALATE_STRATEGY_CHANGE.
def test_orchestration_can_emit_escalate_strategy_change(gate):
    """Orchestration may emit phase-advance hints directly."""
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.ESCALATE_STRATEGY_CHANGE,
            payload={"next_action_hint": "skip_to_kernel"},
        ),
    )
