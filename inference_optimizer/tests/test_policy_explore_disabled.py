# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""--no-explore fail-closed gate: when ``explore_enabled=False`` the phase
interleave grey channel must not let KERNEL propose/delegate an ``explore``
grid. The denial is independent of ``strict_phase`` (always fail-closed).
``specialist`` / ``integrate_patch`` stay allowed in KERNEL because they serve
kernel work (specialist research + patch integration)."""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.policy import PolicyDenied, PolicyGate
from inference_optimizer.orchestrator.phase_state import PHASE_KERNEL
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.protocol.intent import Intent, IntentType


def _gate(state: SharedState, *, strict_phase: bool) -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=state,
        strict_phase=strict_phase,
    )


def _delegate_explore() -> Intent:
    return Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "explore",
            "params": {"grid": [{"name": "v1"}]},
        },
    )


@pytest.mark.parametrize("strict_phase", [False, True])
def test_kernel_explore_denied_when_explore_disabled(strict_phase):
    state = SharedState(phase=PHASE_KERNEL, explore_enabled=False)
    gate = _gate(state, strict_phase=strict_phase)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _delegate_explore())
    # Always fail-closed: the dedicated rule fires regardless of strict_phase.
    assert exc.value.rule == "explore_disabled"


def test_kernel_explore_allowed_when_explore_enabled():
    state = SharedState(phase=PHASE_KERNEL, explore_enabled=True)
    gate = _gate(state, strict_phase=False)
    # interleave default-on: explore is a valid KERNEL interleave action.
    gate.validate_intent("orchestration", _delegate_explore())


def test_kernel_specialist_and_integrate_patch_still_allowed_when_no_explore():
    state = SharedState(phase=PHASE_KERNEL, explore_enabled=False)
    gate = _gate(state, strict_phase=False)
    # specialist stays allowed (research feeds kernel patches).
    gate.validate_intent(
        "orchestration",
        Intent(
            type=IntentType.DELEGATE,
            payload={
                "action_name": "specialist",
                "params": {
                    "tags": ["kernel"],
                    "gap_canonical_id": "gap.test.session-x",
                },
            },
        ),
    )
