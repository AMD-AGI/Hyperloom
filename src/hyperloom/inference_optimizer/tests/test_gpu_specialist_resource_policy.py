# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU-specialist resource policy."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy.denial import PolicyDenied
from hyperloom.orchestrator.policy.gate import (
    PolicyGate,
)
from hyperloom.orchestrator.policy.projection import ResourceFacts
from hyperloom.orchestrator.state.shared_state import SharedState


def _policy(state: SharedState) -> PolicyGate:
    """A gate whose resource rules read facts read off ``state``."""
    facts = ResourceFacts()
    facts.update(state)
    return PolicyGate(
        role_registry=default_role_registry(),
        shared_state=state,
        resources=facts,
    )


def _gate(research_lane_capacity: int = 1) -> PolicyGate:
    s = SharedState()
    s.phase = "FRAMEWORK_AGENT"
    s.research_lane_capacity = research_lane_capacity
    return _policy(s)


def _specialist_delegate(params: dict) -> Intent:
    merged = {
        "tags": ["kernel_agent"],
        "gap_canonical_id": "gap.kernel.microbench.session-test",
    }
    merged.update(params)
    return Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "specialist",
            "params": merged,
        },
    )


def test_gpu_specialist_denied_when_pool_disabled():
    gate = _gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            _specialist_delegate({"needs_gpu": True, "gpu_count": 1}),
        )
    assert exc.value.rule == "specialist_gpu_pool_disabled"


def test_gpu_specialist_allowed_within_capacity(monkeypatch):
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "TP"):
        monkeypatch.delenv(name, raising=False)
    s = SharedState()
    s.phase = "FRAMEWORK_AGENT"
    s.gpu_specialist_capacity = 2
    gate = _policy(s)
    gate.validate_intent(
        "orchestration",
        _specialist_delegate({"needs_gpu": True, "gpu_count": 1}),
    )


def test_gpu_specialist_denies_above_capacity():
    s = SharedState()
    s.phase = "FRAMEWORK_AGENT"
    s.gpu_specialist_capacity = 1
    gate = _policy(s)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            _specialist_delegate({"needs_gpu": True, "gpu_count": 2}),
        )
    assert exc.value.rule == "specialist_gpu_request_exceeds_capacity"
