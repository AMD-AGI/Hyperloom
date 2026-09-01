# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Explore-grid provenance + GPU-specialist resource policy.

Grids of any provenance mix pass PolicyGate, and the GPU specialist pool
resource invariant still holds.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy.gate import (
    PolicyDenied,
    PolicyGate,
)
from hyperloom.orchestrator.state.shared_state import SharedState


def _gate(research_lane_capacity: int = 1) -> PolicyGate:
    s = SharedState()
    s.phase = "FRAMEWORK_AGENT"
    s.research_lane_capacity = research_lane_capacity
    return PolicyGate(role_registry=default_role_registry(), shared_state=s)


def _specialist_variants(n: int) -> list[dict]:
    return [{"name": f"v{i}", "provenance": "specialist:serving_specialist"} for i in range(n)]


def _delegate(grid: list[dict]) -> Intent:
    return Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "explore",
            "params": {"grid": grid},
        },
    )


def _propose(grid: list[dict]) -> Intent:
    return Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "explore",
            "params": {"grid": grid},
        },
    )


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


def test_allows_many_specialist_variants():
    gate = _gate()
    gate.validate_intent("orchestration", _delegate(_specialist_variants(5)))


def test_allows_specialist_variants_across_domains():
    gate = _gate()
    intent = _delegate(
        [
            {"name": "a", "provenance": "specialist:serving_specialist"},
            {"name": "b", "provenance": "specialist:kernel_switch_specialist"},
        ]
    )
    gate.validate_intent("orchestration", intent)  # no raise


def test_allows_multiple_dynamic_variants():
    gate = _gate()
    intent = _delegate(
        [
            {"name": "a", "provenance": "dynamic"},
            {"name": "b", "provenance": "dynamic"},
        ]
    )
    gate.validate_intent("orchestration", intent)  # no raise


def test_allows_three_default_grid_variants():
    gate = _gate()
    intent = _delegate(
        [
            {"name": "a", "provenance": "default_grid"},
            {"name": "b", "provenance": "default_grid"},
            {"name": "c", "provenance": "default_grid"},
        ]
    )
    gate.validate_intent("orchestration", intent)  # no raise


def test_empty_grid_skips_size_check():
    """An empty/omitted grid falls through to the executor's ``empty_grid`` surfacing; PolicyGate must not preempt."""
    gate = _gate()
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "explore",
            "params": {},  # no grid
        },
    )
    gate.validate_intent("orchestration", intent)  # no raise


def test_propose_allows_many_specialist_variants():
    gate = _gate()
    gate.validate_intent("orchestration", _propose(_specialist_variants(5)))


def test_propose_allows_all_llm_direct_grid():
    gate = _gate()
    intent = _propose(
        [
            {"name": "a", "provenance": "llm_direct"},
            {"name": "b", "provenance": "llm_direct"},
        ]
    )
    gate.validate_intent("orchestration", intent)  # no raise


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
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=s)
    gate.validate_intent(
        "orchestration",
        _specialist_delegate({"needs_gpu": True, "gpu_count": 1}),
    )


def test_gpu_specialist_denies_above_capacity():
    s = SharedState()
    s.phase = "FRAMEWORK_AGENT"
    s.gpu_specialist_capacity = 1
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=s)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            _specialist_delegate({"needs_gpu": True, "gpu_count": 2}),
        )
    assert exc.value.rule == "specialist_gpu_request_exceeds_capacity"
