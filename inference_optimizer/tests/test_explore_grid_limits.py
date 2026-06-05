"""Tests for ``explore_specialist_grid_max_one``.

Covers the PolicyGate rule that caps ``provenance='specialist:*'``
variants per explore round (tracking ``research_lane_capacity`` clamped
to the GPU-derived ceiling) on both the delegate and propose_action
channels. All-``llm_direct`` grids are accepted; provenance is audit-only
except for specialist/dynamic per-round caps."""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.policy import (
    MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS,
    PolicyDenied,
    PolicyGate,
)
from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gate(research_lane_capacity: int = 1) -> PolicyGate:
    s = SharedState()
    s.phase = "EXPLORE"
    s.research_lane_capacity = research_lane_capacity
    return PolicyGate(role_registry=default_role_registry(), shared_state=s)


def _specialist_variants(n: int) -> list[dict]:
    return [
        {"name": f"v{i}", "provenance": "specialist:serving_specialist"}
        for i in range(n)
    ]


def _delegate(grid: list[dict]) -> Intent:
    return Intent(type=IntentType.DELEGATE, payload={
        "action_name": "explore",
        "params": {"grid": grid},
    })


def _propose(grid: list[dict]) -> Intent:
    return Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "explore",
        "params": {"grid": grid},
    })


def _specialist_delegate(params: dict) -> Intent:
    merged = {
        "tags": ["kernel"],
        "gap_canonical_id": "gap.kernel.microbench.session-test",
    }
    merged.update(params)
    return Intent(type=IntentType.DELEGATE, payload={
        "action_name": "specialist",
        "params": merged,
    })


# ---------------------------------------------------------------------------
# explore_specialist_grid_max_one (delegate channel)
# ---------------------------------------------------------------------------
def test_constant_is_one():
    assert MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS == 1


def test_denies_two_specialist_variants():
    gate = _gate()
    intent = _delegate([
        {"name": "a", "provenance": "specialist:serving_specialist"},
        {"name": "b", "provenance": "specialist:serving_specialist"},
    ])
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "explore_specialist_grid_max_one"


def test_denies_two_specialist_variants_across_domains():
    """Cap is on total specialist:* count, not per-domain."""
    gate = _gate()
    intent = _delegate([
        {"name": "a", "provenance": "specialist:serving_specialist"},
        {"name": "b", "provenance": "specialist:kernel_switch_specialist"},
    ])
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "explore_specialist_grid_max_one"


def test_allows_single_specialist_variant():
    gate = _gate()
    intent = _delegate([
        {"name": "a", "provenance": "specialist:serving_specialist"},
    ])
    gate.validate_intent("orchestration", intent)  # no raise


def test_allows_three_default_grid_variants():
    """default_grid is uncapped — cold-start may emit several."""
    gate = _gate()
    intent = _delegate([
        {"name": "a", "provenance": "default_grid"},
        {"name": "b", "provenance": "default_grid"},
        {"name": "c", "provenance": "default_grid"},
    ])
    gate.validate_intent("orchestration", intent)  # no raise


def test_allows_one_specialist_plus_many_default_grid():
    """Mixing 1 specialist + N default_grid is permitted."""
    gate = _gate()
    intent = _delegate([
        {"name": "a", "provenance": "specialist:serving_specialist"},
        {"name": "b", "provenance": "default_grid"},
        {"name": "c", "provenance": "default_grid"},
    ])
    gate.validate_intent("orchestration", intent)  # no raise


def test_empty_grid_skips_size_check():
    """An empty / omitted grid falls through to the executor's
    ``empty_grid`` surfacing — PolicyGate must not preempt with
    a size error."""
    gate = _gate()
    intent = Intent(type=IntentType.DELEGATE, payload={
        "action_name": "explore",
        "params": {},  # no grid
    })
    gate.validate_intent("orchestration", intent)  # no raise


# ---------------------------------------------------------------------------
# propose_action channel — mirrors delegate-channel size caps
# ---------------------------------------------------------------------------
def test_propose_denies_two_specialist_variants():
    """The new explore_specialist_grid_max_one rule applies to the
    propose_action channel too, not just delegate."""
    gate = _gate()
    intent = _propose([
        {"name": "a", "provenance": "specialist:serving_specialist"},
        {"name": "b", "provenance": "specialist:serving_specialist"},
    ])
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "explore_specialist_grid_max_one"


def test_propose_allows_all_llm_direct_grid():
    """All-``llm_direct`` grids are accepted on the propose channel; the
    only enforced explore-grid limit is the specialist fan-out cap."""
    gate = _gate()
    intent = _propose([
        {"name": "a", "provenance": "llm_direct"},
        {"name": "b", "provenance": "llm_direct"},
    ])
    gate.validate_intent("orchestration", intent)  # no raise


def test_propose_allows_single_specialist_variant():
    gate = _gate()
    intent = _propose([
        {"name": "a", "provenance": "specialist:serving_specialist"},
    ])
    gate.validate_intent("orchestration", intent)  # no raise


# ---------------------------------------------------------------------------
# Dynamic cap tracking research_lane_capacity
# ---------------------------------------------------------------------------
def test_cap_tracks_research_lane_capacity_allows_four(monkeypatch):
    """With research_lane_capacity=4 (ceiling >= 4), a 4-specialist grid
    is permitted."""
    from inference_optimizer.orchestrator import policy as policy_mod

    monkeypatch.setattr(policy_mod, "detect_gpu_count", lambda: 4)
    gate = _gate(research_lane_capacity=4)
    gate.validate_intent("orchestration", _delegate(_specialist_variants(4)))  # no raise


def test_cap_tracks_research_lane_capacity_denies_five(monkeypatch):
    """With research_lane_capacity=4, a 5-specialist grid is denied."""
    from inference_optimizer.orchestrator import policy as policy_mod

    monkeypatch.setattr(policy_mod, "detect_gpu_count", lambda: 4)
    gate = _gate(research_lane_capacity=4)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _delegate(_specialist_variants(5)))
    assert exc.value.rule == "explore_specialist_grid_max_one"


def test_cap_clamps_to_research_lane_ceiling(monkeypatch):
    """Capacity above the GPU-derived ceiling clamps down.

    With 4 visible GPUs the ceiling is 8: research_lane_capacity=32 ->
    effective cap 8: an 8-variant grid passes, a 9-variant grid is
    denied.
    """
    from inference_optimizer.orchestrator import policy as policy_mod

    monkeypatch.setattr(policy_mod, "detect_gpu_count", lambda: 4)
    gate = _gate(research_lane_capacity=32)
    gate.validate_intent("orchestration", _delegate(_specialist_variants(8)))  # no raise
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _delegate(_specialist_variants(9)))
    assert exc.value.rule == "explore_specialist_grid_max_one"


def test_cap_falls_back_to_one_when_capacity_unset():
    """research_lane_capacity=0 falls back to the hard-1 default."""
    gate = _gate(research_lane_capacity=0)
    gate.validate_intent(
        "orchestration",
        _delegate(_specialist_variants(1)),
    )  # no raise
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _delegate(_specialist_variants(2)))
    assert exc.value.rule == "explore_specialist_grid_max_one"


# ---------------------------------------------------------------------------
# GPU specialist request policy
# ---------------------------------------------------------------------------
def test_gpu_specialist_denied_when_pool_disabled():
    gate = _gate()
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            _specialist_delegate({"needs_gpu": True, "gpu_count": 1}),
        )
    assert exc.value.rule == "specialist_gpu_pool_disabled"


def test_gpu_specialist_allowed_within_capacity():
    s = SharedState()
    s.phase = "EXPLORE"
    s.gpu_specialist_capacity = 2
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=s)
    gate.validate_intent(
        "orchestration",
        _specialist_delegate({"needs_gpu": True, "gpu_count": 1}),
    )


def test_gpu_specialist_denies_above_capacity():
    s = SharedState()
    s.phase = "EXPLORE"
    s.gpu_specialist_capacity = 1
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=s)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            _specialist_delegate({"needs_gpu": True, "gpu_count": 2}),
        )
    assert exc.value.rule == "specialist_gpu_request_exceeds_capacity"
