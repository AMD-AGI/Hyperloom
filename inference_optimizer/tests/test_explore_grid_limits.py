"""Tests for ``explore_specialist_grid_max_one`` and the propose_action
provenance backstop.

Covers the new PolicyGate rule that caps ``provenance='specialist:*'``
variants at 1 per explore round, plus the previously-missing
``_validate_explore_provenance`` / grid_size mount on the propose_action
channel. The all-llm_direct and empty-grid happy-paths are exercised
in test_discard_single_agent_explore.py; this file focuses on the new
cap and the propose-side backstop.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
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
# propose_action channel — backstops for both PR-A9 + the new size cap
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


def test_propose_denies_all_llm_direct_grid():
    """Without this backstop, an LLM that cannot reach the delegate
    channel could ship a llm_direct explore via propose_action and
    have the Coordinator materialise it. PR-A9 enforced on propose."""
    gate = _gate()
    intent = _propose([
        {"name": "a", "provenance": "llm_direct"},
        {"name": "b", "provenance": "llm_direct"},
    ])
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "explore_requires_specialist_provenance"


def test_propose_allows_single_specialist_variant():
    gate = _gate()
    intent = _propose([
        {"name": "a", "provenance": "specialist:serving_specialist"},
    ])
    gate.validate_intent("orchestration", intent)  # no raise


# ---------------------------------------------------------------------------
# Dynamic cap tracking research_lane_capacity
# ---------------------------------------------------------------------------
def test_cap_tracks_research_lane_capacity_allows_four():
    """With research_lane_capacity=4, a 4-specialist grid is permitted."""
    gate = _gate(research_lane_capacity=4)
    gate.validate_intent("orchestration", _delegate(_specialist_variants(4)))  # no raise


def test_cap_tracks_research_lane_capacity_denies_five():
    """With research_lane_capacity=4, a 5-specialist grid is denied."""
    gate = _gate(research_lane_capacity=4)
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _delegate(_specialist_variants(5)))
    assert exc.value.rule == "explore_specialist_grid_max_one"


def test_cap_clamps_to_max_research_lane_capacity():
    """Capacity above the hard ceiling clamps to MAX_RESEARCH_LANE_CAPACITY.

    research_lane_capacity=32 -> effective cap 6: a 6-variant grid passes,
    a 7-variant grid is denied.
    """
    gate = _gate(research_lane_capacity=32)
    gate.validate_intent("orchestration", _delegate(_specialist_variants(6)))  # no raise
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", _delegate(_specialist_variants(7)))
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
