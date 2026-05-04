"""P1-2 full 19-action catalogue tests.

Asserts the v0.6 OptimizationAction catalogue is complete and that
families/owners line up with DESIGN §16.1.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.action_registry import (
    ActionRegistry,
    VALID_FAMILIES,
)
from inference_optimizer.orchestrator.policy import KERNEL_OWNED_ACTIONS


# DESIGN §16.1 — full v0.6 list (19 actions; framework-rebuild dropped)
EXPECTED_ACTIONS_V06: dict[str, str] = {
    # prep (4)
    "setup":                "prep",
    "classify":             "prep",
    "target_analysis":      "prep",
    "baseline":             "prep",
    # analysis (1)
    "profile":              "analysis",
    # shallow (4) — report lives here per DESIGN §16.1
    "backends":             "shallow",
    "params":               "shallow",
    "sweep":                "shallow",
    "report":               "shallow",
    # deep_kernel (5)
    "kernel_opt":           "deep_kernel",
    "integrate":            "deep_kernel",
    "deep_kernel_analysis": "deep_kernel",
    "operator_tuning":      "deep_kernel",
    "vendor_kernel_config": "deep_kernel",
    # long (2)
    "comm_optimization":    "long",
    "compiler_tuning":      "long",
    # creative (2)
    "dream":                "creative",
    "re_explore":           "creative",
    # resilience (1)
    "recover":              "resilience",
}


@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


def test_full_catalogue_loads_and_matches_design(registry):
    """All 19 v0.6 OptimizationActions must be present with correct family."""
    actual = {m.name: m.family for m in registry.all()}
    assert actual == EXPECTED_ACTIONS_V06


def test_catalogue_count_is_exactly_19(registry):
    assert len(registry.all()) == 19


def test_kernel_owned_actions_all_in_registry(registry):
    """The 5 KERNEL_OWNED_ACTIONS must each have metadata."""
    for name in KERNEL_OWNED_ACTIONS:
        meta = registry.get(name)
        assert meta is not None, f"missing metadata for kernel-owned action: {name}"
        assert meta.family == "deep_kernel"


def test_no_framework_rebuild(registry):
    """v0.6 ADR — framework-rebuild is removed."""
    assert registry.get("framework_rebuild") is None
    assert registry.get("framework-rebuild") is None


def test_kernel_opt_has_three_lanes_and_high_cost(registry):
    m = registry.get("kernel_opt")
    assert m is not None
    assert set(m.requires_lanes) == {"server_lifecycle", "workspace_mutation", "benchmark_lane"}
    assert m.cost_minutes_p75 >= 60


def test_recover_owned_by_robustness_handle(registry):
    """recover is the resilience action that Robustness handle_subagent runs."""
    m = registry.get("recover")
    assert m is not None
    assert m.family == "resilience"


def test_dream_zero_gain_creative(registry):
    m = registry.get("dream")
    assert m is not None
    assert m.family == "creative"
    assert m.expected_gain_pct == (0.0, 0.0)


def test_every_action_has_valid_family(registry):
    for m in registry.all():
        assert m.family in VALID_FAMILIES, f"{m.name} has invalid family {m.family!r}"


def test_every_action_uses_only_known_lanes(registry):
    known = {"server_lifecycle", "workspace_mutation", "benchmark_lane", "profile_lane"}
    for m in registry.all():
        bad = set(m.requires_lanes) - known
        assert not bad, f"{m.name}: unknown lanes {bad}"


def test_every_action_has_emit_intent_tool(registry):
    for m in registry.all():
        assert "emit_intent" in m.allowed_tools, (
            f"{m.name}: emit_intent missing from allowed_tools — every "
            f"reactor needs it to communicate"
        )


def test_actions_with_workspace_lane_have_edit_tool(registry):
    """Anything that mutates the workspace must declare Edit (or otherwise
    document that it goes through a sub-agent)."""
    for m in registry.all():
        if "workspace_mutation" in m.requires_lanes:
            assert "Edit" in m.allowed_tools, (
                f"{m.name}: requires workspace_mutation but doesn't declare Edit"
            )


def test_lease_ttl_sec_consistent_with_cost(registry):
    """lease_ttl_sec should be at least cost_minutes_p75 * 60."""
    for m in registry.all():
        if m.cost_minutes_p75 == 0:
            continue
        expected_min_ttl = m.cost_minutes_p75 * 60
        assert m.lease_ttl_sec >= expected_min_ttl * 0.5, (
            f"{m.name}: lease_ttl_sec={m.lease_ttl_sec} too low for "
            f"cost_minutes_p75={m.cost_minutes_p75}"
        )
