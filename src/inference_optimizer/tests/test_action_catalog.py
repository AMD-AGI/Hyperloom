"""Catalog-level tests — IMPL-CHECKLIST §4.22‒§4.43.

Locks in the *names* of every action that ships with the skill, plus the
mode-distribution invariants documented in DESIGN §12. Adding or removing
a yaml without updating this file is intentional friction.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.resource_lock import KNOWN_LANES


SKILL_ACTIONS_DIR = (
    Path(__file__).resolve().parents[3]
    / ".cursor" / "skills" / "inference-optimizer" / "actions"
)


_PREP = {"setup", "classify", "target_analysis", "baseline", "bench_runner"}
_ANALYSIS = {"profile"}
_SHALLOW = {"backends", "params", "sweep", "param_sweep_run", "report"}
_DEEP_KERNEL = {
    "kernel_opt",
    "integrate",
    "deep_kernel_analysis",
    "operator_tuning",
    "vendor_kernel_config",
}
_LONG = {"framework_rebuild", "comm_optimization", "compiler_tuning"}
_CREATIVE = {"dream", "re_explore"}
_RESILIENCE = {"recover"}

ALL_EXPECTED = (
    _PREP | _ANALYSIS | _SHALLOW | _DEEP_KERNEL
    | _LONG | _CREATIVE | _RESILIENCE
)


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    return ActionRegistry(SKILL_ACTIONS_DIR).load()


def test_full_catalog_present(registry: ActionRegistry):
    actual = set(registry.names())
    missing = ALL_EXPECTED - actual
    extra = actual - ALL_EXPECTED
    assert not missing, f"missing actions: {sorted(missing)}"
    assert not extra, f"unexpected extra actions: {sorted(extra)}"


@pytest.mark.parametrize("name", sorted(_PREP))
def test_prep_actions_in_every_mode(registry: ActionRegistry, name: str):
    a = registry.get(name)
    assert a is not None
    assert ExecutionMode.QUICK_PARAM_SWEEP in a.allowed_modes


@pytest.mark.parametrize("name", sorted(_DEEP_KERNEL))
def test_deep_kernel_actions_excluded_from_quick(registry: ActionRegistry, name: str):
    a = registry.get(name)
    assert a is not None
    assert ExecutionMode.QUICK_PARAM_SWEEP not in a.allowed_modes


@pytest.mark.parametrize(
    "name", sorted({"deep_kernel_analysis", "operator_tuning",
                    "framework_rebuild", "comm_optimization",
                    "compiler_tuning", "dream"})
)
def test_marathon_only_actions(registry: ActionRegistry, name: str):
    a = registry.get(name)
    assert a is not None
    assert a.allowed_modes == (ExecutionMode.MARATHON_MULTI_AGENT,)


def test_every_action_uses_only_known_lanes(registry: ActionRegistry):
    bad: list[str] = []
    for a in registry.all():
        for lane in a.requires_lanes:
            if lane not in KNOWN_LANES:
                bad.append(f"{a.name}:{lane}")
    assert not bad, f"unknown lanes: {bad}"


def test_every_action_declares_emit_intent_tool(registry: ActionRegistry):
    """All actions need ``emit_intent`` so PolicyGate.allowed_tools_for_action
    won't reduce to an empty set."""
    no_emit: list[str] = []
    for a in registry.all():
        if "emit_intent" not in a.allowed_tools:
            no_emit.append(a.name)
    assert not no_emit, f"actions missing emit_intent: {no_emit}"


def test_every_action_has_markdown_body(registry: ActionRegistry):
    missing: list[str] = []
    for a in registry.all():
        body = registry.system_prompt_for(a.name)
        if not body.strip():
            missing.append(a.name)
    assert not missing, f"actions missing .md body: {missing}"


def test_integrate_requires_three_lanes(registry: ActionRegistry):
    a = registry.get("integrate")
    assert a is not None
    needed = {"workspace_mutation", "server_lifecycle", "benchmark_lane"}
    assert needed <= set(a.requires_lanes)


def test_recover_requires_server_and_workspace(registry: ActionRegistry):
    a = registry.get("recover")
    assert a is not None
    assert "server_lifecycle" in a.requires_lanes
    assert "workspace_mutation" in a.requires_lanes


def test_creative_actions_have_zero_risk(registry: ActionRegistry):
    for n in _CREATIVE:
        a = registry.get(n)
        assert a is not None
        assert a.accuracy_risk == 0.0
        assert a.crash_risk == 0.0


def test_long_family_p75_above_30min(registry: ActionRegistry):
    """Long-running actions must declare lease ttl long enough to cover p75."""
    for n in _LONG:
        a = registry.get(n)
        assert a is not None
        assert a.cost_minutes_p75 >= 30.0
        assert a.lease_ttl_sec >= int(a.cost_minutes_p75 * 60 * 0.9)
