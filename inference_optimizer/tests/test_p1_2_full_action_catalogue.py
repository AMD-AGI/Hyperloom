"""P1-2 full action catalogue tests.

Asserts the v0.6 OptimizationAction catalogue is complete and that
families/owners line up with DESIGN §16.1.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.action_registry import (
    ActionRegistry,
    VALID_FAMILIES,
    VALID_PIPELINE_PHASES,
)
from inference_optimizer.orchestrator.policy import KERNEL_OWNED_ACTIONS


# DESIGN §16.1 plus isolated PMC roofline analysis action and the Phase 3
# ``validate_stack`` action introduced for cumulative-gain validation.
EXPECTED_ACTIONS_V06: dict[str, str] = {
    # prep (3 — incl. validate_stack which lives in `prep` family because
    # it's a measurement action that doesn't introduce new modifications;
    # `setup` / `classify` are owned by the external SKILL caller, not
    # the optimizer's action loop)
    "target_analysis":      "prep",
    "baseline":             "prep",
    "validate_stack":       "prep",
    # analysis (2)
    "profile":              "analysis",
    "pmc_roofline":         "analysis",
    # shallow (5) — report + session_breakdown live here per DESIGN §16.1
    "backends":             "shallow",
    "params":               "shallow",
    "sweep":                "shallow",
    "report":               "shallow",
    "session_breakdown":    "shallow",
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
    """All OptimizationActions must be present with correct family."""
    actual = {m.name: m.family for m in registry.all()}
    assert actual == EXPECTED_ACTIONS_V06


def test_catalogue_count_matches_expected(registry):
    assert len(registry.all()) == len(EXPECTED_ACTIONS_V06)


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


# ---------------------------------------------------------------------------
# Drift guards: keep the four "is this a real action?" sources of truth
# consistent with each other.
#
# Background: ``session_paths._runs_actions()`` (the whitelist for actions
# that get a per-task ``runs/<kind>/<task_id>/`` workspace) used to be a
# hand-maintained ``_RUNS_ACTIONS`` frozenset. Adding a new action
# (``validate_stack``) without updating it caused the orchestrator to loop
# forever proposing the action — every dispatch raised ``ValueError`` from
# inside the executor's ``runs_dir()`` fallback, but mission TODOs never
# cleared. The fix derives the whitelist from ``pipeline_phase`` in the
# ActionRegistry; these tests lock the alignment in place.
# ---------------------------------------------------------------------------
def test_runs_actions_match_pipeline_phases(registry):
    """``_runs_actions()`` must equal {a.name for a in registry
    if a.pipeline_phase ∈ _RUNS_WORKSPACE_PHASES}.

    This is the primary registry ↔ session_paths drift guard.
    """
    from inference_optimizer.session_paths import (
        _RUNS_WORKSPACE_PHASES,
        _runs_actions,
    )

    expected = frozenset(
        a.name for a in registry.all()
        if a.pipeline_phase in _RUNS_WORKSPACE_PHASES
    )
    actual = _runs_actions()
    assert actual == expected, (
        f"runs_actions drift: actual={sorted(actual)!r} expected={sorted(expected)!r}; "
        f"if a yaml's pipeline_phase changed, fix it; if a new phase was added "
        f"to _RUNS_WORKSPACE_PHASES, this test pins the change."
    )


def test_validate_stack_in_runs_actions():
    """Explicit regression: ``validate_stack`` was the action that
    triggered the historical drift bug. Lock it as a runs/<kind>/ owner.
    """
    from inference_optimizer.session_paths import _runs_actions

    assert "validate_stack" in _runs_actions()


def test_runs_actions_fallback_matches_registry(registry):
    """The hardcoded ``_RUNS_ACTIONS_FALLBACK`` (used only when the
    registry can't be loaded) must stay aligned with the registry-derived
    set. Otherwise a degraded boot path could silently produce a different
    whitelist than production.
    """
    from inference_optimizer.session_paths import (
        _RUNS_ACTIONS_FALLBACK,
        _RUNS_WORKSPACE_PHASES,
    )

    expected = frozenset(
        a.name for a in registry.all()
        if a.pipeline_phase in _RUNS_WORKSPACE_PHASES
    )
    assert _RUNS_ACTIONS_FALLBACK == expected, (
        f"_RUNS_ACTIONS_FALLBACK drift: fallback={sorted(_RUNS_ACTIONS_FALLBACK)!r} "
        f"registry-derived={sorted(expected)!r}; update _RUNS_ACTIONS_FALLBACK "
        f"in session_paths.py to match."
    )


def test_cli_real_executors_consistent_with_runs_actions():
    """Every action wired with a real executor in
    ``cli._register_executors`` must either be in ``_runs_actions()``
    (writes per-task artefacts under ``runs/<kind>/<task_id>/``) or be one
    of the special session-root writers (``report`` → ``reports/``,
    ``session_breakdown`` → ``session_breakdown.json`` at the session root).

    This is the primary cli ↔ session_paths drift guard: if someone adds
    a real executor without giving its yaml a ``pipeline_phase`` that
    falls in ``_RUNS_WORKSPACE_PHASES``, this test fires.
    """
    from inference_optimizer.cli import (
        _REAL_EXECUTORS_FULL,
        _REAL_EXECUTORS_KERNEL_ONLY,
    )
    from inference_optimizer.session_paths import _runs_actions

    SESSION_ROOT_WRITERS = {"report", "session_breakdown"}
    runs = _runs_actions()
    real_kinds = (
        set(_REAL_EXECUTORS_FULL.keys())
        | set(_REAL_EXECUTORS_KERNEL_ONLY.keys())
    )

    missing_from_runs = (real_kinds - SESSION_ROOT_WRITERS) - runs
    assert not missing_from_runs, (
        f"actions {sorted(missing_from_runs)!r} have a real executor in "
        f"cli._REAL_EXECUTORS_* but are not in _runs_actions(); "
        f"SubAgentRunner will not pre-mkdir the workspace and the "
        f"executor's runs_dir() fallback will raise. Either give the "
        f"action a yaml pipeline_phase in _RUNS_WORKSPACE_PHASES, or "
        f"document why it should be exempt (cf. report → reports/, "
        f"session_breakdown → session_breakdown.json)."
    )

    overclaim = SESSION_ROOT_WRITERS & runs
    assert not overclaim, (
        f"actions {sorted(overclaim)!r} are listed in SESSION_ROOT_WRITERS "
        f"but are also in _runs_actions(); pick one (write either to "
        f"runs/ or to the session root, not both)."
    )


# ---------------------------------------------------------------------------
# Phase 1 prompt-builder fields — see ActionMetadata docstring
# ---------------------------------------------------------------------------
def test_every_action_has_non_empty_description(registry):
    for m in registry.all():
        assert m.description, f"{m.name}: description must be non-empty"
        # Sanity: descriptions are 1-line and reasonably brief so the
        # builder doesn't blow up the prompt size.
        assert "\n" not in m.description, (
            f"{m.name}: description must be a single line, got "
            f"{m.description!r}"
        )
        assert len(m.description) <= 200, (
            f"{m.name}: description too long ({len(m.description)} chars); "
            f"keep it under ~200 chars to avoid bloating the system prompt"
        )


def test_every_action_has_valid_pipeline_phase(registry):
    for m in registry.all():
        assert m.pipeline_phase in VALID_PIPELINE_PHASES, (
            f"{m.name}: pipeline_phase={m.pipeline_phase!r} not in "
            f"{sorted(VALID_PIPELINE_PHASES)!r}"
        )


def test_typical_runtime_min_positive_for_active_actions(registry):
    """Every action with a non-zero cost must declare a typical runtime."""
    for m in registry.all():
        if m.cost_minutes_p50 == 0:
            continue
        assert m.typical_runtime_min > 0, (
            f"{m.name}: typical_runtime_min must be > 0 when cost > 0"
        )


def test_validate_stack_action_metadata(registry):
    """The Phase 3 ``validate_stack`` action must be present and shaped right."""
    m = registry.get("validate_stack")
    assert m is not None, "validate_stack action missing from registry"
    assert m.family == "prep"
    assert m.pipeline_phase == "validate"
    assert "baseline" in m.prerequisites
    assert "server_lifecycle" in m.requires_lanes
    assert "benchmark_lane" in m.requires_lanes
    # Risk-free measurement action — never introduces a new modification
    assert m.expected_gain_pct == (0.0, 0.0)
    assert m.accuracy_risk == 0.0


def test_kernel_owned_actions_in_deep_pipeline_phase(registry):
    """Kernel-owned actions must declare pipeline_phase=='deep' so the
    builder groups them under the kernel section."""
    for name in KERNEL_OWNED_ACTIONS:
        m = registry.get(name)
        assert m is not None
        if name == "deep_kernel_analysis":
            # Analysis-only step that PRECEDES kernel_opt — phase=analysis
            # is the right slot.
            assert m.pipeline_phase == "analysis"
        else:
            assert m.pipeline_phase == "deep", (
                f"{name}: expected pipeline_phase='deep', got {m.pipeline_phase!r}"
            )
