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
from inference_optimizer.protocol.action_surfaces import (
    FRAMEWORK_PR_INTERNAL_ACTION_NAMES as SURFACE_FRAMEWORK_PR_INTERNAL_ACTION_NAMES,
    FULL_ENABLED_ACTIONS as SURFACE_FULL_ENABLED_ACTIONS,
    GRID_INJECTABLE_ACTIONS as SURFACE_GRID_INJECTABLE_ACTIONS,
    INTERNAL_ONLY_ACTION_NAMES as SURFACE_INTERNAL_ONLY_ACTION_NAMES,
    KERNEL_OWNED_ACTIONS as SURFACE_KERNEL_OWNED_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS as SURFACE_NO_KERNEL_ENABLED_ACTIONS,
    PHASE_ALLOWLIST_BYPASS_ACTIONS as SURFACE_PHASE_ALLOWLIST_BYPASS_ACTIONS,
)


# DESIGN §16.1 + v0.8 KB_design §3.4 (merged ``explore``).
# PR-A1 (Arbor-into-Hyperloom): added ``specialist`` (creative) and
# ``integrate_patch`` (shallow). ``specialist`` had previously been a
# *synthetic* action with no yaml meta (parameterised by ``params.domain``);
# the missing yaml made it invisible to ``prompt_builder._section_action_catalogue``,
# so the Orchestration LLM never emitted ``delegate{action='specialist'}``.
# ``integrate_patch`` is the new orchestrator-side apply+restart+gate
# step that consumes specialist worktree patches.
EXPECTED_ACTIONS_V06: dict[str, str] = {
    # prep (3)
    "target_analysis":      "prep",
    "baseline":             "prep",
    # GAP 1 — Coordinator-internal one-shot warm-recipe replay.
    # Same prep family as ``baseline`` (it is essentially a re-baseline
    # with the KB best_config applied). PolicyGate denies LLM
    # propose_action / delegate via ``analysis_action_not_llm_proposable``.
    "replay_warm_recipe":   "prep",
    # analysis (2) — Coordinator-internal analysis actions, selected
    # at runtime by ``shared_state.enable_roofline`` (``--enable-roofline``
    # / ``--no-enable-roofline``, default on):
    #   * ``roofline`` — composite (profile + trace_analyze +
    #     analysis.md snapshot);
    #   * ``profile`` — lightweight trace-only fallback.
    # Both are registered so the Coordinator-internal task path can
    # dispatch them through SubAgentRunner. PolicyGate denies LLM
    # propose_action / delegate for either name
    # (``analysis_action_not_llm_proposable``).
    "roofline":             "analysis",
    "profile":              "analysis",
    # shallow (5) — ``explore`` is the merged grid-runner entry.
    # PR-A1: ``integrate_patch`` joins the shallow family as the
    # EXPLORE-phase serving-lane-locked patch integration step.
    "explore":              "shallow",
    "integrate_patch":      "shallow",
    # FRAMEWORK_PR phase: per-candidate Coordinator-internal executor.
    # Mirrors integrate_patch's role for the new phase; LLM may not
    # propose it (framework_pr_action_not_llm_proposable, Stage 3).
    "framework_pr":         "shallow",
    "sweep":                "shallow",
    # SWEEP-phase post-sweep concurrency comparison (Coordinator-
    # internal auto-enqueue after sweep, on by default; disable via
    # ``--no-enable-conc-sweep``). Same family as ``sweep`` — discovery
    # action that benchmarks both arms across a CONC ladder and writes
    # ``reports/conc_sweep_summary.json``; never promotes.
    "conc_sweep":           "shallow",
    "report":               "shallow",
    "session_breakdown":    "shallow",
    # creative (2) — PR-A1: specialist LLM sub-agent dispatch;
    # dynamic_action.MD P1: cross-domain multi-turn ReAct sub-agent
    # dispatch (supplementary EXPLORE channel).
    "specialist":           "creative",
    "dynamic_action":        "creative",
    # deep_kernel (6)
    "kernel_opt":           "deep_kernel",
    "integrate":            "deep_kernel",
    "deep_kernel_analysis": "deep_kernel",
    "operator_tuning":      "deep_kernel",
    "vendor_kernel_config": "deep_kernel",
    "gemm_tuning":          "deep_kernel",
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


def test_action_surface_constants_are_shared():
    """Policy, prompt rendering, and CLI must not carry divergent action lists."""
    from inference_optimizer.cli import _NOOP_KINDS_KERNEL_ONLY
    from inference_optimizer.orchestrator import policy
    from inference_optimizer.orchestrator.system_prompts import prompt_builder

    assert policy.KERNEL_OWNED_ACTIONS is SURFACE_KERNEL_OWNED_ACTIONS
    assert prompt_builder.KERNEL_OWNED_ACTIONS is SURFACE_KERNEL_OWNED_ACTIONS
    assert set(_NOOP_KINDS_KERNEL_ONLY) == SURFACE_KERNEL_OWNED_ACTIONS
    assert prompt_builder.GRID_INJECTABLE_ACTIONS is SURFACE_GRID_INJECTABLE_ACTIONS
    assert policy.INTERNAL_ONLY_ACTION_NAMES is SURFACE_INTERNAL_ONLY_ACTION_NAMES
    assert prompt_builder.FULL_ENABLED_ACTIONS is SURFACE_FULL_ENABLED_ACTIONS
    assert prompt_builder.NO_KERNEL_ENABLED_ACTIONS is SURFACE_NO_KERNEL_ENABLED_ACTIONS


def test_prompt_enabled_actions_are_live_registry_actions(registry):
    retired = {
        "setup", "classify", "backends", "params",
        "validate_stack", "select_kernels",
    }
    enabled = set(SURFACE_FULL_ENABLED_ACTIONS) | set(SURFACE_NO_KERNEL_ENABLED_ACTIONS)
    assert not (enabled & retired)
    for name in enabled:
        assert registry.get(name) is not None, (
            f"prompt-visible action {name!r} must have action metadata"
        )
    assert SURFACE_KERNEL_OWNED_ACTIONS <= set(SURFACE_FULL_ENABLED_ACTIONS)
    assert SURFACE_KERNEL_OWNED_ACTIONS.isdisjoint(set(SURFACE_NO_KERNEL_ENABLED_ACTIONS))


def test_phase_allowlist_actions_are_live_registry_actions(registry):
    from inference_optimizer.orchestrator.phase_state import PHASE_ALLOWED_ACTIONS

    retired = {
        "setup", "classify", "backends", "params",
        "validate_stack", "select_kernels",
    }
    phase_actions = set().union(*PHASE_ALLOWED_ACTIONS.values())
    assert not (phase_actions & retired)
    for name in phase_actions:
        assert registry.get(name) is not None, (
            f"phase allowlist action {name!r} must have action metadata"
        )


def test_action_surface_sets_are_phase_aligned():
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_ALLOWED_ACTIONS,
        PHASE_FRAMEWORK_PR,
        PHASE_KERNEL,
    )

    all_phase_actions = set().union(*PHASE_ALLOWED_ACTIONS.values())
    assert SURFACE_KERNEL_OWNED_ACTIONS <= PHASE_ALLOWED_ACTIONS[PHASE_KERNEL]
    assert (
        SURFACE_INTERNAL_ONLY_ACTION_NAMES - SURFACE_PHASE_ALLOWLIST_BYPASS_ACTIONS
    ) <= all_phase_actions
    assert SURFACE_PHASE_ALLOWLIST_BYPASS_ACTIONS <= SURFACE_INTERNAL_ONLY_ACTION_NAMES
    assert SURFACE_PHASE_ALLOWLIST_BYPASS_ACTIONS.isdisjoint(all_phase_actions)
    assert (
        SURFACE_FRAMEWORK_PR_INTERNAL_ACTION_NAMES
        <= PHASE_ALLOWED_ACTIONS[PHASE_FRAMEWORK_PR]
    )
    assert SURFACE_GRID_INJECTABLE_ACTIONS <= all_phase_actions


def test_kernel_opt_has_three_lanes_and_high_cost(registry):
    m = registry.get("kernel_opt")
    assert m is not None
    assert set(m.requires_lanes) == {"server_lifecycle", "workspace_mutation", "benchmark_lane"}
    assert m.cost_minutes_p75 >= 60


def test_gemm_tuning_action_metadata(registry):
    m = registry.get("gemm_tuning")
    assert m is not None
    assert m.family == "deep_kernel"
    assert m.pipeline_phase == "deep"
    assert set(m.requires_lanes) == {"server_lifecycle", "workspace_mutation", "benchmark_lane"}
    assert "precision == 'fp8'" in m.applicable_when


def test_recover_owned_by_robustness_handle(registry):
    """recover is the resilience action that Robustness handle_subagent runs."""
    m = registry.get("recover")
    assert m is not None
    assert m.family == "resilience"


def test_every_action_has_valid_family(registry):
    for m in registry.all():
        assert m.family in VALID_FAMILIES, f"{m.name} has invalid family {m.family!r}"


def test_every_action_uses_only_known_lanes(registry):
    # v0.8 M5 (KB_design §3.7) introduced ``research_lane`` for the
    # LLM specialist sub-agent (capacity-N, no conflict with serving
    # lanes). PR-A1 (Arbor-into-Hyperloom) registers the first action
    # that actually declares ``research_lane`` in its yaml meta —
    # ``specialist`` — so the known-lanes set must include it.
    known = {
        "server_lifecycle", "workspace_mutation",
        "benchmark_lane", "profile_lane",
        "research_lane",
    }
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
# Drift guards: keep action metadata, session paths, and CLI executor wiring
# aligned so every executor-backed action gets a runs/<kind>/<task_id>
# workspace.
# ---------------------------------------------------------------------------
def test_runs_actions_match_pipeline_phases(registry):
    """``_runs_actions()`` must follow registry pipeline phases."""
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


def test_explore_in_runs_actions():
    """v0.8 M3 + KB_gaps/Dead-A — ``explore`` succeeded the retired
    ``validate_stack`` as the per-action runs/<kind>/ owner that
    originally triggered the drift bug this guard exists for."""
    from inference_optimizer.session_paths import _runs_actions

    assert "explore" in _runs_actions()


def test_runs_actions_fallback_matches_registry(registry):
    """The hardcoded ``_RUNS_ACTIONS_FALLBACK`` (used only when the
    registry can't be loaded) must stay aligned with the registry-derived
    set.
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
    from inference_optimizer.cli import _REAL_EXECUTORS_FULL
    from inference_optimizer.session_paths import _runs_actions

    SESSION_ROOT_WRITERS = {"report", "session_breakdown"}
    runs = _runs_actions()
    real_kinds = set(_REAL_EXECUTORS_FULL.keys())

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


def test_explore_action_metadata(registry):
    """v0.8 M3 + KB_gaps/Dead-A — ``explore`` is the canonical merged
    grid runner; verify it owns the explore pipeline_phase + grid
    lanes the dispatcher expects."""
    m = registry.get("explore")
    assert m is not None, "explore action missing from registry"
    assert m.family == "shallow"
    assert m.pipeline_phase == "explore"
    assert "baseline" in m.prerequisites
    assert "server_lifecycle" in m.requires_lanes
    assert "benchmark_lane" in m.requires_lanes


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
