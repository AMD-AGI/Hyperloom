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
    # shallow (5) — v0.8 M3 + KB_gaps/Dead-A merged the v0.6
    # ``backends`` / ``params`` / ``validate_stack`` actions into
    # ``explore``; their yamls + executors were physically deleted.
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
    # creative (3) — PR-A1: specialist LLM sub-agent dispatch;
    # IR-7 (Saturday May 2026): assess_remaining_gaps is a thin
    # wrapper that dispatches the session_steward_specialist domain;
    # dynamic_action.MD P1: cross-domain multi-turn ReAct sub-agent
    # dispatch (supplementary EXPLORE channel).
    "specialist":           "creative",
    "assess_remaining_gaps": "creative",
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

# v0.8 KB_gaps/Gap-13 + Dead-A — actions removed in KB_design §3.15 §2.3
# / §3.4. ``dream`` / ``re_explore`` / ``comm_optimization`` /
# ``compiler_tuning`` are replaced by specialist sub-agents;
# ``backends`` / ``params`` / ``validate_stack`` are merged into
# ``explore``. ``pmc_roofline`` is superseded by the F1 composite
# ``roofline`` action (rocprof-based PMC gathering retired together
# with the ``roofline_integration`` / ``pmc_workload_params`` modules).
# All eight yamls were physically deleted; a future regression
# re-introducing any of them fails loudly here.
_REMOVED_LEGACY_ACTIONS: tuple[str, ...] = (
    "backends",
    "comm_optimization",
    "compiler_tuning",
    "dream",
    "params",
    "pmc_roofline",
    "re_explore",
    "validate_stack",
)


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


@pytest.mark.parametrize("name", _REMOVED_LEGACY_ACTIONS)
def test_removed_legacy_actions_not_in_registry(registry, name):
    """v0.8 KB_gaps/Gap-13 — KB_design §3.15 §2.3 retired
    ``dream`` / ``re_explore`` / ``comm_optimization`` /
    ``compiler_tuning`` in favour of specialist sub-agents.
    The yaml meta files were deleted; a regression that re-adds
    one of them must fail loudly here before it can leak into the
    Orchestration prompt catalogue."""
    assert registry.get(name) is None, (
        f"{name!r} is removed in v0.8; restore via specialist domain instead"
    )


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
# Drift guards: keep the four "is this a real action?" sources of truth
# consistent with each other.
#
# Background: ``session_paths._runs_actions()`` (the whitelist for actions
# that get a per-task ``runs/<kind>/<task_id>/`` workspace) used to be a
# hand-maintained ``_RUNS_ACTIONS`` frozenset. Adding a new action
# (``explore``) without updating it caused the orchestrator to loop
# forever proposing the action — every dispatch raised ``ValueError`` from
# inside the executor's ``runs_dir()`` fallback, but mission TODOs never
# cleared. The fix derives the whitelist from ``pipeline_phase`` in the
# ActionRegistry; these tests lock the alignment in place.
# ---------------------------------------------------------------------------
# v0.8 M5 (KB_design §3.5 §10) — ``specialist`` is the LLM sub-agent
# action. It's parameterised by ``params.domain`` rather than a yaml
# meta, so the registry-derived runs_actions set won't list it but it
# still needs a per-task workspace (``runs/specialist/<task_id>/``).
# We add it explicitly to the registry-derived expected set in the
# drift tests below.
_SPECIALIST_RUNS_ACTION_NAME = "specialist"


def test_runs_actions_match_pipeline_phases(registry):
    """``_runs_actions()`` must equal {a.name for a in registry
    if a.pipeline_phase ∈ _RUNS_WORKSPACE_PHASES} ∪ {'specialist'}.

    This is the primary registry ↔ session_paths drift guard. The
    ``specialist`` exception covers the yaml-less M5 action surface
    (KB_design §3.5 §10).
    """
    from inference_optimizer.session_paths import (
        _RUNS_WORKSPACE_PHASES,
        _runs_actions,
    )

    expected = frozenset(
        a.name for a in registry.all()
        if a.pipeline_phase in _RUNS_WORKSPACE_PHASES
    ) | frozenset({_SPECIALIST_RUNS_ACTION_NAME})
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
    set ∪ {'specialist'}. The yaml-less ``specialist`` (KB_design §3.5
    §10) is the one well-known exception.
    """
    from inference_optimizer.session_paths import (
        _RUNS_ACTIONS_FALLBACK,
        _RUNS_WORKSPACE_PHASES,
    )

    expected = frozenset(
        a.name for a in registry.all()
        if a.pipeline_phase in _RUNS_WORKSPACE_PHASES
    ) | frozenset({_SPECIALIST_RUNS_ACTION_NAME})
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
