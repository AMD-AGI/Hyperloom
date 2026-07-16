# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for framework_agent.enablement_ops (discovery + authoring).

Folds the former ``test_enablement_authoring`` (mandate builder) and
``test_enablement_discovery`` (search plan + ranking) suites now that both
halves live in the single :mod:`hyperloom.agents.framework.enablement_ops`
module.
"""

from __future__ import annotations

from hyperloom.agents.framework.enablement import EnablementRequest, classify_failure
from hyperloom.agents.framework.enablement_ops import (
    ENABLEMENT_PATCH_INVARIANTS,
    ENABLEMENT_SETUP_GUIDANCE,
    build_mandate,
    build_search_plan,
    rank_titles,
    score_enablement_title,
)


_SGLANG = "https://github.com/sgl-project/sglang.git"


def _req(log: str = "") -> EnablementRequest:
    return EnablementRequest.from_dict(
        {
            "framework": "sglang",
            "model": "zai-org/GLM-5",
            "repo_url": "https://github.com/sgl-project/sglang.git",
            "launch_log": log or "ValueError: Model architecture 'Glm5ForCausalLM' is not supported",
        }
    )


# --- authoring: build_mandate ----------------------------------------------


def test_mandate_always_lists_framework_and_rocm_hip_roots() -> None:
    """Default-on: both the framework and ROCm/HIP source roots are in scope."""
    mandate = build_mandate(_req())
    assert any("serving-framework" in h for h in mandate.allowed_root_hints)
    assert any("ROCm" in h for h in mandate.allowed_root_hints)


def test_mandate_rocm_hip_root_present_for_hip_failure() -> None:
    """A HIP failure still carries the ROCm/HIP source root family (always allowed)."""
    mandate = build_mandate(_req(log="RuntimeError: hipErrorNoBinaryForGpu"))
    assert any("ROCm" in h for h in mandate.allowed_root_hints)


def test_task_description_carries_failure_context() -> None:
    """The rendered mandate embeds model, failure class, and symbol."""
    mandate = build_mandate(_req())
    td = mandate.task_description
    assert "GLM-5" in td
    assert "missing_model_arch" in td
    assert "Glm5ForCausalLM" in td


def test_task_description_lists_candidate_refs() -> None:
    """Provided candidate refs appear in the mandate, best-first order preserved."""
    mandate = build_mandate(_req(), candidate_refs=("PR:123", "PR:456"))
    td = mandate.task_description
    assert td.index("PR:123") < td.index("PR:456")
    assert mandate.candidate_refs == ("PR:123", "PR:456")


def test_mandate_authorizes_and_requires_recording_env_setup() -> None:
    """Q3: the mandate authorizes installs AND tells the specialist to record them."""
    td = build_mandate(_req()).task_description
    assert "ENVIRONMENT SETUP" in td
    assert "pip install" in td
    assert "setup_commands" in td
    assert ENABLEMENT_SETUP_GUIDANCE
    assert any("record" in g.lower() for g in ENABLEMENT_SETUP_GUIDANCE)


def test_invariants_present_and_mention_runnable_gate() -> None:
    """Every mandate carries the invariants; runnability (not perf) is explicit."""
    mandate = build_mandate(_req())
    assert mandate.invariants == ENABLEMENT_PATCH_INVARIANTS
    assert any("RUNNABILITY" in inv for inv in mandate.invariants)
    assert any("git apply --check" in inv for inv in mandate.invariants)


def test_empty_candidate_refs_filtered() -> None:
    """Blank refs are dropped."""
    mandate = build_mandate(_req(), candidate_refs=("", "PR:7", ""))
    assert mandate.candidate_refs == ("PR:7",)


def test_source_context_rendered_when_provided() -> None:
    """A non-empty source_context is injected under a SOURCE CONTEXT block."""
    ctx = "  100| def resolve_arch(name):\n  101|     raise ValueError(name)"
    mandate = build_mandate(_req(), source_context=ctx)
    td = mandate.task_description
    assert "SOURCE CONTEXT (near offending site):" in td
    assert "raise ValueError(name)" in td


def test_source_context_omitted_when_empty() -> None:
    """No SOURCE CONTEXT block is rendered when context is empty/blank."""
    mandate = build_mandate(_req(), source_context="   ")
    assert "SOURCE CONTEXT" not in mandate.task_description


# --- discovery: build_search_plan + ranking --------------------------------


def test_plan_hip_failure_unions_bridge_repos() -> None:
    """A HIP failure always unions the ROCm/HIP/aiter bridge repos (default-on)."""
    sig = classify_failure("RuntimeError: hipErrorNoBinaryForGpu: no kernel image is available")
    plan = build_search_plan(sig, framework_repo_url=_SGLANG)
    assert plan.repos[0] == _SGLANG
    assert "https://github.com/ROCm/aiter.git" in plan.repos
    assert "https://github.com/ROCm/HIP.git" in plan.repos


def test_plan_framework_layer_has_no_bridge_repos() -> None:
    """A framework-layer failure adds no bridge repos (bridge_layer == 'framework' -> ())."""
    sig = classify_failure("ValueError: Model architecture 'FooForCausalLM' is not supported")
    plan = build_search_plan(sig, framework_repo_url=_SGLANG)
    assert plan.repos == (_SGLANG,)


def test_plan_keywords_include_arch_tokens_and_model() -> None:
    """Offending arch name is tokenized (CamelCase) into ranking keywords."""
    sig = classify_failure("ValueError: Model architecture 'Glm5ForCausalLM' is not supported")
    plan = build_search_plan(sig, framework_repo_url=_SGLANG, model="zai-org/GLM-5")
    assert "glm" in plan.keywords
    assert "causal" in plan.keywords


def test_ranking_prefers_enablement_intent() -> None:
    """A title that 'adds support' outranks a generic perf title for the same arch."""
    sig = classify_failure("ValueError: Model architecture 'Glm5ForCausalLM' is not supported")
    plan = build_search_plan(sig, framework_repo_url=_SGLANG, model="GLM-5")
    enable = score_enablement_title("Add GLM support to model registry", plan)
    perf = score_enablement_title("Optimize GLM attention throughput", plan)
    assert enable > perf


def test_rank_titles_sorts_descending() -> None:
    """rank_titles returns (title, score) sorted best-first."""
    sig = classify_failure("ModuleNotFoundError: No module named 'aiter.ops'")
    plan = build_search_plan(sig, framework_repo_url=_SGLANG)
    ranked = rank_titles(
        [
            "Unrelated docs update",
            "Fix aiter build import error on ROCm",
        ],
        plan,
    )
    assert ranked[0][0] == "Fix aiter build import error on ROCm"
    assert ranked[0][1] >= ranked[1][1]


def test_empty_title_scores_zero() -> None:
    """An empty title scores 0.0 without raising."""
    sig = classify_failure("NotImplementedError: x")
    plan = build_search_plan(sig, framework_repo_url=_SGLANG)
    assert score_enablement_title("", plan) == 0.0
