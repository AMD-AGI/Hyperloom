# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for framework_agent.enablement_discovery (search plan + ranking)."""

from __future__ import annotations

from hyperloom.agents.framework.enablement import classify_failure
from hyperloom.agents.framework.enablement_discovery import (
    build_search_plan,
    score_enablement_title,
)


_SGLANG = "https://github.com/sgl-project/sglang.git"


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


def test_empty_title_scores_zero() -> None:
    """An empty title scores 0.0 without raising."""
    sig = classify_failure("NotImplementedError: x")
    plan = build_search_plan(sig, framework_repo_url=_SGLANG)
    assert score_enablement_title("", plan) == 0.0
