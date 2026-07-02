# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for framework_agent.enablement_authoring (mandate builder)."""

from __future__ import annotations

from framework_agent.enablement import EnablementRequest
from framework_agent.enablement_authoring import (
    ENABLEMENT_PATCH_INVARIANTS,
    build_mandate,
)


def _req(log: str = "") -> EnablementRequest:
    return EnablementRequest.from_dict(
        {
            "framework": "sglang",
            "model": "zai-org/GLM-5",
            "repo_url": "https://github.com/sgl-project/sglang.git",
            "launch_log": log or "ValueError: Model architecture 'Glm5ForCausalLM' is not supported",
        }
    )


def test_mandate_always_lists_framework_and_rocm_hip_roots() -> None:
    """Default-on: both the framework and ROCm/HIP source roots are in scope."""
    mandate = build_mandate(_req())
    assert any("serving-framework" in h for h in mandate.allowed_root_hints)
    assert any("ROCm" in h for h in mandate.allowed_root_hints)


def test_mandate_rocm_hip_root_present_for_hip_failure() -> None:
    """A HIP failure still carries the ROCm/HIP root family (always allowed)."""
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
